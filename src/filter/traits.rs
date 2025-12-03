//! Core traits for filters and components
//!
//! This module defines the trait hierarchy for the modular filter architecture.
//! The design separates concerns into distinct traits:
//!
//! - [`Filter`] - Main filter interface for running tracking algorithms
//! - [`Associator`] - Data association algorithms (LBP, Gibbs, Murty)
//! - [`Updater`] - Track update strategies (marginal vs hard assignment)
//! - [`Merger`] - Multi-sensor track fusion strategies
//!
//! This trait-based design allows mixing and matching components. For example,
//! an LMB filter can use LBP or Gibbs for association, while LMBM uses Murty
//! with hard assignment updates.

use crate::association::{AssociationMatrices, PosteriorGrid};
use crate::common::association::gibbs as legacy_gibbs;
use crate::common::association::lbp as legacy_lbp;
use crate::common::association::murtys as legacy_murtys;
use crate::common::rng as legacy_rng;
use crate::types::{AssociationConfig, StateEstimate, Track};

use super::errors::{AssociationError, FilterError};

/// Adapter to bridge `rand::Rng` to the legacy `common::rng::Rng` trait.
///
/// The legacy association implementations use a custom RNG trait for
/// deterministic cross-language testing with MATLAB. This adapter allows
/// using standard `rand::Rng` implementations with those functions.
struct RngAdapter<'a, R: rand::Rng>(&'a mut R);

impl<'a, R: rand::Rng> legacy_rng::Rng for RngAdapter<'a, R> {
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
}

/// Core filter trait implemented by all tracking filters
///
/// This provides a unified interface for running filters regardless
/// of the underlying algorithm (LMB, LMBM, multi-sensor variants).
///
/// # Type Parameters
/// - `State` - Internal state representation (e.g., `Vec<Track>` or `Vec<LmbmHypothesis>`)
pub trait Filter {
    /// Type of internal filter state
    type State;

    /// Type of measurements expected (single or multi-sensor)
    type Measurements;

    /// Process one timestep and return state estimates
    ///
    /// # Arguments
    /// * `rng` - Random number generator for stochastic algorithms
    /// * `measurements` - Measurements at this timestep
    /// * `timestep` - Current timestep index
    ///
    /// # Returns
    /// State estimates or error
    fn step<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError>;

    /// Get current internal state (read-only)
    fn state(&self) -> &Self::State;

    /// Reset filter to initial state
    fn reset(&mut self);

    /// Get state dimension
    fn x_dim(&self) -> usize;

    /// Get measurement dimension
    fn z_dim(&self) -> usize;
}

/// Result of data association containing marginal probabilities and optional samples.
///
/// Data association solves the measurement-to-track assignment problem. Given n tracks
/// and m measurements, we need to determine which measurements originated from which
/// tracks. This is inherently ambiguous, so we compute probabilities.
///
/// The result contains:
/// - **Marginal weights**: P(track i associated with measurement j) for each (i,j) pair
/// - **Miss weights**: P(track i not detected) for each track
/// - **Sampled associations**: For LMBM, discrete association events sampled from the posterior
///
/// Different algorithms produce these in different ways:
/// - LBP: Iterative message passing converges to marginal probabilities
/// - Gibbs: MCMC sampling estimates marginals from sample frequencies
/// - Murty: K-best assignments weighted by likelihood give approximate marginals
#[derive(Debug, Clone)]
pub struct AssociationResult {
    /// Marginal association probabilities matrix (n_tracks × n_measurements).
    /// Entry `[i,j]` is P(track i generated measurement j).
    pub marginal_weights: nalgebra::DMatrix<f64>,

    /// Miss (non-detection) probabilities vector (n_tracks).
    /// Entry `[i]` is P(track i exists but was not detected).
    pub miss_weights: nalgebra::DVector<f64>,

    /// Discrete association event samples for hypothesis-based filters (LMBM).
    /// Each inner Vec represents one sampled association: `v[i]` is the measurement
    /// index assigned to track i, or -1 if track i missed detection.
    pub sampled_associations: Option<Vec<Vec<i32>>>,

    /// Number of iterations (LBP) or samples (Gibbs/Murty) used.
    pub iterations: usize,

    /// Whether the algorithm converged (meaningful for LBP).
    pub converged: bool,
}

impl AssociationResult {
    /// Create a new association result
    pub fn new(
        marginal_weights: nalgebra::DMatrix<f64>,
        miss_weights: nalgebra::DVector<f64>,
    ) -> Self {
        Self {
            marginal_weights,
            miss_weights,
            sampled_associations: None,
            iterations: 0,
            converged: true,
        }
    }

    /// Get the most likely association for a track (MAP estimate).
    ///
    /// Returns `Some(j)` if measurement j has higher probability than missing,
    /// or `None` if the track most likely missed detection.
    pub fn best_association(&self, track_idx: usize) -> Option<usize> {
        let n_meas = self.marginal_weights.ncols();
        let mut best_j = None;
        let mut best_weight = self.miss_weights[track_idx];

        for j in 0..n_meas {
            let w = self.marginal_weights[(track_idx, j)];
            if w > best_weight {
                best_weight = w;
                best_j = Some(j);
            }
        }

        best_j
    }
}

/// Data association algorithm trait.
///
/// Data association computes the probability that each measurement originated
/// from each track. This is the core computational problem in multi-object tracking,
/// as the true associations are unknown and must be inferred probabilistically.
///
/// The trait is implemented by different algorithms with different trade-offs:
/// - [`LbpAssociator`] - Loopy Belief Propagation: fast, approximate message passing
/// - [`GibbsAssociator`] - Gibbs sampling: MCMC-based, produces samples for LMBM
/// - [`MurtyAssociator`] - Murty's k-best: exact top-k assignments, good for LMBM
///
/// All implementations are `Send + Sync` to support parallel filter execution.
pub trait Associator: Send + Sync {
    /// Compute association probabilities from pre-computed likelihood matrices.
    ///
    /// The input `matrices` contains likelihood ratios and related quantities
    /// computed from track predictions and measurements. The output contains
    /// marginal association probabilities and optionally discrete samples.
    fn associate<R: rand::Rng>(
        &self,
        matrices: &AssociationMatrices,
        config: &AssociationConfig,
        rng: &mut R,
    ) -> Result<AssociationResult, AssociationError>;

    /// Algorithm name for logging and debugging.
    fn name(&self) -> &'static str;
}

/// Multi-sensor track fusion strategy.
///
/// In multi-sensor tracking, each sensor produces independent track posteriors.
/// The merger combines these into a single unified track set. Different fusion
/// strategies have different properties:
///
/// - `ArithmeticAverageMerger` (AA-LMB): Simple weighted average of densities
/// - `GeometricAverageMerger` (GA-LMB): Geometric mean, handles sensor disagreement better
/// - `ParallelUpdateMerger` (PU-LMB): Sequential Bayesian updates
/// - `IteratedCorrectorMerger` (IC-LMB): Iterative refinement for consistency
///
/// The choice of merger affects tracking accuracy and computational cost.
pub trait Merger: Send + Sync {
    /// Fuse per-sensor track posteriors into a unified track set.
    fn merge(&self, per_sensor_tracks: &[Vec<Track>], weights: Option<&[f64]>) -> Vec<Track>;

    /// Merger name for logging and debugging.
    fn name(&self) -> &'static str;
}

/// Track update strategy after data association.
///
/// Once association probabilities are computed, tracks must be updated with
/// measurement information. The update strategy differs fundamentally between
/// filter types:
///
/// - [`MarginalUpdater`] (LMB): Maintains Gaussian mixture by reweighting components
///   according to marginal association probabilities. Each track becomes a weighted
///   mixture of "missed" and "detected with measurement j" hypotheses.
///
/// - [`HardAssignmentUpdater`] (LMBM): Makes hard (deterministic) assignments from
///   sampled association events. Each track gets a single component based on its
///   assigned measurement. Used with hypothesis-based filters.
///
/// The updater modifies tracks in-place, updating their Gaussian components
/// based on the association result and pre-computed posterior parameters.
pub trait Updater: Send + Sync {
    /// Apply measurement update to tracks based on association results.
    ///
    /// Modifies tracks in-place, updating their Gaussian mixture components
    /// using the posterior parameters for each (track, measurement) pair.
    fn update(
        &self,
        tracks: &mut [Track],
        result: &AssociationResult,
        posteriors: &PosteriorGrid,
    );

    /// Updater name for logging and debugging.
    fn name(&self) -> &'static str;
}

// ============================================================================
// Associator implementations
// ============================================================================

/// Loopy Belief Propagation (LBP) data association.
///
/// LBP treats the association problem as inference on a factor graph and uses
/// iterative message passing to compute approximate marginal probabilities.
/// It's the fastest association method and works well when associations are
/// relatively unambiguous.
///
/// The algorithm iterates until convergence (messages change less than tolerance)
/// or reaches the maximum iteration count. Convergence is typically fast (10-50
/// iterations) for well-separated tracks.
///
/// LBP is deterministic and does not produce discrete samples, making it suitable
/// for LMB filters but not for LMBM which requires sampled association events.
#[derive(Debug, Clone, Default)]
pub struct LbpAssociator;

impl Associator for LbpAssociator {
    fn associate<R: rand::Rng>(
        &self,
        matrices: &AssociationMatrices,
        config: &AssociationConfig,
        _rng: &mut R,
    ) -> Result<AssociationResult, AssociationError> {
        // Convert to legacy format
        let legacy_matrices = legacy_lbp::AssociationMatrices {
            psi: matrices.psi.clone(),
            phi: matrices.phi.clone(),
            eta: matrices.eta.clone(),
        };

        // Call legacy LBP implementation
        let lbp_result = legacy_lbp::loopy_belief_propagation(
            &legacy_matrices,
            config.lbp_tolerance,
            config.lbp_max_iterations,
        );

        // Convert result: legacy W is (n x (m+1)) where first col is miss probability
        // Our format separates miss weights from marginal weights
        let n = lbp_result.w.nrows();
        let m_plus_1 = lbp_result.w.ncols();

        // Extract miss probabilities (first column)
        let miss_weights = lbp_result.w.column(0).into_owned();

        // Extract marginal weights (remaining columns)
        let marginal_weights = if m_plus_1 > 1 {
            lbp_result.w.columns(1, m_plus_1 - 1).into_owned()
        } else {
            nalgebra::DMatrix::zeros(n, 0)
        };

        Ok(AssociationResult {
            marginal_weights,
            miss_weights,
            sampled_associations: None,
            iterations: config.lbp_max_iterations,
            converged: true, // LBP with convergence check always converges or hits max iters
        })
    }

    fn name(&self) -> &'static str {
        "LBP"
    }
}

/// Gibbs sampling data association.
///
/// Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method that generates
/// samples from the posterior distribution over association events. Unlike LBP
/// which computes marginals directly, Gibbs produces discrete samples that can
/// be used to approximate marginals and to generate hypothesis sets for LMBM.
///
/// The algorithm works by iteratively sampling each track's association conditioned
/// on all other tracks' current associations. After a burn-in period, the samples
/// approximate the true posterior distribution.
///
/// Gibbs is stochastic (requires RNG) and slower than LBP, but produces the
/// discrete samples needed for LMBM filters. The quality of approximation improves
/// with more samples but at increased computational cost.
#[derive(Debug, Clone, Default)]
pub struct GibbsAssociator;

impl Associator for GibbsAssociator {
    fn associate<R: rand::Rng>(
        &self,
        matrices: &AssociationMatrices,
        config: &AssociationConfig,
        rng: &mut R,
    ) -> Result<AssociationResult, AssociationError> {
        let n = matrices.num_tracks();
        let m = matrices.num_measurements();

        if n == 0 || m == 0 {
            return Ok(AssociationResult {
                marginal_weights: nalgebra::DMatrix::zeros(n, m),
                miss_weights: nalgebra::DVector::zeros(n),
                sampled_associations: Some(Vec::new()),
                iterations: 0,
                converged: true,
            });
        }

        // Convert to legacy Gibbs format
        // P: sampling probabilities (n x m) - use sampling_prob without miss column
        let p = matrices.sampling_prob.columns(0, m).into_owned();

        // L: likelihood matrix (n x (m+1)) with [eta, L1, L2, ...]
        // We use eta column followed by psi values
        let mut l = nalgebra::DMatrix::zeros(n, m + 1);
        for i in 0..n {
            l[(i, 0)] = matrices.eta[i];
            for j in 0..m {
                // psi contains L / eta, so L = psi * eta
                l[(i, j + 1)] = matrices.psi[(i, j)] * matrices.eta[i];
            }
        }

        // R: existence ratio matrix (n x (m+1)) with [phi/eta, 1, 1, ...]
        let mut r_mat = nalgebra::DMatrix::from_element(n, m + 1, 1.0);
        for i in 0..n {
            if matrices.eta[i].abs() > super::NUMERICAL_ZERO {
                r_mat[(i, 0)] = matrices.phi[i] / matrices.eta[i];
            } else {
                r_mat[(i, 0)] = 0.0;
            }
        }

        // C: cost matrix (n x m) - use negative log-likelihood
        let c = matrices.cost.clone();

        let gibbs_matrices = legacy_gibbs::GibbsAssociationMatrices {
            p,
            l,
            r: r_mat,
            c,
        };

        // Wrap RNG for legacy interface
        let mut rng_adapter = RngAdapter(rng);

        // Call legacy Gibbs sampling
        let gibbs_result =
            legacy_gibbs::lmb_gibbs_sampling(&mut rng_adapter, &gibbs_matrices, config.gibbs_samples);

        // Convert result: legacy W is (n x (m+1)) where first col is miss probability
        let miss_weights = gibbs_result.w.column(0).into_owned();
        let marginal_weights = if m > 0 {
            gibbs_result.w.columns(1, m).into_owned()
        } else {
            nalgebra::DMatrix::zeros(n, 0)
        };

        // Convert sampled associations to our format
        let sampled_associations: Vec<Vec<i32>> = (0..gibbs_result.v_samples.nrows())
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let val = gibbs_result.v_samples[(i, j)];
                        if val == 0 {
                            -1 // miss
                        } else {
                            (val - 1) as i32 // 0-indexed measurement
                        }
                    })
                    .collect()
            })
            .collect();

        Ok(AssociationResult {
            marginal_weights,
            miss_weights,
            sampled_associations: Some(sampled_associations),
            iterations: config.gibbs_samples,
            converged: true,
        })
    }

    fn name(&self) -> &'static str {
        "Gibbs"
    }
}

/// Murty's k-best assignment algorithm for data association.
///
/// Murty's algorithm finds the k most likely (lowest cost) assignments between
/// tracks and measurements. Unlike LBP which approximates marginals, Murty finds
/// exact solutions ranked by likelihood.
///
/// The algorithm works by solving the optimal assignment (Hungarian algorithm),
/// then systematically partitioning the solution space to find the next best
/// assignments. This produces discrete association events with known likelihoods.
///
/// Murty is deterministic and produces ranked assignments suitable for LMBM.
/// The marginal probabilities are computed by weighting each assignment by its
/// likelihood. Computational cost grows with k, but typically k=100-1000 suffices.
#[derive(Debug, Clone, Default)]
pub struct MurtyAssociator;

impl Associator for MurtyAssociator {
    fn associate<R: rand::Rng>(
        &self,
        matrices: &AssociationMatrices,
        config: &AssociationConfig,
        _rng: &mut R,
    ) -> Result<AssociationResult, AssociationError> {
        let n = matrices.num_tracks();
        let m = matrices.num_measurements();

        if n == 0 || m == 0 {
            return Ok(AssociationResult {
                marginal_weights: nalgebra::DMatrix::zeros(n, m),
                miss_weights: nalgebra::DVector::zeros(n),
                sampled_associations: Some(Vec::new()),
                iterations: 0,
                converged: true,
            });
        }

        // Call legacy Murty's algorithm with cost matrix
        let murty_result =
            legacy_murtys::murtys_algorithm_wrapper(&matrices.cost, config.murty_assignments);

        if murty_result.assignments.nrows() == 0 {
            return Err(AssociationError::NoValidAssignments);
        }

        // Convert assignments to marginal probabilities
        // Each assignment has a cost; compute weights as exp(-cost)
        let num_assignments = murty_result.assignments.nrows();
        let mut assignment_weights: Vec<f64> = murty_result
            .costs
            .iter()
            .map(|&c| (-c).exp())
            .collect();

        // Normalize weights
        let total_weight: f64 = assignment_weights.iter().sum();
        if total_weight > super::NUMERICAL_ZERO {
            for w in &mut assignment_weights {
                *w /= total_weight;
            }
        } else {
            // Uniform weights if all costs are very high
            for w in &mut assignment_weights {
                *w = 1.0 / num_assignments as f64;
            }
        }

        // Compute marginal probabilities from weighted assignments
        let mut marginal_weights = nalgebra::DMatrix::zeros(n, m);
        let mut miss_weights = nalgebra::DVector::zeros(n);

        for (k, assignment_weight) in assignment_weights.iter().enumerate() {
            for i in 0..n {
                let assigned_meas = murty_result.assignments[(k, i)];
                if assigned_meas == 0 {
                    // Miss/clutter
                    miss_weights[i] += assignment_weight;
                } else {
                    // Assigned to measurement (1-indexed in Murty result)
                    let j = assigned_meas - 1;
                    if j < m {
                        marginal_weights[(i, j)] += assignment_weight;
                    } else {
                        // Dummy assignment (clutter)
                        miss_weights[i] += assignment_weight;
                    }
                }
            }
        }

        // Convert assignments to sampled_associations format
        let sampled_associations: Vec<Vec<i32>> = (0..num_assignments)
            .map(|k| {
                (0..n)
                    .map(|i| {
                        let val = murty_result.assignments[(k, i)];
                        if val == 0 || val > m {
                            -1 // miss
                        } else {
                            (val - 1) as i32 // 0-indexed measurement
                        }
                    })
                    .collect()
            })
            .collect();

        Ok(AssociationResult {
            marginal_weights,
            miss_weights,
            sampled_associations: Some(sampled_associations),
            iterations: num_assignments,
            converged: true,
        })
    }

    fn name(&self) -> &'static str {
        "Murty"
    }
}

/// LMB marginal update strategy using soft (probabilistic) associations.
///
/// This updater implements the standard LMB measurement update, which maintains
/// uncertainty about associations by creating a Gaussian mixture. For each track,
/// the posterior is a weighted mixture of:
///
/// 1. **Miss hypothesis**: The track was not detected. Uses prior state with
///    weight proportional to miss probability.
///
/// 2. **Detection hypotheses**: The track generated measurement j. Uses Kalman-
///    updated posterior with weight proportional to marginal association probability.
///
/// The resulting mixture grows as (m+1) × prior_components, so pruning is applied:
/// - Components below `weight_threshold` are discarded
/// - Only top `max_components` are kept
/// - Remaining weights are renormalized
///
/// This approach preserves uncertainty about associations, which is important when
/// measurements are ambiguous. However, it can lead to component explosion in
/// dense scenarios.
#[derive(Debug, Clone, Default)]
pub struct MarginalUpdater {
    /// Components with weight below this threshold are pruned.
    pub weight_threshold: f64,
    /// Maximum number of Gaussian components to retain per track.
    pub max_components: usize,
}

impl MarginalUpdater {
    /// Create a new marginal updater with default settings
    pub fn new() -> Self {
        Self {
            weight_threshold: 1e-4,
            max_components: 100,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(weight_threshold: f64, max_components: usize) -> Self {
        Self {
            weight_threshold,
            max_components,
        }
    }
}

impl Updater for MarginalUpdater {
    fn update(
        &self,
        tracks: &mut [Track],
        result: &AssociationResult,
        posteriors: &PosteriorGrid,
    ) {
        let n = tracks.len();
        let m = result.marginal_weights.ncols();

        for i in 0..n {
            let track = &mut tracks[i];
            let num_prior_components = track.components.len();

            // Build new components: for each (measurement, prior_component) pair
            // Total potential components = (m + 1) * num_prior_components
            // (m+1 because we include miss + m measurements)
            let mut new_weights = Vec::new();
            let mut new_means = Vec::new();
            let mut new_covs = Vec::new();

            // Miss case: prior components weighted by miss probability
            let miss_prob = result.miss_weights[i];
            for comp in track.components.iter() {
                new_weights.push(miss_prob * comp.weight);
                new_means.push(comp.mean.clone());
                new_covs.push(comp.covariance.clone());
            }

            // Measurement cases: posterior components weighted by marginal probabilities
            for j in 0..m {
                let meas_prob = result.marginal_weights[(i, j)];
                if meas_prob < super::NUMERICAL_ZERO {
                    continue; // Skip negligible associations
                }

                // For each prior component, create a posterior component
                for comp_idx in 0..num_prior_components {
                    if let (Some(post_mean), Some(post_cov)) = (
                        posteriors.get_mean(i, j),
                        posteriors.get_covariance(i, j),
                    ) {
                        let prior_weight = track.components[comp_idx].weight;
                        new_weights.push(meas_prob * prior_weight);
                        new_means.push(post_mean.clone());
                        new_covs.push(post_cov.clone());
                    }
                }
            }

            // Build weighted components for pruning
            let weighted_components: Vec<_> = new_weights
                .into_iter()
                .zip(new_means.into_iter())
                .zip(new_covs.into_iter())
                .map(|((w, m), c)| (w, m, c))
                .collect();

            // Prune, truncate, and normalize using shared helper
            track.components = super::common_ops::prune_weighted_components(
                weighted_components,
                self.weight_threshold,
                self.max_components,
            );
        }
    }

    fn name(&self) -> &'static str {
        "Marginal"
    }
}

/// LMBM hard assignment update strategy using discrete association events.
///
/// Unlike [`MarginalUpdater`] which maintains uncertainty through Gaussian mixtures,
/// this updater makes hard (deterministic) assignments. Each track is updated
/// based on a single association event, resulting in single-component posteriors.
///
/// This is the natural update for LMBM (Labeled Multi-Bernoulli Mixture) filters,
/// which maintain multiple weighted hypotheses. Each hypothesis represents one
/// possible "world state" with definite associations:
///
/// - **Detection**: Track existence is set to 1.0 (definitely exists) and state
///   is updated using the Kalman posterior for the assigned measurement.
///
/// - **Miss**: Track keeps its prior state. Existence probability is updated
///   according to missed detection likelihood.
///
/// The `sample_index` selects which association event from `sampled_associations`
/// to use. For LMBM, different hypotheses use different sample indices to
/// represent different association possibilities.
#[derive(Debug, Clone, Default)]
pub struct HardAssignmentUpdater {
    /// Which association sample to apply (0 = first/best sample).
    pub sample_index: usize,
}

impl HardAssignmentUpdater {
    /// Create a new hard assignment updater
    pub fn new() -> Self {
        Self { sample_index: 0 }
    }

    /// Create with specific sample index
    pub fn with_sample_index(sample_index: usize) -> Self {
        Self { sample_index }
    }
}

impl Updater for HardAssignmentUpdater {
    fn update(
        &self,
        tracks: &mut [Track],
        result: &AssociationResult,
        posteriors: &PosteriorGrid,
    ) {
        let n = tracks.len();

        // Get the association sample to use
        let assignments = match &result.sampled_associations {
            Some(samples) if !samples.is_empty() => {
                let idx = self.sample_index.min(samples.len() - 1);
                &samples[idx]
            }
            _ => {
                // Fall back to best association if no samples available
                // Apply best associations directly
                for i in 0..n {
                    let track = &mut tracks[i];
                    let meas_idx = result.best_association(i);

                    match meas_idx {
                        Some(j) => {
                            // Track associated with measurement j
                            if let (Some(post_mean), Some(post_cov)) = (
                                posteriors.get_mean(i, j),
                                posteriors.get_covariance(i, j),
                            ) {
                                // Replace all components with single posterior
                                track.components.clear();
                                track.components.push(crate::types::GaussianComponent::new(
                                    1.0,
                                    post_mean.clone(),
                                    post_cov.clone(),
                                ));
                                // Set existence to 1.0 for detected objects
                                track.existence = 1.0;
                            }
                        }
                        None => {
                            // Track missed - keep prior mean, prior covariance
                            // (components unchanged, just normalize weight)
                            if !track.components.is_empty() {
                                let total: f64 = track.components.iter().map(|c| c.weight).sum();
                                if total > super::NUMERICAL_ZERO {
                                    for c in track.components.iter_mut() {
                                        c.weight /= total;
                                    }
                                }
                            }
                        }
                    }
                }
                return;
            }
        };

        // Apply sampled assignments
        for i in 0..n {
            let track = &mut tracks[i];
            let assigned_meas = assignments.get(i).copied().unwrap_or(-1);

            if assigned_meas >= 0 {
                // Track associated with measurement
                let j = assigned_meas as usize;
                if let (Some(post_mean), Some(post_cov)) = (
                    posteriors.get_mean(i, j),
                    posteriors.get_covariance(i, j),
                ) {
                    // Replace all components with single posterior
                    track.components.clear();
                    track.components.push(crate::types::GaussianComponent::new(
                        1.0,
                        post_mean.clone(),
                        post_cov.clone(),
                    ));
                    // Set existence to 1.0 for detected objects
                    track.existence = 1.0;
                }
            } else {
                // Track missed - keep prior state
                // (components unchanged, just normalize weight)
                if !track.components.is_empty() {
                    let total: f64 = track.components.iter().map(|c| c.weight).sum();
                    if total > super::NUMERICAL_ZERO {
                        for c in track.components.iter_mut() {
                            c.weight /= total;
                        }
                    }
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "HardAssignment"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_association_result() {
        let weights = nalgebra::DMatrix::from_row_slice(2, 3, &[0.1, 0.8, 0.1, 0.2, 0.3, 0.5]);
        let miss = nalgebra::DVector::from_vec(vec![0.0, 0.0]);

        let result = AssociationResult::new(weights, miss);

        // Track 0 best association is measurement 1 (weight 0.8)
        assert_eq!(result.best_association(0), Some(1));
        // Track 1 best association is measurement 2 (weight 0.5)
        assert_eq!(result.best_association(1), Some(2));
    }

    #[test]
    fn test_associator_names() {
        assert_eq!(LbpAssociator.name(), "LBP");
        assert_eq!(GibbsAssociator.name(), "Gibbs");
        assert_eq!(MurtyAssociator.name(), "Murty");
    }

    #[test]
    fn test_updater_names() {
        assert_eq!(MarginalUpdater::new().name(), "Marginal");
        assert_eq!(HardAssignmentUpdater::new().name(), "HardAssignment");
    }
}
