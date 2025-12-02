//! Core traits for filters and components
//!
//! This module defines the trait hierarchy used by all filters.

use crate::association::{AssociationMatrices, PosteriorGrid};
use crate::common::association::gibbs as legacy_gibbs;
use crate::common::association::lbp as legacy_lbp;
use crate::common::association::murtys as legacy_murtys;
use crate::common::rng as legacy_rng;
use crate::types::{AssociationConfig, StateEstimate, Track};

use super::errors::{AssociationError, FilterError};

/// Adapter to use rand::Rng with legacy crate::common::rng::Rng trait
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

/// Result of data association
#[derive(Debug, Clone)]
pub struct AssociationResult {
    /// Marginal association probabilities: W[i,j] = P(track i associated with measurement j)
    pub marginal_weights: nalgebra::DMatrix<f64>,

    /// Marginal miss probabilities: miss_weights[i] = P(track i not associated)
    pub miss_weights: nalgebra::DVector<f64>,

    /// For LMBM: sampled association events
    /// Each inner Vec is one sample: v[i] = measurement index for track i (-1 for miss)
    pub sampled_associations: Option<Vec<Vec<i32>>>,

    /// Number of iterations/samples used
    pub iterations: usize,

    /// Whether the algorithm converged (for iterative methods)
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

    /// Get most likely association for a track
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

/// Data association algorithm trait
///
/// Implementations:
/// - `LbpAssociator` - Loopy Belief Propagation
/// - `GibbsAssociator` - Gibbs sampling
/// - `MurtyAssociator` - Murty's k-best algorithm
pub trait Associator: Send + Sync {
    /// Perform data association
    ///
    /// # Arguments
    /// * `matrices` - Pre-computed association matrices (likelihoods, etc.)
    /// * `config` - Association algorithm configuration
    /// * `rng` - Random number generator (for stochastic methods)
    ///
    /// # Returns
    /// Association result with marginal probabilities
    fn associate<R: rand::Rng>(
        &self,
        matrices: &AssociationMatrices,
        config: &AssociationConfig,
        rng: &mut R,
    ) -> Result<AssociationResult, AssociationError>;

    /// Get algorithm name
    fn name(&self) -> &'static str;
}

/// Multi-sensor track merging trait
///
/// Implementations:
/// - `ArithmeticAverageMerger` - AA-LMB
/// - `GeometricAverageMerger` - GA-LMB
/// - `ParallelUpdateMerger` - PU-LMB
/// - `IteratedCorrectorMerger` - IC-LMB
pub trait Merger: Send + Sync {
    /// Merge per-sensor track posteriors into unified tracks
    ///
    /// # Arguments
    /// * `per_sensor_tracks` - Tracks from each sensor after individual updates
    /// * `weights` - Optional sensor weights (for weighted averaging)
    ///
    /// # Returns
    /// Merged tracks
    fn merge(&self, per_sensor_tracks: &[Vec<Track>], weights: Option<&[f64]>) -> Vec<Track>;

    /// Get merger name
    fn name(&self) -> &'static str;
}

/// Track update strategy trait
///
/// Differs between LMB (marginal reweighting) and LMBM (hard selection).
///
/// Implementations:
/// - `MarginalUpdater` - LMB: reweight GM components by marginal probabilities
/// - `HardAssignmentUpdater` - LMBM: select based on sampled association events
pub trait Updater: Send + Sync {
    /// Update tracks based on association results
    ///
    /// # Arguments
    /// * `tracks` - Tracks to update (modified in place)
    /// * `result` - Association result with marginal/sampled assignments
    /// * `posteriors` - Pre-computed posterior parameters
    fn update(
        &self,
        tracks: &mut [Track],
        result: &AssociationResult,
        posteriors: &PosteriorGrid,
    );

    /// Get updater name
    fn name(&self) -> &'static str;
}

// ============================================================================
// Placeholder implementations (to be fleshed out later)
// ============================================================================

/// LBP (Loopy Belief Propagation) associator
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

/// Gibbs sampling associator
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
            if matrices.eta[i].abs() > 1e-15 {
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

/// Murty's k-best algorithm associator
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
        if total_weight > 1e-15 {
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

/// LMB marginal update strategy
///
/// Updates tracks using marginal association probabilities from LBP or similar.
/// Each track's GM components are reweighted by the marginal probabilities,
/// creating new components for each (prior component, measurement) pair.
#[derive(Debug, Clone, Default)]
pub struct MarginalUpdater {
    /// Weight threshold for pruning small components
    pub weight_threshold: f64,
    /// Maximum number of GM components to keep
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
                if meas_prob < 1e-15 {
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

            // Normalize weights
            let total_weight: f64 = new_weights.iter().sum();
            if total_weight > 1e-15 {
                for w in &mut new_weights {
                    *w /= total_weight;
                }
            }

            // Prune and keep top components
            let mut indexed: Vec<(usize, f64)> = new_weights
                .iter()
                .enumerate()
                .map(|(idx, &w)| (idx, w))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Keep components above threshold, up to max
            track.components.clear();
            let mut kept_weight = 0.0;
            for (orig_idx, weight) in indexed.into_iter() {
                if weight < self.weight_threshold {
                    break;
                }
                if track.components.len() >= self.max_components {
                    break;
                }
                track.components.push(crate::types::GaussianComponent::new(
                    weight,
                    new_means[orig_idx].clone(),
                    new_covs[orig_idx].clone(),
                ));
                kept_weight += weight;
            }

            // Renormalize kept components
            if kept_weight > 1e-15 && !track.components.is_empty() {
                for comp in track.components.iter_mut() {
                    comp.weight /= kept_weight;
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "Marginal"
    }
}

/// LMBM hard assignment update strategy
///
/// Updates tracks using hard (deterministic) assignment from sampled associations.
/// Unlike `MarginalUpdater`, this selects a single posterior for each track
/// based on the association event, resulting in single-component tracks.
///
/// This is used with LMBM which maintains multiple hypotheses, where each
/// hypothesis has tracks with single components.
#[derive(Debug, Clone, Default)]
pub struct HardAssignmentUpdater {
    /// Index of the association sample to use (for multi-hypothesis LMBM)
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
                                if total > 1e-15 {
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
                    if total > 1e-15 {
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
