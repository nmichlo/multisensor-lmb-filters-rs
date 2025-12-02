//! Core traits for filters and components
//!
//! This module defines the trait hierarchy used by all filters.

use crate::association::{AssociationMatrices, PosteriorGrid};
use crate::types::{AssociationConfig, StateEstimate, Track};

use super::errors::{AssociationError, FilterError};

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
        // TODO: Call existing LBP implementation from common/association/lbp.rs
        // For now, return placeholder
        let _n = matrices.num_tracks();
        let _m = matrices.num_measurements();

        // Placeholder: return psi normalized as marginal weights
        let marginal_weights = matrices.psi.clone();
        let miss_weights = matrices.phi.clone();

        Ok(AssociationResult {
            marginal_weights,
            miss_weights,
            sampled_associations: None,
            iterations: config.lbp_max_iterations,
            converged: true,
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
        _rng: &mut R,
    ) -> Result<AssociationResult, AssociationError> {
        // TODO: Call existing Gibbs implementation from common/association/gibbs.rs
        let _n = matrices.num_tracks();
        let _m = matrices.num_measurements();

        let marginal_weights = matrices.psi.clone();
        let miss_weights = matrices.phi.clone();

        Ok(AssociationResult {
            marginal_weights,
            miss_weights,
            sampled_associations: Some(Vec::new()), // Placeholder
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
        // TODO: Call existing Murty implementation from common/association/murtys.rs
        let _n = matrices.num_tracks();
        let _m = matrices.num_measurements();

        let marginal_weights = matrices.psi.clone();
        let miss_weights = matrices.phi.clone();

        Ok(AssociationResult {
            marginal_weights,
            miss_weights,
            sampled_associations: Some(Vec::new()),
            iterations: config.murty_assignments,
            converged: true,
        })
    }

    fn name(&self) -> &'static str {
        "Murty"
    }
}

/// LMB marginal update strategy
#[derive(Debug, Clone, Default)]
pub struct MarginalUpdater;

impl Updater for MarginalUpdater {
    fn update(
        &self,
        _tracks: &mut [Track],
        _result: &AssociationResult,
        _posteriors: &PosteriorGrid,
    ) {
        // TODO: Implement marginal reweighting for LMB
        // For now, placeholder
    }

    fn name(&self) -> &'static str {
        "Marginal"
    }
}

/// LMBM hard assignment update strategy
#[derive(Debug, Clone, Default)]
pub struct HardAssignmentUpdater;

impl Updater for HardAssignmentUpdater {
    fn update(
        &self,
        _tracks: &mut [Track],
        _result: &AssociationResult,
        _posteriors: &PosteriorGrid,
    ) {
        // TODO: Implement hard assignment for LMBM
        // For now, placeholder
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
        assert_eq!(MarginalUpdater.name(), "Marginal");
        assert_eq!(HardAssignmentUpdater.name(), "HardAssignment");
    }
}
