//! Single-sensor LMBM (Labeled Multi-Bernoulli Mixture) filter.
//!
//! The LMBM filter is a multi-object tracking algorithm that maintains multiple
//! weighted hypotheses, where each hypothesis represents a possible "world state"
//! with definite (hard) data associations.
//!
//! Unlike the LMB filter which uses Gaussian mixtures to represent association
//! uncertainty within a single hypothesis, LMBM represents uncertainty through
//! a mixture of hypotheses, each with single-component tracks.
//!
//! Key differences from LMB:
//! - **Multiple hypotheses**: Maintains weighted set of possible association histories
//! - **Hard assignments**: Each hypothesis has deterministic track-to-measurement associations
//! - **Single-component tracks**: Each track in a hypothesis has exactly one Gaussian component
//! - **Hypothesis management**: Normalization, gating, merging, and pruning of hypotheses
//!
//! The filter uses Gibbs sampling or Murty's algorithm to generate association hypotheses,
//! then maintains the top-k most likely hypotheses over time.

use nalgebra::DVector;

use crate::association::AssociationBuilder;
use crate::types::{
    AssociationConfig, BirthModel, FilterParams, LmbmConfig, LmbmHypothesis, MotionModel,
    SensorModel, StateEstimate, Trajectory,
};

use super::errors::FilterError;
use super::traits::{
    AssociationResult, Associator, Filter, GibbsAssociator, HardAssignmentUpdater, Updater,
};

/// Log-likelihood floor to prevent underflow when computing ln(x) for very small x.
/// Approximately ln(UNDERFLOW_THRESHOLD), used when likelihood values are below f64 precision.
const LOG_UNDERFLOW: f64 = -700.0;

/// Single-sensor LMBM filter.
///
/// This filter maintains multiple weighted hypotheses, where each hypothesis
/// represents one possible association history. Unlike LMB which has Gaussian
/// mixtures per track, LMBM has single-component tracks within each hypothesis.
///
/// The filter is generic over the association algorithm `A`, typically:
/// - [`GibbsAssociator`] for sampling-based association (default)
/// - [`MurtyAssociator`] for k-best deterministic assignments
///
/// # Default Associator
///
/// LMBM defaults to [`GibbsAssociator`] because:
/// - LMBM requires discrete/hard associations (one measurement per track per hypothesis)
/// - Gibbs sampling efficiently generates diverse hypothesis samples
/// - Works well with the hypothesis-weighted posterior
///
/// For deterministic k-best assignments, use `with_associator_type(MurtyAssociator)`.
///
/// # Note: No GM Pruning
///
/// Unlike [`LmbFilter`], LMBM does not have `with_gm_pruning()` because each track
/// in a hypothesis has exactly one Gaussian component (single-component tracks).
/// Hypothesis pruning is controlled via [`LmbmConfig`] instead.
///
/// # Type Parameters
///
/// * `A` - The data association algorithm, must implement [`Associator`]
pub struct LmbmFilter<A: Associator = GibbsAssociator> {
    /// Motion model (dynamics, survival probability)
    motion: MotionModel,
    /// Sensor model (observation, detection probability, clutter)
    sensor: SensorModel,
    /// Birth model (where new objects can appear)
    birth: BirthModel,
    /// Association algorithm configuration
    association_config: AssociationConfig,
    /// LMBM-specific configuration
    lmbm_config: LmbmConfig,

    /// Current hypotheses (weighted mixture of track sets)
    hypotheses: Vec<LmbmHypothesis>,
    /// Complete trajectories for all discarded long-lived tracks
    trajectories: Vec<Trajectory>,

    /// Existence probability threshold for gating tracks
    existence_threshold: f64,
    /// Minimum trajectory length to keep when pruning
    min_trajectory_length: usize,

    /// The associator to use
    associator: A,
}

impl LmbmFilter<GibbsAssociator> {
    /// Create a new LMBM filter with the given parameters using the default Gibbs associator.
    pub fn new(
        motion: MotionModel,
        sensor: SensorModel,
        birth: BirthModel,
        association_config: AssociationConfig,
        lmbm_config: LmbmConfig,
    ) -> Self {
        // Start with a single empty hypothesis with weight 1.0
        let initial_hypothesis = LmbmHypothesis::new(0.0, Vec::new()); // log(1.0) = 0.0

        Self {
            motion,
            sensor,
            birth,
            association_config,
            lmbm_config,
            hypotheses: vec![initial_hypothesis],
            trajectories: Vec::new(),
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            associator: GibbsAssociator,
        }
    }

    /// Create from FilterParams (convenience constructor) using the default Gibbs associator.
    ///
    /// Note: This constructor extracts the single sensor from the SensorVariant.
    pub fn from_params(params: &FilterParams) -> Self {
        Self::new(
            params.motion.clone(),
            params.sensor.single().clone(),
            params.birth.clone(),
            params.association.clone(),
            params.lmbm.clone(),
        )
    }
}

impl<A: Associator> LmbmFilter<A> {
    /// Create a new LMBM filter with a custom associator.
    pub fn with_associator_type(
        motion: MotionModel,
        sensor: SensorModel,
        birth: BirthModel,
        association_config: AssociationConfig,
        lmbm_config: LmbmConfig,
        associator: A,
    ) -> Self {
        let initial_hypothesis = LmbmHypothesis::new(0.0, Vec::new());

        Self {
            motion,
            sensor,
            birth,
            association_config,
            lmbm_config,
            hypotheses: vec![initial_hypothesis],
            trajectories: Vec::new(),
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            associator,
        }
    }

    /// Set the existence threshold for gating.
    pub fn with_existence_threshold(mut self, threshold: f64) -> Self {
        self.existence_threshold = threshold;
        self
    }

    /// Set the minimum trajectory length for keeping discarded tracks.
    pub fn with_min_trajectory_length(mut self, length: usize) -> Self {
        self.min_trajectory_length = length;
        self
    }

    /// Predict all hypotheses forward in time.
    fn predict_hypotheses(&mut self, timestep: usize) {
        super::common_ops::predict_all_hypotheses(
            &mut self.hypotheses,
            &self.motion,
            &self.birth,
            timestep,
        );
    }

    /// Generate posterior hypotheses from association samples.
    ///
    /// For each prior hypothesis and each association sample, creates a new
    /// posterior hypothesis with the appropriate track updates.
    fn generate_posterior_hypotheses(
        &mut self,
        result: &AssociationResult,
        posteriors: &crate::association::PosteriorGrid,
        log_likelihoods: &nalgebra::DMatrix<f64>,
    ) {
        let samples = match &result.sampled_associations {
            Some(s) if !s.is_empty() => s,
            _ => return,
        };

        let mut new_hypotheses = Vec::new();

        for prior_hyp in &self.hypotheses {
            for (sample_idx, assignments) in samples.iter().enumerate() {
                // Create new hypothesis by cloning prior
                let mut new_hyp = prior_hyp.clone();

                // Compute log weight contribution from this assignment
                let mut log_likelihood_sum = 0.0;
                for (track_idx, &meas_assignment) in assignments.iter().enumerate() {
                    if track_idx < new_hyp.tracks.len() {
                        // Assignment is -1 for miss, 0..m-1 for measurements
                        let col_idx = if meas_assignment < 0 {
                            0 // miss column
                        } else {
                            (meas_assignment + 1) as usize // measurement columns are 1-indexed
                        };
                        if col_idx < log_likelihoods.ncols() {
                            log_likelihood_sum += log_likelihoods[(track_idx, col_idx)];
                        }
                    }
                }

                // Update hypothesis weight
                new_hyp.log_weight += log_likelihood_sum;

                // Apply hard assignment updates to tracks
                let updater = HardAssignmentUpdater::with_sample_index(sample_idx);
                updater.update(&mut new_hyp.tracks, result, posteriors);

                // Update existence probabilities for detected tracks
                for (track_idx, &meas_assignment) in assignments.iter().enumerate() {
                    if track_idx < new_hyp.tracks.len() {
                        if meas_assignment >= 0 {
                            // Detected - existence is certain
                            new_hyp.tracks[track_idx].existence = 1.0;
                        }
                        // Miss case: existence already updated by the standard LMB formula
                    }
                }

                new_hypotheses.push(new_hyp);
            }
        }

        self.hypotheses = new_hypotheses;
    }

    /// Update existence probabilities when there are no measurements.
    fn update_existence_no_measurements(&mut self) {
        let p_d = self.sensor.detection_probability;
        for hyp in &mut self.hypotheses {
            for track in &mut hyp.tracks {
                track.existence =
                    crate::components::update::update_existence_no_detection(track.existence, p_d);
            }
        }
    }

    /// Normalize and gate hypotheses.
    fn normalize_and_gate_hypotheses(&mut self) {
        super::common_ops::normalize_and_gate_hypotheses(
            &mut self.hypotheses,
            self.lmbm_config.hypothesis_weight_threshold,
            self.lmbm_config.max_hypotheses,
        );
    }

    /// Gate tracks by existence probability across all hypotheses.
    fn gate_tracks(&mut self) {
        super::common_ops::gate_hypothesis_tracks(
            &mut self.hypotheses,
            &mut self.trajectories,
            self.existence_threshold,
            self.min_trajectory_length,
        );
    }

    /// Extract state estimates from the hypothesis mixture.
    fn extract_estimates(&self, timestamp: usize) -> StateEstimate {
        super::common_ops::extract_hypothesis_estimates(
            &self.hypotheses,
            timestamp,
            self.lmbm_config.use_eap,
        )
    }

    /// Update track trajectories after measurement update.
    fn update_trajectories(&mut self, timestamp: usize) {
        super::common_ops::update_hypothesis_trajectories(&mut self.hypotheses, timestamp);
    }

    /// Initialize trajectory recording for new birth tracks.
    fn init_birth_trajectories(&mut self, max_length: usize) {
        super::common_ops::init_hypothesis_birth_trajectories(&mut self.hypotheses, max_length);
    }

    /// Build log-likelihood matrix for computing hypothesis weights.
    ///
    /// The matrix is (n × (m+1)) where:
    /// - First column: log P(miss | track i)
    /// - Remaining columns: log P(measurement j | track i)
    fn build_log_likelihood_matrix(
        &self,
        matrices: &crate::association::AssociationMatrices,
    ) -> nalgebra::DMatrix<f64> {
        let n = matrices.eta.len();
        let m = matrices.psi.ncols();

        let mut log_likelihood = nalgebra::DMatrix::zeros(n, m + 1);

        for i in 0..n {
            // Miss column: log(phi_i)
            log_likelihood[(i, 0)] = if matrices.phi[i] > super::UNDERFLOW_THRESHOLD {
                matrices.phi[i].ln()
            } else {
                LOG_UNDERFLOW
            };

            // Measurement columns: log(eta_i * psi_ij) = log(L_ij)
            for j in 0..m {
                let likelihood_ij = matrices.eta[i] * matrices.psi[(i, j)];
                log_likelihood[(i, j + 1)] = if likelihood_ij > super::UNDERFLOW_THRESHOLD {
                    likelihood_ij.ln()
                } else {
                    LOG_UNDERFLOW
                };
            }
        }

        log_likelihood
    }
}

impl<A: Associator> Filter for LmbmFilter<A> {
    type State = Vec<LmbmHypothesis>;
    type Measurements = Vec<DVector<f64>>;

    fn step<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        // ══════════════════════════════════════════════════════════════════════
        // STEP 1: Prediction - propagate tracks forward and add birth components
        // ══════════════════════════════════════════════════════════════════════
        self.predict_hypotheses(timestep);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2: Initialize trajectory recording for new birth tracks
        // ══════════════════════════════════════════════════════════════════════
        self.init_birth_trajectories(super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 3: Measurement update - data association and track updates
        // ══════════════════════════════════════════════════════════════════════
        if !measurements.is_empty() {
            // For LMBM, we process each prior hypothesis
            // and generate multiple posterior hypotheses from association samples

            // Build association matrices from highest-weight hypothesis
            // (all hypotheses should have same track structure at this point)
            if !self.hypotheses.is_empty() && !self.hypotheses[0].tracks.is_empty() {
                let tracks = &self.hypotheses[0].tracks;

                let mut builder = AssociationBuilder::new(tracks, &self.sensor);
                let matrices = builder.build(measurements);

                // Build log-likelihood matrix for hypothesis weight computation
                let log_likelihood = self.build_log_likelihood_matrix(&matrices);

                // Run data association to get samples
                let result = self
                    .associator
                    .associate(&matrices, &self.association_config, rng)
                    .map_err(FilterError::Association)?;

                // Generate posterior hypotheses
                self.generate_posterior_hypotheses(&result, &matrices.posteriors, &log_likelihood);
            }
        } else {
            // No measurements: update existence for missed detection
            self.update_existence_no_measurements();
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 4: Hypothesis management (LMBM only) - normalize and gate hypotheses
        // ══════════════════════════════════════════════════════════════════════
        self.normalize_and_gate_hypotheses();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 5: Track gating - prune low-existence tracks, archive trajectories
        // ══════════════════════════════════════════════════════════════════════
        self.gate_tracks();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 6: Update trajectories - append current state to track histories
        // ══════════════════════════════════════════════════════════════════════
        self.update_trajectories(timestep);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 7: Extract estimates - return current state estimate
        // ══════════════════════════════════════════════════════════════════════
        Ok(self.extract_estimates(timestep))
    }

    fn state(&self) -> &Self::State {
        &self.hypotheses
    }

    fn reset(&mut self) {
        self.hypotheses.clear();
        self.hypotheses
            .push(LmbmHypothesis::new(0.0, Vec::new()));
        self.trajectories.clear();
    }

    fn x_dim(&self) -> usize {
        self.motion.x_dim()
    }

    fn z_dim(&self) -> usize {
        self.sensor.z_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BirthLocation;
    use nalgebra::DMatrix;

    fn create_test_filter() -> LmbmFilter {
        let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);
        let sensor = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);

        let birth_loc = BirthLocation::new(
            0,
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            DMatrix::identity(4, 4) * 100.0,
        );
        let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);

        let association_config = AssociationConfig::default();
        let lmbm_config = LmbmConfig::default();

        LmbmFilter::new(motion, sensor, birth, association_config, lmbm_config)
    }

    #[test]
    fn test_filter_creation() {
        let filter = create_test_filter();
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
        assert_eq!(filter.hypotheses.len(), 1);
        assert!(filter.hypotheses[0].tracks.is_empty());
    }

    #[test]
    fn test_filter_step_no_measurements() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        let estimate = filter.step(&mut rng, &vec![], 0).unwrap();

        // Should have processed but may have no estimates yet
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_filter_step_with_measurements() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        let measurements = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![5.0, 5.0]),
        ];

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();

        // Should process measurements
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_filter_multiple_steps() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        for t in 0..5 {
            let measurements = vec![DVector::from_vec(vec![t as f64, t as f64])];
            let _estimate = filter.step(&mut rng, &measurements, t).unwrap();
        }

        // Filter should have hypotheses after multiple steps
        assert!(!filter.hypotheses.is_empty());
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        // Run a step to create hypotheses
        let _ = filter.step(&mut rng, &vec![], 0);

        // Reset
        filter.reset();

        assert_eq!(filter.hypotheses.len(), 1);
        assert!(filter.hypotheses[0].tracks.is_empty());
        assert!(filter.trajectories.is_empty());
    }

    #[test]
    fn test_hypothesis_normalization() {
        let mut filter = create_test_filter();

        // Create multiple hypotheses with different weights
        filter.hypotheses = vec![
            LmbmHypothesis::new(0.0, Vec::new()),   // weight = 1.0
            LmbmHypothesis::new(-1.0, Vec::new()),  // weight ≈ 0.368
            LmbmHypothesis::new(-10.0, Vec::new()), // weight ≈ 4.5e-5
        ];

        filter.normalize_and_gate_hypotheses();

        // Should be normalized (weights sum to 1)
        let sum: f64 = filter.hypotheses.iter().map(|h| h.weight()).sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should be sorted by descending weight
        for i in 1..filter.hypotheses.len() {
            assert!(filter.hypotheses[i - 1].log_weight >= filter.hypotheses[i].log_weight);
        }
    }
}
