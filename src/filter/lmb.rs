//! Single-sensor LMB (Labeled Multi-Bernoulli) filter.
//!
//! The LMB filter is a principled multi-object tracking algorithm based on
//! random finite set (RFS) theory. Each potential object is represented as
//! a Bernoulli component with:
//!
//! - An existence probability (how likely the object exists)
//! - A spatial distribution (Gaussian mixture for state uncertainty)
//! - A unique label (birth time + birth location)
//!
//! The filter performs prediction (motion model + birth) and update (data
//! association + measurement incorporation) at each timestep, outputting
//! state estimates for objects believed to exist.

use nalgebra::DVector;

use crate::association::AssociationBuilder;
use crate::components::prediction::predict_tracks;
use crate::types::{
    AssociationConfig, BirthModel, FilterParams, MotionModel, SensorModel, StateEstimate, Track,
    Trajectory,
};

use super::errors::FilterError;
use super::traits::{AssociationResult, Associator, Filter, LbpAssociator, MarginalUpdater, Updater};

/// Single-sensor LMB filter.
///
/// This is the main entry point for single-sensor multi-object tracking using
/// the LMB algorithm. It maintains a set of tracks (Bernoulli components) and
/// updates them based on measurements.
///
/// The filter is generic over the association algorithm `A`, allowing different
/// data association strategies (LBP, Gibbs, Murty) to be plugged in at compile time.
///
/// The filter uses:
/// - Configurable associator `A` for data association (default: [`LbpAssociator`])
/// - [`MarginalUpdater`] for soft association updates
/// - MAP cardinality estimation for state extraction
///
/// # Type Parameters
///
/// * `A` - The data association algorithm, must implement [`Associator`]
pub struct LmbFilter<A: Associator = LbpAssociator> {
    /// Motion model (dynamics, survival probability)
    motion: MotionModel,
    /// Sensor model (observation, detection probability, clutter)
    sensor: SensorModel,
    /// Birth model (where new objects can appear)
    birth: BirthModel,
    /// Association algorithm configuration
    association_config: AssociationConfig,

    /// Current tracks (Bernoulli components)
    tracks: Vec<Track>,
    /// Complete trajectories for all tracks
    trajectories: Vec<Trajectory>,

    /// Existence probability threshold for gating
    existence_threshold: f64,
    /// Minimum trajectory length to keep when pruning
    min_trajectory_length: usize,
    /// GM component weight threshold
    gm_weight_threshold: f64,
    /// Maximum GM components per track
    max_gm_components: usize,

    /// The associator to use
    associator: A,
    /// The updater to use
    updater: MarginalUpdater,
}

impl LmbFilter<LbpAssociator> {
    /// Create a new LMB filter with the given parameters using the default LBP associator.
    pub fn new(
        motion: MotionModel,
        sensor: SensorModel,
        birth: BirthModel,
        association_config: AssociationConfig,
    ) -> Self {
        let updater = MarginalUpdater::new();

        Self {
            motion,
            sensor,
            birth,
            association_config,
            tracks: Vec::new(),
            trajectories: Vec::new(),
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            gm_weight_threshold: super::DEFAULT_GM_WEIGHT_THRESHOLD,
            max_gm_components: super::DEFAULT_MAX_GM_COMPONENTS,
            associator: LbpAssociator,
            updater,
        }
    }

    /// Create from FilterParams (convenience constructor) using the default LBP associator.
    ///
    /// Note: This constructor extracts the single sensor from the SensorVariant.
    /// Use [`LmbFilter::with_associator`] if you need a different associator.
    pub fn from_params(params: &FilterParams) -> Self {
        Self::new(
            params.motion.clone(),
            params.sensor.single().clone(),
            params.birth.clone(),
            params.association.clone(),
        )
    }
}

impl<A: Associator> LmbFilter<A> {
    /// Create a new LMB filter with a custom associator.
    pub fn with_associator_type(
        motion: MotionModel,
        sensor: SensorModel,
        birth: BirthModel,
        association_config: AssociationConfig,
        associator: A,
    ) -> Self {
        let updater = MarginalUpdater::new();

        Self {
            motion,
            sensor,
            birth,
            association_config,
            tracks: Vec::new(),
            trajectories: Vec::new(),
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            gm_weight_threshold: super::DEFAULT_GM_WEIGHT_THRESHOLD,
            max_gm_components: super::DEFAULT_MAX_GM_COMPONENTS,
            associator,
            updater,
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

    /// Set the GM component pruning parameters.
    pub fn with_gm_pruning(mut self, weight_threshold: f64, max_components: usize) -> Self {
        self.gm_weight_threshold = weight_threshold;
        self.max_gm_components = max_components;
        self.updater = MarginalUpdater::with_thresholds(weight_threshold, max_components);
        self
    }

    /// Gate tracks by existence probability.
    fn gate_tracks(&mut self) {
        super::common_ops::gate_tracks(
            &mut self.tracks,
            &mut self.trajectories,
            self.existence_threshold,
            self.min_trajectory_length,
        );
    }

    /// Extract state estimates using MAP cardinality estimation.
    fn extract_estimates(&self, timestamp: usize) -> StateEstimate {
        super::common_ops::extract_estimates(&self.tracks, timestamp)
    }

    /// Update track trajectories after measurement update.
    fn update_trajectories(&mut self, timestamp: usize) {
        super::common_ops::update_trajectories(&mut self.tracks, timestamp);
    }

    /// Initialize trajectory recording for new birth tracks.
    fn init_birth_trajectories(&mut self, max_length: usize) {
        super::common_ops::init_birth_trajectories(&mut self.tracks, max_length);
    }

    /// Update existence probabilities from association result.
    ///
    /// The LBP/Gibbs/Murty algorithms compute posterior existence probabilities
    /// which we need to apply to our tracks.
    fn update_existence_from_association(&mut self, result: &AssociationResult) {
        // The miss_weights give us P(track i not associated | measurements)
        // The marginal_weights give us P(track i associated with meas j | measurements)
        //
        // For LMB, the posterior existence is:
        // r' = (miss_weight + sum_j marginal_weight[j]) * prior_factor
        //
        // However, the legacy implementation computes r directly in the LBP result.
        // For now, we use a simplified update based on whether the track was likely detected.
        for (i, track) in self.tracks.iter_mut().enumerate() {
            // Total association weight = P(detected) + P(not detected)
            // The association algorithms already normalize, so we use the
            // miss weight to scale existence for missed detections
            let miss_weight = result.miss_weights[i];
            let detection_weight: f64 = (0..result.marginal_weights.ncols())
                .map(|j| result.marginal_weights[(i, j)])
                .sum();

            // If strongly associated with measurements, boost existence
            // If mostly miss, reduce existence
            let total = miss_weight + detection_weight;
            if total > 1e-15 {
                // Weighted update: detection increases confidence
                let detection_ratio = detection_weight / total;
                // Interpolate between current existence and 1.0 based on detection
                track.existence = track.existence * (1.0 - detection_ratio * 0.5)
                    + detection_ratio * 0.5 * 1.0_f64.min(track.existence * 2.0);
            }
        }
    }
}

impl<A: Associator> Filter for LmbFilter<A> {
    type State = Vec<Track>;
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
        predict_tracks(&mut self.tracks, &self.motion, &self.birth, timestep, false);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2: Initialize trajectory recording for new birth tracks
        // ══════════════════════════════════════════════════════════════════════
        self.init_birth_trajectories(super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 3: Measurement update - data association and track updates
        // ══════════════════════════════════════════════════════════════════════
        if !measurements.is_empty() {
            // Build association matrices
            let mut builder = AssociationBuilder::new(&self.tracks, &self.sensor);
            let matrices = builder.build(measurements);

            // Run data association
            let result = self
                .associator
                .associate(&matrices, &self.association_config, rng)
                .map_err(FilterError::Association)?;

            // Update existence probabilities from association result
            self.update_existence_from_association(&result);

            // Update track spatial distributions
            self.updater
                .update(&mut self.tracks, &result, &matrices.posteriors);
        } else {
            // No measurements: update existence for missed detection
            for track in &mut self.tracks {
                track.existence = crate::components::update::update_existence_no_detection(
                    track.existence,
                    self.sensor.detection_probability,
                );
            }
        }

        // (STEP 4 skipped - hypothesis management is LMBM only)

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
        &self.tracks
    }

    fn reset(&mut self) {
        self.tracks.clear();
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
    use nalgebra::DMatrix;

    fn create_test_filter() -> LmbFilter {
        let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);
        let sensor = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);

        let birth_loc = crate::types::BirthLocation::new(
            0,
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            DMatrix::identity(4, 4) * 100.0,
        );
        let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);

        let association_config = AssociationConfig::default();

        LmbFilter::new(motion, sensor, birth, association_config)
    }

    #[test]
    fn test_filter_creation() {
        let filter = create_test_filter();
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
        assert!(filter.tracks.is_empty());
    }

    #[test]
    fn test_filter_step_no_measurements() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        let estimate = filter.step(&mut rng, &vec![], 0).unwrap();

        // Should have birth tracks but low existence, so maybe none extracted
        assert!(estimate.tracks.len() <= filter.birth.locations.len());
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
        assert!(estimate.timestamp == 0);
    }

    #[test]
    fn test_filter_multiple_steps() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        for t in 0..5 {
            let measurements = vec![DVector::from_vec(vec![t as f64, t as f64])];
            let _estimate = filter.step(&mut rng, &measurements, t).unwrap();
        }

        // Filter should have some tracks after multiple steps with measurements
        // (exact number depends on association and gating)
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        // Run a step to create tracks
        let _ = filter.step(&mut rng, &vec![], 0);

        // Reset
        filter.reset();

        assert!(filter.tracks.is_empty());
        assert!(filter.trajectories.is_empty());
    }
}
