//! Multi-sensor LMB (Labeled Multi-Bernoulli) filter.
//!
//! The multi-sensor LMB filter extends the single-sensor LMB to handle measurements
//! from multiple sensors. The key challenge is fusing per-sensor posterior updates
//! into a unified track estimate.
//!
//! # Fusion Strategies
//!
//! Four fusion strategies are supported via the [`Merger`] trait:
//!
//! - **Arithmetic Average (AA)**: Simple weighted combination of GM components.
//!   Fast and robust, but doesn't account for correlation between sensors.
//!
//! - **Geometric Average (GA)**: Covariance intersection using canonical form.
//!   Produces single-component result with conservative covariance.
//!
//! - **Parallel Update (PU)**: Information-form fusion with decorrelation.
//!   Theoretically optimal for independent sensors.
//!
//! - **Iterated Corrector (IC)**: Sequential sensor updates (not truly parallel).
//!   Simple but order-dependent.
//!
//! # Type Aliases
//!
//! For convenience, type aliases are provided:
//! - [`AaLmbFilter`] - Arithmetic average fusion
//! - [`GaLmbFilter`] - Geometric average fusion
//! - [`PuLmbFilter`] - Parallel update fusion
//! - [`IcLmbFilter`] - Iterated corrector

use nalgebra::DVector;

use crate::association::AssociationBuilder;
use crate::components::prediction::predict_tracks;

use super::super::builder::{FilterBuilder, LmbFilterBuilder};
use super::super::config::{AssociationConfig, BirthModel, MotionModel, MultisensorConfig};
use super::super::errors::FilterError;
use super::super::output::{StateEstimate, Trajectory};
use super::super::traits::{Associator, Filter, LbpAssociator, MarginalUpdater, Merger, Updater};
use super::super::types::{SensorUpdateOutput, StepDetailedOutput, Track};

// Re-export fusion strategies for backwards compatibility
pub use super::fusion::{
    ArithmeticAverageMerger, GeometricAverageMerger, IteratedCorrectorMerger, ParallelUpdateMerger,
};

// ============================================================================
// MultisensorLmbFilter
// ============================================================================

/// Multi-sensor LMB filter with configurable fusion strategy.
///
/// This filter processes measurements from multiple sensors and fuses the
/// per-sensor updates using the provided [`Merger`] strategy.
///
/// The filter is generic over:
/// - `A`: Data association algorithm (default: [`LbpAssociator`])
/// - `M`: Fusion/merging strategy (implements [`Merger`])
///
/// # Example
///
/// ```ignore
/// use multisensor_lmb_filters_rs::filter::{MultisensorLmbFilter, ArithmeticAverageMerger};
///
/// // Create AA-LMB filter for 2 sensors
/// let merger = ArithmeticAverageMerger::uniform(2, 100);
/// let filter = MultisensorLmbFilter::new(motion, sensors, birth, config, merger);
/// ```
pub struct MultisensorLmbFilter<A: Associator = LbpAssociator, M: Merger = ArithmeticAverageMerger>
{
    /// Motion model (dynamics, survival probability)
    motion: MotionModel,
    /// Per-sensor models
    sensors: MultisensorConfig,
    /// Birth model (where new objects can appear)
    birth: BirthModel,
    /// Association algorithm configuration
    association_config: AssociationConfig,

    /// Current tracks
    tracks: Vec<Track>,
    /// Complete trajectories for discarded tracks
    trajectories: Vec<Trajectory>,

    /// Existence probability threshold for gating
    existence_threshold: f64,
    /// Minimum trajectory length to keep
    min_trajectory_length: usize,
    /// GM component pruning parameters
    gm_weight_threshold: f64,
    max_gm_components: usize,
    /// Mahalanobis distance threshold for GM component merging
    gm_merge_threshold: f64,

    /// The associator to use
    associator: A,
    /// The merger/fusion strategy
    merger: M,
    /// The updater to use
    updater: MarginalUpdater,
}

impl<M: Merger> MultisensorLmbFilter<LbpAssociator, M> {
    /// Create a new multi-sensor LMB filter with default LBP associator.
    pub fn new(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
        merger: M,
    ) -> Self {
        Self {
            motion,
            sensors,
            birth,
            association_config,
            tracks: Vec::new(),
            trajectories: Vec::new(),
            existence_threshold: super::super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            gm_weight_threshold: super::super::DEFAULT_GM_WEIGHT_THRESHOLD,
            max_gm_components: super::super::DEFAULT_MAX_GM_COMPONENTS,
            gm_merge_threshold: super::super::DEFAULT_GM_MERGE_THRESHOLD,
            associator: LbpAssociator,
            merger,
            updater: MarginalUpdater::new(),
        }
    }
}

impl<A: Associator, M: Merger> MultisensorLmbFilter<A, M> {
    /// Create with custom associator.
    pub fn with_associator_type(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
        merger: M,
        associator: A,
    ) -> Self {
        Self {
            motion,
            sensors,
            birth,
            association_config,
            tracks: Vec::new(),
            trajectories: Vec::new(),
            existence_threshold: super::super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            gm_weight_threshold: super::super::DEFAULT_GM_WEIGHT_THRESHOLD,
            max_gm_components: super::super::DEFAULT_MAX_GM_COMPONENTS,
            gm_merge_threshold: super::super::DEFAULT_GM_MERGE_THRESHOLD,
            associator,
            merger,
            updater: MarginalUpdater::new(),
        }
    }

    /// Number of sensors.
    pub fn num_sensors(&self) -> usize {
        self.sensors.num_sensors()
    }

    /// Set GM pruning parameters.
    ///
    /// This also updates the internal updater to use the new thresholds.
    pub fn with_gm_pruning(mut self, weight_threshold: f64, max_components: usize) -> Self {
        self.gm_weight_threshold = weight_threshold;
        self.max_gm_components = max_components;
        self.updater = MarginalUpdater::with_thresholds(
            weight_threshold,
            max_components,
            self.gm_merge_threshold,
        );
        self
    }

    /// Set the GM component merge threshold for Mahalanobis-distance merging.
    ///
    /// This controls how similar components must be (in Mahalanobis distance)
    /// to be merged together. Set to `f64::INFINITY` to disable merging.
    ///
    /// This also updates the internal updater to use the new threshold.
    pub fn with_gm_merge_threshold(mut self, merge_threshold: f64) -> Self {
        self.gm_merge_threshold = merge_threshold;
        self.updater = MarginalUpdater::with_thresholds(
            self.gm_weight_threshold,
            self.max_gm_components,
            merge_threshold,
        );
        self
    }

    /// Gate tracks by existence probability.
    fn gate_tracks(&mut self) {
        super::super::common_ops::gate_tracks(
            &mut self.tracks,
            &mut self.trajectories,
            self.existence_threshold,
            self.min_trajectory_length,
        );
    }

    /// Extract state estimates using MAP cardinality estimation.
    fn extract_estimates(&self, timestamp: usize) -> StateEstimate {
        super::super::common_ops::extract_estimates(&self.tracks, timestamp)
    }

    /// Update track trajectories.
    fn update_trajectories(&mut self, timestamp: usize) {
        super::super::common_ops::update_trajectories(&mut self.tracks, timestamp);
    }

    /// Initialize trajectory recording for birth tracks.
    fn init_birth_trajectories(&mut self, max_length: usize) {
        super::super::common_ops::init_birth_trajectories(&mut self.tracks, max_length);
    }

    /// Update existence for missed detection (all sensors).
    fn update_existence_no_measurements(&mut self) {
        let detection_probs: Vec<f64> = self
            .sensors
            .sensors
            .iter()
            .map(|s| s.detection_probability)
            .collect();
        for track in &mut self.tracks {
            track.existence = crate::components::update::update_existence_no_detection_multisensor(
                track.existence,
                &detection_probs,
            );
        }
    }

    // ========================================================================
    // Testing/Fixture Validation Methods
    // ========================================================================

    /// Set the internal tracks directly (for fixture testing).
    pub fn set_tracks(&mut self, tracks: Vec<Track>) {
        self.tracks = tracks;
    }

    /// Get the current tracks (for fixture testing).
    pub fn get_tracks(&self) -> Vec<Track> {
        self.tracks.clone()
    }

    /// Detailed step that returns all intermediate data for fixture validation.
    ///
    /// Note: Multi-sensor step_detailed does NOT return association matrices
    /// because each sensor produces its own matrices. This returns the fused
    /// result directly.
    pub fn step_detailed<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &[Vec<DVector<f64>>],
        timestep: usize,
    ) -> Result<StepDetailedOutput, FilterError> {
        let num_sensors = self.num_sensors();

        // Validate measurements
        if measurements.len() != num_sensors {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                num_sensors,
                measurements.len()
            )));
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 1: Prediction
        // ══════════════════════════════════════════════════════════════════════
        predict_tracks(&mut self.tracks, &self.motion, &self.birth, timestep, false);
        let predicted_tracks = self.tracks.clone();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2-3: Per-sensor association and fusion
        // ══════════════════════════════════════════════════════════════════════
        let has_any_measurements = measurements.iter().any(|m| !m.is_empty());

        // Collect per-sensor intermediate data for downstream systems
        let mut sensor_updates: Vec<SensorUpdateOutput> = Vec::with_capacity(num_sensors);

        if has_any_measurements && !self.tracks.is_empty() {
            let is_sequential = self.merger.is_sequential();
            let mut per_sensor_tracks: Vec<Vec<Track>> = Vec::with_capacity(num_sensors);

            // For sequential mergers (IC-LMB): each sensor uses previous sensor's output
            // For parallel mergers (AA, GA, PU): all sensors use the same prior
            let mut current_tracks = self.tracks.clone();

            for (sensor_idx, (sensor, sensor_measurements)) in self
                .sensors
                .sensors
                .iter()
                .zip(measurements.iter())
                .enumerate()
            {
                // Sequential: use current_tracks (updated by previous sensor)
                // Parallel: use original prior (self.tracks)
                let input_tracks = if is_sequential {
                    current_tracks.clone()
                } else {
                    self.tracks.clone()
                };
                let mut sensor_tracks = input_tracks.clone();
                let mut assoc_matrices = None;
                let mut assoc_result = None;

                if !sensor_measurements.is_empty() {
                    let mut builder = AssociationBuilder::new(&sensor_tracks, sensor);
                    let matrices = builder.build(sensor_measurements);
                    let result = self
                        .associator
                        .associate(&matrices, &self.association_config, rng)
                        .map_err(FilterError::Association)?;

                    self.updater
                        .update(&mut sensor_tracks, &result, &matrices.posteriors);
                    super::super::common_ops::update_existence_from_marginals(
                        &mut sensor_tracks,
                        &result,
                    );

                    // Capture per-sensor intermediate data
                    assoc_matrices = Some(matrices);
                    assoc_result = Some(result);
                } else {
                    let p_d = sensor.detection_probability;
                    for track in &mut sensor_tracks {
                        track.existence = crate::components::update::update_existence_no_detection(
                            track.existence,
                            p_d,
                        );
                    }
                }

                // Store per-sensor output (before fusion)
                sensor_updates.push(SensorUpdateOutput::new(
                    sensor_idx,
                    input_tracks,
                    assoc_matrices,
                    assoc_result,
                    sensor_tracks.clone(),
                ));

                // For sequential: update current_tracks for next sensor
                if is_sequential {
                    current_tracks = sensor_tracks.clone();
                }

                per_sensor_tracks.push(sensor_tracks);
            }

            // Set prior for PU-LMB decorrelation (no-op for other mergers)
            self.merger.set_prior(self.tracks.clone());
            self.tracks = self.merger.merge(&per_sensor_tracks, None);
        } else if !has_any_measurements {
            self.update_existence_no_measurements();
            // No measurements - store empty sensor updates with miss-detected tracks
            for sensor_idx in 0..num_sensors {
                let input_tracks = self.tracks.clone();
                let p_d = self.sensors.sensors[sensor_idx].detection_probability;
                let mut miss_tracks = self.tracks.clone();
                for track in &mut miss_tracks {
                    track.existence = crate::components::update::update_existence_no_detection(
                        track.existence,
                        p_d,
                    );
                }
                sensor_updates.push(SensorUpdateOutput::new(
                    sensor_idx,
                    input_tracks,
                    None,
                    None,
                    miss_tracks,
                ));
            }
        }
        let updated_tracks = self.tracks.clone();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 4: Cardinality extraction
        // ══════════════════════════════════════════════════════════════════════
        let cardinality = super::super::common_ops::compute_cardinality(&self.tracks);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 5: Gating
        // ══════════════════════════════════════════════════════════════════════
        self.gate_tracks();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 6: Extract final estimate
        // ══════════════════════════════════════════════════════════════════════
        let final_estimate = self.extract_estimates(timestep);

        Ok(StepDetailedOutput {
            predicted_tracks,
            association_matrices: None, // Per-sensor matrices available in sensor_updates
            association_result: None,   // Per-sensor results available in sensor_updates
            updated_tracks,
            cardinality,
            final_estimate,
            // Per-sensor intermediate data for downstream systems
            sensor_updates: Some(sensor_updates),
            // LMB doesn't have LMBM-specific fields
            predicted_hypotheses: None,
            pre_normalization_hypotheses: None,
            normalized_hypotheses: None,
            objects_likely_to_exist: None,
        })
    }
}

/// Multi-sensor measurements: one measurement set per sensor.
pub type MultisensorMeasurements = Vec<Vec<DVector<f64>>>;

impl<A: Associator, M: Merger> Filter for MultisensorLmbFilter<A, M> {
    type State = Vec<Track>;
    type Measurements = MultisensorMeasurements;

    fn step<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        let num_sensors = self.num_sensors();

        // Validate measurements
        if measurements.len() != num_sensors {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                num_sensors,
                measurements.len()
            )));
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 1: Prediction - propagate tracks forward and add birth components
        // ══════════════════════════════════════════════════════════════════════
        predict_tracks(&mut self.tracks, &self.motion, &self.birth, timestep, false);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2: Initialize trajectory recording for new birth tracks
        // ══════════════════════════════════════════════════════════════════════
        self.init_birth_trajectories(super::super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 3: Measurement update - data association and track updates
        // ══════════════════════════════════════════════════════════════════════
        let has_any_measurements = measurements.iter().any(|m| !m.is_empty());

        if has_any_measurements && !self.tracks.is_empty() {
            let is_sequential = self.merger.is_sequential();

            // --- STEP 3a: Per-sensor measurement updates ---
            let mut per_sensor_tracks: Vec<Vec<Track>> = Vec::with_capacity(num_sensors);

            // For sequential mergers (IC-LMB): each sensor uses previous sensor's output
            // For parallel mergers (AA, GA, PU): all sensors use the same prior
            let mut current_tracks = self.tracks.clone();

            for (sensor, sensor_measurements) in
                self.sensors.sensors.iter().zip(measurements.iter())
            {
                // Sequential: use current_tracks (updated by previous sensor)
                // Parallel: use original prior (self.tracks)
                let mut sensor_tracks = if is_sequential {
                    current_tracks.clone()
                } else {
                    self.tracks.clone()
                };

                if !sensor_measurements.is_empty() {
                    // Build association matrices for this sensor
                    let mut builder = AssociationBuilder::new(&sensor_tracks, sensor);
                    let matrices = builder.build(sensor_measurements);

                    // Run data association
                    let result = self
                        .associator
                        .associate(&matrices, &self.association_config, rng)
                        .map_err(FilterError::Association)?;

                    // Update tracks
                    self.updater
                        .update(&mut sensor_tracks, &result, &matrices.posteriors);

                    // Update existence from association
                    super::super::common_ops::update_existence_from_marginals(
                        &mut sensor_tracks,
                        &result,
                    );
                } else {
                    // No measurements for this sensor - missed detection update
                    let p_d = sensor.detection_probability;
                    for track in &mut sensor_tracks {
                        track.existence = crate::components::update::update_existence_no_detection(
                            track.existence,
                            p_d,
                        );
                    }
                }

                // For sequential: update current_tracks for next sensor
                if is_sequential {
                    current_tracks = sensor_tracks.clone();
                }

                per_sensor_tracks.push(sensor_tracks);
            }

            // --- STEP 3b: Fuse per-sensor updates ---
            // Set prior for PU-LMB decorrelation (no-op for other mergers)
            self.merger.set_prior(self.tracks.clone());
            self.tracks = self.merger.merge(&per_sensor_tracks, None);
        } else if !has_any_measurements {
            // No measurements from any sensor
            self.update_existence_no_measurements();
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
        self.sensors.z_dim()
    }
}

// ============================================================================
// Builder Trait Implementations
// ============================================================================

impl<A: Associator, M: Merger> FilterBuilder for MultisensorLmbFilter<A, M> {
    fn existence_threshold_mut(&mut self) -> &mut f64 {
        &mut self.existence_threshold
    }

    fn min_trajectory_length_mut(&mut self) -> &mut usize {
        &mut self.min_trajectory_length
    }
}

impl<A: Associator, M: Merger> LmbFilterBuilder for MultisensorLmbFilter<A, M> {
    fn gm_weight_threshold_mut(&mut self) -> &mut f64 {
        &mut self.gm_weight_threshold
    }

    fn max_gm_components_mut(&mut self) -> &mut usize {
        &mut self.max_gm_components
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

/// Arithmetic Average LMB filter (AA-LMB).
pub type AaLmbFilter<A = LbpAssociator> = MultisensorLmbFilter<A, ArithmeticAverageMerger>;

/// Geometric Average LMB filter (GA-LMB).
pub type GaLmbFilter<A = LbpAssociator> = MultisensorLmbFilter<A, GeometricAverageMerger>;

/// Parallel Update LMB filter (PU-LMB).
pub type PuLmbFilter<A = LbpAssociator> = MultisensorLmbFilter<A, ParallelUpdateMerger>;

/// Iterated Corrector LMB filter (IC-LMB).
pub type IcLmbFilter<A = LbpAssociator> = MultisensorLmbFilter<A, IteratedCorrectorMerger>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmb::{BirthLocation, SensorModel};
    use nalgebra::DMatrix;

    fn create_test_sensors() -> MultisensorConfig {
        let sensor1 = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
        let sensor2 = SensorModel::position_sensor_2d(1.0, 0.85, 8.0, 100.0);
        MultisensorConfig::new(vec![sensor1, sensor2])
    }

    fn create_test_filter() -> MultisensorLmbFilter<LbpAssociator, ArithmeticAverageMerger> {
        let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);
        let sensors = create_test_sensors();

        let birth_loc = BirthLocation::new(
            0,
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            DMatrix::identity(4, 4) * 100.0,
        );
        let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);

        let association_config = AssociationConfig::default();
        let merger = ArithmeticAverageMerger::uniform(2, 100);

        MultisensorLmbFilter::new(motion, sensors, birth, association_config, merger)
    }

    #[test]
    fn test_filter_creation() {
        let filter = create_test_filter();
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
        assert_eq!(filter.num_sensors(), 2);
        assert!(filter.tracks.is_empty());
    }

    #[test]
    fn test_filter_step_no_measurements() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        // Empty measurements for both sensors
        let measurements = vec![vec![], vec![]];
        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();

        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_filter_step_with_measurements() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        // Measurements for each sensor
        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])], // sensor 1
            vec![DVector::from_vec(vec![0.5, 0.5])], // sensor 2
        ];

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_filter_multiple_steps() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        for t in 0..5 {
            let measurements = vec![
                vec![DVector::from_vec(vec![t as f64, t as f64])],
                vec![DVector::from_vec(vec![t as f64 + 0.1, t as f64 + 0.1])],
            ];
            let _estimate = filter.step(&mut rng, &measurements, t).unwrap();
        }
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        let measurements = vec![vec![], vec![]];
        let _ = filter.step(&mut rng, &measurements, 0);

        filter.reset();

        assert!(filter.tracks.is_empty());
        assert!(filter.trajectories.is_empty());
    }

    #[test]
    fn test_aa_merger() {
        let merger = ArithmeticAverageMerger::uniform(2, 100);
        assert_eq!(merger.sensor_weights, vec![0.5, 0.5]);
        assert_eq!(merger.name(), "ArithmeticAverage");
    }

    #[test]
    fn test_ga_merger() {
        let merger = GeometricAverageMerger::uniform(2);
        assert_eq!(merger.sensor_weights, vec![0.5, 0.5]);
        assert_eq!(merger.name(), "GeometricAverage");
    }

    #[test]
    fn test_ic_merger() {
        let merger = IteratedCorrectorMerger::new();
        assert_eq!(merger.name(), "IteratedCorrector");
    }
}
