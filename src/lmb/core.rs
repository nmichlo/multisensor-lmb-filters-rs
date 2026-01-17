//! Unified LMB filter core.
//!
//! This module provides the [`LmbFilterCore`] struct, a generic LMB filter
//! implementation parameterized by associator and update scheduler. This
//! unifies the single-sensor and multi-sensor LMB filter implementations
//! into a single codebase.
//!
//! # Type Aliases
//!
//! For convenience, type aliases are provided for common filter configurations:
//!
//! - [`LmbFilter`] - Single-sensor LMB filter
//! - [`IcLmbFilter`] - Iterated Corrector multi-sensor LMB
//! - [`AaLmbFilter`] - Arithmetic Average multi-sensor LMB
//! - [`GaLmbFilter`] - Geometric Average multi-sensor LMB
//! - [`PuLmbFilter`] - Parallel Update multi-sensor LMB
//!
//! # Architecture
//!
//! The filter core delegates sensor processing to the [`UpdateScheduler`]:
//! - Sequential schedulers process sensors in order (output of N → input to N+1)
//! - Parallel schedulers process all sensors from the same prior, then fuse
//!
//! This design eliminates the ~60% code duplication between single-sensor
//! and multi-sensor filter implementations.

use nalgebra::DVector;

use crate::association::{AssociationBuilder, AssociationMatrices};
use crate::components::prediction::predict_tracks;

use super::builder::{FilterBuilder, LmbFilterBuilder};
use super::config::{
    AssociationConfig, BirthModel, FilterConfigSnapshot, MotionModel, MultisensorConfig,
    SensorModel,
};
use super::errors::FilterError;
use super::multisensor::fusion::{
    ArithmeticAverageMerger, GeometricAverageMerger, ParallelUpdateMerger,
};
use super::output::{StateEstimate, Trajectory};
use super::scheduler::{
    ParallelScheduler, SequentialScheduler, SingleSensorScheduler, UpdateScheduler,
};
use super::traits::{
    AssociationResult, Associator, Filter, LbpAssociator, MarginalUpdater, Merger, Updater,
};
use super::types::{SensorUpdateOutput, StepDetailedOutput, Track};

// ============================================================================
// SensorSet - Unified sensor configuration
// ============================================================================

/// Unified sensor configuration for LMB filters.
///
/// Abstracts over single-sensor and multi-sensor configurations, allowing
/// the filter core to handle both cases uniformly.
#[derive(Debug, Clone)]
pub enum SensorSet {
    /// Single sensor configuration.
    Single(SensorModel),
    /// Multi-sensor configuration.
    Multi(MultisensorConfig),
}

impl SensorSet {
    /// Returns the number of sensors.
    pub fn num_sensors(&self) -> usize {
        match self {
            SensorSet::Single(_) => 1,
            SensorSet::Multi(config) => config.num_sensors(),
        }
    }

    /// Returns the measurement dimension (assumes same for all sensors).
    pub fn z_dim(&self) -> usize {
        match self {
            SensorSet::Single(sensor) => sensor.z_dim(),
            SensorSet::Multi(config) => config.z_dim(),
        }
    }

    /// Returns the sensor at the given index.
    pub fn get(&self, index: usize) -> Option<&SensorModel> {
        match self {
            SensorSet::Single(sensor) if index == 0 => Some(sensor),
            SensorSet::Single(_) => None,
            SensorSet::Multi(config) => config.sensors.get(index),
        }
    }

    /// Returns an iterator over all sensors.
    pub fn iter(&self) -> SensorSetIter<'_> {
        SensorSetIter {
            sensors: self,
            index: 0,
        }
    }

    /// Returns detection probabilities for all sensors.
    pub fn detection_probabilities(&self) -> Vec<f64> {
        match self {
            SensorSet::Single(sensor) => vec![sensor.detection_probability],
            SensorSet::Multi(config) => config
                .sensors
                .iter()
                .map(|s| s.detection_probability)
                .collect(),
        }
    }
}

impl From<SensorModel> for SensorSet {
    fn from(sensor: SensorModel) -> Self {
        SensorSet::Single(sensor)
    }
}

impl From<MultisensorConfig> for SensorSet {
    fn from(config: MultisensorConfig) -> Self {
        SensorSet::Multi(config)
    }
}

/// Iterator over sensors in a [`SensorSet`].
pub struct SensorSetIter<'a> {
    sensors: &'a SensorSet,
    index: usize,
}

impl<'a> Iterator for SensorSetIter<'a> {
    type Item = &'a SensorModel;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.sensors.get(self.index);
        self.index += 1;
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.sensors.num_sensors().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for SensorSetIter<'a> {}

// ============================================================================
// Helper Functions
// ============================================================================

/// Update tracks with a single sensor's measurements.
///
/// This is a standalone function to avoid borrow checker issues when accessing
/// multiple fields of a filter struct simultaneously.
fn update_single_sensor<A: Associator, R: rand::Rng>(
    associator: &A,
    updater: &MarginalUpdater,
    association_config: &AssociationConfig,
    sensor: &SensorModel,
    measurements: &[DVector<f64>],
    tracks: &mut [Track],
    rng: &mut R,
) -> Result<(Option<AssociationMatrices>, Option<AssociationResult>), FilterError> {
    if measurements.is_empty() {
        // No measurements: missed detection update
        let p_d = sensor.detection_probability;
        for track in tracks.iter_mut() {
            track.existence =
                crate::components::update::update_existence_no_detection(track.existence, p_d);
        }
        return Ok((None, None));
    }

    // Build association matrices
    let mut builder = AssociationBuilder::new(tracks, sensor);
    let matrices = builder.build(measurements);

    // Run data association
    let result = associator
        .associate(&matrices, association_config, rng)
        .map_err(FilterError::Association)?;

    // Update tracks
    updater.update(tracks, &result, &matrices.posteriors);
    super::common_ops::update_existence_from_marginals(tracks, &result);

    Ok((Some(matrices), Some(result)))
}

// ============================================================================
// LmbFilterCore
// ============================================================================

/// Generic LMB filter core parameterized by associator and scheduler.
///
/// This struct unifies single-sensor and multi-sensor LMB filter implementations.
/// The scheduler determines how sensors are processed during the update phase:
/// - Sequential schedulers: process sensors in order (IC-LMB, single-sensor)
/// - Parallel schedulers: process all sensors from same prior, then fuse (AA, GA, PU)
///
/// # Type Parameters
///
/// * `A` - The data association algorithm (default: [`LbpAssociator`])
/// * `S` - The update scheduler (determines single/multi-sensor behavior)
///
/// # Example
///
/// ```ignore
/// use multisensor_lmb_filters_rs::lmb::{LmbFilterCore, LbpAssociator, SingleSensorScheduler};
///
/// // Single-sensor LMB filter
/// let filter: LmbFilterCore<LbpAssociator, SingleSensorScheduler> = ...;
///
/// // Or use the type alias
/// let filter: LmbFilter = LmbFilter::new(motion, sensor, birth, config);
/// ```
pub struct LmbFilterCore<A: Associator = LbpAssociator, S: UpdateScheduler = SingleSensorScheduler>
{
    /// Motion model (dynamics, survival probability)
    motion: MotionModel,
    /// Sensor configuration (single or multi-sensor)
    sensors: SensorSet,
    /// Birth model (where new objects can appear)
    birth: BirthModel,
    /// Association algorithm configuration
    association_config: AssociationConfig,

    /// Current tracks (Bernoulli components)
    tracks: Vec<Track>,
    /// Complete trajectories for discarded tracks
    trajectories: Vec<Trajectory>,

    /// Existence probability threshold for gating
    existence_threshold: f64,
    /// Minimum trajectory length to keep
    min_trajectory_length: usize,
    /// GM component weight threshold
    gm_weight_threshold: f64,
    /// Maximum GM components per track
    max_gm_components: usize,
    /// Mahalanobis distance threshold for GM component merging
    gm_merge_threshold: f64,

    /// The associator to use
    associator: A,
    /// The update scheduler
    scheduler: S,
    /// The updater to use
    updater: MarginalUpdater,
}

// ============================================================================
// Constructor implementations for SingleSensorScheduler (single-sensor LMB)
// ============================================================================

impl LmbFilterCore<LbpAssociator, SingleSensorScheduler> {
    /// Create a new single-sensor LMB filter with default LBP associator.
    ///
    /// This is the standard constructor for single-sensor LMB filters.
    ///
    /// # Arguments
    ///
    /// * `motion` - Motion model (dynamics and survival probability)
    /// * `sensor` - Sensor model (observation model, detection probability, clutter)
    /// * `birth` - Birth model (where new objects can appear)
    /// * `association_config` - Association algorithm configuration
    pub fn new(
        motion: MotionModel,
        sensor: SensorModel,
        birth: BirthModel,
        association_config: AssociationConfig,
    ) -> Self {
        Self::with_scheduler(
            motion,
            sensor.into(),
            birth,
            association_config,
            LbpAssociator,
            SingleSensorScheduler::new(),
        )
    }
}

impl<A: Associator> LmbFilterCore<A, SingleSensorScheduler> {
    /// Create a new single-sensor LMB filter with a custom associator.
    pub fn with_associator(
        motion: MotionModel,
        sensor: SensorModel,
        birth: BirthModel,
        association_config: AssociationConfig,
        associator: A,
    ) -> Self {
        Self::with_scheduler(
            motion,
            sensor.into(),
            birth,
            association_config,
            associator,
            SingleSensorScheduler::new(),
        )
    }
}

// ============================================================================
// Constructor implementations for SequentialScheduler (IC-LMB)
// ============================================================================

impl LmbFilterCore<LbpAssociator, SequentialScheduler> {
    /// Create a new IC-LMB filter with default LBP associator.
    ///
    /// IC-LMB (Iterated Corrector) processes sensors sequentially, where
    /// the output of sensor N becomes the input to sensor N+1.
    pub fn new_ic(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
    ) -> Self {
        Self::with_scheduler(
            motion,
            sensors.into(),
            birth,
            association_config,
            LbpAssociator,
            SequentialScheduler::new(),
        )
    }
}

impl<A: Associator> LmbFilterCore<A, SequentialScheduler> {
    /// Create a new IC-LMB filter with a custom associator.
    pub fn with_associator_ic(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
        associator: A,
    ) -> Self {
        Self::with_scheduler(
            motion,
            sensors.into(),
            birth,
            association_config,
            associator,
            SequentialScheduler::new(),
        )
    }
}

// ============================================================================
// Constructor implementations for ParallelScheduler (AA, GA, PU-LMB)
// ============================================================================

impl<M: Merger> LmbFilterCore<LbpAssociator, ParallelScheduler<M>> {
    /// Create a new parallel multi-sensor LMB filter with default LBP associator.
    ///
    /// Use this for AA-LMB, GA-LMB, or PU-LMB filters by providing the
    /// appropriate merger in the scheduler.
    pub fn new_parallel(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
        scheduler: ParallelScheduler<M>,
    ) -> Self {
        Self::with_scheduler(
            motion,
            sensors.into(),
            birth,
            association_config,
            LbpAssociator,
            scheduler,
        )
    }
}

impl<A: Associator, M: Merger> LmbFilterCore<A, ParallelScheduler<M>> {
    /// Create a new parallel multi-sensor LMB filter with a custom associator.
    pub fn with_associator_parallel(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
        associator: A,
        scheduler: ParallelScheduler<M>,
    ) -> Self {
        Self::with_scheduler(
            motion,
            sensors.into(),
            birth,
            association_config,
            associator,
            scheduler,
        )
    }
}

// ============================================================================
// Generic implementation for all scheduler types
// ============================================================================

impl<A: Associator, S: UpdateScheduler> LmbFilterCore<A, S> {
    /// Create a filter with explicit scheduler.
    ///
    /// This is the most general constructor, allowing any associator and scheduler.
    pub fn with_scheduler(
        motion: MotionModel,
        sensors: SensorSet,
        birth: BirthModel,
        association_config: AssociationConfig,
        associator: A,
        scheduler: S,
    ) -> Self {
        Self {
            motion,
            sensors,
            birth,
            association_config,
            tracks: Vec::new(),
            trajectories: Vec::new(),
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            gm_weight_threshold: super::DEFAULT_GM_WEIGHT_THRESHOLD,
            max_gm_components: super::DEFAULT_MAX_GM_COMPONENTS,
            gm_merge_threshold: super::DEFAULT_GM_MERGE_THRESHOLD,
            associator,
            scheduler,
            updater: MarginalUpdater::new(),
        }
    }

    /// Number of sensors.
    pub fn num_sensors(&self) -> usize {
        self.sensors.num_sensors()
    }

    /// Returns a reference to the scheduler.
    pub fn scheduler(&self) -> &S {
        &self.scheduler
    }

    /// Set GM component pruning parameters.
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
    pub fn with_gm_merge_threshold(mut self, merge_threshold: f64) -> Self {
        self.gm_merge_threshold = merge_threshold;
        self.updater = MarginalUpdater::with_thresholds(
            self.gm_weight_threshold,
            self.max_gm_components,
            merge_threshold,
        );
        self
    }

    // ========================================================================
    // Internal helper methods
    // ========================================================================

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

    /// Update track trajectories.
    fn update_trajectories(&mut self, timestamp: usize) {
        super::common_ops::update_trajectories(&mut self.tracks, timestamp);
    }

    /// Initialize trajectory recording for birth tracks.
    fn init_birth_trajectories(&mut self, max_length: usize) {
        super::common_ops::init_birth_trajectories(&mut self.tracks, max_length);
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
}

// ============================================================================
// Step implementation for sequential schedulers (single-sensor and IC-LMB)
// ============================================================================

impl<A: Associator> LmbFilterCore<A, SingleSensorScheduler> {
    /// Process single-sensor measurements.
    fn step_impl<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &[DVector<f64>],
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        // Prediction
        predict_tracks(&mut self.tracks, &self.motion, &self.birth, timestep, false);
        self.init_birth_trajectories(super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        // Single sensor update
        let sensor = self
            .sensors
            .get(0)
            .expect("SingleSensorScheduler requires 1 sensor");
        update_single_sensor(
            &self.associator,
            &self.updater,
            &self.association_config,
            sensor,
            measurements,
            &mut self.tracks,
            rng,
        )?;

        // Gate and extract
        self.gate_tracks();
        self.update_trajectories(timestep);
        Ok(self.extract_estimates(timestep))
    }

    /// Detailed step for fixture validation.
    pub fn step_detailed<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &[DVector<f64>],
        timestep: usize,
    ) -> Result<StepDetailedOutput, FilterError> {
        // Prediction
        predict_tracks(&mut self.tracks, &self.motion, &self.birth, timestep, false);
        let predicted_tracks = self.tracks.clone();

        // Single sensor update
        let sensor = self
            .sensors
            .get(0)
            .expect("SingleSensorScheduler requires 1 sensor");
        let (matrices, result) = update_single_sensor(
            &self.associator,
            &self.updater,
            &self.association_config,
            sensor,
            measurements,
            &mut self.tracks,
            rng,
        )?;
        let updated_tracks = self.tracks.clone();

        // Cardinality and gating
        let cardinality = super::common_ops::compute_cardinality(&self.tracks);
        self.gate_tracks();
        let final_estimate = self.extract_estimates(timestep);

        Ok(StepDetailedOutput {
            predicted_tracks,
            association_matrices: matrices,
            association_result: result,
            updated_tracks,
            cardinality,
            final_estimate,
            sensor_updates: None,
            predicted_hypotheses: None,
            pre_normalization_hypotheses: None,
            normalized_hypotheses: None,
            objects_likely_to_exist: None,
        })
    }

    /// Get configuration snapshot for debugging.
    pub fn get_config(&self) -> FilterConfigSnapshot {
        match &self.sensors {
            SensorSet::Single(sensor) => FilterConfigSnapshot::single_sensor_lmb(
                "LmbFilter",
                &self.motion,
                sensor,
                &self.birth,
                &self.association_config,
                self.existence_threshold,
                self.gm_weight_threshold,
                self.max_gm_components,
                self.min_trajectory_length,
                self.gm_merge_threshold,
            ),
            SensorSet::Multi(_) => panic!("SingleSensorScheduler with multi-sensor config"),
        }
    }
}

impl<A: Associator> Filter for LmbFilterCore<A, SingleSensorScheduler> {
    type State = Vec<Track>;
    type Measurements = Vec<DVector<f64>>;

    fn step<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.step_impl(rng, measurements, timestep)
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
// Step implementation for SequentialScheduler (IC-LMB)
// ============================================================================

impl<A: Associator> LmbFilterCore<A, SequentialScheduler> {
    /// Process multi-sensor measurements sequentially.
    fn step_sequential<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &[Vec<DVector<f64>>],
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        let num_sensors = self.num_sensors();

        if measurements.len() != num_sensors {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                num_sensors,
                measurements.len()
            )));
        }

        // Prediction
        predict_tracks(&mut self.tracks, &self.motion, &self.birth, timestep, false);
        self.init_birth_trajectories(super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        // Sequential sensor updates
        for (sensor_idx, sensor_measurements) in measurements.iter().enumerate() {
            let sensor = self.sensors.get(sensor_idx).unwrap();
            update_single_sensor(
                &self.associator,
                &self.updater,
                &self.association_config,
                sensor,
                sensor_measurements,
                &mut self.tracks,
                rng,
            )?;
        }

        // Gate and extract
        self.gate_tracks();
        self.update_trajectories(timestep);
        Ok(self.extract_estimates(timestep))
    }

    /// Detailed step for fixture validation.
    pub fn step_detailed<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &[Vec<DVector<f64>>],
        timestep: usize,
    ) -> Result<StepDetailedOutput, FilterError> {
        let num_sensors = self.num_sensors();

        if measurements.len() != num_sensors {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                num_sensors,
                measurements.len()
            )));
        }

        // Prediction
        predict_tracks(&mut self.tracks, &self.motion, &self.birth, timestep, false);
        let predicted_tracks = self.tracks.clone();

        // Sequential sensor updates with intermediate capture
        let mut sensor_updates: Vec<SensorUpdateOutput> = Vec::with_capacity(num_sensors);

        for (sensor_idx, sensor_measurements) in measurements.iter().enumerate() {
            let input_tracks = self.tracks.clone();
            let sensor = self.sensors.get(sensor_idx).unwrap();
            let (matrices, result) = update_single_sensor(
                &self.associator,
                &self.updater,
                &self.association_config,
                sensor,
                sensor_measurements,
                &mut self.tracks,
                rng,
            )?;

            sensor_updates.push(SensorUpdateOutput::new(
                sensor_idx,
                input_tracks,
                matrices,
                result,
                self.tracks.clone(),
            ));
        }

        let updated_tracks = self.tracks.clone();
        let cardinality = super::common_ops::compute_cardinality(&self.tracks);
        self.gate_tracks();
        let final_estimate = self.extract_estimates(timestep);

        Ok(StepDetailedOutput {
            predicted_tracks,
            association_matrices: None,
            association_result: None,
            updated_tracks,
            cardinality,
            final_estimate,
            sensor_updates: Some(sensor_updates),
            predicted_hypotheses: None,
            pre_normalization_hypotheses: None,
            normalized_hypotheses: None,
            objects_likely_to_exist: None,
        })
    }

    /// Get configuration snapshot for debugging.
    pub fn get_config(&self) -> FilterConfigSnapshot {
        match &self.sensors {
            SensorSet::Multi(config) => FilterConfigSnapshot::multi_sensor_lmb(
                "IcLmbFilter",
                &self.motion,
                config,
                &self.birth,
                &self.association_config,
                self.existence_threshold,
                self.gm_weight_threshold,
                self.max_gm_components,
                self.min_trajectory_length,
                self.gm_merge_threshold,
            ),
            SensorSet::Single(sensor) => {
                // Can still work with single sensor in sequential mode
                FilterConfigSnapshot::single_sensor_lmb(
                    "IcLmbFilter(single)",
                    &self.motion,
                    sensor,
                    &self.birth,
                    &self.association_config,
                    self.existence_threshold,
                    self.gm_weight_threshold,
                    self.max_gm_components,
                    self.min_trajectory_length,
                    self.gm_merge_threshold,
                )
            }
        }
    }
}

impl<A: Associator> Filter for LmbFilterCore<A, SequentialScheduler> {
    type State = Vec<Track>;
    type Measurements = Vec<Vec<DVector<f64>>>;

    fn step<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.step_sequential(rng, measurements, timestep)
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
// Step implementation for ParallelScheduler (AA, GA, PU-LMB)
// ============================================================================

impl<A: Associator, M: Merger> LmbFilterCore<A, ParallelScheduler<M>> {
    /// Process multi-sensor measurements in parallel with fusion.
    fn step_parallel<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &[Vec<DVector<f64>>],
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        let num_sensors = self.num_sensors();

        if measurements.len() != num_sensors {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                num_sensors,
                measurements.len()
            )));
        }

        // Prediction
        predict_tracks(&mut self.tracks, &self.motion, &self.birth, timestep, false);
        self.init_birth_trajectories(super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        let has_any_measurements = measurements.iter().any(|m| !m.is_empty());

        if has_any_measurements && !self.tracks.is_empty() {
            // Parallel sensor updates: each sensor uses the same prior
            let mut per_sensor_tracks: Vec<Vec<Track>> = Vec::with_capacity(num_sensors);

            for (sensor_idx, sensor_measurements) in measurements.iter().enumerate() {
                let mut sensor_tracks = self.tracks.clone();
                let sensor = self.sensors.get(sensor_idx).unwrap();
                update_single_sensor(
                    &self.associator,
                    &self.updater,
                    &self.association_config,
                    sensor,
                    sensor_measurements,
                    &mut sensor_tracks,
                    rng,
                )?;
                per_sensor_tracks.push(sensor_tracks);
            }

            // Set prior for PU-LMB decorrelation (no-op for other mergers)
            self.scheduler.merger_mut().set_prior(self.tracks.clone());

            // Fuse per-sensor posteriors
            self.tracks = self.scheduler.merger().merge(&per_sensor_tracks, None);
        } else if !has_any_measurements {
            // No measurements: update existence for all sensors' missed detection
            let detection_probs = self.sensors.detection_probabilities();
            for track in &mut self.tracks {
                track.existence =
                    crate::components::update::update_existence_no_detection_multisensor(
                        track.existence,
                        &detection_probs,
                    );
            }
        }

        // Gate and extract
        self.gate_tracks();
        self.update_trajectories(timestep);
        Ok(self.extract_estimates(timestep))
    }

    /// Detailed step for fixture validation.
    pub fn step_detailed<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &[Vec<DVector<f64>>],
        timestep: usize,
    ) -> Result<StepDetailedOutput, FilterError> {
        let num_sensors = self.num_sensors();

        if measurements.len() != num_sensors {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                num_sensors,
                measurements.len()
            )));
        }

        // Prediction
        predict_tracks(&mut self.tracks, &self.motion, &self.birth, timestep, false);
        let predicted_tracks = self.tracks.clone();

        let has_any_measurements = measurements.iter().any(|m| !m.is_empty());
        let mut sensor_updates: Vec<SensorUpdateOutput> = Vec::with_capacity(num_sensors);

        if has_any_measurements && !self.tracks.is_empty() {
            let mut per_sensor_tracks: Vec<Vec<Track>> = Vec::with_capacity(num_sensors);

            for (sensor_idx, sensor_measurements) in measurements.iter().enumerate() {
                let input_tracks = self.tracks.clone();
                let mut sensor_tracks = self.tracks.clone();
                let sensor = self.sensors.get(sensor_idx).unwrap();
                let (matrices, result) = update_single_sensor(
                    &self.associator,
                    &self.updater,
                    &self.association_config,
                    sensor,
                    sensor_measurements,
                    &mut sensor_tracks,
                    rng,
                )?;

                sensor_updates.push(SensorUpdateOutput::new(
                    sensor_idx,
                    input_tracks,
                    matrices,
                    result,
                    sensor_tracks.clone(),
                ));

                per_sensor_tracks.push(sensor_tracks);
            }

            // Fusion
            self.scheduler.merger_mut().set_prior(self.tracks.clone());
            self.tracks = self.scheduler.merger().merge(&per_sensor_tracks, None);
        } else if !has_any_measurements {
            let detection_probs = self.sensors.detection_probabilities();
            for (sensor_idx, &p_d) in detection_probs.iter().enumerate() {
                let input_tracks = self.tracks.clone();
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
            // Update main tracks for missed detection
            for track in &mut self.tracks {
                track.existence =
                    crate::components::update::update_existence_no_detection_multisensor(
                        track.existence,
                        &detection_probs,
                    );
            }
        }

        let updated_tracks = self.tracks.clone();
        let cardinality = super::common_ops::compute_cardinality(&self.tracks);
        self.gate_tracks();
        let final_estimate = self.extract_estimates(timestep);

        Ok(StepDetailedOutput {
            predicted_tracks,
            association_matrices: None,
            association_result: None,
            updated_tracks,
            cardinality,
            final_estimate,
            sensor_updates: Some(sensor_updates),
            predicted_hypotheses: None,
            pre_normalization_hypotheses: None,
            normalized_hypotheses: None,
            objects_likely_to_exist: None,
        })
    }

    /// Get configuration snapshot for debugging.
    pub fn get_config(&self) -> FilterConfigSnapshot {
        let filter_type = format!(
            "{}LmbFilter",
            match self.scheduler.merger().name() {
                "ArithmeticAverage" => "Aa",
                "GeometricAverage" => "Ga",
                "ParallelUpdate" => "Pu",
                _ => "Custom",
            }
        );

        match &self.sensors {
            SensorSet::Multi(config) => FilterConfigSnapshot::multi_sensor_lmb(
                &filter_type,
                &self.motion,
                config,
                &self.birth,
                &self.association_config,
                self.existence_threshold,
                self.gm_weight_threshold,
                self.max_gm_components,
                self.min_trajectory_length,
                self.gm_merge_threshold,
            ),
            SensorSet::Single(sensor) => FilterConfigSnapshot::single_sensor_lmb(
                &filter_type,
                &self.motion,
                sensor,
                &self.birth,
                &self.association_config,
                self.existence_threshold,
                self.gm_weight_threshold,
                self.max_gm_components,
                self.min_trajectory_length,
                self.gm_merge_threshold,
            ),
        }
    }

    /// Returns a mutable reference to the scheduler.
    ///
    /// Useful for accessing the merger (e.g., for PU-LMB's `set_prior`).
    pub fn scheduler_mut(&mut self) -> &mut ParallelScheduler<M> {
        &mut self.scheduler
    }
}

impl<A: Associator, M: Merger> Filter for LmbFilterCore<A, ParallelScheduler<M>> {
    type State = Vec<Track>;
    type Measurements = Vec<Vec<DVector<f64>>>;

    fn step<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.step_parallel(rng, measurements, timestep)
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

impl<A: Associator, S: UpdateScheduler> FilterBuilder for LmbFilterCore<A, S> {
    fn existence_threshold_mut(&mut self) -> &mut f64 {
        &mut self.existence_threshold
    }

    fn min_trajectory_length_mut(&mut self) -> &mut usize {
        &mut self.min_trajectory_length
    }
}

impl<A: Associator, S: UpdateScheduler> LmbFilterBuilder for LmbFilterCore<A, S> {
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

/// Single-sensor LMB filter with default LBP associator.
///
/// This is the standard LMB filter for tracking with a single sensor.
/// For multi-sensor tracking, use [`IcLmbFilter`], [`AaLmbFilter`],
/// [`GaLmbFilter`], or [`PuLmbFilter`].
///
/// # Example
///
/// ```ignore
/// use multisensor_lmb_filters_rs::lmb::LmbFilter;
///
/// let filter = LmbFilter::new(motion, sensor, birth, config);
/// let estimate = filter.step(&mut rng, &measurements, timestep)?;
/// ```
pub type LmbFilter<A = LbpAssociator> = LmbFilterCore<A, SingleSensorScheduler>;

/// Iterated Corrector LMB filter (IC-LMB).
///
/// Processes sensors sequentially, where the output of sensor N becomes
/// the input to sensor N+1. Simple but order-dependent.
pub type IcLmbFilter<A = LbpAssociator> = LmbFilterCore<A, SequentialScheduler>;

/// Arithmetic Average LMB filter (AA-LMB).
///
/// Processes sensors in parallel, then fuses using weighted arithmetic average.
/// Fast and robust, but doesn't account for sensor correlation.
pub type AaLmbFilter<A = LbpAssociator> =
    LmbFilterCore<A, ParallelScheduler<ArithmeticAverageMerger>>;

/// Geometric Average LMB filter (GA-LMB).
///
/// Processes sensors in parallel, then fuses using covariance intersection.
/// Produces conservative estimates, robust to unknown correlations.
pub type GaLmbFilter<A = LbpAssociator> =
    LmbFilterCore<A, ParallelScheduler<GeometricAverageMerger>>;

/// Parallel Update LMB filter (PU-LMB).
///
/// Processes sensors in parallel, then fuses using information-form fusion
/// with decorrelation. Theoretically optimal for independent sensors.
pub type PuLmbFilter<A = LbpAssociator> = LmbFilterCore<A, ParallelScheduler<ParallelUpdateMerger>>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmb::config::BirthLocation;
    use nalgebra::DMatrix;

    fn create_motion() -> MotionModel {
        MotionModel::constant_velocity_2d(1.0, 0.1, 0.99)
    }

    fn create_sensor() -> SensorModel {
        SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0)
    }

    fn create_multi_sensor() -> MultisensorConfig {
        let sensor1 = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
        let sensor2 = SensorModel::position_sensor_2d(1.0, 0.85, 8.0, 100.0);
        MultisensorConfig::new(vec![sensor1, sensor2])
    }

    fn create_birth() -> BirthModel {
        let birth_loc = BirthLocation::new(
            0,
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            DMatrix::identity(4, 4) * 100.0,
        );
        BirthModel::new(vec![birth_loc], 0.1, 0.01)
    }

    #[test]
    fn test_sensor_set_single() {
        let sensor = create_sensor();
        let set: SensorSet = sensor.into();
        assert_eq!(set.num_sensors(), 1);
        assert!(set.get(0).is_some());
        assert!(set.get(1).is_none());
        assert_eq!(set.iter().count(), 1);
    }

    #[test]
    fn test_sensor_set_multi() {
        let sensors = create_multi_sensor();
        let set: SensorSet = sensors.into();
        assert_eq!(set.num_sensors(), 2);
        assert!(set.get(0).is_some());
        assert!(set.get(1).is_some());
        assert!(set.get(2).is_none());
        assert_eq!(set.iter().count(), 2);
    }

    #[test]
    fn test_lmb_filter_creation() {
        let filter = LmbFilter::new(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
        assert_eq!(filter.num_sensors(), 1);
    }

    #[test]
    fn test_lmb_filter_step_no_measurements() {
        let mut filter = LmbFilter::new(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
        );
        let mut rng = rand::thread_rng();

        let estimate = filter.step(&mut rng, &vec![], 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_lmb_filter_step_with_measurements() {
        let mut filter = LmbFilter::new(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
        );
        let mut rng = rand::thread_rng();

        let measurements = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![5.0, 5.0]),
        ];

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_ic_lmb_filter_creation() {
        let filter = IcLmbFilter::new_ic(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
        assert_eq!(filter.num_sensors(), 2);
    }

    #[test]
    fn test_ic_lmb_filter_step() {
        let mut filter = IcLmbFilter::new_ic(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
        );
        let mut rng = rand::thread_rng();

        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])],
            vec![DVector::from_vec(vec![0.5, 0.5])],
        ];

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_aa_lmb_filter_creation() {
        let merger = ArithmeticAverageMerger::uniform(2, 100);
        let scheduler = ParallelScheduler::new(merger);

        let filter = AaLmbFilter::new_parallel(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            scheduler,
        );

        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.num_sensors(), 2);
        assert_eq!(filter.scheduler().merger().name(), "ArithmeticAverage");
    }

    #[test]
    fn test_aa_lmb_filter_step() {
        let merger = ArithmeticAverageMerger::uniform(2, 100);
        let scheduler = ParallelScheduler::new(merger);

        let mut filter = AaLmbFilter::new_parallel(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            scheduler,
        );
        let mut rng = rand::thread_rng();

        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])],
            vec![DVector::from_vec(vec![0.5, 0.5])],
        ];

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_ga_lmb_filter_step() {
        let merger = GeometricAverageMerger::uniform(2);
        let scheduler = ParallelScheduler::new(merger);

        let mut filter = GaLmbFilter::new_parallel(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            scheduler,
        );
        let mut rng = rand::thread_rng();

        let measurements = vec![vec![], vec![]]; // No measurements
        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_pu_lmb_filter_step() {
        let merger = ParallelUpdateMerger::new(Vec::new());
        let scheduler = ParallelScheduler::new(merger);

        let mut filter = PuLmbFilter::new_parallel(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            scheduler,
        );
        let mut rng = rand::thread_rng();

        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])],
            vec![DVector::from_vec(vec![0.5, 0.5])],
        ];

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = LmbFilter::new(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
        );
        let mut rng = rand::thread_rng();

        let _ = filter.step(&mut rng, &vec![], 0);
        filter.reset();

        assert!(filter.get_tracks().is_empty());
    }

    #[test]
    fn test_gm_pruning_configuration() {
        let filter = LmbFilter::new(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
        )
        .with_gm_pruning(1e-3, 50)
        .with_gm_merge_threshold(4.0);

        // Filter was configured successfully
        assert_eq!(filter.gm_weight_threshold, 1e-3);
        assert_eq!(filter.max_gm_components, 50);
        assert_eq!(filter.gm_merge_threshold, 4.0);
    }

    #[test]
    fn test_type_alias_equivalence() {
        // Verify type aliases match expected types
        let _: LmbFilter = LmbFilter::new(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
        );

        let _: IcLmbFilter = IcLmbFilter::new_ic(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
        );
    }
}
