//! Update strategy trait and implementations for unified filter architecture.
//!
//! This module provides the `UpdateStrategy` trait which abstracts over different
//! tracking algorithms (LMB, LMBM, etc.) allowing them to share common filter
//! infrastructure while maintaining their distinct behaviors.
//!
//! # Key Types
//!
//! - [`UpdateStrategy`] - Core trait abstracting predict/update/prune/extract operations
//! - [`CommonPruneConfig`] - Shared pruning configuration (existence threshold, trajectory length)
//! - [`UpdateContext`] - Context passed to strategy methods (sensors, motion, birth models)
//!
//! # Strategy Implementations
//!
//! - [`LmbStrategy`] - LMB algorithm with marginal GM posteriors
//! - [`LmbmStrategy`] - LMBM algorithm with hypothesis mixture management

use nalgebra::{DMatrix, DVector};
use rand::Rng;

use super::config::{AssociationConfig, BirthModel, MotionModel, SensorConfig, SensorModel};
use super::errors::FilterError;
use super::multisensor::traits::AssociatorMultisensor;
use super::multisensor::MeasurementsMultisensor;
use super::output::{StateEstimate, Trajectory};
use super::scheduler::{
    ParallelScheduler, SequentialScheduler, SingleSensorScheduler, UpdateScheduler,
};
use super::traits::{
    AssociationResult, Associator, Merger, Updater, UpdaterHardAssignment, UpdaterMarginal,
};
use super::types::{GaussianComponent, Hypothesis, Track};
use crate::association::{AssociationBuilder, AssociationMatrices};
use crate::common::linalg::{log_gaussian_normalizing_constant, robust_inverse};
use crate::components::prediction::predict_tracks;

// ============================================================================
// Configuration Types
// ============================================================================

/// Shared pruning configuration used by both LMB and LMBM strategies.
///
/// These parameters control when tracks are removed from the filter state.
#[derive(Debug, Clone)]
pub struct CommonPruneConfig {
    /// Minimum existence probability to keep a track (default: 0.001)
    pub existence_threshold: f64,
    /// Minimum trajectory length to archive when pruning (default: 3)
    pub min_trajectory_length: usize,
}

impl Default for CommonPruneConfig {
    fn default() -> Self {
        Self {
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
        }
    }
}

/// LMB-specific pruning configuration (GM component management).
///
/// These parameters control Gaussian mixture component pruning within tracks.
#[derive(Debug, Clone)]
pub struct LmbPruneConfig {
    /// Minimum weight to keep a GM component (default: 1e-4)
    pub gm_weight_threshold: f64,
    /// Maximum number of GM components per track (default: 100)
    pub max_gm_components: usize,
    /// Mahalanobis distance threshold for merging GM components (default: inf)
    pub gm_merge_threshold: f64,
}

impl Default for LmbPruneConfig {
    fn default() -> Self {
        Self {
            gm_weight_threshold: super::DEFAULT_GM_WEIGHT_THRESHOLD,
            max_gm_components: super::DEFAULT_MAX_GM_COMPONENTS,
            gm_merge_threshold: f64::INFINITY,
        }
    }
}

/// LMBM-specific pruning configuration (hypothesis management).
///
/// These parameters control hypothesis pruning in LMBM filters.
#[derive(Debug, Clone)]
pub struct LmbmPruneConfig {
    /// Minimum weight to keep a hypothesis (default: 1e-6)
    pub hypothesis_weight_threshold: f64,
    /// Maximum number of hypotheses to maintain (default: 1000)
    pub max_hypotheses: usize,
    /// Whether to use Expected A Posteriori (EAP) vs Maximum A Posteriori (MAP)
    pub use_eap: bool,
}

impl Default for LmbmPruneConfig {
    fn default() -> Self {
        Self {
            hypothesis_weight_threshold: 1e-6,
            max_hypotheses: 1000,
            use_eap: false,
        }
    }
}

/// Context passed to strategy methods during filter operations.
///
/// Contains references to all shared filter configuration that strategies
/// need to perform their operations, but no strategy-specific fields.
pub struct UpdateContext<'a> {
    /// Motion model (prediction dynamics)
    pub motion: &'a MotionModel,
    /// Sensor configuration (single or multi-sensor)
    pub sensors: &'a SensorConfig,
    /// Birth model (where new objects appear)
    pub birth: &'a BirthModel,
    /// Association algorithm configuration
    pub association_config: &'a AssociationConfig,
    /// Common pruning configuration (shared by all strategies)
    pub common_prune: &'a CommonPruneConfig,
}

// ============================================================================
// Update Intermediate Results
// ============================================================================

/// Intermediate results from the update step for detailed output.
///
/// Used by `step_detailed()` to provide fixture validation data.
#[derive(Debug, Clone, Default)]
pub struct UpdateIntermediate {
    /// Association matrices (if available - single sensor only)
    pub association_matrices: Option<AssociationMatrices>,
    /// Association result (if available - single sensor only)
    pub association_result: Option<AssociationResult>,
    /// Per-sensor update details (multisensor filters only)
    pub sensor_updates: Option<Vec<super::types::SensorUpdateOutput>>,
}

// ============================================================================
// UpdateStrategy Trait
// ============================================================================

/// Core trait for filter update strategies.
///
/// This trait abstracts over different tracking paradigms (LMB, LMBM, etc.)
/// by defining the key operations each algorithm must implement:
///
/// - **Predict**: Forward motion model prediction and birth injection
/// - **Update**: Measurement association and state update
/// - **Prune**: Remove low-weight hypotheses/components and archive trajectories
/// - **Extract**: Generate state estimates from current hypothesis mixture
///
/// # Type Parameters
///
/// * `Self::Measurements` - The measurement type this strategy accepts
///   - Single-sensor: `&[DVector<f64>]`
///   - Multi-sensor: `&[Vec<DVector<f64>>]`
///
/// # Implementation Notes
///
/// Each strategy owns its specific configuration internally:
/// - `LmbStrategy` owns `LmbPruneConfig` (GM weight threshold, max components, merge threshold)
/// - `LmbmStrategy` owns `LmbmPruneConfig` (hypothesis weight threshold, max hypotheses, use_eap)
///
/// This avoids a "god config" with unused fields and makes illegal states unrepresentable.
pub trait UpdateStrategy: Send + Sync + Clone {
    /// Type of measurements this strategy accepts.
    type Measurements: ?Sized;

    /// Predict hypotheses forward in time and inject birth tracks.
    ///
    /// # Arguments
    ///
    /// * `hypotheses` - Current hypotheses (modified in place)
    /// * `ctx` - Update context with motion model and birth config
    /// * `timestep` - Current timestep index
    fn predict(&self, hypotheses: &mut Vec<Hypothesis>, ctx: &UpdateContext, timestep: usize);

    /// Perform association and update with measurements.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator for stochastic algorithms
    /// * `hypotheses` - Current hypotheses (modified in place)
    /// * `measurements` - Measurements at current timestep
    /// * `ctx` - Update context with sensor and association config
    ///
    /// # Returns
    ///
    /// Ok with optional intermediate data for detailed output, or error on failure.
    fn update<R: Rng>(
        &mut self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        ctx: &UpdateContext,
    ) -> Result<UpdateIntermediate, FilterError>;

    /// Prune hypotheses, components, and archive dead tracks.
    ///
    /// # Arguments
    ///
    /// * `hypotheses` - Current hypotheses (modified in place)
    /// * `trajectories` - Trajectory archive (dead tracks added here)
    /// * `ctx` - Update context with common prune config
    ///
    /// # Returns
    ///
    /// Vector of booleans indicating which tracks were likely to exist before pruning.
    fn prune(
        &self,
        hypotheses: &mut Vec<Hypothesis>,
        trajectories: &mut Vec<Trajectory>,
        ctx: &UpdateContext,
    ) -> Vec<bool>;

    /// Extract state estimates from current hypotheses.
    ///
    /// # Arguments
    ///
    /// * `hypotheses` - Current hypotheses
    /// * `timestamp` - Current timestamp for output
    /// * `ctx` - Update context
    ///
    /// # Returns
    ///
    /// State estimate containing estimated tracks and cardinality.
    fn extract(
        &self,
        hypotheses: &[Hypothesis],
        timestamp: usize,
        ctx: &UpdateContext,
    ) -> StateEstimate;

    /// Get the algorithm name for logging and debugging.
    fn name(&self) -> &'static str;

    /// Whether this strategy maintains multiple hypotheses.
    ///
    /// Returns `true` for LMBM (multiple hypotheses), `false` for LMB (single hypothesis).
    fn is_hypothesis_based(&self) -> bool;

    /// Update track trajectories after measurement update.
    ///
    /// Default implementation delegates to common_ops.
    fn update_trajectories(&self, hypotheses: &mut Vec<Hypothesis>, timestamp: usize) {
        super::common_ops::update_hypothesis_trajectories(hypotheses, timestamp);
    }

    /// Initialize trajectory recording for birth tracks.
    ///
    /// Default implementation delegates to common_ops.
    fn init_birth_trajectories(&self, hypotheses: &mut Vec<Hypothesis>, max_length: usize) {
        super::common_ops::init_hypothesis_birth_trajectories(hypotheses, max_length);
    }

    // ========================================================================
    // Config getters for debugging/serialization
    // ========================================================================

    /// Get GM weight threshold (for LMB filters).
    ///
    /// Returns `Some(threshold)` for LMB strategies, `None` for LMBM strategies.
    /// This avoids LSP violations where LMBM would return a meaningless 0.0.
    fn gm_weight_threshold(&self) -> Option<f64> {
        None
    }

    /// Get max GM components (for LMB filters).
    ///
    /// Returns `Some(max)` for LMB strategies, `None` for LMBM strategies.
    /// This avoids LSP violations where LMBM would return a meaningless 0.
    fn max_gm_components(&self) -> Option<usize> {
        None
    }

    /// Get LMBM prune configuration (for LMBM filters).
    ///
    /// Returns `Some(config)` for LMBM strategies, `None` for LMB strategies.
    /// This avoids LSP violations where LMB would return a meaningless default.
    fn lmbm_config(&self) -> Option<LmbmPruneConfig> {
        None
    }

    // ========================================================================
    // Polymorphic hypothesis capture methods (eliminate is_hypothesis_based() checks)
    // ========================================================================

    /// Capture hypotheses for detailed output if this strategy needs it.
    ///
    /// Returns `None` for LMB strategies (single hypothesis, not captured),
    /// Returns `Some(clone)` for LMBM strategies (multiple hypotheses, captured).
    fn capture_hypotheses(
        &self,
        _hypotheses: &[super::types::Hypothesis],
    ) -> Option<Vec<super::types::Hypothesis>> {
        None
    }

    /// Wrap objects_likely_to_exist mask for detailed output.
    ///
    /// Returns `None` for LMB strategies,
    /// Returns `Some(mask)` for LMBM strategies.
    fn wrap_objects_likely_to_exist(&self, _keep_mask: Vec<bool>) -> Option<Vec<bool>> {
        None
    }

    /// Build configuration snapshot for debugging/serialization.
    ///
    /// Each strategy knows how to build its own config snapshot,
    /// eliminating the need to branch on `is_hypothesis_based()`.
    fn build_config_snapshot(
        &self,
        motion: &super::config::MotionModel,
        sensors: &super::config::SensorConfig,
        birth: &super::config::BirthModel,
        association_config: &super::config::AssociationConfig,
        common_prune: &CommonPruneConfig,
    ) -> super::config::FilterConfigSnapshot;
}

// ============================================================================
// LmbStrategy - LMB filter update strategy
// ============================================================================

/// LMB filter update strategy.
///
/// This strategy implements the LMB (Labeled Multi-Bernoulli) tracking algorithm
/// with marginal Gaussian mixture posteriors. It maintains a single hypothesis
/// containing multi-component tracks.
///
/// # Type Parameters
///
/// * `A` - The data association algorithm (default: [`AssociatorLbp`])
/// * `S` - The update scheduler determining single/multi-sensor behavior
///
/// # Configuration
///
/// The strategy owns its LMB-specific pruning configuration:
/// - GM component weight threshold
/// - Maximum GM components per track
/// - GM component merge threshold
///
/// Common pruning configuration (existence threshold, trajectory length) is
/// passed via `UpdateContext`.
#[derive(Clone)]
pub struct LmbStrategy<A: Associator, S: UpdateScheduler> {
    /// The data association algorithm
    pub(crate) associator: A,
    /// The update scheduler
    pub(crate) scheduler: S,
    /// The marginal updater (applies soft assignments)
    pub(crate) updater: UpdaterMarginal,
    /// LMB-specific pruning configuration
    pub(crate) prune_config: LmbPruneConfig,
}

impl<A: Associator, S: UpdateScheduler> LmbStrategy<A, S> {
    /// Create a new LMB strategy with the given components.
    pub fn new(associator: A, scheduler: S, prune_config: LmbPruneConfig) -> Self {
        let updater = UpdaterMarginal::with_thresholds(
            prune_config.gm_weight_threshold,
            prune_config.max_gm_components,
            prune_config.gm_merge_threshold,
        );
        Self {
            associator,
            scheduler,
            updater,
            prune_config,
        }
    }

    /// Create with default pruning configuration.
    pub fn with_defaults(associator: A, scheduler: S) -> Self {
        Self::new(associator, scheduler, LmbPruneConfig::default())
    }

    /// Get mutable reference to scheduler (for parallel mergers).
    pub fn scheduler_mut(&mut self) -> &mut S {
        &mut self.scheduler
    }

    /// Get mutable reference to prune config.
    pub fn prune_config_mut(&mut self) -> &mut LmbPruneConfig {
        &mut self.prune_config
    }
}

/// Update tracks with a single sensor's measurements.
///
/// Helper function to avoid borrow checker issues.
fn update_single_sensor<A: Associator, R: rand::Rng>(
    associator: &A,
    updater: &UpdaterMarginal,
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
// LmbStrategy for SingleSensorScheduler
// ============================================================================

impl<A: Associator + Clone> UpdateStrategy for LmbStrategy<A, SingleSensorScheduler> {
    type Measurements = [DVector<f64>];

    fn predict(&self, hypotheses: &mut Vec<Hypothesis>, ctx: &UpdateContext, timestep: usize) {
        // Ensure we have exactly one hypothesis
        if hypotheses.is_empty() {
            hypotheses.push(Hypothesis::empty());
        }

        // Predict tracks in the single hypothesis
        predict_tracks(
            &mut hypotheses[0].tracks,
            ctx.motion,
            ctx.birth,
            timestep,
            false,
        );
    }

    fn update<R: Rng>(
        &mut self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        ctx: &UpdateContext,
    ) -> Result<UpdateIntermediate, FilterError> {
        if hypotheses.is_empty() {
            return Ok(UpdateIntermediate::default());
        }

        let sensor = ctx.sensors.single();
        let (matrices, result) = update_single_sensor(
            &self.associator,
            &self.updater,
            ctx.association_config,
            sensor,
            measurements,
            &mut hypotheses[0].tracks,
            rng,
        )?;

        Ok(UpdateIntermediate {
            association_matrices: matrices,
            association_result: result,
            sensor_updates: None, // Single sensor - no per-sensor data
        })
    }

    fn prune(
        &self,
        hypotheses: &mut Vec<Hypothesis>,
        trajectories: &mut Vec<Trajectory>,
        ctx: &UpdateContext,
    ) -> Vec<bool> {
        if hypotheses.is_empty() {
            return Vec::new();
        }

        // LMB uses single hypothesis - gate tracks by existence
        super::common_ops::gate_tracks(
            &mut hypotheses[0].tracks,
            trajectories,
            ctx.common_prune.existence_threshold,
            ctx.common_prune.min_trajectory_length,
        );

        // Return empty since LMB doesn't track objects_likely_to_exist
        Vec::new()
    }

    fn extract(
        &self,
        hypotheses: &[Hypothesis],
        timestamp: usize,
        _ctx: &UpdateContext,
    ) -> StateEstimate {
        if hypotheses.is_empty() {
            return StateEstimate {
                timestamp,
                tracks: Vec::new(),
            };
        }

        super::common_ops::extract_estimates(&hypotheses[0].tracks, timestamp)
    }

    fn name(&self) -> &'static str {
        "LMB"
    }

    fn is_hypothesis_based(&self) -> bool {
        false
    }

    fn gm_weight_threshold(&self) -> Option<f64> {
        Some(self.prune_config.gm_weight_threshold)
    }

    fn max_gm_components(&self) -> Option<usize> {
        Some(self.prune_config.max_gm_components)
    }

    fn build_config_snapshot(
        &self,
        motion: &super::config::MotionModel,
        sensors: &super::config::SensorConfig,
        birth: &super::config::BirthModel,
        association_config: &super::config::AssociationConfig,
        common_prune: &CommonPruneConfig,
    ) -> super::config::FilterConfigSnapshot {
        super::config::FilterConfigSnapshot::single_sensor_lmb(
            self.name(),
            motion,
            sensors.single(),
            birth,
            association_config,
            common_prune.existence_threshold,
            self.prune_config.gm_weight_threshold,
            self.prune_config.max_gm_components,
            common_prune.min_trajectory_length,
            self.prune_config.gm_merge_threshold,
        )
    }
}

// ============================================================================
// LmbStrategy for SequentialScheduler (IC-LMB)
// ============================================================================

impl<A: Associator + Clone> UpdateStrategy for LmbStrategy<A, SequentialScheduler> {
    type Measurements = [Vec<DVector<f64>>];

    fn predict(&self, hypotheses: &mut Vec<Hypothesis>, ctx: &UpdateContext, timestep: usize) {
        if hypotheses.is_empty() {
            hypotheses.push(Hypothesis::empty());
        }

        predict_tracks(
            &mut hypotheses[0].tracks,
            ctx.motion,
            ctx.birth,
            timestep,
            false,
        );
    }

    fn update<R: Rng>(
        &mut self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        ctx: &UpdateContext,
    ) -> Result<UpdateIntermediate, FilterError> {
        use super::types::SensorUpdateOutput;

        let num_sensors = ctx.sensors.num_sensors();

        if measurements.len() != num_sensors {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                num_sensors,
                measurements.len()
            )));
        }

        if hypotheses.is_empty() {
            return Ok(UpdateIntermediate::default());
        }

        let mut sensor_updates: Vec<SensorUpdateOutput> = Vec::with_capacity(num_sensors);

        // Sequential sensor updates: each sensor uses output of previous sensor
        for (sensor_idx, sensor_measurements) in measurements.iter().enumerate() {
            // Capture input tracks (output of previous sensor, or predicted tracks for first)
            let input_tracks = hypotheses[0].tracks.clone();

            let sensor = ctx.sensors.sensor(sensor_idx).unwrap();
            let (association_matrices, association_result) = update_single_sensor(
                &self.associator,
                &self.updater,
                ctx.association_config,
                sensor,
                sensor_measurements,
                &mut hypotheses[0].tracks,
                rng,
            )?;

            // Capture per-sensor update data
            sensor_updates.push(SensorUpdateOutput::new(
                sensor_idx,
                input_tracks,
                association_matrices,
                association_result,
                hypotheses[0].tracks.clone(),
            ));
        }

        Ok(UpdateIntermediate {
            association_matrices: None,
            association_result: None,
            sensor_updates: Some(sensor_updates),
        })
    }

    fn prune(
        &self,
        hypotheses: &mut Vec<Hypothesis>,
        trajectories: &mut Vec<Trajectory>,
        ctx: &UpdateContext,
    ) -> Vec<bool> {
        if hypotheses.is_empty() {
            return Vec::new();
        }

        super::common_ops::gate_tracks(
            &mut hypotheses[0].tracks,
            trajectories,
            ctx.common_prune.existence_threshold,
            ctx.common_prune.min_trajectory_length,
        );

        Vec::new()
    }

    fn extract(
        &self,
        hypotheses: &[Hypothesis],
        timestamp: usize,
        _ctx: &UpdateContext,
    ) -> StateEstimate {
        if hypotheses.is_empty() {
            return StateEstimate {
                timestamp,
                tracks: Vec::new(),
            };
        }

        super::common_ops::extract_estimates(&hypotheses[0].tracks, timestamp)
    }

    fn name(&self) -> &'static str {
        "IC-LMB"
    }

    fn is_hypothesis_based(&self) -> bool {
        false
    }

    fn gm_weight_threshold(&self) -> Option<f64> {
        Some(self.prune_config.gm_weight_threshold)
    }

    fn max_gm_components(&self) -> Option<usize> {
        Some(self.prune_config.max_gm_components)
    }

    fn build_config_snapshot(
        &self,
        motion: &super::config::MotionModel,
        sensors: &super::config::SensorConfig,
        birth: &super::config::BirthModel,
        association_config: &super::config::AssociationConfig,
        common_prune: &CommonPruneConfig,
    ) -> super::config::FilterConfigSnapshot {
        super::config::FilterConfigSnapshot::multi_sensor_lmb(
            self.name(),
            motion,
            sensors,
            birth,
            association_config,
            common_prune.existence_threshold,
            self.prune_config.gm_weight_threshold,
            self.prune_config.max_gm_components,
            common_prune.min_trajectory_length,
            self.prune_config.gm_merge_threshold,
        )
    }
}

// ============================================================================
// LmbStrategy for ParallelScheduler (AA, GA, PU-LMB)
// ============================================================================

impl<A: Associator + Clone, M: Merger + Clone> UpdateStrategy
    for LmbStrategy<A, ParallelScheduler<M>>
{
    type Measurements = [Vec<DVector<f64>>];

    fn predict(&self, hypotheses: &mut Vec<Hypothesis>, ctx: &UpdateContext, timestep: usize) {
        if hypotheses.is_empty() {
            hypotheses.push(Hypothesis::empty());
        }

        predict_tracks(
            &mut hypotheses[0].tracks,
            ctx.motion,
            ctx.birth,
            timestep,
            false,
        );
    }

    fn update<R: Rng>(
        &mut self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        ctx: &UpdateContext,
    ) -> Result<UpdateIntermediate, FilterError> {
        use super::types::SensorUpdateOutput;

        let num_sensors = ctx.sensors.num_sensors();

        if measurements.len() != num_sensors {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                num_sensors,
                measurements.len()
            )));
        }

        if hypotheses.is_empty() {
            return Ok(UpdateIntermediate::default());
        }

        let has_any_measurements = measurements.iter().any(|m| !m.is_empty());
        let mut sensor_updates: Vec<SensorUpdateOutput> = Vec::with_capacity(num_sensors);

        if has_any_measurements && !hypotheses[0].tracks.is_empty() {
            // Parallel sensor updates: each sensor uses the same prior
            let mut per_sensor_tracks: Vec<Vec<Track>> = Vec::with_capacity(num_sensors);
            let input_tracks = hypotheses[0].tracks.clone(); // Same for all sensors in parallel

            for (sensor_idx, sensor_measurements) in measurements.iter().enumerate() {
                let mut sensor_tracks = hypotheses[0].tracks.clone();
                let sensor = ctx.sensors.sensor(sensor_idx).unwrap();
                let (association_matrices, association_result) = update_single_sensor(
                    &self.associator,
                    &self.updater,
                    ctx.association_config,
                    sensor,
                    sensor_measurements,
                    &mut sensor_tracks,
                    rng,
                )?;

                // Capture per-sensor update data
                sensor_updates.push(SensorUpdateOutput::new(
                    sensor_idx,
                    input_tracks.clone(),
                    association_matrices,
                    association_result,
                    sensor_tracks.clone(),
                ));

                per_sensor_tracks.push(sensor_tracks);
            }

            // Set prior for PU-LMB decorrelation (required for information form fusion)
            self.scheduler
                .merger_mut()
                .set_prior(hypotheses[0].tracks.clone());

            // Fuse per-sensor posteriors
            hypotheses[0].tracks = self.scheduler.merger().merge(&per_sensor_tracks, None);
        } else if !has_any_measurements {
            // No measurements: update existence for all sensors' missed detection
            let input_tracks = hypotheses[0].tracks.clone();
            let detection_probs = ctx.sensors.detection_probabilities();
            for track in &mut hypotheses[0].tracks {
                track.existence =
                    crate::components::update::update_existence_no_detection_multisensor(
                        track.existence,
                        &detection_probs,
                    );
            }

            // Still capture per-sensor data (with no association since no measurements)
            for sensor_idx in 0..num_sensors {
                sensor_updates.push(SensorUpdateOutput::new(
                    sensor_idx,
                    input_tracks.clone(),
                    None,
                    None,
                    hypotheses[0].tracks.clone(),
                ));
            }
        }

        Ok(UpdateIntermediate {
            association_matrices: None,
            association_result: None,
            sensor_updates: Some(sensor_updates),
        })
    }

    fn prune(
        &self,
        hypotheses: &mut Vec<Hypothesis>,
        trajectories: &mut Vec<Trajectory>,
        ctx: &UpdateContext,
    ) -> Vec<bool> {
        if hypotheses.is_empty() {
            return Vec::new();
        }

        super::common_ops::gate_tracks(
            &mut hypotheses[0].tracks,
            trajectories,
            ctx.common_prune.existence_threshold,
            ctx.common_prune.min_trajectory_length,
        );

        Vec::new()
    }

    fn extract(
        &self,
        hypotheses: &[Hypothesis],
        timestamp: usize,
        _ctx: &UpdateContext,
    ) -> StateEstimate {
        if hypotheses.is_empty() {
            return StateEstimate {
                timestamp,
                tracks: Vec::new(),
            };
        }

        super::common_ops::extract_estimates(&hypotheses[0].tracks, timestamp)
    }

    fn name(&self) -> &'static str {
        match self.scheduler.merger().name() {
            "ArithmeticAverage" => "AA-LMB",
            "GeometricAverage" => "GA-LMB",
            "ParallelUpdate" => "PU-LMB",
            _ => "Parallel-LMB",
        }
    }

    fn is_hypothesis_based(&self) -> bool {
        false
    }

    fn gm_weight_threshold(&self) -> Option<f64> {
        Some(self.prune_config.gm_weight_threshold)
    }

    fn max_gm_components(&self) -> Option<usize> {
        Some(self.prune_config.max_gm_components)
    }

    fn build_config_snapshot(
        &self,
        motion: &super::config::MotionModel,
        sensors: &super::config::SensorConfig,
        birth: &super::config::BirthModel,
        association_config: &super::config::AssociationConfig,
        common_prune: &CommonPruneConfig,
    ) -> super::config::FilterConfigSnapshot {
        super::config::FilterConfigSnapshot::multi_sensor_lmb(
            self.name(),
            motion,
            sensors,
            birth,
            association_config,
            common_prune.existence_threshold,
            self.prune_config.gm_weight_threshold,
            self.prune_config.max_gm_components,
            common_prune.min_trajectory_length,
            self.prune_config.gm_merge_threshold,
        )
    }
}

// ============================================================================
// LmbmAssociator trait - Unified association for LMBM filters
// ============================================================================

/// Log-likelihood floor to prevent underflow when computing ln(x) for very small x.
const LOG_UNDERFLOW: f64 = -700.0;

/// Unified LMBM association trait.
///
/// This trait abstracts over the different association patterns used by single-sensor
/// and multi-sensor LMBM filters. It combines association AND posterior hypothesis
/// generation into one operation, since the inputs and logic differ significantly.
pub trait LmbmAssociator: Send + Sync {
    /// Type of measurements this associator accepts.
    type Measurements;

    /// Perform association and generate posterior hypotheses.
    fn associate_and_update<R: Rng>(
        &self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        sensor_config: &SensorConfig,
        motion: &MotionModel,
        association_config: &AssociationConfig,
    ) -> Result<Option<LmbmAssociationIntermediate>, FilterError>;

    /// Update existence probabilities when there are no measurements.
    fn update_existence_no_measurements(
        &self,
        hypotheses: &mut Vec<Hypothesis>,
        sensor_config: &SensorConfig,
    );

    /// Check if measurements are empty.
    fn measurements_empty(&self, measurements: &Self::Measurements) -> bool;

    /// Validate measurements match expected sensor count.
    fn validate_measurements(
        &self,
        measurements: &Self::Measurements,
        sensor_config: &SensorConfig,
    ) -> Result<(), FilterError>;

    /// Algorithm name for logging and debugging.
    fn name(&self) -> &'static str;
}

/// Intermediate association data for detailed step output.
#[derive(Debug, Clone)]
pub struct LmbmAssociationIntermediate {
    /// Association matrices (single-sensor only).
    pub matrices: Option<AssociationMatrices>,
    /// Association result (single-sensor only).
    pub result: Option<AssociationResult>,
}

// ============================================================================
// SingleSensorLmbmStrategy - Wraps Associator for single-sensor LMBM
// ============================================================================

/// Single-sensor LMBM association strategy.
#[derive(Debug, Clone, Default)]
pub struct SingleSensorLmbmStrategy<A: Associator = super::traits::AssociatorGibbs> {
    pub associator: A,
}

impl<A: Associator> SingleSensorLmbmStrategy<A> {
    /// Build log-likelihood matrix for computing hypothesis weights.
    fn build_log_likelihood_matrix(matrices: &AssociationMatrices) -> DMatrix<f64> {
        let n = matrices.eta.len();
        let m = matrices.psi.ncols();
        let mut log_likelihood = DMatrix::zeros(n, m + 1);

        for i in 0..n {
            log_likelihood[(i, 0)] = if matrices.eta[i] > super::UNDERFLOW_THRESHOLD {
                matrices.eta[i].ln()
            } else {
                LOG_UNDERFLOW
            };

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

    /// Generate posterior hypotheses from sampled associations.
    fn generate_posterior_hypotheses(
        hypotheses: &mut Vec<Hypothesis>,
        result: &AssociationResult,
        matrices: &AssociationMatrices,
        log_likelihoods: &DMatrix<f64>,
    ) {
        let samples = match &result.sampled_associations {
            Some(s) if !s.is_empty() => s,
            _ => return,
        };

        let miss_posterior_r = matrices.miss_posterior_existence();
        let mut new_hypotheses = Vec::new();

        for prior_hyp in hypotheses.iter() {
            for (sample_idx, assignments) in samples.iter().enumerate() {
                let mut new_hyp = prior_hyp.clone();

                for (track_idx, track) in new_hyp.tracks.iter_mut().enumerate() {
                    if track_idx < miss_posterior_r.len() {
                        track.existence = miss_posterior_r[track_idx];
                    }
                }

                let mut log_likelihood_sum = 0.0;
                for (track_idx, &meas_assignment) in assignments.iter().enumerate() {
                    if track_idx < new_hyp.tracks.len() {
                        let col_idx = if meas_assignment < 0 {
                            0
                        } else {
                            (meas_assignment + 1) as usize
                        };
                        if col_idx < log_likelihoods.ncols() {
                            log_likelihood_sum += log_likelihoods[(track_idx, col_idx)];
                        }
                    }
                }

                new_hyp.log_weight += log_likelihood_sum;

                let updater = UpdaterHardAssignment::with_sample_index(sample_idx);
                updater.update(&mut new_hyp.tracks, result, &matrices.posteriors);

                for (track_idx, &meas_assignment) in assignments.iter().enumerate() {
                    if track_idx < new_hyp.tracks.len() && meas_assignment >= 0 {
                        new_hyp.tracks[track_idx].existence = 1.0;
                    }
                }

                new_hypotheses.push(new_hyp);
            }
        }

        *hypotheses = new_hypotheses;
    }
}

impl<A: Associator> LmbmAssociator for SingleSensorLmbmStrategy<A> {
    type Measurements = Vec<DVector<f64>>;

    fn associate_and_update<R: Rng>(
        &self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        sensor_config: &SensorConfig,
        _motion: &MotionModel,
        association_config: &AssociationConfig,
    ) -> Result<Option<LmbmAssociationIntermediate>, FilterError> {
        if hypotheses.is_empty() || hypotheses[0].tracks.is_empty() {
            return Ok(None);
        }

        let sensor = sensor_config.single();
        let tracks = &hypotheses[0].tracks;

        let mut builder = AssociationBuilder::new(tracks, sensor);
        let matrices = builder.build(measurements);
        let log_likelihood = Self::build_log_likelihood_matrix(&matrices);

        let result = self
            .associator
            .associate(&matrices, association_config, rng)
            .map_err(FilterError::Association)?;

        Self::generate_posterior_hypotheses(hypotheses, &result, &matrices, &log_likelihood);

        Ok(Some(LmbmAssociationIntermediate {
            matrices: Some(matrices),
            result: Some(result),
        }))
    }

    fn update_existence_no_measurements(
        &self,
        hypotheses: &mut Vec<Hypothesis>,
        sensor_config: &SensorConfig,
    ) {
        let p_d = sensor_config.single().detection_probability;
        for hyp in hypotheses.iter_mut() {
            for track in &mut hyp.tracks {
                track.existence =
                    crate::components::update::update_existence_no_detection(track.existence, p_d);
            }
        }
    }

    fn measurements_empty(&self, measurements: &Self::Measurements) -> bool {
        measurements.is_empty()
    }

    fn validate_measurements(
        &self,
        _measurements: &Self::Measurements,
        sensor_config: &SensorConfig,
    ) -> Result<(), FilterError> {
        if sensor_config.num_sensors() != 1 {
            return Err(FilterError::InvalidInput(format!(
                "SingleSensorLmbmStrategy requires 1 sensor, got {}",
                sensor_config.num_sensors()
            )));
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SingleSensorLmbm"
    }
}

// ============================================================================
// MultisensorLmbmStrategy - Wraps MultisensorAssociator for multi-sensor LMBM
// ============================================================================

/// Multi-sensor LMBM association strategy.
#[derive(Debug, Clone, Default)]
pub struct MultisensorLmbmStrategy<
    A: AssociatorMultisensor = super::multisensor::AssociatorMultisensorGibbs,
> {
    pub associator: A,
}

/// Posterior parameters for a single entry in the flattened likelihood tensor.
#[derive(Clone)]
struct MultisensorPosterior {
    existence: f64,
    mean: DVector<f64>,
    covariance: DMatrix<f64>,
}

impl<A: AssociatorMultisensor> MultisensorLmbmStrategy<A> {
    /// Convert linear index to Cartesian coordinates (MATLAB-style, 1-indexed).
    fn linear_to_cartesian(mut ell: usize, page_sizes: &[usize]) -> Vec<usize> {
        let m = page_sizes.len();
        let mut u = vec![0; m];

        for i in 0..m {
            let j = m - i - 1;
            let zeta = ell / page_sizes[j];
            let eta = ell % page_sizes[j];
            u[j] = zeta + if eta != 0 { 1 } else { 0 };
            ell -= page_sizes[j] * (zeta - if eta == 0 { 1 } else { 0 });
        }
        u
    }

    /// Convert Cartesian coordinates to linear index (MATLAB-style, 1-indexed).
    fn cartesian_to_linear(u: &[usize], dimensions: &[usize]) -> usize {
        let mut ell = u[0];
        let mut pi = 1;

        for i in 1..u.len() {
            pi *= dimensions[i - 1];
            ell += pi * (u[i] - 1);
        }
        ell - 1
    }

    /// Generate multi-sensor association matrices.
    fn generate_association_matrices(
        tracks: &[super::types::Track],
        measurements: &MeasurementsMultisensor,
        sensors: &SensorConfig,
        motion: &MotionModel,
    ) -> (Vec<f64>, Vec<MultisensorPosterior>, Vec<usize>) {
        let num_sensors = sensors.num_sensors();
        let num_objects = tracks.len();
        let x_dim = motion.x_dim();

        let mut dimensions = vec![0; num_sensors + 1];
        for s in 0..num_sensors {
            dimensions[s] = measurements[s].len() + 1;
        }
        dimensions[num_sensors] = num_objects;

        let num_entries: usize = dimensions.iter().product();

        let mut page_sizes = vec![1; num_sensors + 1];
        for i in 1..=num_sensors {
            page_sizes[i] = page_sizes[i - 1] * dimensions[i - 1];
        }

        let mut log_likelihoods = vec![0.0; num_entries];
        let mut posteriors = vec![
            MultisensorPosterior {
                existence: 0.0,
                mean: DVector::zeros(x_dim),
                covariance: DMatrix::zeros(x_dim, x_dim),
            };
            num_entries
        ];

        for ell in 0..num_entries {
            let u = Self::linear_to_cartesian(ell + 1, &page_sizes);
            let obj_idx = u[num_sensors] - 1;
            let associations: Vec<usize> = u[0..num_sensors].iter().map(|&x| x - 1).collect();

            let (log_l, posterior) = Self::compute_log_likelihood(
                obj_idx,
                &associations,
                tracks,
                measurements,
                sensors,
                motion,
            );

            log_likelihoods[ell] = log_l;
            posteriors[ell] = posterior;
        }

        (log_likelihoods, posteriors, dimensions)
    }

    /// Compute log-likelihood and posterior for a single object-association pair.
    fn compute_log_likelihood(
        obj_idx: usize,
        associations: &[usize],
        tracks: &[super::types::Track],
        measurements: &MeasurementsMultisensor,
        sensors: &SensorConfig,
        motion: &MotionModel,
    ) -> (f64, MultisensorPosterior) {
        let track = &tracks[obj_idx];
        let (prior_mean, prior_cov) = match (track.primary_mean(), track.primary_covariance()) {
            (Some(m), Some(c)) => (m.clone(), c.clone()),
            _ => {
                return (
                    f64::NEG_INFINITY,
                    MultisensorPosterior {
                        existence: 0.0,
                        mean: DVector::zeros(motion.x_dim()),
                        covariance: DMatrix::identity(motion.x_dim(), motion.x_dim()),
                    },
                );
            }
        };

        let num_sensors = associations.len();
        let detecting: Vec<bool> = associations.iter().map(|&a| a > 0).collect();
        let num_detections: usize = detecting.iter().filter(|&&x| x).count();

        if num_detections > 0 {
            let z_dim = sensors.sensors[0].z_dim();
            let z_dim_total = z_dim * num_detections;
            let x_dim = motion.x_dim();

            let mut z = DVector::zeros(z_dim_total);
            let mut c = DMatrix::zeros(z_dim_total, x_dim);
            let mut q_blocks = Vec::new();

            let mut counter = 0;
            for s in 0..num_sensors {
                if detecting[s] {
                    let sensor = &sensors.sensors[s];
                    let meas_idx = associations[s] - 1;
                    let start = z_dim * counter;

                    z.rows_mut(start, z_dim)
                        .copy_from(&measurements[s][meas_idx]);
                    c.view_mut((start, 0), (z_dim, x_dim))
                        .copy_from(&sensor.observation_matrix);
                    q_blocks.push(sensor.measurement_noise.clone());
                    counter += 1;
                }
            }

            let mut q = DMatrix::zeros(z_dim_total, z_dim_total);
            let mut offset = 0;
            for q_block in &q_blocks {
                q.view_mut((offset, offset), (z_dim, z_dim))
                    .copy_from(q_block);
                offset += z_dim;
            }

            let nu = &z - &c * &prior_mean;
            let s_mat = &c * &prior_cov * c.transpose() + &q;

            let s_inv = match robust_inverse(&s_mat) {
                Some(inv) => inv,
                None => {
                    return (
                        f64::NEG_INFINITY,
                        MultisensorPosterior {
                            existence: 0.0,
                            mean: prior_mean,
                            covariance: prior_cov,
                        },
                    );
                }
            };

            let k = &prior_cov * c.transpose() * &s_inv;
            let log_eta = log_gaussian_normalizing_constant(&s_mat, z_dim_total);

            let mut log_pd = 0.0;
            for (sensor, &is_detecting) in sensors.sensors.iter().zip(detecting.iter()) {
                let p_d = sensor.detection_probability;
                log_pd += if is_detecting {
                    p_d.ln()
                } else {
                    (1.0 - p_d).ln()
                };
            }

            let log_kappa: f64 = detecting
                .iter()
                .enumerate()
                .filter(|(_, &d)| d)
                .map(|(s, _)| sensors.sensors[s].clutter_density().ln())
                .sum();

            let log_l =
                track.existence.ln() + log_pd + log_eta - 0.5 * nu.dot(&(&s_inv * &nu)) - log_kappa;

            let post_mean = &prior_mean + &k * &nu;
            let post_cov = (DMatrix::identity(x_dim, x_dim) - &k * &c) * &prior_cov;

            (
                log_l,
                MultisensorPosterior {
                    existence: 1.0,
                    mean: post_mean,
                    covariance: post_cov,
                },
            )
        } else {
            let mut prob_no_detect = 1.0;
            for sensor in &sensors.sensors {
                prob_no_detect *= 1.0 - sensor.detection_probability;
            }

            let r = track.existence;
            let numerator = r * prob_no_detect;
            let denominator = 1.0 - r + numerator;

            let log_l = denominator.ln();
            let post_r = numerator / denominator;

            (
                log_l,
                MultisensorPosterior {
                    existence: post_r,
                    mean: prior_mean,
                    covariance: prior_cov,
                },
            )
        }
    }

    /// Generate posterior hypotheses from association samples.
    fn generate_posterior_hypotheses(
        hypotheses: &mut Vec<Hypothesis>,
        samples: &[Vec<usize>],
        log_likelihoods: &[f64],
        posteriors: &[MultisensorPosterior],
        dimensions: &[usize],
    ) {
        if samples.is_empty() {
            return;
        }

        let num_sensors = dimensions.len() - 1;
        let num_objects = dimensions[num_sensors];
        let mut new_hypotheses = Vec::new();

        for prior_hyp in hypotheses.iter() {
            for sample in samples {
                let mut new_hyp = prior_hyp.clone();
                let mut log_likelihood_sum = 0.0;

                for i in 0..num_objects.min(new_hyp.tracks.len()) {
                    let mut u: Vec<usize> = Vec::with_capacity(num_sensors + 1);
                    for s in 0..num_sensors {
                        let v_is = sample[s * num_objects + i];
                        u.push(v_is + 1);
                    }
                    u.push(i + 1);

                    let ell = Self::cartesian_to_linear(&u, dimensions);
                    log_likelihood_sum += log_likelihoods[ell];

                    let posterior = &posteriors[ell];
                    new_hyp.tracks[i].existence = posterior.existence;
                    new_hyp.tracks[i].components.clear();
                    new_hyp.tracks[i].components.push(GaussianComponent {
                        weight: 1.0,
                        mean: posterior.mean.clone(),
                        covariance: posterior.covariance.clone(),
                    });
                }

                new_hyp.log_weight += log_likelihood_sum;
                new_hypotheses.push(new_hyp);
            }
        }

        *hypotheses = new_hypotheses;
    }
}

impl<A: AssociatorMultisensor> LmbmAssociator for MultisensorLmbmStrategy<A> {
    type Measurements = MeasurementsMultisensor;

    fn associate_and_update<R: Rng>(
        &self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        sensor_config: &SensorConfig,
        motion: &MotionModel,
        association_config: &AssociationConfig,
    ) -> Result<Option<LmbmAssociationIntermediate>, FilterError> {
        if hypotheses.is_empty() || hypotheses[0].tracks.is_empty() {
            return Ok(None);
        }

        let tracks = &hypotheses[0].tracks;
        let sensors = sensor_config;

        let (log_likelihoods, posteriors, dimensions) =
            Self::generate_association_matrices(tracks, measurements, sensors, motion);

        let result =
            self.associator
                .associate(rng, &log_likelihoods, &dimensions, association_config)?;

        Self::generate_posterior_hypotheses(
            hypotheses,
            &result.samples,
            &log_likelihoods,
            &posteriors,
            &dimensions,
        );

        Ok(None)
    }

    fn update_existence_no_measurements(
        &self,
        hypotheses: &mut Vec<Hypothesis>,
        sensor_config: &SensorConfig,
    ) {
        let sensors = sensor_config;
        let mut prob_no_detect = 1.0;
        for sensor in &sensors.sensors {
            prob_no_detect *= 1.0 - sensor.detection_probability;
        }

        for hyp in hypotheses.iter_mut() {
            for track in &mut hyp.tracks {
                let r = track.existence;
                let numerator = r * prob_no_detect;
                let denominator = 1.0 - r + numerator;
                track.existence = numerator / denominator;
            }
        }
    }

    fn measurements_empty(&self, measurements: &Self::Measurements) -> bool {
        measurements.iter().all(|s| s.is_empty())
    }

    fn validate_measurements(
        &self,
        measurements: &Self::Measurements,
        sensor_config: &SensorConfig,
    ) -> Result<(), FilterError> {
        if measurements.len() != sensor_config.num_sensors() {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensor measurements, got {}",
                sensor_config.num_sensors(),
                measurements.len()
            )));
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "MultisensorLmbm"
    }
}

// ============================================================================
// LmbmStrategy - LMBM filter update strategy
// ============================================================================

/// LMBM filter update strategy.
///
/// This strategy implements the LMBM (Labeled Multi-Bernoulli Mixture) tracking
/// algorithm with hard assignment posteriors and multiple hypotheses.
///
/// # Type Parameters
///
/// * `S` - The LMBM associator determining single/multi-sensor behavior
///
/// # Configuration
///
/// The strategy owns its LMBM-specific pruning configuration:
/// - Hypothesis weight threshold
/// - Maximum number of hypotheses
/// - Whether to use EAP (Expected A Posteriori) vs MAP estimation
///
/// Common pruning configuration (existence threshold, trajectory length) is
/// passed via `UpdateContext`.
#[derive(Clone)]
pub struct LmbmStrategy<S: LmbmAssociator + Clone> {
    /// The LMBM associator (handles association AND posterior generation)
    pub(crate) inner: S,
    /// LMBM-specific pruning configuration
    pub(crate) prune_config: LmbmPruneConfig,
}

impl<S: LmbmAssociator + Clone> LmbmStrategy<S> {
    /// Create a new LMBM strategy with the given associator and configuration.
    pub fn new(inner: S, prune_config: LmbmPruneConfig) -> Self {
        Self {
            inner,
            prune_config,
        }
    }

    /// Create with default pruning configuration.
    pub fn with_defaults(inner: S) -> Self {
        Self::new(inner, LmbmPruneConfig::default())
    }
}

/// Helper to predict hypotheses for LMBM.
fn predict_hypotheses(
    hypotheses: &mut [Hypothesis],
    motion: &MotionModel,
    birth: &BirthModel,
    timestep: usize,
) {
    super::common_ops::predict_all_hypotheses(hypotheses, motion, birth, timestep);
}

// ============================================================================
// LmbmStrategy for single-sensor (SingleSensorLmbmStrategy)
// ============================================================================

impl<A: Associator + Clone> UpdateStrategy for LmbmStrategy<SingleSensorLmbmStrategy<A>> {
    type Measurements = [DVector<f64>];

    fn predict(&self, hypotheses: &mut Vec<Hypothesis>, ctx: &UpdateContext, timestep: usize) {
        if hypotheses.is_empty() {
            hypotheses.push(Hypothesis::empty());
        }
        predict_hypotheses(hypotheses, ctx.motion, ctx.birth, timestep);
    }

    fn update<R: Rng>(
        &mut self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        ctx: &UpdateContext,
    ) -> Result<UpdateIntermediate, FilterError> {
        // Convert slice to Vec for trait interface
        let measurements_vec: Vec<DVector<f64>> = measurements.to_vec();

        if measurements_vec.is_empty() {
            self.inner
                .update_existence_no_measurements(hypotheses, ctx.sensors);
            return Ok(UpdateIntermediate::default());
        }

        if hypotheses.is_empty() || hypotheses[0].tracks.is_empty() {
            return Ok(UpdateIntermediate::default());
        }

        let intermediate = self.inner.associate_and_update(
            rng,
            hypotheses,
            &measurements_vec,
            ctx.sensors,
            ctx.motion,
            ctx.association_config,
        )?;

        match intermediate {
            Some(i) => Ok(UpdateIntermediate {
                association_matrices: i.matrices,
                association_result: i.result,
                sensor_updates: None, // LMBM - no per-sensor data in current implementation
            }),
            None => Ok(UpdateIntermediate::default()),
        }
    }

    fn prune(
        &self,
        hypotheses: &mut Vec<Hypothesis>,
        trajectories: &mut Vec<Trajectory>,
        ctx: &UpdateContext,
    ) -> Vec<bool> {
        super::common_ops::normalize_gate_and_prune_tracks(
            hypotheses,
            trajectories,
            self.prune_config.hypothesis_weight_threshold,
            self.prune_config.max_hypotheses,
            ctx.common_prune.existence_threshold,
            ctx.common_prune.min_trajectory_length,
        )
    }

    fn extract(
        &self,
        hypotheses: &[Hypothesis],
        timestamp: usize,
        _ctx: &UpdateContext,
    ) -> StateEstimate {
        super::common_ops::extract_hypothesis_estimates(
            hypotheses,
            timestamp,
            self.prune_config.use_eap,
        )
    }

    fn name(&self) -> &'static str {
        "LMBM"
    }

    fn is_hypothesis_based(&self) -> bool {
        true
    }

    fn lmbm_config(&self) -> Option<LmbmPruneConfig> {
        Some(self.prune_config.clone())
    }

    fn capture_hypotheses(
        &self,
        hypotheses: &[super::types::Hypothesis],
    ) -> Option<Vec<super::types::Hypothesis>> {
        Some(hypotheses.to_vec())
    }

    fn wrap_objects_likely_to_exist(&self, keep_mask: Vec<bool>) -> Option<Vec<bool>> {
        Some(keep_mask)
    }

    fn build_config_snapshot(
        &self,
        motion: &super::config::MotionModel,
        sensors: &super::config::SensorConfig,
        birth: &super::config::BirthModel,
        association_config: &super::config::AssociationConfig,
        common_prune: &CommonPruneConfig,
    ) -> super::config::FilterConfigSnapshot {
        super::config::FilterConfigSnapshot::single_sensor_lmbm(
            self.name(),
            motion,
            sensors.single(),
            birth,
            association_config,
            common_prune.existence_threshold,
            common_prune.min_trajectory_length,
            &self.prune_config,
        )
    }
}

// ============================================================================
// LmbmStrategy for multi-sensor (MultisensorLmbmStrategy)
// ============================================================================

impl<A: AssociatorMultisensor + Clone> UpdateStrategy for LmbmStrategy<MultisensorLmbmStrategy<A>> {
    type Measurements = MeasurementsMultisensor;

    fn predict(&self, hypotheses: &mut Vec<Hypothesis>, ctx: &UpdateContext, timestep: usize) {
        if hypotheses.is_empty() {
            hypotheses.push(Hypothesis::empty());
        }
        predict_hypotheses(hypotheses, ctx.motion, ctx.birth, timestep);
    }

    fn update<R: Rng>(
        &mut self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        ctx: &UpdateContext,
    ) -> Result<UpdateIntermediate, FilterError> {
        self.inner
            .validate_measurements(measurements, ctx.sensors)?;

        let has_measurements = !self.inner.measurements_empty(measurements);

        if !has_measurements {
            self.inner
                .update_existence_no_measurements(hypotheses, ctx.sensors);
            return Ok(UpdateIntermediate::default());
        }

        if hypotheses.is_empty() {
            return Ok(UpdateIntermediate::default());
        }

        self.inner.associate_and_update(
            rng,
            hypotheses,
            measurements,
            ctx.sensors,
            ctx.motion,
            ctx.association_config,
        )?;

        // Multi-sensor LMBM doesn't expose intermediate data
        Ok(UpdateIntermediate::default())
    }

    fn prune(
        &self,
        hypotheses: &mut Vec<Hypothesis>,
        trajectories: &mut Vec<Trajectory>,
        ctx: &UpdateContext,
    ) -> Vec<bool> {
        // For multi-sensor LMBM, we do normalize + gate separately
        super::common_ops::normalize_and_gate_hypotheses(
            hypotheses,
            self.prune_config.hypothesis_weight_threshold,
            self.prune_config.max_hypotheses,
        );

        let objects_likely_to_exist = super::common_ops::compute_objects_likely_to_exist(
            hypotheses,
            ctx.common_prune.existence_threshold,
        );

        super::common_ops::gate_hypothesis_tracks(
            hypotheses,
            trajectories,
            ctx.common_prune.existence_threshold,
            ctx.common_prune.min_trajectory_length,
        );

        objects_likely_to_exist
    }

    fn extract(
        &self,
        hypotheses: &[Hypothesis],
        timestamp: usize,
        _ctx: &UpdateContext,
    ) -> StateEstimate {
        super::common_ops::extract_hypothesis_estimates(
            hypotheses,
            timestamp,
            self.prune_config.use_eap,
        )
    }

    fn name(&self) -> &'static str {
        "Multisensor-LMBM"
    }

    fn is_hypothesis_based(&self) -> bool {
        true
    }

    fn lmbm_config(&self) -> Option<LmbmPruneConfig> {
        Some(self.prune_config.clone())
    }

    fn capture_hypotheses(
        &self,
        hypotheses: &[super::types::Hypothesis],
    ) -> Option<Vec<super::types::Hypothesis>> {
        Some(hypotheses.to_vec())
    }

    fn wrap_objects_likely_to_exist(&self, keep_mask: Vec<bool>) -> Option<Vec<bool>> {
        Some(keep_mask)
    }

    fn build_config_snapshot(
        &self,
        motion: &super::config::MotionModel,
        sensors: &super::config::SensorConfig,
        birth: &super::config::BirthModel,
        association_config: &super::config::AssociationConfig,
        common_prune: &CommonPruneConfig,
    ) -> super::config::FilterConfigSnapshot {
        super::config::FilterConfigSnapshot::multi_sensor_lmbm(
            self.name(),
            motion,
            sensors,
            birth,
            association_config,
            common_prune.existence_threshold,
            common_prune.min_trajectory_length,
            &self.prune_config,
        )
    }
}

// ============================================================================
// Strategy Type Aliases
// ============================================================================

use super::multisensor::fusion::{
    MergerAverageArithmetic, MergerAverageGeometric, MergerParallelUpdate,
};
use super::multisensor::traits::AssociatorMultisensorGibbs;
use super::traits::{AssociatorGibbs, AssociatorLbp};

/// Single-sensor LMB strategy with LBP associator.
pub type LmbStrategyLbp = LmbStrategy<AssociatorLbp, SingleSensorScheduler>;

/// Multi-sensor LMB strategy with sequential (IC) update and LBP associator.
pub type IcLmbStrategyLbp = LmbStrategy<AssociatorLbp, SequentialScheduler>;

/// Multi-sensor LMB strategy with arithmetic average fusion and LBP associator.
pub type AaLmbStrategyLbp = LmbStrategy<AssociatorLbp, ParallelScheduler<MergerAverageArithmetic>>;

/// Multi-sensor LMB strategy with geometric average fusion and LBP associator.
pub type GaLmbStrategyLbp = LmbStrategy<AssociatorLbp, ParallelScheduler<MergerAverageGeometric>>;

/// Multi-sensor LMB strategy with parallel update fusion and LBP associator.
pub type PuLmbStrategyLbp = LmbStrategy<AssociatorLbp, ParallelScheduler<MergerParallelUpdate>>;

/// Single-sensor LMBM strategy with Gibbs associator.
pub type LmbmStrategyGibbs = LmbmStrategy<SingleSensorLmbmStrategy<AssociatorGibbs>>;

/// Multi-sensor LMBM strategy with Gibbs associator.
pub type MultisensorLmbmStrategyGibbs =
    LmbmStrategy<MultisensorLmbmStrategy<AssociatorMultisensorGibbs>>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_prune_config_default() {
        let config = CommonPruneConfig::default();
        assert!(config.existence_threshold > 0.0);
        assert!(config.min_trajectory_length > 0);
    }

    #[test]
    fn test_lmb_prune_config_default() {
        let config = LmbPruneConfig::default();
        assert!(config.gm_weight_threshold > 0.0);
        assert!(config.max_gm_components > 0);
        assert!(config.gm_merge_threshold.is_infinite());
    }

    #[test]
    fn test_lmbm_prune_config_default() {
        let config = LmbmPruneConfig::default();
        assert!(config.hypothesis_weight_threshold > 0.0);
        assert!(config.max_hypotheses > 0);
        assert!(!config.use_eap);
    }
}
