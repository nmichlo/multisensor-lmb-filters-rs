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

use rand::Rng;

use super::config::{AssociationConfig, BirthModel, MotionModel, SensorSet};
use super::errors::FilterError;
use super::output::{StateEstimate, Trajectory};
use super::traits::AssociationResult;
use super::types::Hypothesis;
use crate::association::AssociationMatrices;

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
    pub sensors: &'a SensorSet,
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
    /// Association matrices (if available)
    pub association_matrices: Option<AssociationMatrices>,
    /// Association result (if available)
    pub association_result: Option<AssociationResult>,
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
        &self,
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
}

// ============================================================================
// LmbStrategy - LMB filter update strategy
// ============================================================================

use nalgebra::DVector;

use crate::association::AssociationBuilder;
use crate::components::prediction::predict_tracks;

use super::config::SensorModel;
use super::scheduler::{ParallelScheduler, SequentialScheduler, SingleSensorScheduler, UpdateScheduler};
use super::traits::{Associator, MarginalUpdater, Merger, Updater};
use super::types::Track;

/// LMB filter update strategy.
///
/// This strategy implements the LMB (Labeled Multi-Bernoulli) tracking algorithm
/// with marginal Gaussian mixture posteriors. It maintains a single hypothesis
/// containing multi-component tracks.
///
/// # Type Parameters
///
/// * `A` - The data association algorithm (default: [`LbpAssociator`])
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
    pub(crate) updater: MarginalUpdater,
    /// LMB-specific pruning configuration
    pub(crate) prune_config: LmbPruneConfig,
}

impl<A: Associator, S: UpdateScheduler> LmbStrategy<A, S> {
    /// Create a new LMB strategy with the given components.
    pub fn new(associator: A, scheduler: S, prune_config: LmbPruneConfig) -> Self {
        let updater = MarginalUpdater::with_thresholds(
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
        &self,
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
        &self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        ctx: &UpdateContext,
    ) -> Result<UpdateIntermediate, FilterError> {
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

        // Sequential sensor updates
        for (sensor_idx, sensor_measurements) in measurements.iter().enumerate() {
            let sensor = ctx.sensors.get(sensor_idx).unwrap();
            update_single_sensor(
                &self.associator,
                &self.updater,
                ctx.association_config,
                sensor,
                sensor_measurements,
                &mut hypotheses[0].tracks,
                rng,
            )?;
        }

        // Sequential doesn't expose single-sensor association data
        Ok(UpdateIntermediate::default())
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
}

// ============================================================================
// LmbStrategy for ParallelScheduler (AA, GA, PU-LMB)
// ============================================================================

impl<A: Associator + Clone, M: Merger + Clone> UpdateStrategy for LmbStrategy<A, ParallelScheduler<M>> {
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
        &self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        ctx: &UpdateContext,
    ) -> Result<UpdateIntermediate, FilterError> {
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

        if has_any_measurements && !hypotheses[0].tracks.is_empty() {
            // Parallel sensor updates: each sensor uses the same prior
            let mut per_sensor_tracks: Vec<Vec<Track>> = Vec::with_capacity(num_sensors);

            for (sensor_idx, sensor_measurements) in measurements.iter().enumerate() {
                let mut sensor_tracks = hypotheses[0].tracks.clone();
                let sensor = ctx.sensors.get(sensor_idx).unwrap();
                update_single_sensor(
                    &self.associator,
                    &self.updater,
                    ctx.association_config,
                    sensor,
                    sensor_measurements,
                    &mut sensor_tracks,
                    rng,
                )?;
                per_sensor_tracks.push(sensor_tracks);
            }

            // Set prior for PU-LMB decorrelation (no-op for other mergers)
            // Note: We can't modify self here since &self is immutable
            // The UnifiedFilter will handle this at a higher level

            // Fuse per-sensor posteriors
            hypotheses[0].tracks = self.scheduler.merger().merge(&per_sensor_tracks, None);
        } else if !has_any_measurements {
            // No measurements: update existence for all sensors' missed detection
            let detection_probs = ctx.sensors.detection_probabilities();
            for track in &mut hypotheses[0].tracks {
                track.existence =
                    crate::components::update::update_existence_no_detection_multisensor(
                        track.existence,
                        &detection_probs,
                    );
            }
        }

        // Parallel doesn't expose per-sensor association data in intermediate
        Ok(UpdateIntermediate::default())
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
}

// ============================================================================
// LmbmStrategy - LMBM filter update strategy
// ============================================================================

use super::core_lmbm::LmbmAssociator;
use super::multisensor::MultisensorMeasurements;

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
        Self { inner, prune_config }
    }

    /// Create with default pruning configuration.
    pub fn with_defaults(inner: S) -> Self {
        Self::new(inner, LmbmPruneConfig::default())
    }
}

/// Helper to predict hypotheses for LMBM.
fn predict_hypotheses(
    hypotheses: &mut Vec<Hypothesis>,
    motion: &MotionModel,
    birth: &BirthModel,
    timestep: usize,
) {
    super::common_ops::predict_all_hypotheses(hypotheses, motion, birth, timestep);
}

// ============================================================================
// LmbmStrategy for single-sensor (SingleSensorLmbmStrategy)
// ============================================================================

use super::core_lmbm::SingleSensorLmbmStrategy;

impl<A: Associator + Clone> UpdateStrategy for LmbmStrategy<SingleSensorLmbmStrategy<A>> {
    type Measurements = [DVector<f64>];

    fn predict(&self, hypotheses: &mut Vec<Hypothesis>, ctx: &UpdateContext, timestep: usize) {
        if hypotheses.is_empty() {
            hypotheses.push(Hypothesis::empty());
        }
        predict_hypotheses(hypotheses, ctx.motion, ctx.birth, timestep);
    }

    fn update<R: Rng>(
        &self,
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
}

// ============================================================================
// LmbmStrategy for multi-sensor (MultisensorLmbmStrategy)
// ============================================================================

use super::core_lmbm::MultisensorLmbmStrategy;
use super::multisensor::traits::MultisensorAssociator;

impl<A: MultisensorAssociator + Clone> UpdateStrategy
    for LmbmStrategy<MultisensorLmbmStrategy<A>>
{
    type Measurements = MultisensorMeasurements;

    fn predict(&self, hypotheses: &mut Vec<Hypothesis>, ctx: &UpdateContext, timestep: usize) {
        if hypotheses.is_empty() {
            hypotheses.push(Hypothesis::empty());
        }
        predict_hypotheses(hypotheses, ctx.motion, ctx.birth, timestep);
    }

    fn update<R: Rng>(
        &self,
        rng: &mut R,
        hypotheses: &mut Vec<Hypothesis>,
        measurements: &Self::Measurements,
        ctx: &UpdateContext,
    ) -> Result<UpdateIntermediate, FilterError> {
        self.inner.validate_measurements(measurements, ctx.sensors)?;

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
}

// ============================================================================
// Strategy Type Aliases
// ============================================================================

use super::multisensor::fusion::{
    ArithmeticAverageMerger, GeometricAverageMerger, ParallelUpdateMerger,
};
use super::multisensor::traits::MultisensorGibbsAssociator;
use super::traits::{GibbsAssociator, LbpAssociator};

/// Single-sensor LMB strategy with LBP associator.
pub type LmbStrategyLbp = LmbStrategy<LbpAssociator, SingleSensorScheduler>;

/// Multi-sensor LMB strategy with sequential (IC) update and LBP associator.
pub type IcLmbStrategyLbp = LmbStrategy<LbpAssociator, SequentialScheduler>;

/// Multi-sensor LMB strategy with arithmetic average fusion and LBP associator.
pub type AaLmbStrategyLbp = LmbStrategy<LbpAssociator, ParallelScheduler<ArithmeticAverageMerger>>;

/// Multi-sensor LMB strategy with geometric average fusion and LBP associator.
pub type GaLmbStrategyLbp = LmbStrategy<LbpAssociator, ParallelScheduler<GeometricAverageMerger>>;

/// Multi-sensor LMB strategy with parallel update fusion and LBP associator.
pub type PuLmbStrategyLbp = LmbStrategy<LbpAssociator, ParallelScheduler<ParallelUpdateMerger>>;

/// Single-sensor LMBM strategy with Gibbs associator.
pub type LmbmStrategyGibbs = LmbmStrategy<SingleSensorLmbmStrategy<GibbsAssociator>>;

/// Multi-sensor LMBM strategy with Gibbs associator.
pub type MultisensorLmbmStrategyGibbs =
    LmbmStrategy<MultisensorLmbmStrategy<MultisensorGibbsAssociator>>;

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
