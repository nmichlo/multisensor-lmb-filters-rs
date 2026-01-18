//! Unified filter architecture for LMB-family filters.
//!
//! This module provides [`UnifiedFilter`], a generic filter struct parameterized
//! by an [`UpdateStrategy`]. This enables a single filter implementation to
//! support all LMB and LMBM variants through strategy composition.
//! ```

use nalgebra::DVector;
use rand::Rng;

use super::config::{AssociationConfig, BirthModel, MotionModel, SensorSet};
use super::errors::FilterError;
use super::multisensor::MultisensorMeasurements;
use super::output::{StateEstimate, Trajectory};
use super::scheduler::{ParallelScheduler, SequentialScheduler, SingleSensorScheduler};
use super::strategy::{
    CommonPruneConfig, LmbStrategy, LmbmStrategy, MultisensorLmbmStrategy,
    SingleSensorLmbmStrategy, UpdateContext, UpdateIntermediate, UpdateStrategy,
};
use super::traits::Filter;
use super::types::{Hypothesis, StepDetailedOutput, Track};

// ============================================================================
// UnifiedFilter
// ============================================================================

/// Generic filter parameterized by update strategy.
///
/// This struct provides a single implementation that works with any [`UpdateStrategy`],
/// enabling code reuse across all LMB and LMBM filter variants.
///
/// # Type Parameters
///
/// * `S` - The update strategy determining the tracking algorithm
///
/// # Internal Representation
///
/// The filter state is stored as `Vec<Hypothesis>`:
/// - **LMB strategies**: Single hypothesis with `log_weight=0`, multi-component tracks
/// - **LMBM strategies**: Multiple hypotheses with varying weights, single-component tracks
#[derive(Clone)]
pub struct UnifiedFilter<S: UpdateStrategy> {
    /// Motion model (dynamics, survival probability)
    motion: MotionModel,
    /// Sensor configuration (single or multi-sensor)
    sensors: SensorSet,
    /// Birth model (where new objects can appear)
    birth: BirthModel,
    /// Association algorithm configuration
    association_config: AssociationConfig,
    /// Common pruning configuration
    common_prune: CommonPruneConfig,

    /// Current hypotheses (single for LMB, multiple for LMBM)
    hypotheses: Vec<Hypothesis>,
    /// Archived trajectories for discarded tracks
    trajectories: Vec<Trajectory>,

    /// The update strategy
    strategy: S,
}

impl<S: UpdateStrategy> UnifiedFilter<S> {
    /// Create a new unified filter.
    pub fn new(
        motion: MotionModel,
        sensors: SensorSet,
        birth: BirthModel,
        association_config: AssociationConfig,
        common_prune: CommonPruneConfig,
        strategy: S,
    ) -> Self {
        Self {
            motion,
            sensors,
            birth,
            association_config,
            common_prune,
            hypotheses: Vec::new(),
            trajectories: Vec::new(),
            strategy,
        }
    }

    /// Create with default common pruning configuration.
    pub fn with_defaults(
        motion: MotionModel,
        sensors: SensorSet,
        birth: BirthModel,
        association_config: AssociationConfig,
        strategy: S,
    ) -> Self {
        Self::new(
            motion,
            sensors,
            birth,
            association_config,
            CommonPruneConfig::default(),
            strategy,
        )
    }

    /// Get the update strategy.
    pub fn strategy(&self) -> &S {
        &self.strategy
    }

    /// Get mutable reference to the update strategy.
    pub fn strategy_mut(&mut self) -> &mut S {
        &mut self.strategy
    }

    /// Get the algorithm name from the strategy.
    pub fn algorithm_name(&self) -> &'static str {
        self.strategy.name()
    }

    /// Whether this filter maintains multiple hypotheses.
    pub fn is_hypothesis_based(&self) -> bool {
        self.strategy.is_hypothesis_based()
    }

    /// Number of sensors.
    pub fn num_sensors(&self) -> usize {
        self.sensors.num_sensors()
    }

    /// Get the current hypotheses.
    pub fn hypotheses(&self) -> &[Hypothesis] {
        &self.hypotheses
    }

    /// Get the archived trajectories.
    pub fn trajectories(&self) -> &[Trajectory] {
        &self.trajectories
    }

    /// Get all tracks from the highest-weight hypothesis.
    pub fn get_tracks(&self) -> Vec<Track> {
        if self.hypotheses.is_empty() {
            return Vec::new();
        }

        if self.hypotheses.len() == 1 {
            return self.hypotheses[0].tracks.clone();
        }

        // For multiple hypotheses, return tracks from the highest-weight one
        self.hypotheses
            .iter()
            .max_by(|a, b| a.log_weight.partial_cmp(&b.log_weight).unwrap())
            .map(|h| h.tracks.clone())
            .unwrap_or_default()
    }

    /// Set the internal hypotheses (for fixture testing).
    pub fn set_hypotheses(&mut self, hypotheses: Vec<Hypothesis>) {
        self.hypotheses = hypotheses;
    }

    /// Set tracks directly (for fixture testing).
    /// This sets a single hypothesis with the given tracks.
    pub fn set_tracks(&mut self, tracks: Vec<Track>) {
        self.hypotheses = vec![Hypothesis::lmb(tracks)];
    }

    /// Get configuration as FilterConfigSnapshot (for debugging).
    pub fn get_config(&self) -> super::config::FilterConfigSnapshot {
        self.strategy.build_config_snapshot(
            &self.motion,
            &self.sensors,
            &self.birth,
            &self.association_config,
            &self.common_prune,
        )
    }

    /// Core step implementation.
    fn step_core<R: Rng>(
        &mut self,
        rng: &mut R,
        measurements: &S::Measurements,
        timestep: usize,
    ) -> Result<UpdateIntermediate, FilterError> {
        // Prediction
        {
            let ctx = UpdateContext {
                motion: &self.motion,
                sensors: &self.sensors,
                birth: &self.birth,
                association_config: &self.association_config,
                common_prune: &self.common_prune,
            };
            self.strategy.predict(&mut self.hypotheses, &ctx, timestep);
        }

        self.strategy
            .init_birth_trajectories(&mut self.hypotheses, super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        // Update
        let intermediate = {
            let ctx = UpdateContext {
                motion: &self.motion,
                sensors: &self.sensors,
                birth: &self.birth,
                association_config: &self.association_config,
                common_prune: &self.common_prune,
            };
            self.strategy
                .update(rng, &mut self.hypotheses, measurements, &ctx)?
        };

        // Prune
        {
            let ctx = UpdateContext {
                motion: &self.motion,
                sensors: &self.sensors,
                birth: &self.birth,
                association_config: &self.association_config,
                common_prune: &self.common_prune,
            };
            self.strategy
                .prune(&mut self.hypotheses, &mut self.trajectories, &ctx);
        }

        // Update trajectories
        self.strategy
            .update_trajectories(&mut self.hypotheses, timestep);

        Ok(intermediate)
    }

    /// Build update context (for use by callers, not internally).
    fn build_context(&self) -> UpdateContext<'_> {
        UpdateContext {
            motion: &self.motion,
            sensors: &self.sensors,
            birth: &self.birth,
            association_config: &self.association_config,
            common_prune: &self.common_prune,
        }
    }

    /// Detailed step implementation that captures intermediate state for fixture validation.
    ///
    /// This method runs the full predict-update-prune cycle while capturing
    /// intermediate state at each stage for debugging and fixture validation.
    ///
    /// # Returns
    /// `StepDetailedOutput` containing intermediate state at each processing stage.
    pub fn step_detailed<R: Rng>(
        &mut self,
        rng: &mut R,
        measurements: &S::Measurements,
        timestep: usize,
    ) -> Result<StepDetailedOutput, FilterError> {
        // Prediction (use block to limit context lifetime)
        {
            let ctx = UpdateContext {
                motion: &self.motion,
                sensors: &self.sensors,
                birth: &self.birth,
                association_config: &self.association_config,
                common_prune: &self.common_prune,
            };
            self.strategy.predict(&mut self.hypotheses, &ctx, timestep);
        }
        self.strategy
            .init_birth_trajectories(&mut self.hypotheses, super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        // Capture predicted tracks (from first hypothesis for LMB, or all for LMBM)
        let predicted_tracks = self.get_tracks();
        // LMBM strategies capture hypotheses, LMB strategies return None
        let predicted_hypotheses = self.strategy.capture_hypotheses(&self.hypotheses);

        // Update (use block to limit context lifetime)
        let intermediate = {
            let ctx = UpdateContext {
                motion: &self.motion,
                sensors: &self.sensors,
                birth: &self.birth,
                association_config: &self.association_config,
                common_prune: &self.common_prune,
            };
            self.strategy
                .update(rng, &mut self.hypotheses, measurements, &ctx)?
        };

        // Capture pre-normalization hypotheses (after update, before prune)
        // LMBM strategies capture hypotheses, LMB strategies return None
        let pre_normalization_hypotheses = self.strategy.capture_hypotheses(&self.hypotheses);

        // Capture updated tracks before pruning
        let updated_tracks = self.get_tracks();

        // Prune (use block to limit context lifetime)
        // For LMBM, prune() normalizes hypotheses first, then computes objects_likely_to_exist,
        // then prunes tracks. It returns the keep_mask (objects_likely_to_exist).
        let objects_likely_to_exist = {
            let ctx = UpdateContext {
                motion: &self.motion,
                sensors: &self.sensors,
                birth: &self.birth,
                association_config: &self.association_config,
                common_prune: &self.common_prune,
            };
            let keep_mask = self
                .strategy
                .prune(&mut self.hypotheses, &mut self.trajectories, &ctx);
            // LMBM strategies wrap the mask, LMB strategies return None
            self.strategy.wrap_objects_likely_to_exist(keep_mask)
        };

        // Capture normalized hypotheses (after prune, for LMBM)
        // LMBM strategies capture hypotheses, LMB strategies return None
        let normalized_hypotheses = self.strategy.capture_hypotheses(&self.hypotheses);

        // Update trajectories
        self.strategy
            .update_trajectories(&mut self.hypotheses, timestep);

        // Extract final estimates (use block to limit context lifetime)
        let final_estimate = {
            let ctx = UpdateContext {
                motion: &self.motion,
                sensors: &self.sensors,
                birth: &self.birth,
                association_config: &self.association_config,
                common_prune: &self.common_prune,
            };
            self.strategy.extract(&self.hypotheses, timestep, &ctx)
        };

        // Compute cardinality from updated tracks (before pruning)
        // MATLAB computes cardinality from fused tracks before the gating step
        let cardinality = super::common_ops::compute_cardinality(&updated_tracks);

        Ok(StepDetailedOutput {
            predicted_tracks,
            association_matrices: intermediate.association_matrices,
            association_result: intermediate.association_result,
            updated_tracks,
            cardinality,
            final_estimate,
            sensor_updates: intermediate.sensor_updates,
            predicted_hypotheses,
            pre_normalization_hypotheses,
            normalized_hypotheses,
            objects_likely_to_exist,
        })
    }
}

// ============================================================================
// Filter trait implementation for single-sensor LMB
// ============================================================================

impl<A: super::traits::Associator + Clone> Filter
    for UnifiedFilter<LmbStrategy<A, SingleSensorScheduler>>
{
    type State = Vec<Hypothesis>;
    type Measurements = Vec<DVector<f64>>;

    fn step<R: Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.step_core(rng, measurements, timestep)?;
        let ctx = self.build_context();
        Ok(self.strategy.extract(&self.hypotheses, timestep, &ctx))
    }

    fn state(&self) -> &Self::State {
        &self.hypotheses
    }

    fn reset(&mut self) {
        self.hypotheses.clear();
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
// Filter trait implementation for multi-sensor LMB (Sequential)
// ============================================================================

impl<A: super::traits::Associator + Clone> Filter
    for UnifiedFilter<LmbStrategy<A, SequentialScheduler>>
{
    type State = Vec<Hypothesis>;
    type Measurements = Vec<Vec<DVector<f64>>>;

    fn step<R: Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.step_core(rng, measurements, timestep)?;
        let ctx = self.build_context();
        Ok(self.strategy.extract(&self.hypotheses, timestep, &ctx))
    }

    fn state(&self) -> &Self::State {
        &self.hypotheses
    }

    fn reset(&mut self) {
        self.hypotheses.clear();
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
// Filter trait implementation for multi-sensor LMB (Parallel)
// ============================================================================

impl<A: super::traits::Associator + Clone, M: super::traits::Merger + Clone> Filter
    for UnifiedFilter<LmbStrategy<A, ParallelScheduler<M>>>
{
    type State = Vec<Hypothesis>;
    type Measurements = Vec<Vec<DVector<f64>>>;

    fn step<R: Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.step_core(rng, measurements, timestep)?;
        let ctx = self.build_context();
        Ok(self.strategy.extract(&self.hypotheses, timestep, &ctx))
    }

    fn state(&self) -> &Self::State {
        &self.hypotheses
    }

    fn reset(&mut self) {
        self.hypotheses.clear();
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
// Filter trait implementation for single-sensor LMBM
// ============================================================================

impl<A: super::traits::Associator + Clone> Filter
    for UnifiedFilter<LmbmStrategy<SingleSensorLmbmStrategy<A>>>
{
    type State = Vec<Hypothesis>;
    type Measurements = Vec<DVector<f64>>;

    fn step<R: Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.step_core(rng, measurements, timestep)?;
        let ctx = self.build_context();
        Ok(self.strategy.extract(&self.hypotheses, timestep, &ctx))
    }

    fn state(&self) -> &Self::State {
        &self.hypotheses
    }

    fn reset(&mut self) {
        self.hypotheses.clear();
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
// Filter trait implementation for multi-sensor LMBM
// ============================================================================

impl<A: super::multisensor::traits::MultisensorAssociator + Clone> Filter
    for UnifiedFilter<LmbmStrategy<MultisensorLmbmStrategy<A>>>
{
    type State = Vec<Hypothesis>;
    type Measurements = MultisensorMeasurements;

    fn step<R: Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.step_core(rng, measurements, timestep)?;
        let ctx = self.build_context();
        Ok(self.strategy.extract(&self.hypotheses, timestep, &ctx))
    }

    fn state(&self) -> &Self::State {
        &self.hypotheses
    }

    fn reset(&mut self) {
        self.hypotheses.clear();
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmb::config::{BirthLocation, SensorModel};
    use crate::lmb::strategy::LmbStrategyLbp;
    use crate::lmb::traits::LbpAssociator;
    use nalgebra::DMatrix;

    fn create_motion() -> MotionModel {
        MotionModel::constant_velocity_2d(1.0, 0.1, 0.99)
    }

    fn create_sensor() -> SensorModel {
        SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0)
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
    fn test_unified_filter_creation() {
        let strategy = LmbStrategy::with_defaults(LbpAssociator, SingleSensorScheduler);

        let filter: UnifiedFilter<LmbStrategyLbp> = UnifiedFilter::with_defaults(
            create_motion(),
            create_sensor().into(),
            create_birth(),
            AssociationConfig::default(),
            strategy,
        );

        assert_eq!(filter.algorithm_name(), "LMB");
        assert!(!filter.is_hypothesis_based());
        assert_eq!(filter.num_sensors(), 1);
    }

    #[test]
    fn test_unified_lmb_filter_step() {
        let strategy = LmbStrategy::with_defaults(LbpAssociator, SingleSensorScheduler);

        let mut filter: UnifiedFilter<LmbStrategyLbp> = UnifiedFilter::with_defaults(
            create_motion(),
            create_sensor().into(),
            create_birth(),
            AssociationConfig::default(),
            strategy,
        );

        let mut rng = rand::thread_rng();
        let measurements = vec![DVector::from_vec(vec![0.0, 0.0])];

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_unified_filter_reset() {
        let strategy = LmbStrategy::with_defaults(LbpAssociator, SingleSensorScheduler);

        let mut filter: UnifiedFilter<LmbStrategyLbp> = UnifiedFilter::with_defaults(
            create_motion(),
            create_sensor().into(),
            create_birth(),
            AssociationConfig::default(),
            strategy,
        );

        let mut rng = rand::thread_rng();
        let _ = filter.step(&mut rng, &vec![], 0);

        filter.reset();
        assert!(filter.hypotheses.is_empty());
        assert!(filter.trajectories.is_empty());
    }
}
