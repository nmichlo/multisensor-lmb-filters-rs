//! Factory functions for creating LMB and LMBM filters.
//!
//! These functions provide simple, one-call construction for common filter configurations.
//! For custom associators or schedulers, use [`UnifiedFilter::new`] with a custom strategy.
//!
//! # LMB Filters
//!
//! - [`lmb_filter`] - Single-sensor LMB with LBP associator
//! - [`ic_lmb_filter`] - Iterated Corrector multi-sensor LMB
//! - [`aa_lmb_filter`] - Arithmetic Average multi-sensor LMB
//! - [`ga_lmb_filter`] - Geometric Average multi-sensor LMB
//! - [`pu_lmb_filter`] - Parallel Update multi-sensor LMB
//!
//! # LMBM Filters
//!
//! - [`lmbm_filter`] - Single-sensor LMBM with Gibbs associator
//! - [`multisensor_lmbm_filter`] - Multi-sensor LMBM with Gibbs associator

use super::config::{AssociationConfig, BirthModel, MotionModel, SensorConfig, SensorModel};
use super::multisensor::fusion::{
    MergerAverageArithmetic, MergerAverageGeometric, MergerParallelUpdate,
};
use super::multisensor::traits::AssociatorMultisensorGibbs;
use super::scheduler::{ParallelScheduler, SequentialScheduler, SingleSensorScheduler};
use super::strategy::{
    AaLmbStrategyLbp, CommonPruneConfig, GaLmbStrategyLbp, IcLmbStrategyLbp, LmbPruneConfig,
    LmbStrategy, LmbStrategyLbp, LmbmPruneConfig, LmbmStrategy, LmbmStrategyGibbs,
    MultisensorLmbmStrategy, MultisensorLmbmStrategyGibbs, PuLmbStrategyLbp,
    SingleSensorLmbmStrategy,
};
use super::traits::{AssociatorGibbs, AssociatorLbp};
use super::unified::UnifiedFilter;

// ============================================================================
// LMB Filter Factory Functions
// ============================================================================

/// Create a single-sensor LMB filter with LBP associator.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensor` - Sensor model (observation model, detection probability, clutter)
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
/// * `common_prune` - Common pruning config (existence threshold, trajectory length)
/// * `lmb_prune` - LMB-specific pruning config (GM component management)
pub fn lmb_filter(
    motion: MotionModel,
    sensor: SensorModel,
    birth: BirthModel,
    association: AssociationConfig,
    common_prune: CommonPruneConfig,
    lmb_prune: LmbPruneConfig,
) -> UnifiedFilter<LmbStrategyLbp> {
    let strategy = LmbStrategy::new(AssociatorLbp, SingleSensorScheduler::new(), lmb_prune);
    UnifiedFilter::new(
        motion,
        sensor.into(),
        birth,
        association,
        common_prune,
        strategy,
    )
}

/// Create an IC-LMB (Iterated Corrector) multi-sensor filter.
///
/// Processes sensors sequentially, where the output of sensor N becomes
/// the input to sensor N+1.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensors` - Multi-sensor configuration
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
/// * `common_prune` - Common pruning config (existence threshold, trajectory length)
/// * `lmb_prune` - LMB-specific pruning config (GM component management)
pub fn ic_lmb_filter(
    motion: MotionModel,
    sensors: SensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
    common_prune: CommonPruneConfig,
    lmb_prune: LmbPruneConfig,
) -> UnifiedFilter<IcLmbStrategyLbp> {
    let strategy = LmbStrategy::new(AssociatorLbp, SequentialScheduler::new(), lmb_prune);
    UnifiedFilter::new(motion, sensors, birth, association, common_prune, strategy)
}

/// Create an AA-LMB (Arithmetic Average) multi-sensor filter.
///
/// Processes sensors in parallel, then fuses using weighted arithmetic average.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensors` - Multi-sensor configuration
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
/// * `common_prune` - Common pruning config (existence threshold, trajectory length)
/// * `lmb_prune` - LMB-specific pruning config (GM component management)
/// * `max_hypotheses` - Maximum number of hypotheses for fusion
pub fn aa_lmb_filter(
    motion: MotionModel,
    sensors: SensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
    common_prune: CommonPruneConfig,
    lmb_prune: LmbPruneConfig,
    max_hypotheses: usize,
) -> UnifiedFilter<AaLmbStrategyLbp> {
    let num_sensors = sensors.num_sensors();
    let merger = MergerAverageArithmetic::uniform(num_sensors, max_hypotheses);
    let strategy = LmbStrategy::new(AssociatorLbp, ParallelScheduler::new(merger), lmb_prune);
    UnifiedFilter::new(motion, sensors, birth, association, common_prune, strategy)
}

/// Create a GA-LMB (Geometric Average) multi-sensor filter.
///
/// Processes sensors in parallel, then fuses using covariance intersection.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensors` - Multi-sensor configuration
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
/// * `common_prune` - Common pruning config (existence threshold, trajectory length)
/// * `lmb_prune` - LMB-specific pruning config (GM component management)
pub fn ga_lmb_filter(
    motion: MotionModel,
    sensors: SensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
    common_prune: CommonPruneConfig,
    lmb_prune: LmbPruneConfig,
) -> UnifiedFilter<GaLmbStrategyLbp> {
    let num_sensors = sensors.num_sensors();
    let merger = MergerAverageGeometric::uniform(num_sensors);
    let strategy = LmbStrategy::new(AssociatorLbp, ParallelScheduler::new(merger), lmb_prune);
    UnifiedFilter::new(motion, sensors, birth, association, common_prune, strategy)
}

/// Create a PU-LMB (Parallel Update) multi-sensor filter.
///
/// Processes sensors in parallel, then fuses using information-form fusion
/// with decorrelation.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensors` - Multi-sensor configuration
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
/// * `common_prune` - Common pruning config (existence threshold, trajectory length)
/// * `lmb_prune` - LMB-specific pruning config (GM component management)
pub fn pu_lmb_filter(
    motion: MotionModel,
    sensors: SensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
    common_prune: CommonPruneConfig,
    lmb_prune: LmbPruneConfig,
) -> UnifiedFilter<PuLmbStrategyLbp> {
    let merger = MergerParallelUpdate::new(Vec::new());
    let strategy = LmbStrategy::new(AssociatorLbp, ParallelScheduler::new(merger), lmb_prune);
    UnifiedFilter::new(motion, sensors, birth, association, common_prune, strategy)
}

// ============================================================================
// LMBM Filter Factory Functions
// ============================================================================

/// Create a single-sensor LMBM filter with Gibbs associator.
///
/// LMBM maintains a set of hypotheses, each representing a possible
/// data association. Uses Gibbs sampling for association.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensor` - Sensor model (observation model, detection probability, clutter)
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
/// * `common_prune` - Common pruning config (existence threshold, trajectory length)
/// * `lmbm_prune` - LMBM-specific pruning config (hypothesis management)
pub fn lmbm_filter(
    motion: MotionModel,
    sensor: SensorModel,
    birth: BirthModel,
    association: AssociationConfig,
    common_prune: CommonPruneConfig,
    lmbm_prune: LmbmPruneConfig,
) -> UnifiedFilter<LmbmStrategyGibbs> {
    let inner = SingleSensorLmbmStrategy {
        associator: AssociatorGibbs,
    };
    let strategy = LmbmStrategy::new(inner, lmbm_prune);
    UnifiedFilter::new(
        motion,
        sensor.into(),
        birth,
        association,
        common_prune,
        strategy,
    )
}

/// Create a multi-sensor LMBM filter with Gibbs associator.
///
/// Multi-sensor LMBM processes all sensors jointly using a multi-sensor
/// Gibbs sampler for association.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensors` - Multi-sensor configuration
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
/// * `common_prune` - Common pruning config (existence threshold, trajectory length)
/// * `lmbm_prune` - LMBM-specific pruning config (hypothesis management)
pub fn multisensor_lmbm_filter(
    motion: MotionModel,
    sensors: SensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
    common_prune: CommonPruneConfig,
    lmbm_prune: LmbmPruneConfig,
) -> UnifiedFilter<MultisensorLmbmStrategyGibbs> {
    let inner = MultisensorLmbmStrategy {
        associator: AssociatorMultisensorGibbs,
    };
    let strategy = LmbmStrategy::new(inner, lmbm_prune);
    UnifiedFilter::new(motion, sensors, birth, association, common_prune, strategy)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmb::config::BirthLocation;
    use crate::lmb::traits::Filter;
    use nalgebra::DMatrix;

    fn create_motion() -> MotionModel {
        MotionModel::constant_velocity_2d(1.0, 0.1, 0.99)
    }

    fn create_sensor() -> SensorModel {
        SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0)
    }

    fn create_multi_sensor() -> SensorConfig {
        let sensor1 = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
        let sensor2 = SensorModel::position_sensor_2d(1.0, 0.85, 8.0, 100.0);
        SensorConfig::new(vec![sensor1, sensor2])
    }

    fn create_birth() -> BirthModel {
        let birth_loc = BirthLocation::new(
            0,
            nalgebra::DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            DMatrix::identity(4, 4) * 100.0,
        );
        BirthModel::new(vec![birth_loc], 0.1, 0.01)
    }

    #[test]
    fn test_lmb_filter_factory() {
        let filter = lmb_filter(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
            CommonPruneConfig::default(),
            LmbPruneConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
    }

    #[test]
    fn test_ic_lmb_filter_factory() {
        let filter = ic_lmb_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            CommonPruneConfig::default(),
            LmbPruneConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.num_sensors(), 2);
    }

    #[test]
    fn test_aa_lmb_filter_factory() {
        let filter = aa_lmb_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            CommonPruneConfig::default(),
            LmbPruneConfig::default(),
            100,
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.num_sensors(), 2);
    }

    #[test]
    fn test_ga_lmb_filter_factory() {
        let filter = ga_lmb_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            CommonPruneConfig::default(),
            LmbPruneConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.num_sensors(), 2);
    }

    #[test]
    fn test_pu_lmb_filter_factory() {
        let filter = pu_lmb_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            CommonPruneConfig::default(),
            LmbPruneConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.num_sensors(), 2);
    }

    #[test]
    fn test_lmbm_filter_factory() {
        let filter = lmbm_filter(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
            CommonPruneConfig::default(),
            LmbmPruneConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
    }

    #[test]
    fn test_multisensor_lmbm_filter_factory() {
        let filter = multisensor_lmbm_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            CommonPruneConfig::default(),
            LmbmPruneConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.num_sensors(), 2);
    }
}
