//! Factory functions for creating LMB and LMBM filters.
//!
//! These functions provide simple, one-call construction for common filter configurations.
//! For custom associators or schedulers, use [`LmbFilterCore::with_scheduler`] or
//! [`LmbmFilterCore::with_strategy`] directly.
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

use super::config::{
    AssociationConfig, BirthModel, LmbmConfig, MotionModel, MultisensorConfig, SensorModel,
};
use super::core::{AaLmbFilter, GaLmbFilter, IcLmbFilter, LmbFilter, LmbFilterCore, PuLmbFilter};
use super::core_lmbm::{
    LmbmFilter, LmbmFilterCore, MultisensorLmbmFilter, MultisensorLmbmStrategy,
    SingleSensorLmbmStrategy,
};
use super::multisensor::fusion::{
    ArithmeticAverageMerger, GeometricAverageMerger, ParallelUpdateMerger,
};
use super::multisensor::traits::MultisensorGibbsAssociator;
use super::scheduler::{ParallelScheduler, SequentialScheduler, SingleSensorScheduler};
use super::traits::{GibbsAssociator, LbpAssociator};

// ============================================================================
// LMB Filter Factory Functions
// ============================================================================

/// Create a single-sensor LMB filter with default LBP associator.
///
/// This is the standard LMB filter for tracking with a single sensor.
/// Uses Loopy Belief Propagation for data association.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensor` - Sensor model (observation model, detection probability, clutter)
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
///
/// # Example
///
/// ```ignore
/// use multisensor_lmb_filters_rs::lmb::{lmb_filter, MotionModel, SensorModel, BirthModel, AssociationConfig};
///
/// let filter = lmb_filter(motion, sensor, birth, AssociationConfig::default());
/// ```
pub fn lmb_filter(
    motion: MotionModel,
    sensor: SensorModel,
    birth: BirthModel,
    association: AssociationConfig,
) -> LmbFilter {
    LmbFilterCore::with_scheduler(
        motion,
        sensor.into(),
        birth,
        association,
        LbpAssociator,
        SingleSensorScheduler::new(),
    )
}

/// Create an IC-LMB (Iterated Corrector) multi-sensor filter.
///
/// Processes sensors sequentially, where the output of sensor N becomes
/// the input to sensor N+1. Simple but order-dependent.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensors` - Multi-sensor configuration
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
pub fn ic_lmb_filter(
    motion: MotionModel,
    sensors: MultisensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
) -> IcLmbFilter {
    LmbFilterCore::with_scheduler(
        motion,
        sensors.into(),
        birth,
        association,
        LbpAssociator,
        SequentialScheduler::new(),
    )
}

/// Create an AA-LMB (Arithmetic Average) multi-sensor filter.
///
/// Processes sensors in parallel, then fuses using weighted arithmetic average.
/// Fast and robust, but doesn't account for sensor correlation.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensors` - Multi-sensor configuration
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
/// * `max_hypotheses` - Maximum number of hypotheses for fusion
pub fn aa_lmb_filter(
    motion: MotionModel,
    sensors: MultisensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
    max_hypotheses: usize,
) -> AaLmbFilter {
    let num_sensors = sensors.num_sensors();
    let merger = ArithmeticAverageMerger::uniform(num_sensors, max_hypotheses);
    LmbFilterCore::with_scheduler(
        motion,
        sensors.into(),
        birth,
        association,
        LbpAssociator,
        ParallelScheduler::new(merger),
    )
}

/// Create a GA-LMB (Geometric Average) multi-sensor filter.
///
/// Processes sensors in parallel, then fuses using covariance intersection.
/// Produces conservative estimates, robust to unknown correlations.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensors` - Multi-sensor configuration
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
pub fn ga_lmb_filter(
    motion: MotionModel,
    sensors: MultisensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
) -> GaLmbFilter {
    let num_sensors = sensors.num_sensors();
    let merger = GeometricAverageMerger::uniform(num_sensors);
    LmbFilterCore::with_scheduler(
        motion,
        sensors.into(),
        birth,
        association,
        LbpAssociator,
        ParallelScheduler::new(merger),
    )
}

/// Create a PU-LMB (Parallel Update) multi-sensor filter.
///
/// Processes sensors in parallel, then fuses using information-form fusion
/// with decorrelation. Theoretically optimal for independent sensors.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensors` - Multi-sensor configuration
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
pub fn pu_lmb_filter(
    motion: MotionModel,
    sensors: MultisensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
) -> PuLmbFilter {
    let merger = ParallelUpdateMerger::new(Vec::new());
    LmbFilterCore::with_scheduler(
        motion,
        sensors.into(),
        birth,
        association,
        LbpAssociator,
        ParallelScheduler::new(merger),
    )
}

// ============================================================================
// LMBM Filter Factory Functions
// ============================================================================

/// Create a single-sensor LMBM filter with default Gibbs associator.
///
/// LMBM (Labeled Multi-Bernoulli Mixture) maintains a set of hypotheses,
/// each representing a possible data association. Uses Gibbs sampling
/// for association.
///
/// # Arguments
///
/// * `motion` - Motion model (dynamics and survival probability)
/// * `sensor` - Sensor model (observation model, detection probability, clutter)
/// * `birth` - Birth model (where new objects can appear)
/// * `association` - Association algorithm configuration
/// * `lmbm_config` - LMBM-specific configuration (max hypotheses, etc.)
pub fn lmbm_filter(
    motion: MotionModel,
    sensor: SensorModel,
    birth: BirthModel,
    association: AssociationConfig,
    lmbm_config: LmbmConfig,
) -> LmbmFilter {
    let strategy = SingleSensorLmbmStrategy::new(GibbsAssociator);
    LmbmFilterCore::with_strategy(
        motion,
        sensor.into(),
        birth,
        association,
        lmbm_config,
        strategy,
    )
}

/// Create a multi-sensor LMBM filter with default Gibbs associator.
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
/// * `lmbm_config` - LMBM-specific configuration (max hypotheses, etc.)
pub fn multisensor_lmbm_filter(
    motion: MotionModel,
    sensors: MultisensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
    lmbm_config: LmbmConfig,
) -> MultisensorLmbmFilter {
    let strategy = MultisensorLmbmStrategy::new(MultisensorGibbsAssociator);
    LmbmFilterCore::with_strategy(
        motion,
        sensors.into(),
        birth,
        association,
        lmbm_config,
        strategy,
    )
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

    fn create_multi_sensor() -> MultisensorConfig {
        let sensor1 = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
        let sensor2 = SensorModel::position_sensor_2d(1.0, 0.85, 8.0, 100.0);
        MultisensorConfig::new(vec![sensor1, sensor2])
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
            LmbmConfig::default(),
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
            LmbmConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.num_sensors(), 2);
    }
}
