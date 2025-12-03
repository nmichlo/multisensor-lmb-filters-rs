/*!
# Prak - Multi-object tracking library

Rust implementation of multi-object tracking algorithms based on
Labelled Multi-Bernoulli (LMB) filters and their variants.

## Features

- Single-sensor LMB and LMBM filters
- Multi-sensor fusion algorithms (PU-LMB, IC-LMB, GA-LMB, AA-LMB)
- Multiple data association methods (LBP, Gibbs, Murty's algorithm)

## Modules

- [`lmb`] - LMB tracking algorithms and types
- [`components`] - Shared algorithms: prediction, update
- [`association`] - Data association: likelihood computation, matrix building
- [`common`] - Low-level utilities

## Example

```rust,no_run
use prak::lmb::{Filter, LmbFilter, MotionModel, SensorModel, BirthModel, BirthLocation, AssociationConfig};
use nalgebra::{DVector, DMatrix};

// Create filter configuration
let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);
let sensor = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);

// Define a birth location
let birth_loc = BirthLocation::new(
    0,
    DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
    DMatrix::identity(4, 4) * 100.0,
);
let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);
let association = AssociationConfig::default();

// Create filter
let mut filter = LmbFilter::new(motion, sensor, birth, association);

// Process measurements
let mut rng = rand::thread_rng();
let measurements = vec![DVector::from_vec(vec![1.0, 2.0])];
let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
```
*/

// ============================================================================
// Core modules
// ============================================================================

/// LMB (Labeled Multi-Bernoulli) tracking algorithms
///
/// This is the main module containing all LMB-family implementations:
/// - Single-sensor: `LmbFilter`, `LmbmFilter`
/// - Multi-sensor: `MultisensorLmbFilter`, `MultisensorLmbmFilter`
/// - Fusion strategies: `ArithmeticAverageMerger`, `GeometricAverageMerger`, etc.
pub mod lmb;

/// Shared tracking components (prediction, update)
pub mod components;

/// Data association algorithms and utilities
pub mod association;

/// Low-level utilities (linear algebra, RNG, constants)
pub mod common;

// ============================================================================
// Re-exports for convenience
// ============================================================================

// Core types
pub use lmb::{
    Track, TrackLabel, GaussianComponent, LmbmHypothesis,
    MotionModel, SensorModel, FilterParams, BirthModel, BirthLocation,
    StateEstimate, EstimatedTrack, FilterOutput, Trajectory,
    AssociationConfig, FilterThresholds, LmbmConfig,
    MultisensorConfig, SensorVariant,
};

// Errors
pub use lmb::{FilterError, AssociationError};

// Traits
pub use lmb::{Filter, Associator, Merger, Updater};

// Associator implementations
pub use lmb::{LbpAssociator, GibbsAssociator, MurtyAssociator};

// Updater implementations
pub use lmb::{MarginalUpdater, HardAssignmentUpdater};

// Single-sensor filters
pub use lmb::{LmbFilter, LmbmFilter};

// Multi-sensor filters
pub use lmb::{
    MultisensorLmbFilter, MultisensorLmbmFilter,
    AaLmbFilter, GaLmbFilter, PuLmbFilter, IcLmbFilter,
    ArithmeticAverageMerger, GeometricAverageMerger,
    ParallelUpdateMerger, IteratedCorrectorMerger,
    MultisensorMeasurements,
};

// Multi-sensor association
pub use lmb::{
    MultisensorAssociator, MultisensorGibbsAssociator,
    MultisensorAssociationResult,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ============================================================================
// Backwards compatibility re-exports
// ============================================================================

/// Deprecated: Use `lmb` module instead
#[deprecated(since = "2.0.0", note = "Use `lmb` module instead")]
pub mod types {
    //! Deprecated: Types have moved to the `lmb` module.
    pub use super::lmb::{
        Track, TrackLabel, GaussianComponent, LmbmHypothesis, TrajectoryHistory,
        MotionModel, SensorModel, FilterParams, BirthModel, BirthLocation,
        StateEstimate, EstimatedTrack, FilterOutput, Trajectory,
        AssociationConfig, FilterThresholds, LmbmConfig,
        MultisensorConfig, SensorVariant, FilterParamsBuilder,
        DataAssociationMethod,
    };
}

/// Deprecated: Use `lmb` module instead
#[deprecated(since = "2.0.0", note = "Use `lmb` module instead")]
pub mod filter {
    //! Deprecated: Filters have moved to the `lmb` module.
    pub use super::lmb::{
        Filter, Associator, Merger, Updater,
        FilterError, AssociationError,
        LbpAssociator, GibbsAssociator, MurtyAssociator,
        MarginalUpdater, HardAssignmentUpdater,
        LmbFilter, LmbmFilter,
        MultisensorLmbFilter, MultisensorLmbmFilter,
        AaLmbFilter, GaLmbFilter, PuLmbFilter, IcLmbFilter,
        ArithmeticAverageMerger, GeometricAverageMerger,
        ParallelUpdateMerger, IteratedCorrectorMerger,
        MultisensorMeasurements,
        MultisensorAssociator, MultisensorGibbsAssociator,
        MultisensorAssociationResult,
        AssociationResult,
        DEFAULT_EXISTENCE_THRESHOLD, DEFAULT_MIN_TRAJECTORY_LENGTH,
        DEFAULT_GM_WEIGHT_THRESHOLD, DEFAULT_MAX_GM_COMPONENTS,
        DEFAULT_LMBM_MAX_HYPOTHESES, DEFAULT_LMBM_WEIGHT_THRESHOLD,
        DEFAULT_MAX_TRAJECTORY_LENGTH, NUMERICAL_ZERO, UNDERFLOW_THRESHOLD,
    };
}
