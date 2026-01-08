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
use multisensor_lmb_filters_rs::lmb::{Filter, LmbFilter, MotionModel, SensorModel, BirthModel, BirthLocation, AssociationConfig};
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
// Python bindings (optional)
// ============================================================================

#[cfg(feature = "python")]
pub mod python;

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

/// Benchmark utilities (scenario loading, filter factory)
pub mod bench_utils;

// ============================================================================
// Re-exports for convenience
// ============================================================================

// Core types
pub use lmb::{
    AssociationConfig, BirthLocation, BirthModel, EstimatedTrack, FilterOutput, FilterParams,
    FilterThresholds, GaussianComponent, LmbmConfig, LmbmHypothesis, MotionModel,
    MultisensorConfig, SensorModel, SensorVariant, StateEstimate, Track, TrackLabel, Trajectory,
};

// Errors
pub use lmb::{AssociationError, FilterError};

// Traits
pub use lmb::{Associator, Filter, Merger, Updater};

// Associator implementations
pub use lmb::{GibbsAssociator, LbpAssociator, MurtyAssociator};

// Updater implementations
pub use lmb::{HardAssignmentUpdater, MarginalUpdater};

// Single-sensor filters
pub use lmb::{LmbFilter, LmbmFilter};

// Multi-sensor filters
pub use lmb::{
    AaLmbFilter, ArithmeticAverageMerger, GaLmbFilter, GeometricAverageMerger, IcLmbFilter,
    IteratedCorrectorMerger, MultisensorLmbFilter, MultisensorLmbmFilter, MultisensorMeasurements,
    ParallelUpdateMerger, PuLmbFilter,
};

// Multi-sensor association
pub use lmb::{MultisensorAssociationResult, MultisensorAssociator, MultisensorGibbsAssociator};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
