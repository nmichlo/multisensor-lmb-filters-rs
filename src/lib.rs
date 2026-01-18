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
    AssociationConfig, BirthLocation, BirthModel, EstimatedTrack, FilterOutput, GaussianComponent,
    Hypothesis, MotionModel, SensorConfig, SensorModel, StateEstimate, Track, TrackLabel,
    Trajectory,
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
    IteratedCorrectorMerger, MultisensorLmbmFilter, MultisensorMeasurements, ParallelUpdateMerger,
    PuLmbFilter,
};

// Multi-sensor association
pub use lmb::{MultisensorAssociationResult, MultisensorAssociator, MultisensorGibbsAssociator};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
