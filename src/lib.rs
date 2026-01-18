/*!
# multisensor-lmb-filters-rs - Multi-object tracking library

Rust implementation of multi-object tracking algorithms based on
Labelled Multi-Bernoulli (LMB) filters and their variants.

## Features

- Single-sensor LMB and LMBM filters
- Multi-sensor fusion algorithms (PU-LMB, IC-LMB, GA-LMB, AA-LMB)
- Multiple data association methods (LBP, Gibbs, Murty's algorithm)

## Modules

- [`lmb`] - LMB tracking algorithms and types
- [`association`] - Data association: likelihood computation, matrix building
- [`utils`] - Low-level utilities and shared algorithms
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

/// Data association algorithms and utilities
pub mod association;

/// Low-level utilities (linear algebra, RNG, constants)
pub mod utils;

// Core types
pub mod config;
pub mod output;
pub mod types;
pub mod errors;
pub mod factory;
pub mod reporter;
pub mod scheduler;
pub mod strategy;
pub mod traits;
pub mod unified;
// Utilities
pub mod cardinality;
pub mod measurements;
pub mod builder;
pub mod likelihood;
pub mod fusion_ms;
pub mod traits_ms;
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
pub use lmb::{AssociatorGibbs, AssociatorLbp, AssociatorMurty};

// Updater implementations
pub use lmb::{UpdaterHardAssignment, UpdaterMarginal};

// Single-sensor filters
pub use lmb::{FilterLmb, FilterLmbm};

// Multi-sensor filters
pub use lmb::{
    FilterLmbAa, FilterLmbGa, FilterLmbIc, FilterLmbPu, FilterLmbmMultisensor,
    MeasurementsMultisensor, MergerAverageArithmetic, MergerAverageGeometric,
    MergerIteratedCorrector, MergerParallelUpdate,
};

// Multi-sensor association
pub use lmb::{AssociationMultisensorResult, AssociatorMultisensor, AssociatorMultisensorGibbs};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
