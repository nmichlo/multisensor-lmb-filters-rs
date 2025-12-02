/*!
# Prak - Multi-object tracking library

Rust port of the MATLAB multisensor-lmb-filters library.

This library implements various multi-object tracking algorithms based on
Labelled Multi-Bernoulli (LMB) filters and their variants, including:

- Single-sensor LMB and LMBM filters
- Multi-sensor fusion algorithms (PU-LMB, IC-LMB, GA-LMB, AA-LMB)
- Multiple data association methods (LBP, Gibbs, Murty's algorithm)

## New API (v2)

The library is being refactored to a cleaner, trait-based API:

- [`types`] - Core types: `Track`, `FilterParams`, `StateEstimate`
- [`components`] - Shared algorithms: prediction, update
- [`association`] - Data association: likelihood computation, matrix building
- [`filter`] - Filter trait and implementations

## Legacy API

The following modules contain the original MATLAB-ported implementations:

- `common` - Shared utilities, data structures, and algorithms
- `lmb` - Single-sensor LMB filter implementation
- `lmbm` - Single-sensor LMBM filter implementation
- `multisensor_lmb` - Multi-sensor LMB variants
- `multisensor_lmbm` - Multi-sensor LMBM implementation
*/

// ============================================================================
// New API (v2) - Trait-based, modular design
// ============================================================================

pub mod types;
pub mod components;
pub mod association;
pub mod filter;

// Re-export commonly used new types
pub use types::{
    Track, TrackLabel, GaussianComponent, LmbmHypothesis,
    MotionModel, SensorModel, FilterParams, BirthModel,
    StateEstimate, EstimatedTrack, FilterOutput,
};

pub use filter::{Filter, Associator, Merger, FilterError};

// ============================================================================
// Legacy API - Original MATLAB-ported implementations
// ============================================================================

pub mod common;
pub mod lmb;
pub mod lmbm;
pub mod multisensor_lmb;
pub mod multisensor_lmbm;

// Re-export legacy types (for backward compatibility)
pub use common::types::{Model, Object, Measurement, GroundTruth};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
