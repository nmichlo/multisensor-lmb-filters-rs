/*!
# Prak - Multi-object tracking library

Rust port of the MATLAB multisensor-lmb-filters library.

This library implements various multi-object tracking algorithms based on
Labelled Multi-Bernoulli (LMB) filters and their variants, including:

- Single-sensor LMB and LMBM filters
- Multi-sensor fusion algorithms (PU-LMB, IC-LMB, GA-LMB, AA-LMB)
- Multiple data association methods (LBP, Gibbs, Murty's algorithm)

## Modules

- `common` - Shared utilities, data structures, and algorithms
- `lmb` - Single-sensor LMB filter implementation
- `lmbm` - Single-sensor LMBM filter implementation
- `multisensor_lmb` - Multi-sensor LMB variants
- `multisensor_lmbm` - Multi-sensor LMBM implementation
*/

// Use mimalloc allocator when enabled (typically 5-15% faster for allocation-heavy workloads)
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod common;
pub mod lmb;
pub mod lmbm;
pub mod multisensor_lmb;
pub mod multisensor_lmbm;

// Re-export commonly used types
pub use common::types::{Model, Object, Measurement, GroundTruth};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
