//! Common utilities for Python bindings
//!
//! Shared helper functions to reduce boilerplate across filter implementations.

use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::lmb::FilterError;

/// Create RNG from optional seed
pub fn create_rng(seed: Option<u64>) -> StdRng {
    seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64)
}

/// Convert FilterError to PyResult
pub fn wrap_filter_error<T>(result: Result<T, FilterError>) -> PyResult<T> {
    result.map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Filter error: {:?}", e))
    })
}
