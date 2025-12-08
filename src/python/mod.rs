//! Python bindings for multisensor-lmb-filters-rs
//!
//! This module provides PyO3 bindings for the LMB tracking library.

mod common;
mod config;
mod convert;
mod filters;
mod ops;
mod output;
mod types;

use pyo3::prelude::*;

pub use config::*;
pub use filters::*;
pub use output::*;
pub use types::*;

/// Python module definition
#[pymodule]
fn _multisensor_lmb_filters_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Configuration types
    m.add_class::<PyMotionModel>()?;
    m.add_class::<PySensorModel>()?;
    m.add_class::<PyMultisensorConfig>()?;
    m.add_class::<PyBirthLocation>()?;
    m.add_class::<PyBirthModel>()?;
    m.add_class::<PyAssociationConfig>()?;
    m.add_class::<PyFilterThresholds>()?;
    m.add_class::<PyLmbmConfig>()?;

    // Core types
    m.add_class::<PyTrackLabel>()?;
    m.add_class::<PyGaussianComponent>()?;
    m.add_class::<PyTrack>()?;

    // Output types
    m.add_class::<PyEstimatedTrack>()?;
    m.add_class::<PyStateEstimate>()?;
    m.add_class::<PyTrajectory>()?;
    m.add_class::<PyFilterOutput>()?;

    // Single-sensor filters
    m.add_class::<PyLmbFilter>()?;
    m.add_class::<PyLmbmFilter>()?;

    // Multi-sensor filters
    m.add_class::<PyAaLmbFilter>()?;
    m.add_class::<PyGaLmbFilter>()?;
    m.add_class::<PyPuLmbFilter>()?;
    m.add_class::<PyIcLmbFilter>()?;
    m.add_class::<PyMultisensorLmbmFilter>()?;

    // Low-level operations (for testing)
    ops::register_ops(m)?;

    // Version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
