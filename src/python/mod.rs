//! Python bindings for multisensor-lmb-filters-rs
//!
//! This module provides PyO3 bindings for the LMB tracking library.

mod birth;
mod convert;
mod filters;
mod intermediate;
mod models;
mod output;
mod types;

use pyo3::prelude::*;

/// Python module definition
#[pymodule]
fn _multisensor_lmb_filters_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Models
    m.add_class::<models::PyMotionModel>()?;
    m.add_class::<models::PySensorModel>()?;
    m.add_class::<models::PySensorConfigMulti>()?;

    // Birth
    m.add_class::<birth::PyBirthModel>()?;
    m.add_class::<birth::PyBirthLocation>()?;

    // Configuration
    m.add_class::<filters::PyAssociatorConfig>()?;
    m.add_class::<filters::PyFilterThresholds>()?;
    m.add_class::<filters::PyFilterLmbmConfig>()?;

    // Filters - Single-sensor
    m.add_class::<filters::PyFilterLmb>()?;
    m.add_class::<filters::PyFilterLmbm>()?;

    // Filters - Multi-sensor
    m.add_class::<filters::PyFilterAaLmb>()?;
    m.add_class::<filters::PyFilterGaLmb>()?;
    m.add_class::<filters::PyFilterPuLmb>()?;
    m.add_class::<filters::PyFilterIcLmb>()?;
    m.add_class::<filters::PyFilterMultisensorLmbm>()?;

    // Output types
    m.add_class::<output::PyTrackEstimate>()?;
    m.add_class::<output::PyStateEstimate>()?;

    // Core types
    m.add_class::<types::PyTrackLabel>()?;
    m.add_class::<types::PyGaussianComponent>()?;

    // Internal types for testing (underscore-prefixed = private)
    m.add_class::<intermediate::PyTrackData>()?;
    m.add_class::<intermediate::PyAssociationMatrices>()?;
    m.add_class::<intermediate::PyAssociationResult>()?;
    m.add_class::<intermediate::PyCardinalityEstimate>()?;
    m.add_class::<intermediate::PyStepOutput>()?;
    m.add_class::<intermediate::PyPosteriorParameters>()?;
    m.add_class::<intermediate::PyLmbmHypothesis>()?;

    // Version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
