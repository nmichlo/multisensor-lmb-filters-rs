//! Low-level operations exposed for testing.
//!
//! These are private/internal APIs primarily used for fixture equivalence testing.
//! They allow step-by-step verification against MATLAB-generated fixtures.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use super::convert::{dmatrix_to_numpy, dvector_to_numpy, numpy_to_dmatrix, numpy_to_dvector};
use crate::common::linalg::{predict_covariance, predict_mean};

/// Predict a Gaussian component forward in time (internal).
///
/// Returns (predicted_mean, predicted_covariance).
#[pyfunction]
#[pyo3(name = "_predict_component")]
pub fn py_predict_component<'py>(
    py: Python<'py>,
    mean: PyReadonlyArray1<'_, f64>,
    covariance: PyReadonlyArray2<'_, f64>,
    transition_matrix: PyReadonlyArray2<'_, f64>,
    process_noise: PyReadonlyArray2<'_, f64>,
    control_input: PyReadonlyArray1<'_, f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>) {
    let mu = numpy_to_dvector(mean);
    let sigma = numpy_to_dmatrix(covariance);
    let a = numpy_to_dmatrix(transition_matrix);
    let r = numpy_to_dmatrix(process_noise);
    let u = numpy_to_dvector(control_input);

    let pred_mean = predict_mean(&mu, &a, &u);
    let pred_cov = predict_covariance(&sigma, &a, &r);

    (
        dvector_to_numpy(py, &pred_mean),
        dmatrix_to_numpy(py, &pred_cov),
    )
}

/// Predict existence probability (internal).
#[pyfunction]
#[pyo3(name = "_predict_existence")]
pub fn py_predict_existence(existence: f64, survival_probability: f64) -> f64 {
    crate::common::linalg::predict_existence(existence, survival_probability)
}

/// Register low-level operations with the Python module.
pub fn register_ops(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_predict_component, m)?)?;
    m.add_function(wrap_pyfunction!(py_predict_existence, m)?)?;
    Ok(())
}
