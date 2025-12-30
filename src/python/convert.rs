//! Internal numpy <-> nalgebra conversion utilities.
//!
//! These are NOT exposed to Python - purely internal helpers.

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Convert numpy 1D array to nalgebra DVector
pub(crate) fn numpy_to_dvector(arr: PyReadonlyArray1<'_, f64>) -> DVector<f64> {
    DVector::from_vec(arr.as_slice().unwrap().to_vec())
}

/// Convert numpy 2D array to nalgebra DMatrix (row-major)
pub(crate) fn numpy_to_dmatrix(arr: PyReadonlyArray2<'_, f64>) -> DMatrix<f64> {
    let shape = arr.shape();
    let rows = shape[0];
    let cols = shape[1];
    let data: Vec<f64> = arr.as_slice().unwrap().to_vec();
    DMatrix::from_row_slice(rows, cols, &data)
}

/// Convert nalgebra DVector to numpy 1D array
pub(crate) fn dvector_to_numpy<'py>(
    py: Python<'py>,
    v: &DVector<f64>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, v.as_slice())
}

/// Convert nalgebra DMatrix to numpy 2D array (row-major)
pub(crate) fn dmatrix_to_numpy<'py>(
    py: Python<'py>,
    m: &DMatrix<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut data = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            data[i][j] = m[(i, j)];
        }
    }
    PyArray2::from_vec2(py, &data).unwrap()
}

/// Convert list of numpy arrays to Vec<DVector> (single-sensor measurements)
pub(crate) fn numpy_list_to_measurements(
    measurements: Vec<PyReadonlyArray1<'_, f64>>,
) -> Vec<DVector<f64>> {
    measurements.into_iter().map(numpy_to_dvector).collect()
}

/// Convert nested list of numpy arrays to Vec<Vec<DVector>> (multi-sensor measurements)
pub(crate) fn numpy_nested_to_measurements(
    measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
) -> Vec<Vec<DVector<f64>>> {
    measurements
        .into_iter()
        .map(|sensor_meas| sensor_meas.into_iter().map(numpy_to_dvector).collect())
        .collect()
}
