//! Python wrappers for birth model configuration.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::lmb::config::{BirthLocation, BirthModel};

use super::convert::{dmatrix_to_numpy, dvector_to_numpy, numpy_to_dmatrix, numpy_to_dvector};

// =============================================================================
// BirthLocation
// =============================================================================

#[pyclass(name = "BirthLocation")]
#[derive(Clone)]
pub struct PyBirthLocation {
    pub(crate) inner: BirthLocation,
}

#[pymethods]
impl PyBirthLocation {
    #[new]
    #[pyo3(signature = (label, mean, covariance))]
    fn new(
        label: usize,
        mean: PyReadonlyArray1<'_, f64>,
        covariance: PyReadonlyArray2<'_, f64>,
    ) -> Self {
        Self {
            inner: BirthLocation::new(label, numpy_to_dvector(mean), numpy_to_dmatrix(covariance)),
        }
    }

    #[getter]
    fn label(&self) -> usize {
        self.inner.label
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_numpy(py, &self.inner.mean)
    }

    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.covariance)
    }

    fn __repr__(&self) -> String {
        format!(
            "BirthLocation(label={}, x_dim={})",
            self.inner.label,
            self.inner.mean.len()
        )
    }
}

// =============================================================================
// BirthModel
// =============================================================================

#[pyclass(name = "BirthModel")]
#[derive(Clone)]
pub struct PyBirthModel {
    pub(crate) inner: BirthModel,
}

#[pymethods]
impl PyBirthModel {
    #[new]
    #[pyo3(signature = (locations, lmb_existence, lmbm_existence))]
    fn new(locations: Vec<PyBirthLocation>, lmb_existence: f64, lmbm_existence: f64) -> Self {
        Self {
            inner: BirthModel::new(
                locations.into_iter().map(|l| l.inner).collect(),
                lmb_existence,
                lmbm_existence,
            ),
        }
    }

    #[getter]
    fn num_locations(&self) -> usize {
        self.inner.num_locations()
    }

    #[getter]
    fn lmb_existence(&self) -> f64 {
        self.inner.lmb_existence
    }

    #[getter]
    fn lmbm_existence(&self) -> f64 {
        self.inner.lmbm_existence
    }

    fn __len__(&self) -> usize {
        self.inner.num_locations()
    }

    fn __getitem__(&self, idx: usize) -> PyResult<PyBirthLocation> {
        self.inner
            .locations
            .get(idx)
            .map(|l| PyBirthLocation { inner: l.clone() })
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("birth location index out of range")
            })
    }

    fn __repr__(&self) -> String {
        format!(
            "BirthModel(num_locations={}, lmb_existence={:.3}, lmbm_existence={:.3})",
            self.inner.num_locations(),
            self.inner.lmb_existence,
            self.inner.lmbm_existence
        )
    }
}
