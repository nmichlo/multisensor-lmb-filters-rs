//! Python wrappers for core track types.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::lmb::types::{GaussianComponent, TrackLabel};

use super::convert::{dmatrix_to_numpy, dvector_to_numpy};

// =============================================================================
// TrackLabel
// =============================================================================

#[pyclass(name = "TrackLabel")]
#[derive(Clone)]
pub struct PyTrackLabel {
    pub(crate) inner: TrackLabel,
}

#[pymethods]
impl PyTrackLabel {
    #[new]
    fn new(birth_time: usize, birth_location: usize) -> Self {
        Self {
            inner: TrackLabel::new(birth_time, birth_location),
        }
    }

    #[getter]
    fn birth_time(&self) -> usize {
        self.inner.birth_time
    }

    #[getter]
    fn birth_location(&self) -> usize {
        self.inner.birth_location
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackLabel(birth_time={}, birth_location={})",
            self.inner.birth_time, self.inner.birth_location
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

// =============================================================================
// GaussianComponent
// =============================================================================

#[pyclass(name = "GaussianComponent")]
#[derive(Clone)]
pub struct PyGaussianComponent {
    pub(crate) inner: GaussianComponent,
}

#[pymethods]
impl PyGaussianComponent {
    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_numpy(py, &self.inner.mean)
    }

    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.covariance)
    }

    #[getter]
    fn x_dim(&self) -> usize {
        self.inner.x_dim()
    }

    fn __repr__(&self) -> String {
        format!(
            "GaussianComponent(weight={:.4}, x_dim={})",
            self.inner.weight,
            self.inner.x_dim()
        )
    }
}
