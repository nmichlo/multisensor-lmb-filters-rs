//! Python bindings for core track types

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use super::convert::{dmatrix_to_numpy, dvector_to_numpy, numpy_to_dmatrix, numpy_to_dvector};
use crate::lmb::{GaussianComponent, Track, TrackLabel};

/// Track label uniquely identifies a track
#[pyclass(name = "TrackLabel")]
#[derive(Clone)]
pub struct PyTrackLabel {
    pub(crate) inner: TrackLabel,
}

#[pymethods]
impl PyTrackLabel {
    /// Create a new track label
    #[new]
    #[pyo3(signature = (birth_time, birth_location))]
    fn new(birth_time: usize, birth_location: usize) -> Self {
        Self {
            inner: TrackLabel::new(birth_time, birth_location),
        }
    }

    /// Get birth time
    #[getter]
    fn birth_time(&self) -> usize {
        self.inner.birth_time
    }

    /// Get birth location
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

/// Gaussian component with weight, mean, and covariance
#[pyclass(name = "GaussianComponent")]
#[derive(Clone)]
pub struct PyGaussianComponent {
    pub(crate) inner: GaussianComponent,
}

#[pymethods]
impl PyGaussianComponent {
    /// Create a new Gaussian component
    #[new]
    #[pyo3(signature = (weight, mean, covariance))]
    fn new(
        weight: f64,
        mean: PyReadonlyArray1<'_, f64>,
        covariance: PyReadonlyArray2<'_, f64>,
    ) -> Self {
        Self {
            inner: GaussianComponent::new(
                weight,
                numpy_to_dvector(mean),
                numpy_to_dmatrix(covariance),
            ),
        }
    }

    /// Get weight
    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }

    /// Get mean
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_numpy(py, &self.inner.mean)
    }

    /// Get covariance
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.covariance)
    }

    /// Get state dimension
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

/// Track representing a potential object
#[pyclass(name = "Track")]
#[derive(Clone)]
pub struct PyTrack {
    pub(crate) inner: Track,
}

#[pymethods]
impl PyTrack {
    /// Create a new track
    #[new]
    #[pyo3(signature = (label, existence, mean, covariance))]
    fn new(
        label: PyTrackLabel,
        existence: f64,
        mean: PyReadonlyArray1<'_, f64>,
        covariance: PyReadonlyArray2<'_, f64>,
    ) -> Self {
        Self {
            inner: Track::new(
                label.inner,
                existence,
                numpy_to_dvector(mean),
                numpy_to_dmatrix(covariance),
            ),
        }
    }

    /// Get track label
    #[getter]
    fn label(&self) -> PyTrackLabel {
        PyTrackLabel {
            inner: self.inner.label,
        }
    }

    /// Get existence probability
    #[getter]
    fn existence(&self) -> f64 {
        self.inner.existence
    }

    /// Get state dimension
    #[getter]
    fn x_dim(&self) -> usize {
        self.inner.x_dim()
    }

    /// Get number of Gaussian components
    #[getter]
    fn num_components(&self) -> usize {
        self.inner.num_components()
    }

    /// Get components as list
    #[getter]
    fn components(&self) -> Vec<PyGaussianComponent> {
        self.inner
            .components
            .iter()
            .map(|c| PyGaussianComponent { inner: c.clone() })
            .collect()
    }

    /// Get weighted mean across all components
    fn weighted_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_numpy(py, &self.inner.weighted_mean())
    }

    /// Get primary (highest weight) component's mean
    fn primary_mean<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.primary_mean().map(|m| dvector_to_numpy(py, m))
    }

    /// Get primary (highest weight) component's covariance
    fn primary_covariance<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .primary_covariance()
            .map(|c| dmatrix_to_numpy(py, c))
    }

    fn __repr__(&self) -> String {
        format!(
            "Track(label={:?}, existence={:.4}, components={})",
            (self.inner.label.birth_time, self.inner.label.birth_location),
            self.inner.existence,
            self.inner.num_components()
        )
    }
}
