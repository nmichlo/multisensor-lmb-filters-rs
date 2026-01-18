//! Python wrappers for filter output types.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::output::{EstimatedTrack, StateEstimate};

use super::convert::{dmatrix_to_numpy, dvector_to_numpy};
use super::types::PyTrackLabel;

// =============================================================================
// TrackEstimate
// =============================================================================

#[pyclass(name = "TrackEstimate")]
#[derive(Clone)]
pub struct PyTrackEstimate {
    pub(crate) inner: EstimatedTrack,
}

impl PyTrackEstimate {
    pub fn from_inner(inner: EstimatedTrack) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrackEstimate {
    #[getter]
    fn label(&self) -> PyTrackLabel {
        PyTrackLabel {
            inner: self.inner.label,
        }
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
            "TrackEstimate(label=({}, {}), x_dim={})",
            self.inner.label.birth_time,
            self.inner.label.birth_location,
            self.inner.x_dim()
        )
    }
}

// =============================================================================
// StateEstimate
// =============================================================================

#[pyclass(name = "StateEstimate")]
#[derive(Clone)]
pub struct PyStateEstimate {
    pub(crate) inner: StateEstimate,
}

impl PyStateEstimate {
    pub fn from_inner(inner: StateEstimate) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyStateEstimate {
    #[getter]
    fn timestamp(&self) -> usize {
        self.inner.timestamp
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.num_tracks()
    }

    #[getter]
    fn tracks(&self) -> Vec<PyTrackEstimate> {
        self.inner
            .tracks
            .iter()
            .map(|t| PyTrackEstimate::from_inner(t.clone()))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.num_tracks()
    }

    fn __getitem__(&self, idx: usize) -> PyResult<PyTrackEstimate> {
        self.inner
            .tracks
            .get(idx)
            .map(|t| PyTrackEstimate::from_inner(t.clone()))
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("track index out of range"))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyTrackIterator {
        PyTrackIterator {
            tracks: slf.inner.tracks.clone(),
            index: 0,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "StateEstimate(timestamp={}, num_tracks={})",
            self.inner.timestamp,
            self.inner.num_tracks()
        )
    }
}

// =============================================================================
// Iterator for StateEstimate
// =============================================================================

#[pyclass]
pub struct PyTrackIterator {
    tracks: Vec<EstimatedTrack>,
    index: usize,
}

#[pymethods]
impl PyTrackIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyTrackEstimate> {
        if slf.index < slf.tracks.len() {
            let track = slf.tracks[slf.index].clone();
            slf.index += 1;
            Some(PyTrackEstimate::from_inner(track))
        } else {
            None
        }
    }
}
