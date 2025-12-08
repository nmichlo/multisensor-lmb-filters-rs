//! Python bindings for output types

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::lmb::{EstimatedTrack, FilterOutput, StateEstimate, Trajectory};
use nalgebra::{DMatrix, DVector};

use super::types::PyTrackLabel;

/// Convert nalgebra DVector to numpy array
fn dvector_to_numpy<'py>(py: Python<'py>, v: &DVector<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, v.as_slice())
}

/// Convert nalgebra DMatrix to numpy array (row-major)
fn dmatrix_to_numpy<'py>(py: Python<'py>, m: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
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

/// Estimated state of a single track at one timestep
#[pyclass(name = "EstimatedTrack")]
#[derive(Clone)]
pub struct PyEstimatedTrack {
    pub(crate) inner: EstimatedTrack,
}

#[pymethods]
impl PyEstimatedTrack {
    /// Get track label
    #[getter]
    fn label(&self) -> PyTrackLabel {
        PyTrackLabel {
            inner: self.inner.label,
        }
    }

    /// Get state mean
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_numpy(py, &self.inner.mean)
    }

    /// Get state covariance
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
            "EstimatedTrack(label=({}, {}), x_dim={})",
            self.inner.label.birth_time, self.inner.label.birth_location, self.inner.x_dim()
        )
    }
}

/// All track state estimates at a single timestep
#[pyclass(name = "StateEstimate")]
#[derive(Clone)]
pub struct PyStateEstimate {
    pub(crate) inner: StateEstimate,
}

#[pymethods]
impl PyStateEstimate {
    /// Get timestamp
    #[getter]
    fn timestamp(&self) -> usize {
        self.inner.timestamp
    }

    /// Get number of estimated tracks
    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.num_tracks()
    }

    /// Get all estimated tracks
    #[getter]
    fn tracks(&self) -> Vec<PyEstimatedTrack> {
        self.inner
            .tracks
            .iter()
            .map(|t| PyEstimatedTrack { inner: t.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "StateEstimate(timestamp={}, num_tracks={})",
            self.inner.timestamp,
            self.inner.num_tracks()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.num_tracks()
    }
}

/// Complete trajectory of a single track across multiple timesteps
#[pyclass(name = "Trajectory")]
#[derive(Clone)]
pub struct PyTrajectory {
    pub(crate) inner: Trajectory,
}

#[pymethods]
impl PyTrajectory {
    /// Get track label
    #[getter]
    fn label(&self) -> PyTrackLabel {
        PyTrackLabel {
            inner: self.inner.label,
        }
    }

    /// Get all states as list of arrays
    #[getter]
    fn states<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<f64>>> {
        self.inner
            .states
            .iter()
            .map(|s| dvector_to_numpy(py, s))
            .collect()
    }

    /// Get all covariances as list of arrays
    #[getter]
    fn covariances<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.inner
            .covariances
            .iter()
            .map(|c| dmatrix_to_numpy(py, c))
            .collect()
    }

    /// Get timestamps
    #[getter]
    fn timestamps(&self) -> Vec<usize> {
        self.inner.timestamps.clone()
    }

    /// Get trajectory length
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get state at index
    fn get_state<'py>(&self, py: Python<'py>, index: usize) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.get_state(index).map(|s| dvector_to_numpy(py, s))
    }

    /// Get covariance at index
    fn get_covariance<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .get_covariance(index)
            .map(|c| dmatrix_to_numpy(py, c))
    }

    /// Get timestamp at index
    fn get_timestamp(&self, index: usize) -> Option<usize> {
        self.inner.get_timestamp(index)
    }

    fn __repr__(&self) -> String {
        format!(
            "Trajectory(label=({}, {}), len={})",
            self.inner.label.birth_time, self.inner.label.birth_location, self.inner.len()
        )
    }
}

/// Complete output from running a filter over a sequence of measurements
#[pyclass(name = "FilterOutput")]
#[derive(Clone)]
pub struct PyFilterOutput {
    pub(crate) inner: FilterOutput,
}

#[pymethods]
impl PyFilterOutput {
    /// Get all state estimates
    #[getter]
    fn estimates(&self) -> Vec<PyStateEstimate> {
        self.inner
            .estimates
            .iter()
            .map(|e| PyStateEstimate { inner: e.clone() })
            .collect()
    }

    /// Get all trajectories
    #[getter]
    fn trajectories(&self) -> Vec<PyTrajectory> {
        self.inner
            .trajectories
            .iter()
            .map(|t| PyTrajectory { inner: t.clone() })
            .collect()
    }

    /// Get number of timesteps
    #[getter]
    fn num_timesteps(&self) -> usize {
        self.inner.num_timesteps()
    }

    /// Get number of trajectories
    #[getter]
    fn num_trajectories(&self) -> usize {
        self.inner.num_trajectories()
    }

    fn __repr__(&self) -> String {
        format!(
            "FilterOutput(timesteps={}, trajectories={})",
            self.inner.num_timesteps(),
            self.inner.num_trajectories()
        )
    }
}
