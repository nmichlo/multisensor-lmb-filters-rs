//! Python wrappers for motion and sensor models.

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::config::{MotionModel, SensorConfig, SensorModel};

use super::convert::{dmatrix_to_numpy, numpy_to_dmatrix, numpy_to_dvector};

// =============================================================================
// MotionModel
// =============================================================================

#[pyclass(name = "MotionModel")]
#[derive(Clone)]
pub struct PyMotionModel {
    pub(crate) inner: MotionModel,
}

#[pymethods]
impl PyMotionModel {
    #[new]
    #[pyo3(signature = (transition_matrix, process_noise, control_input, survival_probability))]
    fn new(
        transition_matrix: PyReadonlyArray2<'_, f64>,
        process_noise: PyReadonlyArray2<'_, f64>,
        control_input: PyReadonlyArray1<'_, f64>,
        survival_probability: f64,
    ) -> Self {
        Self {
            inner: MotionModel::new(
                numpy_to_dmatrix(transition_matrix),
                numpy_to_dmatrix(process_noise),
                numpy_to_dvector(control_input),
                survival_probability,
            ),
        }
    }

    /// Create a constant velocity motion model for 2D tracking.
    /// State: [x, vx, y, vy]
    #[staticmethod]
    #[pyo3(signature = (dt, process_noise_std, survival_probability))]
    fn constant_velocity_2d(dt: f64, process_noise_std: f64, survival_probability: f64) -> Self {
        Self {
            inner: MotionModel::constant_velocity_2d(dt, process_noise_std, survival_probability),
        }
    }

    #[getter]
    fn x_dim(&self) -> usize {
        self.inner.x_dim()
    }

    #[getter]
    fn survival_probability(&self) -> f64 {
        self.inner.survival_probability
    }

    #[getter]
    fn transition_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.transition_matrix)
    }

    #[getter]
    fn process_noise<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.process_noise)
    }

    fn __repr__(&self) -> String {
        format!(
            "MotionModel(x_dim={}, survival_probability={:.3})",
            self.inner.x_dim(),
            self.inner.survival_probability
        )
    }
}

// =============================================================================
// SensorModel
// =============================================================================

#[pyclass(name = "SensorModel")]
#[derive(Clone)]
pub struct PySensorModel {
    pub(crate) inner: SensorModel,
}

#[pymethods]
impl PySensorModel {
    #[new]
    #[pyo3(signature = (observation_matrix, measurement_noise, detection_probability, clutter_rate, observation_volume))]
    fn new(
        observation_matrix: PyReadonlyArray2<'_, f64>,
        measurement_noise: PyReadonlyArray2<'_, f64>,
        detection_probability: f64,
        clutter_rate: f64,
        observation_volume: f64,
    ) -> Self {
        Self {
            inner: SensorModel::new(
                numpy_to_dmatrix(observation_matrix),
                numpy_to_dmatrix(measurement_noise),
                detection_probability,
                clutter_rate,
                observation_volume,
            ),
        }
    }

    /// Create a position-only sensor for 2D tracking.
    /// Measures [x, y] from state [x, vx, y, vy].
    #[staticmethod]
    #[pyo3(signature = (measurement_noise_std, detection_probability, clutter_rate, observation_volume))]
    fn position_2d(
        measurement_noise_std: f64,
        detection_probability: f64,
        clutter_rate: f64,
        observation_volume: f64,
    ) -> Self {
        Self {
            inner: SensorModel::position_sensor_2d(
                measurement_noise_std,
                detection_probability,
                clutter_rate,
                observation_volume,
            ),
        }
    }

    #[getter]
    fn x_dim(&self) -> usize {
        self.inner.x_dim()
    }

    #[getter]
    fn z_dim(&self) -> usize {
        self.inner.z_dim()
    }

    #[getter]
    fn detection_probability(&self) -> f64 {
        self.inner.detection_probability
    }

    #[getter]
    fn clutter_rate(&self) -> f64 {
        self.inner.clutter_rate
    }

    #[getter]
    fn observation_volume(&self) -> f64 {
        self.inner.observation_volume
    }

    #[getter]
    fn observation_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.observation_matrix)
    }

    #[getter]
    fn measurement_noise<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.measurement_noise)
    }

    fn __repr__(&self) -> String {
        format!(
            "SensorModel(z_dim={}, x_dim={}, detection_probability={:.3})",
            self.inner.z_dim(),
            self.inner.x_dim(),
            self.inner.detection_probability
        )
    }
}

// =============================================================================
// SensorConfigMulti
// =============================================================================

#[pyclass(name = "SensorConfigMulti")]
#[derive(Clone)]
pub struct PySensorConfigMulti {
    pub(crate) inner: SensorConfig,
}

#[pymethods]
impl PySensorConfigMulti {
    #[new]
    fn new(sensors: Vec<PySensorModel>) -> Self {
        Self {
            inner: SensorConfig::new(sensors.into_iter().map(|s| s.inner).collect()),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.num_sensors()
    }

    fn __getitem__(&self, idx: usize) -> PyResult<PySensorModel> {
        self.inner
            .sensor(idx)
            .map(|s| PySensorModel { inner: s.clone() })
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("sensor index out of range"))
    }

    #[getter]
    fn num_sensors(&self) -> usize {
        self.inner.num_sensors()
    }

    fn __repr__(&self) -> String {
        format!(
            "SensorConfigMulti(num_sensors={})",
            self.inner.num_sensors()
        )
    }
}
