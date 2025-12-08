//! Python bindings for configuration types

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::lmb::{
    AssociationConfig, BirthLocation, BirthModel, DataAssociationMethod, FilterThresholds,
    LmbmConfig, MotionModel, MultisensorConfig, SensorModel,
};
use nalgebra::{DMatrix, DVector};

/// Convert numpy array to nalgebra DVector
fn numpy_to_dvector(arr: PyReadonlyArray1<'_, f64>) -> DVector<f64> {
    DVector::from_vec(arr.as_slice().unwrap().to_vec())
}

/// Convert numpy array to nalgebra DMatrix
fn numpy_to_dmatrix(arr: PyReadonlyArray2<'_, f64>) -> DMatrix<f64> {
    let shape = arr.shape();
    let rows = shape[0];
    let cols = shape[1];
    let data: Vec<f64> = arr.as_slice().unwrap().to_vec();
    // numpy is row-major, nalgebra is column-major
    DMatrix::from_row_slice(rows, cols, &data)
}

/// Convert nalgebra DVector to numpy array
fn dvector_to_numpy<'py>(py: Python<'py>, v: &DVector<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, v.as_slice())
}

/// Convert nalgebra DMatrix to numpy array (row-major)
fn dmatrix_to_numpy<'py>(py: Python<'py>, m: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
    let rows = m.nrows();
    let cols = m.ncols();
    // Convert to row-major for numpy
    let mut data = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            data[i][j] = m[(i, j)];
        }
    }
    PyArray2::from_vec2(py, &data).unwrap()
}

/// Motion model for prediction
#[pyclass(name = "MotionModel")]
#[derive(Clone)]
pub struct PyMotionModel {
    pub(crate) inner: MotionModel,
}

#[pymethods]
impl PyMotionModel {
    /// Create a new motion model
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

    /// Create a constant velocity 2D motion model
    ///
    /// State: [x, vx, y, vy]
    #[staticmethod]
    #[pyo3(signature = (dt, process_noise_std, survival_prob))]
    fn constant_velocity_2d(dt: f64, process_noise_std: f64, survival_prob: f64) -> Self {
        Self {
            inner: MotionModel::constant_velocity_2d(dt, process_noise_std, survival_prob),
        }
    }

    /// Get state dimension
    #[getter]
    fn x_dim(&self) -> usize {
        self.inner.x_dim()
    }

    /// Get transition matrix
    #[getter]
    fn transition_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.transition_matrix)
    }

    /// Get process noise covariance
    #[getter]
    fn process_noise<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.process_noise)
    }

    /// Get control input
    #[getter]
    fn control_input<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_numpy(py, &self.inner.control_input)
    }

    /// Get survival probability
    #[getter]
    fn survival_probability(&self) -> f64 {
        self.inner.survival_probability
    }

    fn __repr__(&self) -> String {
        format!(
            "MotionModel(x_dim={}, survival_prob={:.3})",
            self.inner.x_dim(),
            self.inner.survival_probability
        )
    }
}

/// Sensor observation model
#[pyclass(name = "SensorModel")]
#[derive(Clone)]
pub struct PySensorModel {
    pub(crate) inner: SensorModel,
}

#[pymethods]
impl PySensorModel {
    /// Create a new sensor model
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

    /// Create a position-only sensor for 4D state [x, vx, y, vy]
    ///
    /// Measures [x, y]
    #[staticmethod]
    #[pyo3(signature = (measurement_noise_std, detection_prob, clutter_rate, obs_volume))]
    fn position_sensor_2d(
        measurement_noise_std: f64,
        detection_prob: f64,
        clutter_rate: f64,
        obs_volume: f64,
    ) -> Self {
        Self {
            inner: SensorModel::position_sensor_2d(
                measurement_noise_std,
                detection_prob,
                clutter_rate,
                obs_volume,
            ),
        }
    }

    /// Get measurement dimension
    #[getter]
    fn z_dim(&self) -> usize {
        self.inner.z_dim()
    }

    /// Get state dimension
    #[getter]
    fn x_dim(&self) -> usize {
        self.inner.x_dim()
    }

    /// Get observation matrix
    #[getter]
    fn observation_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.observation_matrix)
    }

    /// Get measurement noise covariance
    #[getter]
    fn measurement_noise<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_numpy(py, &self.inner.measurement_noise)
    }

    /// Get detection probability
    #[getter]
    fn detection_probability(&self) -> f64 {
        self.inner.detection_probability
    }

    /// Get clutter rate
    #[getter]
    fn clutter_rate(&self) -> f64 {
        self.inner.clutter_rate
    }

    /// Get observation volume
    #[getter]
    fn observation_volume(&self) -> f64 {
        self.inner.observation_volume
    }

    /// Get clutter density
    #[getter]
    fn clutter_density(&self) -> f64 {
        self.inner.clutter_density()
    }

    fn __repr__(&self) -> String {
        format!(
            "SensorModel(z_dim={}, x_dim={}, p_d={:.3})",
            self.inner.z_dim(),
            self.inner.x_dim(),
            self.inner.detection_probability
        )
    }
}

/// Multi-sensor configuration
#[pyclass(name = "MultisensorConfig")]
#[derive(Clone)]
pub struct PyMultisensorConfig {
    pub(crate) inner: MultisensorConfig,
}

#[pymethods]
impl PyMultisensorConfig {
    /// Create a new multi-sensor configuration
    #[new]
    #[pyo3(signature = (sensors))]
    fn new(sensors: Vec<PySensorModel>) -> Self {
        Self {
            inner: MultisensorConfig::new(sensors.into_iter().map(|s| s.inner).collect()),
        }
    }

    /// Number of sensors
    #[getter]
    fn num_sensors(&self) -> usize {
        self.inner.num_sensors()
    }

    /// Get measurement dimension
    #[getter]
    fn z_dim(&self) -> usize {
        self.inner.z_dim()
    }

    fn __repr__(&self) -> String {
        format!("MultisensorConfig(num_sensors={})", self.inner.num_sensors())
    }

    fn __len__(&self) -> usize {
        self.inner.num_sensors()
    }
}

/// Birth location parameters
#[pyclass(name = "BirthLocation")]
#[derive(Clone)]
pub struct PyBirthLocation {
    pub(crate) inner: BirthLocation,
}

#[pymethods]
impl PyBirthLocation {
    /// Create a new birth location
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

    /// Get label
    #[getter]
    fn label(&self) -> usize {
        self.inner.label
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

    fn __repr__(&self) -> String {
        format!("BirthLocation(label={})", self.inner.label)
    }
}

/// Birth model parameters
#[pyclass(name = "BirthModel")]
#[derive(Clone)]
pub struct PyBirthModel {
    pub(crate) inner: BirthModel,
}

#[pymethods]
impl PyBirthModel {
    /// Create a new birth model
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

    /// Number of birth locations
    #[getter]
    fn num_locations(&self) -> usize {
        self.inner.num_locations()
    }

    /// Get LMB existence probability
    #[getter]
    fn lmb_existence(&self) -> f64 {
        self.inner.lmb_existence
    }

    /// Get LMBM existence probability
    #[getter]
    fn lmbm_existence(&self) -> f64 {
        self.inner.lmbm_existence
    }

    fn __repr__(&self) -> String {
        format!(
            "BirthModel(num_locations={}, lmb_r={:.3})",
            self.inner.num_locations(),
            self.inner.lmb_existence
        )
    }
}

/// Data association configuration
#[pyclass(name = "AssociationConfig")]
#[derive(Clone)]
pub struct PyAssociationConfig {
    pub(crate) inner: AssociationConfig,
}

#[pymethods]
impl PyAssociationConfig {
    /// Create LBP association configuration
    #[staticmethod]
    #[pyo3(signature = (max_iterations=50, tolerance=1e-6))]
    fn lbp(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            inner: AssociationConfig::lbp(max_iterations, tolerance),
        }
    }

    /// Create Gibbs association configuration
    #[staticmethod]
    #[pyo3(signature = (samples=1000))]
    fn gibbs(samples: usize) -> Self {
        Self {
            inner: AssociationConfig::gibbs(samples),
        }
    }

    /// Create Murty association configuration
    #[staticmethod]
    #[pyo3(signature = (assignments=100))]
    fn murty(assignments: usize) -> Self {
        Self {
            inner: AssociationConfig::murty(assignments),
        }
    }

    /// Create default configuration (LBP)
    #[new]
    fn new() -> Self {
        Self {
            inner: AssociationConfig::default(),
        }
    }

    /// Get association method as string
    #[getter]
    fn method(&self) -> &'static str {
        match self.inner.method {
            DataAssociationMethod::Lbp => "lbp",
            DataAssociationMethod::LbpFixed => "lbp_fixed",
            DataAssociationMethod::Gibbs => "gibbs",
            DataAssociationMethod::Murty => "murty",
        }
    }

    fn __repr__(&self) -> String {
        format!("AssociationConfig(method='{}')", self.method())
    }
}

/// Filter threshold configuration
#[pyclass(name = "FilterThresholds")]
#[derive(Clone)]
pub struct PyFilterThresholds {
    pub(crate) inner: FilterThresholds,
}

#[pymethods]
impl PyFilterThresholds {
    /// Create new filter thresholds
    #[new]
    #[pyo3(signature = (existence_threshold=0.5, gm_weight_threshold=1e-4, max_gm_components=100, min_trajectory_length=3))]
    fn new(
        existence_threshold: f64,
        gm_weight_threshold: f64,
        max_gm_components: usize,
        min_trajectory_length: usize,
    ) -> Self {
        Self {
            inner: FilterThresholds::new(
                existence_threshold,
                gm_weight_threshold,
                max_gm_components,
                min_trajectory_length,
            ),
        }
    }

    /// Get existence threshold
    #[getter]
    fn existence_threshold(&self) -> f64 {
        self.inner.existence_threshold
    }

    /// Get GM weight threshold
    #[getter]
    fn gm_weight_threshold(&self) -> f64 {
        self.inner.gm_weight_threshold
    }

    /// Get max GM components
    #[getter]
    fn max_gm_components(&self) -> usize {
        self.inner.max_gm_components
    }

    /// Get min trajectory length
    #[getter]
    fn min_trajectory_length(&self) -> usize {
        self.inner.min_trajectory_length
    }

    fn __repr__(&self) -> String {
        format!(
            "FilterThresholds(existence={:.2}, gm_weight={:.0e})",
            self.inner.existence_threshold, self.inner.gm_weight_threshold
        )
    }
}

/// LMBM-specific configuration
#[pyclass(name = "LmbmConfig")]
#[derive(Clone)]
pub struct PyLmbmConfig {
    pub(crate) inner: LmbmConfig,
}

#[pymethods]
impl PyLmbmConfig {
    /// Create new LMBM configuration
    #[new]
    #[pyo3(signature = (max_hypotheses=1000, hypothesis_weight_threshold=1e-6, use_eap=false))]
    fn new(max_hypotheses: usize, hypothesis_weight_threshold: f64, use_eap: bool) -> Self {
        Self {
            inner: LmbmConfig {
                max_hypotheses,
                hypothesis_weight_threshold,
                use_eap,
            },
        }
    }

    /// Get max hypotheses
    #[getter]
    fn max_hypotheses(&self) -> usize {
        self.inner.max_hypotheses
    }

    /// Get hypothesis weight threshold
    #[getter]
    fn hypothesis_weight_threshold(&self) -> f64 {
        self.inner.hypothesis_weight_threshold
    }

    /// Get use EAP flag
    #[getter]
    fn use_eap(&self) -> bool {
        self.inner.use_eap
    }

    fn __repr__(&self) -> String {
        format!(
            "LmbmConfig(max_hyp={}, use_eap={})",
            self.inner.max_hypotheses, self.inner.use_eap
        )
    }
}
