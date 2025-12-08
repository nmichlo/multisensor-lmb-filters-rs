//! Python bindings for filter implementations

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use super::common::{create_rng, wrap_filter_error};
use super::config::{
    PyAssociationConfig, PyBirthModel, PyLmbmConfig, PyMotionModel, PyMultisensorConfig,
    PySensorModel,
};
use super::convert::{numpy_list_to_measurements, numpy_nested_to_measurements};
use super::output::PyStateEstimate;
use crate::lmb::{
    AaLmbFilter, ArithmeticAverageMerger, Filter, GaLmbFilter, GeometricAverageMerger, IcLmbFilter,
    IteratedCorrectorMerger, LbpAssociator, LmbFilter, LmbmFilter, MultisensorLmbmFilter,
    ParallelUpdateMerger, PuLmbFilter,
};

/// Single-sensor LMB filter
#[pyclass(name = "LmbFilter")]
pub struct PyLmbFilter {
    inner: LmbFilter<LbpAssociator>,
    rng: rand::rngs::StdRng,
}

#[pymethods]
impl PyLmbFilter {
    /// Create a new LMB filter
    #[new]
    #[pyo3(signature = (motion, sensor, birth, association=None, seed=None))]
    fn new(
        motion: PyMotionModel,
        sensor: PySensorModel,
        birth: PyBirthModel,
        association: Option<PyAssociationConfig>,
        seed: Option<u64>,
    ) -> Self {
        let association_config = association.map(|a| a.inner).unwrap_or_default();
        Self {
            inner: LmbFilter::new(motion.inner, sensor.inner, birth.inner, association_config),
            rng: create_rng(seed),
        }
    }

    /// Process one timestep of measurements
    ///
    /// Args:
    ///     measurements: List of measurement arrays, each shape (z_dim,)
    ///     timestep: Current timestep index
    ///
    /// Returns:
    ///     StateEstimate with estimated tracks at this timestep
    #[pyo3(signature = (measurements, timestep))]
    fn step(
        &mut self,
        measurements: Vec<PyReadonlyArray1<'_, f64>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_list_to_measurements(measurements);
        wrap_filter_error(self.inner.step(&mut self.rng, &meas, timestep))
            .map(|estimate| PyStateEstimate { inner: estimate })
    }

    /// Reset the filter to initial state
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get number of current tracks
    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("LmbFilter(num_tracks={})", self.inner.state().len())
    }
}

/// Single-sensor LMBM filter
#[pyclass(name = "LmbmFilter")]
pub struct PyLmbmFilter {
    inner: LmbmFilter,
    rng: rand::rngs::StdRng,
}

#[pymethods]
impl PyLmbmFilter {
    /// Create a new LMBM filter
    #[new]
    #[pyo3(signature = (motion, sensor, birth, association=None, lmbm_config=None, seed=None))]
    fn new(
        motion: PyMotionModel,
        sensor: PySensorModel,
        birth: PyBirthModel,
        association: Option<PyAssociationConfig>,
        lmbm_config: Option<PyLmbmConfig>,
        seed: Option<u64>,
    ) -> Self {
        let association_config = association.map(|a| a.inner).unwrap_or_default();
        let lmbm = lmbm_config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: LmbmFilter::new(
                motion.inner,
                sensor.inner,
                birth.inner,
                association_config,
                lmbm,
            ),
            rng: create_rng(seed),
        }
    }

    /// Process one timestep of measurements
    #[pyo3(signature = (measurements, timestep))]
    fn step(
        &mut self,
        measurements: Vec<PyReadonlyArray1<'_, f64>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_list_to_measurements(measurements);
        wrap_filter_error(self.inner.step(&mut self.rng, &meas, timestep))
            .map(|estimate| PyStateEstimate { inner: estimate })
    }

    /// Reset the filter to initial state
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get number of hypotheses
    #[getter]
    fn num_hypotheses(&self) -> usize {
        self.inner.state().len()
    }

    /// Get number of tracks (max across all hypotheses)
    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner
            .state()
            .iter()
            .map(|h| h.tracks.len())
            .max()
            .unwrap_or(0)
    }

    fn __repr__(&self) -> String {
        format!(
            "LmbmFilter(num_hypotheses={}, num_tracks={})",
            self.inner.state().len(),
            self.num_tracks()
        )
    }
}

/// Multi-sensor LMB filter with Arithmetic Average fusion
#[pyclass(name = "AaLmbFilter")]
pub struct PyAaLmbFilter {
    inner: AaLmbFilter,
    rng: rand::rngs::StdRng,
}

#[pymethods]
impl PyAaLmbFilter {
    /// Create a new AA-LMB filter
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, max_components=100, seed=None))]
    fn new(
        motion: PyMotionModel,
        sensors: PyMultisensorConfig,
        birth: PyBirthModel,
        association: Option<PyAssociationConfig>,
        max_components: usize,
        seed: Option<u64>,
    ) -> Self {
        let association_config = association.map(|a| a.inner).unwrap_or_default();
        let num_sensors = sensors.inner.num_sensors();
        let merger = ArithmeticAverageMerger::uniform(num_sensors, max_components);
        Self {
            inner: AaLmbFilter::new(
                motion.inner,
                sensors.inner,
                birth.inner,
                association_config,
                merger,
            ),
            rng: create_rng(seed),
        }
    }

    /// Process one timestep of measurements from all sensors
    ///
    /// Args:
    ///     measurements: List of sensor measurements, each is a list of arrays
    ///     timestep: Current timestep index
    #[pyo3(signature = (measurements, timestep))]
    fn step(
        &mut self,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_nested_to_measurements(measurements);
        wrap_filter_error(self.inner.step(&mut self.rng, &meas, timestep))
            .map(|estimate| PyStateEstimate { inner: estimate })
    }

    /// Reset the filter to initial state
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get number of current tracks
    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("AaLmbFilter(num_tracks={})", self.inner.state().len())
    }
}

/// Multi-sensor LMB filter with Geometric Average fusion
#[pyclass(name = "GaLmbFilter")]
pub struct PyGaLmbFilter {
    inner: GaLmbFilter,
    rng: rand::rngs::StdRng,
}

#[pymethods]
impl PyGaLmbFilter {
    /// Create a new GA-LMB filter
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, seed=None))]
    fn new(
        motion: PyMotionModel,
        sensors: PyMultisensorConfig,
        birth: PyBirthModel,
        association: Option<PyAssociationConfig>,
        seed: Option<u64>,
    ) -> Self {
        let association_config = association.map(|a| a.inner).unwrap_or_default();
        let num_sensors = sensors.inner.num_sensors();
        let merger = GeometricAverageMerger::uniform(num_sensors);
        Self {
            inner: GaLmbFilter::new(
                motion.inner,
                sensors.inner,
                birth.inner,
                association_config,
                merger,
            ),
            rng: create_rng(seed),
        }
    }

    /// Process one timestep of measurements from all sensors
    #[pyo3(signature = (measurements, timestep))]
    fn step(
        &mut self,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_nested_to_measurements(measurements);
        wrap_filter_error(self.inner.step(&mut self.rng, &meas, timestep))
            .map(|estimate| PyStateEstimate { inner: estimate })
    }

    /// Reset the filter to initial state
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get number of current tracks
    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("GaLmbFilter(num_tracks={})", self.inner.state().len())
    }
}

/// Multi-sensor LMB filter with Parallel Update fusion
#[pyclass(name = "PuLmbFilter")]
pub struct PyPuLmbFilter {
    inner: PuLmbFilter,
    rng: rand::rngs::StdRng,
}

#[pymethods]
impl PyPuLmbFilter {
    /// Create a new PU-LMB filter
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, seed=None))]
    fn new(
        motion: PyMotionModel,
        sensors: PyMultisensorConfig,
        birth: PyBirthModel,
        association: Option<PyAssociationConfig>,
        seed: Option<u64>,
    ) -> Self {
        let association_config = association.map(|a| a.inner).unwrap_or_default();
        let merger = ParallelUpdateMerger::new(Vec::new());
        Self {
            inner: PuLmbFilter::new(
                motion.inner,
                sensors.inner,
                birth.inner,
                association_config,
                merger,
            ),
            rng: create_rng(seed),
        }
    }

    /// Process one timestep of measurements from all sensors
    #[pyo3(signature = (measurements, timestep))]
    fn step(
        &mut self,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_nested_to_measurements(measurements);
        wrap_filter_error(self.inner.step(&mut self.rng, &meas, timestep))
            .map(|estimate| PyStateEstimate { inner: estimate })
    }

    /// Reset the filter to initial state
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get number of current tracks
    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("PuLmbFilter(num_tracks={})", self.inner.state().len())
    }
}

/// Multi-sensor LMB filter with Iterated Corrector fusion
#[pyclass(name = "IcLmbFilter")]
pub struct PyIcLmbFilter {
    inner: IcLmbFilter,
    rng: rand::rngs::StdRng,
}

#[pymethods]
impl PyIcLmbFilter {
    /// Create a new IC-LMB filter
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, seed=None))]
    fn new(
        motion: PyMotionModel,
        sensors: PyMultisensorConfig,
        birth: PyBirthModel,
        association: Option<PyAssociationConfig>,
        seed: Option<u64>,
    ) -> Self {
        let association_config = association.map(|a| a.inner).unwrap_or_default();
        let merger = IteratedCorrectorMerger::new();
        Self {
            inner: IcLmbFilter::new(
                motion.inner,
                sensors.inner,
                birth.inner,
                association_config,
                merger,
            ),
            rng: create_rng(seed),
        }
    }

    /// Process one timestep of measurements from all sensors
    #[pyo3(signature = (measurements, timestep))]
    fn step(
        &mut self,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_nested_to_measurements(measurements);
        wrap_filter_error(self.inner.step(&mut self.rng, &meas, timestep))
            .map(|estimate| PyStateEstimate { inner: estimate })
    }

    /// Reset the filter to initial state
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get number of current tracks
    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("IcLmbFilter(num_tracks={})", self.inner.state().len())
    }
}

/// Multi-sensor LMBM filter
#[pyclass(name = "MultisensorLmbmFilter")]
pub struct PyMultisensorLmbmFilter {
    inner: MultisensorLmbmFilter,
    rng: rand::rngs::StdRng,
}

#[pymethods]
impl PyMultisensorLmbmFilter {
    /// Create a new multi-sensor LMBM filter
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, lmbm_config=None, seed=None))]
    fn new(
        motion: PyMotionModel,
        sensors: PyMultisensorConfig,
        birth: PyBirthModel,
        association: Option<PyAssociationConfig>,
        lmbm_config: Option<PyLmbmConfig>,
        seed: Option<u64>,
    ) -> Self {
        let association_config = association.map(|a| a.inner).unwrap_or_default();
        let lmbm = lmbm_config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: MultisensorLmbmFilter::new(
                motion.inner,
                sensors.inner,
                birth.inner,
                association_config,
                lmbm,
            ),
            rng: create_rng(seed),
        }
    }

    /// Process one timestep of measurements from all sensors
    #[pyo3(signature = (measurements, timestep))]
    fn step(
        &mut self,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_nested_to_measurements(measurements);
        wrap_filter_error(self.inner.step(&mut self.rng, &meas, timestep))
            .map(|estimate| PyStateEstimate { inner: estimate })
    }

    /// Reset the filter to initial state
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get number of hypotheses
    #[getter]
    fn num_hypotheses(&self) -> usize {
        self.inner.state().len()
    }

    /// Get number of tracks (max across all hypotheses)
    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner
            .state()
            .iter()
            .map(|h| h.tracks.len())
            .max()
            .unwrap_or(0)
    }

    fn __repr__(&self) -> String {
        format!(
            "MultisensorLmbmFilter(num_hypotheses={}, num_tracks={})",
            self.inner.state().len(),
            self.num_tracks()
        )
    }
}
