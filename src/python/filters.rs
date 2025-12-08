//! Python bindings for filter implementations

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rand::SeedableRng;

use crate::lmb::{
    AaLmbFilter, ArithmeticAverageMerger, Filter, GaLmbFilter, GeometricAverageMerger, IcLmbFilter,
    IteratedCorrectorMerger, LbpAssociator, LmbFilter, LmbmFilter, MultisensorLmbmFilter,
    ParallelUpdateMerger, PuLmbFilter,
};
use nalgebra::DVector;

use super::config::{
    PyAssociationConfig, PyBirthModel, PyLmbmConfig, PyMotionModel, PyMultisensorConfig,
    PySensorModel,
};
use super::output::PyStateEstimate;

/// Convert numpy array to nalgebra DVector
fn numpy_to_dvector(arr: PyReadonlyArray1<'_, f64>) -> DVector<f64> {
    DVector::from_vec(arr.as_slice().unwrap().to_vec())
}

/// Convert list of numpy arrays to Vec<DVector>
fn numpy_list_to_measurements(measurements: Vec<PyReadonlyArray1<'_, f64>>) -> Vec<DVector<f64>> {
    measurements.into_iter().map(numpy_to_dvector).collect()
}

/// Convert list of list of numpy arrays to Vec<Vec<DVector>> (multisensor)
fn numpy_nested_to_measurements(
    measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
) -> Vec<Vec<DVector<f64>>> {
    measurements
        .into_iter()
        .map(|sensor_meas| sensor_meas.into_iter().map(numpy_to_dvector).collect())
        .collect()
}

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
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        Self {
            inner: LmbFilter::new(motion.inner, sensor.inner, birth.inner, association_config),
            rng,
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
        match self.inner.step(&mut self.rng, &meas, timestep) {
            Ok(estimate) => Ok(PyStateEstimate { inner: estimate }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Filter error: {:?}",
                e
            ))),
        }
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
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        Self {
            inner: LmbmFilter::new(
                motion.inner,
                sensor.inner,
                birth.inner,
                association_config,
                lmbm,
            ),
            rng,
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
        match self.inner.step(&mut self.rng, &meas, timestep) {
            Ok(estimate) => Ok(PyStateEstimate { inner: estimate }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Filter error: {:?}",
                e
            ))),
        }
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

    fn __repr__(&self) -> String {
        format!("LmbmFilter(num_hypotheses={})", self.inner.state().len())
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
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        Self {
            inner: AaLmbFilter::new(
                motion.inner,
                sensors.inner,
                birth.inner,
                association_config,
                merger,
            ),
            rng,
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
        match self.inner.step(&mut self.rng, &meas, timestep) {
            Ok(estimate) => Ok(PyStateEstimate { inner: estimate }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Filter error: {:?}",
                e
            ))),
        }
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
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        Self {
            inner: GaLmbFilter::new(
                motion.inner,
                sensors.inner,
                birth.inner,
                association_config,
                merger,
            ),
            rng,
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
        match self.inner.step(&mut self.rng, &meas, timestep) {
            Ok(estimate) => Ok(PyStateEstimate { inner: estimate }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Filter error: {:?}",
                e
            ))),
        }
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
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        Self {
            inner: PuLmbFilter::new(
                motion.inner,
                sensors.inner,
                birth.inner,
                association_config,
                merger,
            ),
            rng,
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
        match self.inner.step(&mut self.rng, &meas, timestep) {
            Ok(estimate) => Ok(PyStateEstimate { inner: estimate }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Filter error: {:?}",
                e
            ))),
        }
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
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        Self {
            inner: IcLmbFilter::new(
                motion.inner,
                sensors.inner,
                birth.inner,
                association_config,
                merger,
            ),
            rng,
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
        match self.inner.step(&mut self.rng, &meas, timestep) {
            Ok(estimate) => Ok(PyStateEstimate { inner: estimate }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Filter error: {:?}",
                e
            ))),
        }
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
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        Self {
            inner: MultisensorLmbmFilter::new(
                motion.inner,
                sensors.inner,
                birth.inner,
                association_config,
                lmbm,
            ),
            rng,
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
        match self.inner.step(&mut self.rng, &meas, timestep) {
            Ok(estimate) => Ok(PyStateEstimate { inner: estimate }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Filter error: {:?}",
                e
            ))),
        }
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

    fn __repr__(&self) -> String {
        format!(
            "MultisensorLmbmFilter(num_hypotheses={})",
            self.inner.state().len()
        )
    }
}
