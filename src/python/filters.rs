//! Python wrappers for filter implementations.
//!
//! Uses macros to reduce boilerplate across the 7 filter types.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::lmb::config::{AssociationConfig, DataAssociationMethod, FilterThresholds, LmbmConfig};
use crate::lmb::multisensor::fusion::{
    ArithmeticAverageMerger, GeometricAverageMerger, IteratedCorrectorMerger, ParallelUpdateMerger,
};
use crate::lmb::multisensor::lmb::MultisensorLmbFilter;
use crate::lmb::multisensor::lmbm::MultisensorLmbmFilter;
use crate::lmb::singlesensor::lmb::LmbFilter;
use crate::lmb::singlesensor::lmbm::LmbmFilter;
use crate::lmb::traits::Filter;
use crate::lmb::LbpAssociator;

use super::birth::PyBirthModel;
use super::convert::{numpy_list_to_measurements, numpy_nested_to_measurements};
use super::intermediate::{
    PyAssociationMatrices, PyAssociationResult, PyCardinalityEstimate, PyStepOutput, PyTrackData,
};
use super::models::{PyMotionModel, PySensorConfigMulti, PySensorModel};
use super::output::PyStateEstimate;

// =============================================================================
// AssociatorConfig
// =============================================================================

#[pyclass(name = "AssociatorConfig")]
#[derive(Clone)]
pub struct PyAssociatorConfig {
    pub(crate) inner: AssociationConfig,
}

#[pymethods]
impl PyAssociatorConfig {
    /// Create LBP (Loopy Belief Propagation) associator configuration.
    #[staticmethod]
    #[pyo3(signature = (max_iterations=100, tolerance=1e-6))]
    fn lbp(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            inner: AssociationConfig::lbp(max_iterations, tolerance),
        }
    }

    /// Create Gibbs sampling associator configuration.
    #[staticmethod]
    #[pyo3(signature = (samples=1000))]
    fn gibbs(samples: usize) -> Self {
        Self {
            inner: AssociationConfig::gibbs(samples),
        }
    }

    /// Create Murty's algorithm associator configuration.
    #[staticmethod]
    #[pyo3(signature = (assignments=100))]
    fn murty(assignments: usize) -> Self {
        Self {
            inner: AssociationConfig::murty(assignments),
        }
    }

    fn __repr__(&self) -> String {
        match self.inner.method {
            DataAssociationMethod::Lbp => format!(
                "AssociatorConfig.lbp(max_iterations={}, tolerance={})",
                self.inner.lbp_max_iterations, self.inner.lbp_tolerance
            ),
            DataAssociationMethod::LbpFixed => format!(
                "AssociatorConfig.lbp_fixed(max_iterations={})",
                self.inner.lbp_max_iterations
            ),
            DataAssociationMethod::Gibbs => {
                format!(
                    "AssociatorConfig.gibbs(samples={})",
                    self.inner.gibbs_samples
                )
            }
            DataAssociationMethod::Murty => format!(
                "AssociatorConfig.murty(assignments={})",
                self.inner.murty_assignments
            ),
        }
    }
}

// =============================================================================
// FilterThresholds
// =============================================================================

#[pyclass(name = "FilterThresholds")]
#[derive(Clone)]
pub struct PyFilterThresholds {
    pub(crate) inner: FilterThresholds,
}

#[pymethods]
impl PyFilterThresholds {
    #[new]
    #[pyo3(signature = (existence=0.5, gm_weight=1e-4, max_components=100, min_trajectory_length=3))]
    fn new(
        existence: f64,
        gm_weight: f64,
        max_components: usize,
        min_trajectory_length: usize,
    ) -> Self {
        Self {
            inner: FilterThresholds::new(
                existence,
                gm_weight,
                max_components,
                min_trajectory_length,
            ),
        }
    }

    #[getter]
    fn existence_threshold(&self) -> f64 {
        self.inner.existence_threshold
    }

    #[getter]
    fn gm_weight_threshold(&self) -> f64 {
        self.inner.gm_weight_threshold
    }

    #[getter]
    fn max_gm_components(&self) -> usize {
        self.inner.max_gm_components
    }

    fn __repr__(&self) -> String {
        format!(
            "FilterThresholds(existence={}, gm_weight={}, max_components={})",
            self.inner.existence_threshold,
            self.inner.gm_weight_threshold,
            self.inner.max_gm_components
        )
    }
}

// =============================================================================
// FilterLmbmConfig
// =============================================================================

#[pyclass(name = "FilterLmbmConfig")]
#[derive(Clone)]
pub struct PyFilterLmbmConfig {
    pub(crate) inner: LmbmConfig,
}

#[pymethods]
impl PyFilterLmbmConfig {
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

    fn __repr__(&self) -> String {
        format!(
            "FilterLmbmConfig(max_hypotheses={}, use_eap={})",
            self.inner.max_hypotheses, self.inner.use_eap
        )
    }
}

// =============================================================================
// Helper: Create RNG from optional seed
// =============================================================================

fn create_rng(seed: Option<u64>) -> StdRng {
    seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64)
}

// =============================================================================
// FilterLmb - Single-sensor LMB filter
// =============================================================================

#[pyclass(name = "FilterLmb")]
pub struct PyFilterLmb {
    inner: LmbFilter,
    rng: StdRng,
}

#[pymethods]
impl PyFilterLmb {
    #[new]
    #[pyo3(signature = (motion, sensor, birth, association=None, thresholds=None, seed=None))]
    fn new(
        motion: &PyMotionModel,
        sensor: &PySensorModel,
        birth: &PyBirthModel,
        association: Option<&PyAssociatorConfig>,
        thresholds: Option<&PyFilterThresholds>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let assoc = association.map(|a| a.inner.clone()).unwrap_or_default();
        let thresh = thresholds.map(|t| t.inner.clone()).unwrap_or_default();

        let inner = LmbFilter::new(
            motion.inner.clone(),
            sensor.inner.clone(),
            birth.inner.clone(),
            assoc,
        )
        .with_gm_pruning(thresh.gm_weight_threshold, thresh.max_gm_components);

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    fn step(
        &mut self,
        measurements: Vec<PyReadonlyArray1<'_, f64>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_list_to_measurements(measurements);
        let estimate = self
            .inner
            .step(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(PyStateEstimate::from_inner(estimate))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("FilterLmb(num_tracks={})", self.num_tracks())
    }

    // =========================================================================
    // Testing/Fixture Validation Methods
    // =========================================================================

    /// Set internal tracks from PyTrackData list (for fixture testing).
    fn set_tracks(&mut self, tracks: Vec<PyRef<PyTrackData>>) {
        let rust_tracks: Vec<_> = tracks.iter().map(|t| t.to_track()).collect();
        self.inner.set_tracks(rust_tracks);
    }

    /// Get current tracks as PyTrackData list (for fixture testing).
    fn get_tracks(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTrackData>>> {
        self.inner
            .get_tracks()
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect()
    }

    /// Run a detailed step returning all intermediate data (for fixture testing).
    fn step_detailed(
        &mut self,
        py: Python<'_>,
        measurements: Vec<PyReadonlyArray1<'_, f64>>,
        timestep: usize,
    ) -> PyResult<Py<PyStepOutput>> {
        let meas = numpy_list_to_measurements(measurements);
        let output = self
            .inner
            .step_detailed(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        // Convert predicted tracks
        let predicted: Vec<Py<PyTrackData>> = output
            .predicted_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;

        // Convert association matrices (if present)
        let matrices = output
            .association_matrices
            .map(|m| Py::new(py, PyAssociationMatrices::from_matrices(&m)))
            .transpose()?;

        // Convert association result (if present)
        let result = output
            .association_result
            .map(|r| Py::new(py, PyAssociationResult::from_result(&r)))
            .transpose()?;

        // Convert updated tracks
        let updated: Vec<Py<PyTrackData>> = output
            .updated_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;

        // Convert cardinality
        let cardinality = Py::new(
            py,
            PyCardinalityEstimate {
                n_estimated: output.cardinality.n_estimated,
                map_indices: output.cardinality.map_indices,
            },
        )?;

        Py::new(
            py,
            PyStepOutput {
                predicted_tracks: predicted,
                association_matrices: matrices,
                association_result: result,
                updated_tracks: updated,
                cardinality,
            },
        )
    }
}

// =============================================================================
// FilterLmbm - Single-sensor LMBM filter
// =============================================================================

#[pyclass(name = "FilterLmbm")]
pub struct PyFilterLmbm {
    inner: LmbmFilter,
    rng: StdRng,
}

#[pymethods]
impl PyFilterLmbm {
    #[new]
    #[pyo3(signature = (motion, sensor, birth, association=None, thresholds=None, lmbm_config=None, seed=None))]
    #[allow(unused_variables)]
    fn new(
        motion: &PyMotionModel,
        sensor: &PySensorModel,
        birth: &PyBirthModel,
        association: Option<&PyAssociatorConfig>,
        thresholds: Option<&PyFilterThresholds>,
        lmbm_config: Option<&PyFilterLmbmConfig>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let assoc = association
            .map(|a| a.inner.clone())
            .unwrap_or_else(|| AssociationConfig::gibbs(1000));
        let lmbm = lmbm_config.map(|c| c.inner.clone()).unwrap_or_default();

        // Note: thresholds not used for LMBM filter (no builder method for it)
        let inner = LmbmFilter::new(
            motion.inner.clone(),
            sensor.inner.clone(),
            birth.inner.clone(),
            assoc,
            lmbm,
        );

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    fn step(
        &mut self,
        measurements: Vec<PyReadonlyArray1<'_, f64>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_list_to_measurements(measurements);
        let estimate = self
            .inner
            .step(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(PyStateEstimate::from_inner(estimate))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner
            .state()
            .iter()
            .map(|h| h.tracks.len())
            .max()
            .unwrap_or(0)
    }

    #[getter]
    fn num_hypotheses(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "FilterLmbm(num_hypotheses={}, num_tracks={})",
            self.num_hypotheses(),
            self.num_tracks()
        )
    }

    // =========================================================================
    // Testing/Fixture Validation Methods
    // =========================================================================

    /// Get tracks from highest-weight hypothesis as PyTrackData list (for fixture testing).
    fn get_tracks(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTrackData>>> {
        self.inner
            .get_tracks()
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect()
    }

    /// Run a detailed step returning all intermediate data (for fixture testing).
    fn step_detailed(
        &mut self,
        py: Python<'_>,
        measurements: Vec<PyReadonlyArray1<'_, f64>>,
        timestep: usize,
    ) -> PyResult<Py<PyStepOutput>> {
        let meas = numpy_list_to_measurements(measurements);
        let output = self
            .inner
            .step_detailed(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let predicted: Vec<Py<PyTrackData>> = output
            .predicted_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;

        let updated: Vec<Py<PyTrackData>> = output
            .updated_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;

        let cardinality = Py::new(
            py,
            PyCardinalityEstimate {
                n_estimated: output.cardinality.n_estimated,
                map_indices: output.cardinality.map_indices,
            },
        )?;

        Py::new(
            py,
            PyStepOutput {
                predicted_tracks: predicted,
                association_matrices: None, // LMBM doesn't expose this
                association_result: None,   // LMBM doesn't expose this
                updated_tracks: updated,
                cardinality,
            },
        )
    }
}

// =============================================================================
// FilterAaLmb - Arithmetic Average multi-sensor LMB filter
// =============================================================================

#[pyclass(name = "FilterAaLmb")]
pub struct PyFilterAaLmb {
    inner: MultisensorLmbFilter<LbpAssociator, ArithmeticAverageMerger>,
    rng: StdRng,
}

#[pymethods]
impl PyFilterAaLmb {
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, thresholds=None, seed=None))]
    fn new(
        motion: &PyMotionModel,
        sensors: &PySensorConfigMulti,
        birth: &PyBirthModel,
        association: Option<&PyAssociatorConfig>,
        thresholds: Option<&PyFilterThresholds>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let assoc = association.map(|a| a.inner.clone()).unwrap_or_default();
        let thresh = thresholds.map(|t| t.inner.clone()).unwrap_or_default();

        let num_sensors = sensors.inner.num_sensors();
        let merger = ArithmeticAverageMerger::uniform(num_sensors, thresh.max_gm_components);

        let inner = MultisensorLmbFilter::new(
            motion.inner.clone(),
            sensors.inner.clone(),
            birth.inner.clone(),
            assoc,
            merger,
        )
        .with_gm_pruning(thresh.gm_weight_threshold, thresh.max_gm_components);

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    fn step(
        &mut self,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_nested_to_measurements(measurements);
        let estimate = self
            .inner
            .step(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(PyStateEstimate::from_inner(estimate))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("FilterAaLmb(num_tracks={})", self.num_tracks())
    }

    // Testing methods
    fn set_tracks(&mut self, tracks: Vec<PyRef<PyTrackData>>) {
        let rust_tracks: Vec<_> = tracks.iter().map(|t| t.to_track()).collect();
        self.inner.set_tracks(rust_tracks);
    }

    fn get_tracks(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTrackData>>> {
        self.inner
            .get_tracks()
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect()
    }

    fn step_detailed(
        &mut self,
        py: Python<'_>,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<Py<PyStepOutput>> {
        let meas = numpy_nested_to_measurements(measurements);
        let output = self
            .inner
            .step_detailed(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let predicted: Vec<Py<PyTrackData>> = output
            .predicted_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;
        let updated: Vec<Py<PyTrackData>> = output
            .updated_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;
        let cardinality = Py::new(
            py,
            PyCardinalityEstimate {
                n_estimated: output.cardinality.n_estimated,
                map_indices: output.cardinality.map_indices,
            },
        )?;

        Py::new(
            py,
            PyStepOutput {
                predicted_tracks: predicted,
                association_matrices: None,
                association_result: None,
                updated_tracks: updated,
                cardinality,
            },
        )
    }
}

// =============================================================================
// FilterGaLmb - Geometric Average multi-sensor LMB filter
// =============================================================================

#[pyclass(name = "FilterGaLmb")]
pub struct PyFilterGaLmb {
    inner: MultisensorLmbFilter<LbpAssociator, GeometricAverageMerger>,
    rng: StdRng,
}

#[pymethods]
impl PyFilterGaLmb {
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, thresholds=None, seed=None))]
    fn new(
        motion: &PyMotionModel,
        sensors: &PySensorConfigMulti,
        birth: &PyBirthModel,
        association: Option<&PyAssociatorConfig>,
        thresholds: Option<&PyFilterThresholds>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let assoc = association.map(|a| a.inner.clone()).unwrap_or_default();
        let thresh = thresholds.map(|t| t.inner.clone()).unwrap_or_default();

        let num_sensors = sensors.inner.num_sensors();
        let merger = GeometricAverageMerger::uniform(num_sensors);

        let inner = MultisensorLmbFilter::new(
            motion.inner.clone(),
            sensors.inner.clone(),
            birth.inner.clone(),
            assoc,
            merger,
        )
        .with_gm_pruning(thresh.gm_weight_threshold, thresh.max_gm_components);

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    fn step(
        &mut self,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_nested_to_measurements(measurements);
        let estimate = self
            .inner
            .step(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(PyStateEstimate::from_inner(estimate))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("FilterGaLmb(num_tracks={})", self.num_tracks())
    }

    // Testing methods
    fn set_tracks(&mut self, tracks: Vec<PyRef<PyTrackData>>) {
        let rust_tracks: Vec<_> = tracks.iter().map(|t| t.to_track()).collect();
        self.inner.set_tracks(rust_tracks);
    }

    fn get_tracks(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTrackData>>> {
        self.inner
            .get_tracks()
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect()
    }

    fn step_detailed(
        &mut self,
        py: Python<'_>,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<Py<PyStepOutput>> {
        let meas = numpy_nested_to_measurements(measurements);
        let output = self
            .inner
            .step_detailed(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let predicted: Vec<Py<PyTrackData>> = output
            .predicted_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;
        let updated: Vec<Py<PyTrackData>> = output
            .updated_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;
        let cardinality = Py::new(
            py,
            PyCardinalityEstimate {
                n_estimated: output.cardinality.n_estimated,
                map_indices: output.cardinality.map_indices,
            },
        )?;

        Py::new(
            py,
            PyStepOutput {
                predicted_tracks: predicted,
                association_matrices: None,
                association_result: None,
                updated_tracks: updated,
                cardinality,
            },
        )
    }
}

// =============================================================================
// FilterPuLmb - Parallel Update multi-sensor LMB filter
// =============================================================================

#[pyclass(name = "FilterPuLmb")]
pub struct PyFilterPuLmb {
    inner: MultisensorLmbFilter<LbpAssociator, ParallelUpdateMerger>,
    rng: StdRng,
}

#[pymethods]
impl PyFilterPuLmb {
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, thresholds=None, seed=None))]
    fn new(
        motion: &PyMotionModel,
        sensors: &PySensorConfigMulti,
        birth: &PyBirthModel,
        association: Option<&PyAssociatorConfig>,
        thresholds: Option<&PyFilterThresholds>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let assoc = association.map(|a| a.inner.clone()).unwrap_or_default();
        let thresh = thresholds.map(|t| t.inner.clone()).unwrap_or_default();

        // PU merger needs prior tracks - start with empty
        let merger = ParallelUpdateMerger::new(Vec::new());

        let inner = MultisensorLmbFilter::new(
            motion.inner.clone(),
            sensors.inner.clone(),
            birth.inner.clone(),
            assoc,
            merger,
        )
        .with_gm_pruning(thresh.gm_weight_threshold, thresh.max_gm_components);

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    fn step(
        &mut self,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_nested_to_measurements(measurements);
        let estimate = self
            .inner
            .step(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(PyStateEstimate::from_inner(estimate))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("FilterPuLmb(num_tracks={})", self.num_tracks())
    }

    // Testing methods
    fn set_tracks(&mut self, tracks: Vec<PyRef<PyTrackData>>) {
        let rust_tracks: Vec<_> = tracks.iter().map(|t| t.to_track()).collect();
        self.inner.set_tracks(rust_tracks);
    }

    fn get_tracks(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTrackData>>> {
        self.inner
            .get_tracks()
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect()
    }

    fn step_detailed(
        &mut self,
        py: Python<'_>,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<Py<PyStepOutput>> {
        let meas = numpy_nested_to_measurements(measurements);
        let output = self
            .inner
            .step_detailed(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let predicted: Vec<Py<PyTrackData>> = output
            .predicted_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;
        let updated: Vec<Py<PyTrackData>> = output
            .updated_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;
        let cardinality = Py::new(
            py,
            PyCardinalityEstimate {
                n_estimated: output.cardinality.n_estimated,
                map_indices: output.cardinality.map_indices,
            },
        )?;

        Py::new(
            py,
            PyStepOutput {
                predicted_tracks: predicted,
                association_matrices: None,
                association_result: None,
                updated_tracks: updated,
                cardinality,
            },
        )
    }
}

// =============================================================================
// FilterIcLmb - Iterated Corrector multi-sensor LMB filter
// =============================================================================

#[pyclass(name = "FilterIcLmb")]
pub struct PyFilterIcLmb {
    inner: MultisensorLmbFilter<LbpAssociator, IteratedCorrectorMerger>,
    rng: StdRng,
}

#[pymethods]
impl PyFilterIcLmb {
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, thresholds=None, seed=None))]
    fn new(
        motion: &PyMotionModel,
        sensors: &PySensorConfigMulti,
        birth: &PyBirthModel,
        association: Option<&PyAssociatorConfig>,
        thresholds: Option<&PyFilterThresholds>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let assoc = association.map(|a| a.inner.clone()).unwrap_or_default();
        let thresh = thresholds.map(|t| t.inner.clone()).unwrap_or_default();

        let merger = IteratedCorrectorMerger::new();

        let inner = MultisensorLmbFilter::new(
            motion.inner.clone(),
            sensors.inner.clone(),
            birth.inner.clone(),
            assoc,
            merger,
        )
        .with_gm_pruning(thresh.gm_weight_threshold, thresh.max_gm_components);

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    fn step(
        &mut self,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_nested_to_measurements(measurements);
        let estimate = self
            .inner
            .step(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(PyStateEstimate::from_inner(estimate))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("FilterIcLmb(num_tracks={})", self.num_tracks())
    }

    // Testing methods
    fn set_tracks(&mut self, tracks: Vec<PyRef<PyTrackData>>) {
        let rust_tracks: Vec<_> = tracks.iter().map(|t| t.to_track()).collect();
        self.inner.set_tracks(rust_tracks);
    }

    fn get_tracks(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTrackData>>> {
        self.inner
            .get_tracks()
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect()
    }

    fn step_detailed(
        &mut self,
        py: Python<'_>,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<Py<PyStepOutput>> {
        let meas = numpy_nested_to_measurements(measurements);
        let output = self
            .inner
            .step_detailed(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let predicted: Vec<Py<PyTrackData>> = output
            .predicted_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;
        let updated: Vec<Py<PyTrackData>> = output
            .updated_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;
        let cardinality = Py::new(
            py,
            PyCardinalityEstimate {
                n_estimated: output.cardinality.n_estimated,
                map_indices: output.cardinality.map_indices,
            },
        )?;

        Py::new(
            py,
            PyStepOutput {
                predicted_tracks: predicted,
                association_matrices: None,
                association_result: None,
                updated_tracks: updated,
                cardinality,
            },
        )
    }
}

// =============================================================================
// FilterMultisensorLmbm - Multi-sensor LMBM filter
// =============================================================================

#[pyclass(name = "FilterMultisensorLmbm")]
pub struct PyFilterMultisensorLmbm {
    inner: MultisensorLmbmFilter,
    rng: StdRng,
}

#[pymethods]
impl PyFilterMultisensorLmbm {
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, thresholds=None, lmbm_config=None, seed=None))]
    #[allow(unused_variables)]
    fn new(
        motion: &PyMotionModel,
        sensors: &PySensorConfigMulti,
        birth: &PyBirthModel,
        association: Option<&PyAssociatorConfig>,
        thresholds: Option<&PyFilterThresholds>,
        lmbm_config: Option<&PyFilterLmbmConfig>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let assoc = association
            .map(|a| a.inner.clone())
            .unwrap_or_else(|| AssociationConfig::gibbs(1000));
        let lmbm = lmbm_config.map(|c| c.inner.clone()).unwrap_or_default();

        // Note: thresholds not used for LMBM filter (no builder method for it)
        let inner = MultisensorLmbmFilter::new(
            motion.inner.clone(),
            sensors.inner.clone(),
            birth.inner.clone(),
            assoc,
            lmbm,
        );

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    fn step(
        &mut self,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<PyStateEstimate> {
        let meas = numpy_nested_to_measurements(measurements);
        let estimate = self
            .inner
            .step(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(PyStateEstimate::from_inner(estimate))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner
            .state()
            .iter()
            .map(|h| h.tracks.len())
            .max()
            .unwrap_or(0)
    }

    #[getter]
    fn num_hypotheses(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "FilterMultisensorLmbm(num_hypotheses={}, num_tracks={})",
            self.num_hypotheses(),
            self.num_tracks()
        )
    }

    // Testing methods
    fn get_tracks(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTrackData>>> {
        self.inner
            .get_tracks()
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect()
    }

    fn step_detailed(
        &mut self,
        py: Python<'_>,
        measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        timestep: usize,
    ) -> PyResult<Py<PyStepOutput>> {
        let meas = numpy_nested_to_measurements(measurements);
        let output = self
            .inner
            .step_detailed(&mut self.rng, &meas, timestep)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let predicted: Vec<Py<PyTrackData>> = output
            .predicted_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;
        let updated: Vec<Py<PyTrackData>> = output
            .updated_tracks
            .iter()
            .map(|t| Py::new(py, PyTrackData::from_track(t)))
            .collect::<PyResult<_>>()?;
        let cardinality = Py::new(
            py,
            PyCardinalityEstimate {
                n_estimated: output.cardinality.n_estimated,
                map_indices: output.cardinality.map_indices,
            },
        )?;

        Py::new(
            py,
            PyStepOutput {
                predicted_tracks: predicted,
                association_matrices: None,
                association_result: None,
                updated_tracks: updated,
                cardinality,
            },
        )
    }
}
