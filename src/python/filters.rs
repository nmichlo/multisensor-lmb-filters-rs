//! Python wrappers for filter implementations.
//!
//! Uses helper functions and macros to reduce boilerplate across the 7 filter types.

use crate::common::rng::SimpleRng;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::lmb::config::{AssociationConfig, DataAssociationMethod, FilterThresholds, LmbmConfig};
use crate::lmb::errors::FilterError;
use crate::lmb::multisensor::fusion::{
    ArithmeticAverageMerger, GeometricAverageMerger, IteratedCorrectorMerger, ParallelUpdateMerger,
};
use crate::lmb::multisensor::lmb::MultisensorLmbFilter;
use crate::lmb::multisensor::lmbm::MultisensorLmbmFilter;
use crate::lmb::singlesensor::lmb::LmbFilter;
use crate::lmb::singlesensor::lmbm::LmbmFilter;
use crate::lmb::traits::Filter;
use crate::lmb::types::{StepDetailedOutput, Track};
use crate::lmb::{DynamicAssociator, LbpAssociator};

use super::birth::PyBirthModel;
use super::convert::{numpy_list_to_measurements, numpy_nested_to_measurements};
use super::intermediate::{
    PyAssociationMatrices, PyAssociationResult, PyCardinalityEstimate, PySensorUpdateOutput,
    PyStepOutput, PyTrackData,
};
use super::models::{PyMotionModel, PySensorConfigMulti, PySensorModel};
use super::output::PyStateEstimate;

// =============================================================================
// Shared Helper Functions
// =============================================================================

/// Convert filter error to PyO3 RuntimeError
fn filter_error_to_py(e: FilterError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e))
}

/// Convert Track slice to Python list of PyTrackData
fn tracks_to_py(py: Python<'_>, tracks: &[Track]) -> PyResult<Vec<Py<PyTrackData>>> {
    tracks
        .iter()
        .map(|t| Py::new(py, PyTrackData::from_track(t)))
        .collect()
}

/// Convert StepDetailedOutput to PyStepOutput
///
/// # Arguments
/// * `include_association` - Whether to include association matrices and results
/// * `use_log_space_l` - Whether L matrix should be in log space (LMBM) or linear space (LMB)
fn step_output_to_py(
    py: Python<'_>,
    output: StepDetailedOutput,
    include_association: bool,
    use_log_space_l: bool,
) -> PyResult<Py<PyStepOutput>> {
    let predicted = tracks_to_py(py, &output.predicted_tracks)?;
    let updated = tracks_to_py(py, &output.updated_tracks)?;

    // Extract prior data from predicted tracks for posteriorParameters
    // MATLAB's posteriorParameters includes miss hypothesis which uses prior values
    let prior_weights: Vec<Vec<f64>> = output
        .predicted_tracks
        .iter()
        .map(|t| t.components.iter().map(|c| c.weight).collect())
        .collect();

    let prior_means: Vec<Vec<Vec<f64>>> = output
        .predicted_tracks
        .iter()
        .map(|t| {
            t.components
                .iter()
                .map(|c| c.mean.iter().copied().collect())
                .collect()
        })
        .collect();

    let prior_covariances: Vec<Vec<Vec<Vec<f64>>>> = output
        .predicted_tracks
        .iter()
        .map(|t| {
            t.components
                .iter()
                .map(|c| {
                    let nrows = c.covariance.nrows();
                    (0..nrows)
                        .map(|i| c.covariance.row(i).iter().copied().collect())
                        .collect()
                })
                .collect()
        })
        .collect();

    let matrices = if include_association {
        output
            .association_matrices
            .map(|m| {
                Py::new(
                    py,
                    PyAssociationMatrices::from_matrices(
                        &m,
                        &prior_weights,
                        &prior_means,
                        &prior_covariances,
                        use_log_space_l,
                    ),
                )
            })
            .transpose()?
    } else {
        None
    };

    let result = if include_association {
        output
            .association_result
            .map(|r| Py::new(py, PyAssociationResult::from_result(&r)))
            .transpose()?
    } else {
        None
    };

    let cardinality = Py::new(
        py,
        PyCardinalityEstimate {
            n_estimated: output.cardinality.n_estimated,
            map_indices: output.cardinality.map_indices,
        },
    )?;

    // Convert LMBM-specific fields (None for LMB filters)
    let predicted_hypotheses = output
        .predicted_hypotheses
        .map(|hyps| {
            hyps.iter()
                .map(|h| {
                    Py::new(
                        py,
                        super::intermediate::PyLmbmHypothesis::from_hypothesis(h),
                    )
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;

    let pre_normalization_hypotheses = output
        .pre_normalization_hypotheses
        .map(|hyps| {
            hyps.iter()
                .map(|h| {
                    Py::new(
                        py,
                        super::intermediate::PyLmbmHypothesis::from_hypothesis(h),
                    )
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;

    let normalized_hypotheses = output
        .normalized_hypotheses
        .map(|hyps| {
            hyps.iter()
                .map(|h| {
                    Py::new(
                        py,
                        super::intermediate::PyLmbmHypothesis::from_hypothesis(h),
                    )
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;

    // Convert per-sensor updates (for multisensor filters)
    let sensor_updates = output
        .sensor_updates
        .map(|updates| {
            updates
                .into_iter()
                .map(|su| {
                    // Extract prior data from this sensor's INPUT tracks
                    // For sequential mergers (IC-LMB): sensor N uses sensor N-1's output
                    // For parallel mergers: all sensors use predicted_tracks
                    let sensor_prior_weights: Vec<Vec<f64>> = su
                        .input_tracks
                        .iter()
                        .map(|t| t.components.iter().map(|c| c.weight).collect())
                        .collect();

                    let sensor_prior_means: Vec<Vec<Vec<f64>>> = su
                        .input_tracks
                        .iter()
                        .map(|t| {
                            t.components
                                .iter()
                                .map(|c| c.mean.iter().copied().collect())
                                .collect()
                        })
                        .collect();

                    let sensor_prior_covariances: Vec<Vec<Vec<Vec<f64>>>> = su
                        .input_tracks
                        .iter()
                        .map(|t| {
                            t.components
                                .iter()
                                .map(|c| {
                                    let nrows = c.covariance.nrows();
                                    (0..nrows)
                                        .map(|i| c.covariance.row(i).iter().copied().collect())
                                        .collect()
                                })
                                .collect()
                        })
                        .collect();

                    let sensor_matrices = su
                        .association_matrices
                        .map(|m| {
                            Py::new(
                                py,
                                PyAssociationMatrices::from_matrices(
                                    &m,
                                    &sensor_prior_weights,
                                    &sensor_prior_means,
                                    &sensor_prior_covariances,
                                    false, // LMB uses linear space L matrix
                                ),
                            )
                        })
                        .transpose()?;

                    let sensor_result = su
                        .association_result
                        .map(|r| Py::new(py, PyAssociationResult::from_result(&r)))
                        .transpose()?;

                    let sensor_input_tracks = tracks_to_py(py, &su.input_tracks)?;
                    let sensor_updated_tracks = tracks_to_py(py, &su.updated_tracks)?;

                    Py::new(
                        py,
                        PySensorUpdateOutput {
                            sensor_index: su.sensor_index,
                            input_tracks: sensor_input_tracks,
                            association_matrices: sensor_matrices,
                            association_result: sensor_result,
                            updated_tracks: sensor_updated_tracks,
                        },
                    )
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;

    Py::new(
        py,
        PyStepOutput {
            predicted_tracks: predicted,
            association_matrices: matrices,
            association_result: result,
            updated_tracks: updated,
            cardinality,
            // Multisensor-specific fields
            sensor_updates,
            // LMBM-specific fields
            predicted_hypotheses,
            pre_normalization_hypotheses,
            normalized_hypotheses,
            objects_likely_to_exist: output.objects_likely_to_exist,
        },
    )
}

// =============================================================================
// Macros for Reducing Filter Boilerplate
// =============================================================================

/// Implement reset method for all filters
macro_rules! impl_filter_reset {
    ($filter:ty) => {
        #[pymethods]
        impl $filter {
            fn reset(&mut self) {
                self.inner.reset();
            }
        }
    };
}

/// Implement step and step_detailed for single-sensor filters
///
/// # Parameters
/// * `$filter` - The filter type to implement for
/// * `$include_association` - Whether to include association matrices in output
/// * `$use_log_space_l` - Whether L matrix should be in log space (LMBM) or linear space (LMB)
macro_rules! impl_singlesensor_step {
    ($filter:ty, $include_association:expr, $use_log_space_l:expr) => {
        #[pymethods]
        impl $filter {
            fn step(
                &mut self,
                measurements: Vec<PyReadonlyArray1<'_, f64>>,
                timestep: usize,
            ) -> PyResult<PyStateEstimate> {
                let meas = numpy_list_to_measurements(measurements);
                let estimate = self
                    .inner
                    .step(&mut self.rng, &meas, timestep)
                    .map_err(filter_error_to_py)?;
                Ok(PyStateEstimate::from_inner(estimate))
            }

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
                    .map_err(filter_error_to_py)?;
                step_output_to_py(py, output, $include_association, $use_log_space_l)
            }
        }
    };
}

/// Implement step and step_detailed for multi-sensor filters
///
/// # Parameters
/// * `$filter` - The filter type to implement for
/// * `$include_association` - Whether to include association matrices in output
/// * `$use_log_space_l` - Whether L matrix should be in log space (LMBM) or linear space (LMB)
macro_rules! impl_multisensor_step {
    ($filter:ty, $include_association:expr, $use_log_space_l:expr) => {
        #[pymethods]
        impl $filter {
            fn step(
                &mut self,
                measurements: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
                timestep: usize,
            ) -> PyResult<PyStateEstimate> {
                let meas = numpy_nested_to_measurements(measurements);
                let estimate = self
                    .inner
                    .step(&mut self.rng, &meas, timestep)
                    .map_err(filter_error_to_py)?;
                Ok(PyStateEstimate::from_inner(estimate))
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
                    .map_err(filter_error_to_py)?;
                step_output_to_py(py, output, $include_association, $use_log_space_l)
            }
        }
    };
}

/// Implement set_tracks and get_tracks for LMB-style filters (not LMBM)
macro_rules! impl_lmb_track_access {
    ($filter:ty) => {
        #[pymethods]
        impl $filter {
            fn set_tracks(&mut self, tracks: Vec<PyRef<PyTrackData>>) {
                let rust_tracks: Vec<_> = tracks.iter().map(|t| t.to_track()).collect();
                self.inner.set_tracks(rust_tracks);
            }

            fn get_tracks(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTrackData>>> {
                tracks_to_py(py, &self.inner.get_tracks())
            }
        }
    };
}

/// Implement get_tracks for LMBM-style filters (read-only, from best hypothesis)
macro_rules! impl_lmbm_track_access {
    ($filter:ty) => {
        #[pymethods]
        impl $filter {
            fn get_tracks(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTrackData>>> {
                tracks_to_py(py, &self.inner.get_tracks())
            }
        }
    };
}

/// Implement set_hypotheses for single-sensor LMBM filter
macro_rules! impl_lmbm_hypothesis_access {
    ($filter:ty) => {
        #[pymethods]
        impl $filter {
            fn set_hypotheses(
                &mut self,
                hypotheses: Vec<PyRef<super::intermediate::PyLmbmHypothesis>>,
            ) {
                let rust_hypotheses: Vec<_> =
                    hypotheses.iter().map(|h| h.to_hypothesis()).collect();
                self.inner.set_hypotheses(rust_hypotheses);
            }
        }
    };
}

/// Implement get_config methods for all filters
macro_rules! impl_get_config {
    ($filter:ty) => {
        #[pymethods]
        impl $filter {
            /// Get a JSON string with all filter configuration parameters.
            ///
            /// Useful for debugging and comparing configurations across
            /// implementations (Rust, Python, Octave).
            fn get_config_json(&self) -> String {
                self.inner.get_config().to_json()
            }

            /// Get a pretty-printed JSON string with all filter configuration parameters.
            fn get_config_json_pretty(&self) -> String {
                self.inner.get_config().to_json_pretty()
            }
        }
    };
}

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
    #[pyo3(signature = (existence=0.5, gm_weight=1e-4, max_components=100, min_trajectory_length=3, gm_merge=f64::INFINITY))]
    fn new(
        existence: f64,
        gm_weight: f64,
        max_components: usize,
        min_trajectory_length: usize,
        gm_merge: f64,
    ) -> Self {
        Self {
            inner: FilterThresholds::with_merge_threshold(
                existence,
                gm_weight,
                max_components,
                min_trajectory_length,
                gm_merge,
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

    #[getter]
    fn gm_merge_threshold(&self) -> f64 {
        self.inner.gm_merge_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "FilterThresholds(existence={}, gm_weight={}, max_components={}, gm_merge={})",
            self.inner.existence_threshold,
            self.inner.gm_weight_threshold,
            self.inner.max_gm_components,
            self.inner.gm_merge_threshold
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
    /// Track existence threshold for pruning (default: 0.001)
    pub(crate) existence_threshold: f64,
}

#[pymethods]
impl PyFilterLmbmConfig {
    #[new]
    #[pyo3(signature = (max_hypotheses=1000, hypothesis_weight_threshold=1e-6, use_eap=false, existence_threshold=None))]
    fn new(
        max_hypotheses: usize,
        hypothesis_weight_threshold: f64,
        use_eap: bool,
        existence_threshold: Option<f64>,
    ) -> Self {
        use crate::lmb::DEFAULT_EXISTENCE_THRESHOLD;
        Self {
            inner: LmbmConfig {
                max_hypotheses,
                hypothesis_weight_threshold,
                use_eap,
            },
            existence_threshold: existence_threshold.unwrap_or(DEFAULT_EXISTENCE_THRESHOLD),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FilterLmbmConfig(max_hypotheses={}, use_eap={}, existence_threshold={})",
            self.inner.max_hypotheses, self.inner.use_eap, self.existence_threshold
        )
    }
}

// =============================================================================
// Helper: Create RNG from optional seed
// =============================================================================

fn create_rng(seed: Option<u64>) -> SimpleRng {
    // Use SimpleRng for MATLAB equivalence - it matches MATLAB's SimpleRng class exactly
    SimpleRng::new(seed.unwrap_or(42))
}

// =============================================================================
// FilterLmb - Single-sensor LMB filter
// =============================================================================

#[pyclass(name = "FilterLmb")]
pub struct PyFilterLmb {
    inner: LmbFilter<DynamicAssociator>,
    rng: SimpleRng,
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

        // Create the appropriate dynamic associator based on the config
        let associator = DynamicAssociator::from_config(&assoc);

        let inner = LmbFilter::with_associator_type(
            motion.inner.clone(),
            sensor.inner.clone(),
            birth.inner.clone(),
            assoc,
            associator,
        )
        .with_gm_pruning(thresh.gm_weight_threshold, thresh.max_gm_components)
        .with_gm_merge_threshold(thresh.gm_merge_threshold);

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("FilterLmb(num_tracks={})", self.num_tracks())
    }
}

impl_filter_reset!(PyFilterLmb);
impl_singlesensor_step!(PyFilterLmb, true, false); // LMB uses linear space L
impl_lmb_track_access!(PyFilterLmb);
impl_get_config!(PyFilterLmb);

// =============================================================================
// FilterLmbm - Single-sensor LMBM filter
// =============================================================================

#[pyclass(name = "FilterLmbm")]
pub struct PyFilterLmbm {
    inner: LmbmFilter<DynamicAssociator>,
    rng: SimpleRng,
}

#[pymethods]
impl PyFilterLmbm {
    #[new]
    #[pyo3(signature = (motion, sensor, birth, association=None, thresholds=None, lmbm_config=None, seed=None))]
    fn new(
        motion: &PyMotionModel,
        sensor: &PySensorModel,
        birth: &PyBirthModel,
        association: Option<&PyAssociatorConfig>,
        thresholds: Option<&PyFilterThresholds>,
        lmbm_config: Option<&PyFilterLmbmConfig>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        use crate::lmb::builder::FilterBuilder;
        use crate::lmb::DEFAULT_EXISTENCE_THRESHOLD;

        let assoc = association
            .map(|a| a.inner.clone())
            .unwrap_or_else(|| AssociationConfig::gibbs(1000));
        let lmbm = lmbm_config.map(|c| c.inner.clone()).unwrap_or_default();
        let existence_threshold = lmbm_config
            .map(|c| c.existence_threshold)
            .unwrap_or(DEFAULT_EXISTENCE_THRESHOLD);

        // Create the appropriate dynamic associator based on the config
        let associator = DynamicAssociator::from_config(&assoc);

        let mut inner = LmbmFilter::with_associator_type(
            motion.inner.clone(),
            sensor.inner.clone(),
            birth.inner.clone(),
            assoc,
            lmbm,
            associator,
        );

        // Set existence threshold from config
        *inner.existence_threshold_mut() = existence_threshold;

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
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
}

impl_filter_reset!(PyFilterLmbm);
impl_singlesensor_step!(PyFilterLmbm, true, true); // LMBM uses log space L
impl_lmbm_track_access!(PyFilterLmbm);
impl_lmbm_hypothesis_access!(PyFilterLmbm);
impl_get_config!(PyFilterLmbm);

// =============================================================================
// FilterAaLmb - Arithmetic Average multi-sensor LMB filter
// =============================================================================

#[pyclass(name = "FilterAaLmb")]
pub struct PyFilterAaLmb {
    inner: MultisensorLmbFilter<LbpAssociator, ArithmeticAverageMerger>,
    rng: SimpleRng,
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
        .with_gm_pruning(thresh.gm_weight_threshold, thresh.max_gm_components)
        .with_gm_merge_threshold(thresh.gm_merge_threshold);

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("FilterAaLmb(num_tracks={})", self.num_tracks())
    }
}

impl_filter_reset!(PyFilterAaLmb);
impl_multisensor_step!(PyFilterAaLmb, false, false); // LMB uses linear space L
impl_lmb_track_access!(PyFilterAaLmb);
impl_get_config!(PyFilterAaLmb);

// =============================================================================
// FilterGaLmb - Geometric Average multi-sensor LMB filter
// =============================================================================

#[pyclass(name = "FilterGaLmb")]
pub struct PyFilterGaLmb {
    inner: MultisensorLmbFilter<LbpAssociator, GeometricAverageMerger>,
    rng: SimpleRng,
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
        .with_gm_pruning(thresh.gm_weight_threshold, thresh.max_gm_components)
        .with_gm_merge_threshold(thresh.gm_merge_threshold);

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("FilterGaLmb(num_tracks={})", self.num_tracks())
    }
}

impl_filter_reset!(PyFilterGaLmb);
impl_multisensor_step!(PyFilterGaLmb, false, false); // LMB uses linear space L
impl_lmb_track_access!(PyFilterGaLmb);
impl_get_config!(PyFilterGaLmb);

// =============================================================================
// FilterPuLmb - Parallel Update multi-sensor LMB filter
// =============================================================================

#[pyclass(name = "FilterPuLmb")]
pub struct PyFilterPuLmb {
    inner: MultisensorLmbFilter<LbpAssociator, ParallelUpdateMerger>,
    rng: SimpleRng,
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
        .with_gm_pruning(thresh.gm_weight_threshold, thresh.max_gm_components)
        .with_gm_merge_threshold(thresh.gm_merge_threshold);

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("FilterPuLmb(num_tracks={})", self.num_tracks())
    }
}

impl_filter_reset!(PyFilterPuLmb);
impl_multisensor_step!(PyFilterPuLmb, false, false); // LMB uses linear space L
impl_lmb_track_access!(PyFilterPuLmb);
impl_get_config!(PyFilterPuLmb);

// =============================================================================
// FilterIcLmb - Iterated Corrector multi-sensor LMB filter
// =============================================================================

#[pyclass(name = "FilterIcLmb")]
pub struct PyFilterIcLmb {
    inner: MultisensorLmbFilter<LbpAssociator, IteratedCorrectorMerger>,
    rng: SimpleRng,
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
        .with_gm_pruning(thresh.gm_weight_threshold, thresh.max_gm_components)
        .with_gm_merge_threshold(thresh.gm_merge_threshold);

        Ok(Self {
            inner,
            rng: create_rng(seed),
        })
    }

    #[getter]
    fn num_tracks(&self) -> usize {
        self.inner.state().len()
    }

    fn __repr__(&self) -> String {
        format!("FilterIcLmb(num_tracks={})", self.num_tracks())
    }
}

impl_filter_reset!(PyFilterIcLmb);
impl_multisensor_step!(PyFilterIcLmb, false, false); // LMB uses linear space L
impl_lmb_track_access!(PyFilterIcLmb);
impl_get_config!(PyFilterIcLmb);

// =============================================================================
// FilterMultisensorLmbm - Multi-sensor LMBM filter
// =============================================================================

#[pyclass(name = "FilterMultisensorLmbm")]
pub struct PyFilterMultisensorLmbm {
    inner: MultisensorLmbmFilter,
    rng: SimpleRng,
}

#[pymethods]
impl PyFilterMultisensorLmbm {
    #[new]
    #[pyo3(signature = (motion, sensors, birth, association=None, thresholds=None, lmbm_config=None, seed=None))]
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
}

impl_filter_reset!(PyFilterMultisensorLmbm);
impl_multisensor_step!(PyFilterMultisensorLmbm, false, true); // LMBM uses log space L
impl_lmbm_track_access!(PyFilterMultisensorLmbm);
impl_lmbm_hypothesis_access!(PyFilterMultisensorLmbm);
impl_get_config!(PyFilterMultisensorLmbm);
