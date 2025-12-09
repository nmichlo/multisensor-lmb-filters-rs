//! Internal types for step-by-step fixture validation.
//!
//! These types expose intermediate filter data for testing purposes.
//! They are NOT part of the public API and have underscore-prefixed names.

use numpy::ndarray::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, ToPyArray};
use pyo3::prelude::*;

use nalgebra::{DMatrix, DVector};
use smallvec::SmallVec;

use crate::association::AssociationMatrices;
use crate::lmb::traits::AssociationResult;
use crate::lmb::types::{GaussianComponent, Track, TrackLabel};

// =============================================================================
// _TrackData - Full track data matching fixture format
// =============================================================================

#[pyclass(name = "_TrackData")]
pub struct PyTrackData {
    /// Track label as (birth_time, birth_location)
    #[pyo3(get)]
    pub label: (usize, usize),

    /// Existence probability (r in fixtures)
    #[pyo3(get)]
    pub existence: f64,

    /// Gaussian component means: [n_components x state_dim]
    pub means: Vec<Vec<f64>>,

    /// Gaussian component covariances: [n_components x state_dim x state_dim]
    pub covariances: Vec<Vec<Vec<f64>>>,

    /// Gaussian component weights: [n_components]
    pub weights: Vec<f64>,
}

#[pymethods]
impl PyTrackData {
    /// Create a new track from Python data.
    ///
    /// Args:
    ///     label: Tuple of (birth_time, birth_location)
    ///     existence: Existence probability (r in fixtures)
    ///     means: List of mean vectors [n_components x state_dim]
    ///     covariances: List of covariance matrices [n_components x state_dim x state_dim]
    ///     weights: List of GM component weights [n_components]
    #[new]
    #[pyo3(signature = (label, existence, means, covariances, weights))]
    fn new(
        label: (usize, usize),
        existence: f64,
        means: Vec<Vec<f64>>,
        covariances: Vec<Vec<Vec<f64>>>,
        weights: Vec<f64>,
    ) -> Self {
        Self {
            label,
            existence,
            means,
            covariances,
            weights,
        }
    }

    #[getter]
    fn mu<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n_components = self.means.len();
        let state_dim = if n_components > 0 {
            self.means[0].len()
        } else {
            0
        };
        let flat: Vec<f64> = self.means.iter().flatten().copied().collect();
        let arr = Array2::from_shape_vec((n_components, state_dim), flat)
            .unwrap_or_else(|_| Array2::zeros((0, 0)));
        arr.to_pyarray(py)
    }

    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let n_components = self.covariances.len();
        let state_dim = if n_components > 0 && !self.covariances[0].is_empty() {
            self.covariances[0].len()
        } else {
            0
        };
        let flat: Vec<f64> = self
            .covariances
            .iter()
            .flat_map(|cov| cov.iter().flatten())
            .copied()
            .collect();
        let arr = Array3::from_shape_vec((n_components, state_dim, state_dim), flat)
            .unwrap_or_else(|_| Array3::zeros((0, 0, 0)));
        arr.to_pyarray(py)
    }

    #[getter]
    fn w<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.weights.to_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "_TrackData(label={:?}, r={:.4}, n_components={})",
            self.label,
            self.existence,
            self.weights.len()
        )
    }
}

impl PyTrackData {
    pub fn from_track(track: &Track) -> Self {
        let means: Vec<Vec<f64>> = track
            .components
            .iter()
            .map(|c| c.mean.iter().copied().collect())
            .collect();

        let covariances: Vec<Vec<Vec<f64>>> = track
            .components
            .iter()
            .map(|c| {
                let nrows = c.covariance.nrows();
                (0..nrows)
                    .map(|i| c.covariance.row(i).iter().copied().collect())
                    .collect()
            })
            .collect();

        let weights: Vec<f64> = track.components.iter().map(|c| c.weight).collect();

        Self {
            label: (track.label.birth_time, track.label.birth_location),
            existence: track.existence,
            means,
            covariances,
            weights,
        }
    }

    /// Convert this PyTrackData back to a Rust Track.
    pub fn to_track(&self) -> Track {
        let label = TrackLabel {
            birth_time: self.label.0,
            birth_location: self.label.1,
        };

        let components: SmallVec<[GaussianComponent; 4]> = self
            .means
            .iter()
            .zip(self.covariances.iter())
            .zip(self.weights.iter())
            .map(|((mean, cov), &weight)| {
                let mean_vec = DVector::from_vec(mean.clone());
                let n = cov.len();
                let cov_flat: Vec<f64> = cov.iter().flatten().copied().collect();
                let covariance = DMatrix::from_row_slice(n, n, &cov_flat);
                GaussianComponent {
                    weight,
                    mean: mean_vec,
                    covariance,
                }
            })
            .collect();

        Track {
            label,
            existence: self.existence,
            components,
            trajectory: None,
        }
    }
}

// =============================================================================
// _PosteriorParameters - Per-track posterior parameters matching MATLAB format
// =============================================================================

/// Posterior parameters for a single track, matching MATLAB's posteriorParameters[i].
///
/// # MATLAB Fixture Format
///
/// In fixtures at `tests/data/step_by_step/*.json`, each track has:
/// - `posteriorParameters[i].w` → shape `(num_meas + 1, num_comp)`
///   Row 0 is miss hypothesis (equals prior weights), rows 1+ are measurements.
///   Each row sums to 1.0 (likelihood-normalized component weights).
///
/// - `posteriorParameters[i].mu` → shape `(num_meas * num_comp, state_dim)`
///   Flattened posterior means. Index `(j * num_comp + k)` gives measurement j, component k.
///
/// - `posteriorParameters[i].Sigma` → shape `(num_meas * num_comp, state_dim, state_dim)`
///   Flattened posterior covariances.
#[pyclass(name = "_PosteriorParameters")]
pub struct PyPosteriorParameters {
    /// Component weights: [num_meas + 1 x num_comp]
    /// Row 0 = prior weights (miss hypothesis)
    /// Rows 1+ = likelihood-normalized weights per measurement
    w: Vec<Vec<f64>>,

    /// Posterior means: [num_meas * num_comp x state_dim]
    /// Flattened: index = meas_idx * num_comp + comp_idx
    mu: Vec<Vec<f64>>,

    /// Posterior covariances: [num_meas * num_comp x state_dim x state_dim]
    sigma: Vec<Vec<Vec<f64>>>,
}

#[pymethods]
impl PyPosteriorParameters {
    /// Component weights matrix matching MATLAB's posteriorParameters[i].w
    /// Shape: [num_meas + 1, num_comp] where row 0 is miss (prior weights)
    #[getter]
    fn w<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_numpy(py, &self.w)
    }

    /// Posterior means matching MATLAB's posteriorParameters[i].mu
    /// Shape: [num_meas * num_comp, state_dim]
    #[getter]
    fn mu<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_numpy(py, &self.mu)
    }

    /// Posterior covariances matching MATLAB's posteriorParameters[i].Sigma
    /// Shape: [num_meas * num_comp, state_dim, state_dim]
    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let n_entries = self.sigma.len();
        let state_dim = if n_entries > 0 && !self.sigma[0].is_empty() {
            self.sigma[0].len()
        } else {
            0
        };
        let flat: Vec<f64> = self
            .sigma
            .iter()
            .flat_map(|cov| cov.iter().flatten())
            .copied()
            .collect();
        let arr = Array3::from_shape_vec((n_entries, state_dim, state_dim), flat)
            .unwrap_or_else(|_| Array3::zeros((0, 0, 0)));
        arr.to_pyarray(py)
    }

    fn __repr__(&self) -> String {
        let n_meas_plus_miss = self.w.len();
        let n_comp = if n_meas_plus_miss > 0 {
            self.w[0].len()
        } else {
            0
        };
        format!(
            "_PosteriorParameters(n_hypotheses={}, n_components={})",
            n_meas_plus_miss, n_comp
        )
    }
}

// =============================================================================
// _AssociationMatrices - Matrices from association builder
// =============================================================================

#[pyclass(name = "_AssociationMatrices")]
#[allow(dead_code)]
pub struct PyAssociationMatrices {
    /// Cost matrix C: [n_tracks x n_measurements]
    cost: Vec<Vec<f64>>,

    /// Log-likelihood ratios L: [n_tracks x n_measurements]
    likelihood: Vec<Vec<f64>>,

    /// Miss probability matrix R: [n_tracks x n_measurements]
    /// R[i,j] = (1 - P_d) * r[i] / eta[i] for all j (constant per row)
    /// This matches MATLAB's R matrix format.
    miss_prob: Vec<Vec<f64>>,

    /// Sampling probabilities P: [n_tracks x n_measurements]
    /// Computed as psi / (1 + psi) element-wise to match MATLAB format.
    sampling_prob: Vec<Vec<f64>>,

    /// Eta normalization factors: [n_tracks]
    eta: Vec<f64>,

    /// Posterior parameters for each track
    /// This matches MATLAB's posteriorParameters array structure.
    posterior_parameters: Vec<PyPosteriorParameters>,
}

#[pymethods]
impl PyAssociationMatrices {
    #[getter]
    fn cost<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_numpy(py, &self.cost)
    }

    #[getter]
    fn likelihood<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_numpy(py, &self.likelihood)
    }

    /// Miss probability matrix R matching MATLAB format.
    /// R[i,j] = phi[i] / eta[i] = (1 - P_d) * r[i] / eta[i]
    #[getter]
    fn miss_prob<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_numpy(py, &self.miss_prob)
    }

    #[getter]
    fn sampling_prob<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_numpy(py, &self.sampling_prob)
    }

    #[getter]
    fn eta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.eta.to_pyarray(py)
    }

    /// Posterior parameters for each track, matching MATLAB's posteriorParameters array.
    #[getter]
    fn posterior_parameters(&self) -> Vec<PyPosteriorParameters> {
        // Clone the posterior parameters for Python access
        self.posterior_parameters
            .iter()
            .map(|pp| PyPosteriorParameters {
                w: pp.w.clone(),
                mu: pp.mu.clone(),
                sigma: pp.sigma.clone(),
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        let n_tracks = self.cost.len();
        let n_meas = if n_tracks > 0 { self.cost[0].len() } else { 0 };
        format!("_AssociationMatrices(n_tracks={n_tracks}, n_measurements={n_meas})")
    }
}

impl PyAssociationMatrices {
    /// Create from AssociationMatrices with prior track data for posteriorParameters.
    ///
    /// The priors are needed because MATLAB's posteriorParameters includes the
    /// miss hypothesis (row 0), which uses the prior means/covariances unchanged.
    pub fn from_matrices(
        matrices: &AssociationMatrices,
        prior_weights: &[Vec<f64>],
        prior_means: &[Vec<Vec<f64>>],
        prior_covariances: &[Vec<Vec<Vec<f64>>>],
    ) -> Self {
        // Convert cost matrix
        let n_tracks = matrices.cost.nrows();
        let n_meas = matrices.psi.ncols();

        let cost: Vec<Vec<f64>> = (0..n_tracks)
            .map(|i| matrices.cost.row(i).iter().copied().collect())
            .collect();

        // Convert log-likelihood: L = [eta | exp(-cost)]
        // MATLAB format: L[i,0] = eta[i] = 1 - r[i] * P_d, L[i,j+1] = likelihood ratio for measurement j
        let likelihood: Vec<Vec<f64>> = (0..n_tracks)
            .map(|i| {
                // First column: eta[i] = 1 - r[i] * P_d (same as MATLAB's L first column)
                let mut row = vec![matrices.eta[i]];
                // Remaining columns: exp(-cost[i,j]) for each measurement
                for j in 0..n_meas {
                    row.push((-matrices.cost[(i, j)]).exp());
                }
                row
            })
            .collect();

        // Compute R matrix matching MATLAB format:
        // R[i,0] = phi[i] / eta[i] (miss probability for LBP)
        // R[i,j>0] = 1.0 for all measurement columns
        let miss_prob: Vec<Vec<f64>> = (0..n_tracks)
            .map(|i| {
                let miss_val = if matrices.eta[i].abs() > 1e-15 {
                    matrices.phi[i] / matrices.eta[i]
                } else {
                    0.0
                };
                let mut row = vec![miss_val];
                row.extend(std::iter::repeat(1.0).take(n_meas));
                row
            })
            .collect();

        // Compute P = psi / (1 + psi) element-wise to match MATLAB's P matrix
        let sampling_prob: Vec<Vec<f64>> = (0..n_tracks)
            .map(|i| {
                (0..n_meas)
                    .map(|j| {
                        let psi_ij = matrices.psi[(i, j)];
                        psi_ij / (1.0 + psi_ij)
                    })
                    .collect()
            })
            .collect();

        // Convert eta
        let eta: Vec<f64> = matrices.eta.iter().copied().collect();

        // Build posterior parameters for each track
        // Matches MATLAB's posteriorParameters[i] structure
        let posterior_parameters: Vec<PyPosteriorParameters> = (0..n_tracks)
            .map(|track_idx| {
                let n_comp = matrices.posteriors.num_components(track_idx);

                // Build w matrix: [n_meas + 1, n_comp]
                // Row 0 = prior weights (miss hypothesis)
                // Rows 1+ = likelihood-normalized weights per measurement
                let mut w: Vec<Vec<f64>> = Vec::with_capacity(n_meas + 1);

                // Row 0: prior weights - ensure exactly n_comp elements
                let prior_w: Vec<f64> = if track_idx < prior_weights.len()
                    && prior_weights[track_idx].len() == n_comp
                {
                    prior_weights[track_idx].clone()
                } else {
                    // Fallback: uniform weights if prior_weights missing or wrong size
                    vec![1.0 / n_comp as f64; n_comp]
                };
                debug_assert_eq!(
                    prior_w.len(),
                    n_comp,
                    "prior_w length mismatch: {} vs {} for track {}",
                    prior_w.len(),
                    n_comp,
                    track_idx
                );
                w.push(prior_w);

                // Rows 1+: component_weights for each measurement
                for meas_idx in 0..n_meas {
                    let meas_weights: Vec<f64> = (0..n_comp)
                        .map(|comp_idx| {
                            matrices
                                .posteriors
                                .get_component_weight(track_idx, meas_idx, comp_idx)
                                .unwrap_or(1.0 / n_comp as f64)
                        })
                        .collect();
                    w.push(meas_weights);
                }

                // Build mu: [(n_meas + 1) * n_comp, state_dim]
                // MATLAB uses component-major ordering:
                // For each component, iterate hypotheses (miss first, then measurements)
                // Row index = comp_idx * (n_meas + 1) + hypothesis_idx
                let mut mu: Vec<Vec<f64>> = Vec::with_capacity((n_meas + 1) * n_comp);

                for comp_idx in 0..n_comp {
                    // Hypothesis 0: miss (use prior mean for this component)
                    if track_idx < prior_means.len() && comp_idx < prior_means[track_idx].len() {
                        mu.push(prior_means[track_idx][comp_idx].clone());
                    } else {
                        mu.push(Vec::new());
                    }

                    // Hypotheses 1+: measurements
                    for meas_idx in 0..n_meas {
                        let mean = matrices
                            .posteriors
                            .get_mean_for_component(track_idx, meas_idx, comp_idx)
                            .map(|m| m.iter().copied().collect())
                            .unwrap_or_default();
                        mu.push(mean);
                    }
                }

                // Build Sigma: [(n_meas + 1) * n_comp, state_dim, state_dim]
                // Same component-major ordering as mu
                let mut sigma: Vec<Vec<Vec<f64>>> = Vec::with_capacity((n_meas + 1) * n_comp);

                for comp_idx in 0..n_comp {
                    // Hypothesis 0: miss (use prior covariance for this component)
                    if track_idx < prior_covariances.len()
                        && comp_idx < prior_covariances[track_idx].len()
                    {
                        sigma.push(prior_covariances[track_idx][comp_idx].clone());
                    } else {
                        sigma.push(Vec::new());
                    }

                    // Hypotheses 1+: measurements
                    for meas_idx in 0..n_meas {
                        let cov = matrices
                            .posteriors
                            .get_covariance_for_component(track_idx, meas_idx, comp_idx)
                            .map(|cov| {
                                let nrows = cov.nrows();
                                (0..nrows)
                                    .map(|i| cov.row(i).iter().copied().collect())
                                    .collect()
                            })
                            .unwrap_or_default();
                        sigma.push(cov);
                    }
                }

                PyPosteriorParameters { w, mu, sigma }
            })
            .collect();

        Self {
            cost,
            likelihood,
            miss_prob,
            sampling_prob,
            eta,
            posterior_parameters,
        }
    }
}

// =============================================================================
// _AssociationResult - Result from data association algorithm
// =============================================================================

#[pyclass(name = "_AssociationResult")]
pub struct PyAssociationResult {
    /// Marginal weights W: [n_tracks x n_measurements]
    marginal_weights: Vec<Vec<f64>>,

    /// Miss weights (W column 0): [n_tracks]
    miss_weights: Vec<f64>,

    /// Posterior existence probabilities (r from LBP): [n_tracks]
    posterior_existence: Vec<f64>,

    /// Sampled assignments V: [n_samples x n_tracks] (Gibbs/Murty only)
    assignments: Option<Vec<Vec<i32>>>,
}

#[pymethods]
impl PyAssociationResult {
    #[getter]
    fn marginal_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_numpy(py, &self.marginal_weights)
    }

    #[getter]
    fn miss_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.miss_weights.to_pyarray(py)
    }

    #[getter]
    fn posterior_existence<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.posterior_existence.to_pyarray(py)
    }

    #[getter]
    fn assignments<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<i32>>> {
        self.assignments.as_ref().map(|v| {
            let n_samples = v.len();
            let n_tracks = if n_samples > 0 { v[0].len() } else { 0 };
            let flat: Vec<i32> = v.iter().flatten().copied().collect();
            let arr = Array2::from_shape_vec((n_samples, n_tracks), flat)
                .unwrap_or_else(|_| Array2::zeros((0, 0)));
            arr.to_pyarray(py)
        })
    }

    fn __repr__(&self) -> String {
        let n_tracks = self.miss_weights.len();
        let n_meas = if !self.marginal_weights.is_empty() {
            self.marginal_weights[0].len()
        } else {
            0
        };
        let n_samples = self.assignments.as_ref().map_or(0, |v| v.len());
        format!(
            "_AssociationResult(n_tracks={n_tracks}, n_measurements={n_meas}, n_samples={n_samples})"
        )
    }
}

impl PyAssociationResult {
    pub fn from_result(result: &AssociationResult) -> Self {
        // Convert marginal weights
        let n_tracks = result.marginal_weights.nrows();
        let marginal_weights: Vec<Vec<f64>> = (0..n_tracks)
            .map(|i| result.marginal_weights.row(i).iter().copied().collect())
            .collect();

        // Convert miss weights
        let miss_weights: Vec<f64> = result.miss_weights.iter().copied().collect();

        // Convert posterior existence
        let posterior_existence: Vec<f64> = result.posterior_existence.iter().copied().collect();

        // Convert sampled associations
        let assignments: Option<Vec<Vec<i32>>> = result
            .sampled_associations
            .as_ref()
            .map(|samples| samples.iter().map(|sample| sample.to_vec()).collect());

        Self {
            marginal_weights,
            miss_weights,
            posterior_existence,
            assignments,
        }
    }
}

// =============================================================================
// _CardinalityEstimate - Cardinality extraction result
// =============================================================================

#[pyclass(name = "_CardinalityEstimate")]
pub struct PyCardinalityEstimate {
    #[pyo3(get)]
    pub n_estimated: usize,

    /// Indices of estimated tracks
    pub map_indices: Vec<usize>,
}

#[pymethods]
impl PyCardinalityEstimate {
    #[getter]
    fn map_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        self.map_indices.to_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "_CardinalityEstimate(n_estimated={}, map_indices={:?})",
            self.n_estimated, self.map_indices
        )
    }
}

// =============================================================================
// _StepOutput - Full step output for fixture comparison
// =============================================================================

#[pyclass(name = "_StepOutput")]
pub struct PyStepOutput {
    /// Predicted tracks (after step 1)
    #[pyo3(get)]
    pub predicted_tracks: Vec<Py<PyTrackData>>,

    /// Association matrices (after step 2) - None for multi-sensor/LMBM filters
    #[pyo3(get)]
    pub association_matrices: Option<Py<PyAssociationMatrices>>,

    /// Association result (after step 3) - None for multi-sensor/LMBM filters
    #[pyo3(get)]
    pub association_result: Option<Py<PyAssociationResult>>,

    /// Updated tracks (after step 4)
    #[pyo3(get)]
    pub updated_tracks: Vec<Py<PyTrackData>>,

    /// Cardinality estimate (after step 5)
    #[pyo3(get)]
    pub cardinality: Py<PyCardinalityEstimate>,
}

#[pymethods]
impl PyStepOutput {
    fn __repr__(&self) -> String {
        format!(
            "_StepOutput(n_predicted={}, n_updated={})",
            self.predicted_tracks.len(),
            self.updated_tracks.len()
        )
    }
}

// =============================================================================
// Helper functions
// =============================================================================

fn matrix_to_numpy<'py>(py: Python<'py>, data: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
    let nrows = data.len();
    let ncols = if nrows > 0 { data[0].len() } else { 0 };
    let flat: Vec<f64> = data.iter().flatten().copied().collect();
    let arr =
        Array2::from_shape_vec((nrows, ncols), flat).unwrap_or_else(|_| Array2::zeros((0, 0)));
    arr.to_pyarray(py)
}
