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
// _AssociationMatrices - Matrices from association builder
// =============================================================================

#[pyclass(name = "_AssociationMatrices")]
#[allow(dead_code)]
pub struct PyAssociationMatrices {
    /// Cost matrix C: [n_tracks x n_measurements]
    cost: Vec<Vec<f64>>,

    /// Log-likelihood ratios L: [n_tracks x n_measurements]
    likelihood: Vec<Vec<f64>>,

    /// Sampling probabilities P: [n_tracks x n_measurements]
    /// Computed as psi / (1 + psi) element-wise to match MATLAB format.
    sampling_prob: Vec<Vec<f64>>,

    /// Eta normalization factors: [n_tracks]
    eta: Vec<f64>,

    /// Posterior means for each (track, measurement) pair
    posterior_means: Vec<Vec<Vec<f64>>>,

    /// Posterior covariances for each (track, measurement) pair
    posterior_covariances: Vec<Vec<Vec<Vec<f64>>>>,
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

    #[getter]
    fn sampling_prob<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_numpy(py, &self.sampling_prob)
    }

    #[getter]
    fn eta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.eta.to_pyarray(py)
    }

    fn __repr__(&self) -> String {
        let n_tracks = self.cost.len();
        let n_meas = if n_tracks > 0 { self.cost[0].len() } else { 0 };
        format!("_AssociationMatrices(n_tracks={n_tracks}, n_measurements={n_meas})")
    }
}

impl PyAssociationMatrices {
    pub fn from_matrices(matrices: &AssociationMatrices) -> Self {
        // Convert cost matrix
        let n_tracks = matrices.cost.nrows();
        let cost: Vec<Vec<f64>> = (0..n_tracks)
            .map(|i| matrices.cost.row(i).iter().copied().collect())
            .collect();

        // Convert log-likelihood (stored in cost, so compute L = exp(-cost))
        // Actually, we need the raw likelihood ratios. Let's use psi/phi relationship.
        // For simplicity, compute L from cost: L = exp(-cost)
        let likelihood: Vec<Vec<f64>> = cost
            .iter()
            .map(|row| row.iter().map(|&c| (-c).exp()).collect())
            .collect();

        // Compute P = psi / (1 + psi) element-wise to match MATLAB's P matrix
        // This is NOT row-normalized - each element is normalized independently
        let n_meas = matrices.psi.ncols();
        let sampling_prob: Vec<Vec<f64>> = (0..matrices.psi.nrows())
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

        // Convert posteriors - use first component for backwards compatibility
        // New structure is [track][measurement][component] - we take [track][measurement][0]
        let posterior_means: Vec<Vec<Vec<f64>>> = matrices
            .posteriors
            .means
            .iter()
            .map(|track_means| {
                track_means
                    .iter()
                    .filter_map(|meas_comps| meas_comps.first())
                    .map(|m| m.iter().copied().collect())
                    .collect()
            })
            .collect();

        let posterior_covariances: Vec<Vec<Vec<Vec<f64>>>> = matrices
            .posteriors
            .covariances
            .iter()
            .map(|track_covs| {
                track_covs
                    .iter()
                    .filter_map(|meas_comps| meas_comps.first())
                    .map(|cov| {
                        let nrows = cov.nrows();
                        (0..nrows)
                            .map(|i| cov.row(i).iter().copied().collect())
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Self {
            cost,
            likelihood,
            sampling_prob,
            eta,
            posterior_means,
            posterior_covariances,
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
