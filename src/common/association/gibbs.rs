//! Gibbs sampling for data association
//!
//! Implements Gibbs sampling algorithm for computing marginal association probabilities.
//! Matches MATLAB generateGibbsSample.m, initialiseGibbsAssociationVectors.m,
//! and lmbGibbsSampling.m exactly.

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Gibbs sampling result
#[derive(Debug, Clone)]
pub struct GibbsResult {
    /// Posterior existence probabilities (n x 1)
    pub r: DVector<f64>,
    /// Marginal association probabilities (n x (m+1))
    /// Each row contains an object's association probabilities
    pub w: DMatrix<f64>,
    /// Association event samples (n_samples x n)
    pub v_samples: DMatrix<usize>,
}

/// Association matrices for Gibbs sampling
#[derive(Debug, Clone)]
pub struct GibbsAssociationMatrices {
    /// P matrix (n x m): sampling probabilities
    pub p: DMatrix<f64>,
    /// L matrix (n x (m+1)): likelihood matrix [eta, L]
    pub l: DMatrix<f64>,
    /// R matrix (n x (m+1)): existence ratio matrix [phi/eta, ones]
    pub r: DMatrix<f64>,
    /// C matrix (n x m): cost matrix for initialization
    pub c: DMatrix<f64>,
}

/// Initialize Gibbs association vectors using Hungarian/Murty's algorithm
///
/// Finds the most likely assignment as the initial state for Gibbs sampling.
///
/// # Arguments
/// * `c` - Cost matrix (n x m)
///
/// # Returns
/// Tuple of (v, w) association vectors
pub fn initialize_gibbs_association_vectors(c: &DMatrix<f64>) -> (Vec<usize>, Vec<usize>) {
    let n = c.nrows();
    let m = c.ncols();

    // Use Hungarian algorithm to find best assignment
    let hungarian_result = super::hungarian::hungarian(c);

    // Extract v from matching matrix
    let mut v = vec![0; n];
    for i in 0..n {
        for j in 0..m {
            if hungarian_result.matching[(i, j)] == 1.0 {
                v[i] = j + 1; // 1-indexed (0 means not assigned)
                break;
            }
        }
    }

    // Determine w from v
    let mut w = vec![0; m];
    for (obj_idx, &meas_idx) in v.iter().enumerate() {
        if meas_idx > 0 {
            w[meas_idx - 1] = obj_idx + 1; // 1-indexed
        }
    }

    (v, w)
}

/// Generate a new Gibbs sample
///
/// Updates association vectors v and w using Gibbs sampling.
///
/// # Arguments
/// * `p` - Sampling probability matrix (n x m)
/// * `v` - Object-to-measurement association vector (1-indexed, 0 = not assigned)
/// * `w` - Measurement-to-object association vector (1-indexed, 0 = not assigned)
///
/// # Returns
/// Updated (v, w) vectors
pub fn generate_gibbs_sample(
    rng: &mut impl crate::common::rng::Rng,
    p: &DMatrix<f64>,
    mut v: Vec<usize>,
    mut w: Vec<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let n = p.nrows();
    let m = p.ncols();

    // Loop through each object
    for i in 0..n {
        // Start at first measurement or index 0
        let k = if v[i] == 0 { 0 } else { v[i] - 1 }; // Convert to 0-indexed

        // Try each measurement starting from k
        for j in k..m {
            // Sample from a_i^j if column j is otherwise unoccupied
            if w[j] == 0 || w[j] == i + 1 {
                if rng.rand() < p[(i, j)] {
                    // Object i generated measurement j
                    v[i] = j + 1; // Store as 1-indexed
                    w[j] = i + 1;
                    break;
                } else {
                    // Object i does not exist or missed detection
                    v[i] = 0;
                    w[j] = 0;
                }
            }
        }
    }

    (v, w)
}

/// LMB Gibbs sampling
///
/// Determines posterior existence probabilities and marginal association
/// probabilities using Gibbs sampling.
///
/// # Arguments
/// * `matrices` - Gibbs association matrices (P, L, R, C)
/// * `num_samples` - Number of Gibbs samples to generate
///
/// # Returns
/// GibbsResult with existence probabilities, association weights, and samples
pub fn lmb_gibbs_sampling(
    rng: &mut impl crate::common::rng::Rng,
    matrices: &GibbsAssociationMatrices,
    num_samples: usize,
) -> GibbsResult {
    let n = matrices.p.nrows();
    let m = matrices.p.ncols();

    // Initialize association vectors
    let (mut v, mut w) = initialize_gibbs_association_vectors(&matrices.c);

    // Store samples
    let mut v_samples_vec = Vec::new();

    // Generate samples
    for _ in 0..num_samples {
        (v, w) = generate_gibbs_sample(rng, &matrices.p, v, w);
        v_samples_vec.push(v.clone());
    }

    // Convert samples to matrix
    let mut v_samples = DMatrix::zeros(v_samples_vec.len(), n);
    for (i, sample) in v_samples_vec.iter().enumerate() {
        for (j, &val) in sample.iter().enumerate() {
            v_samples[(i, j)] = val;
        }
    }

    // Find unique samples
    let mut unique_samples: HashMap<Vec<usize>, usize> = HashMap::new();
    for sample in &v_samples_vec {
        *unique_samples.entry(sample.clone()).or_insert(0) += 1;
    }

    let unique_v: Vec<Vec<usize>> = unique_samples.keys().cloned().collect();

    // Compute marginal distributions
    // This is complex due to MATLAB's advanced indexing
    let mut sigma = DMatrix::zeros(n, m + 1);

    for v_event in &unique_v {
        // Compute likelihood of this event
        let mut j_values = Vec::new();
        for (obj_idx, &meas_idx) in v_event.iter().enumerate() {
            // L is (n x (m+1)) with columns [miss, meas1, meas2, ...]
            j_values.push(matrices.l[(obj_idx, meas_idx)]);
        }

        let event_likelihood: f64 = j_values.iter().product();

        // Add to marginal for each object-measurement pair
        for (obj_idx, &meas_idx) in v_event.iter().enumerate() {
            sigma[(obj_idx, meas_idx)] += event_likelihood;
        }
    }

    // Normalize: Tau = (Sigma .* R) ./ sum(Sigma, 2)
    let mut tau = DMatrix::zeros(n, m + 1);
    for i in 0..n {
        let row_sum: f64 = sigma.row(i).sum();
        if row_sum > 1e-15 {
            for j in 0..(m + 1) {
                tau[(i, j)] = (sigma[(i, j)] * matrices.r[(i, j)]) / row_sum;
            }
        }
    }

    // Existence probabilities: r = sum(Tau, 2)
    let mut r = DVector::zeros(n);
    for i in 0..n {
        r[i] = tau.row(i).sum();
    }

    // Marginal association probabilities: W = Tau ./ r
    let mut w_result = DMatrix::zeros(n, m + 1);
    for i in 0..n {
        if r[i] > 1e-15 {
            for j in 0..(m + 1) {
                w_result[(i, j)] = tau[(i, j)] / r[i];
            }
        }
    }

    GibbsResult {
        r,
        w: w_result,
        v_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gibbs_sampling_simple() {
        // Simple 2 objects, 2 measurements
        let p = DMatrix::from_row_slice(2, 2, &[
            0.7, 0.3,
            0.4, 0.6,
        ]);

        // L = [eta, L1, L2]
        let l = DMatrix::from_row_slice(2, 3, &[
            0.05, 0.8, 0.15,
            0.05, 0.3, 0.65,
        ]);

        // R = [phi/eta, 1, 1]
        let r = DMatrix::from_row_slice(2, 3, &[
            0.05, 1.0, 1.0,
            0.05, 1.0, 1.0,
        ]);

        let c = DMatrix::from_row_slice(2, 2, &[
            1.0, 3.0,
            3.0, 1.0,
        ]);

        let matrices = GibbsAssociationMatrices { p, l, r, c };

        let result = lmb_gibbs_sampling(&matrices, 100);

        // Verify shapes
        assert_eq!(result.r.len(), 2);
        assert_eq!(result.w.nrows(), 2);
        assert_eq!(result.w.ncols(), 3); // miss + 2 measurements

        // Verify probabilities sum to 1 for each object
        for i in 0..2 {
            let row_sum: f64 = result.w.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sum: {}", i, row_sum);
        }

        // Verify existence probabilities are in [0, 1]
        for i in 0..2 {
            assert!(result.r[i] >= 0.0 && result.r[i] <= 1.0);
        }
    }
}