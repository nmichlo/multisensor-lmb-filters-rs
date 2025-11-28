//! Performance metrics for tracking evaluation
//!
//! Implements OSPA (Optimal Sub-Pattern Assignment) metrics for evaluating
//! tracking performance. Matches MATLAB ospa.m exactly.

use nalgebra::{DMatrix, DVector};
use crate::common::types::OspaParameters;

/// OSPA metric result
#[derive(Debug, Clone)]
pub struct OspaResult {
    /// Total OSPA distance
    pub total: f64,
    /// Localization component
    pub localization: f64,
    /// Cardinality component
    pub cardinality: f64,
}

/// Complete OSPA metrics (Euclidean and Hellinger)
#[derive(Debug, Clone)]
pub struct OspaMetrics {
    /// Euclidean OSPA
    pub e_ospa: OspaResult,
    /// Hellinger OSPA
    pub h_ospa: OspaResult,
}

/// Compute Hellinger distance between two Gaussians
///
/// # Arguments
/// * `mu1` - Mean of first Gaussian
/// * `sigma1` - Covariance of first Gaussian
/// * `mu2` - Mean of second Gaussian
/// * `sigma2` - Covariance of second Gaussian
///
/// # Returns
/// Hellinger distance
fn compute_hellinger_distance(
    mu1: &DVector<f64>,
    sigma1: &DMatrix<f64>,
    mu2: &DVector<f64>,
    sigma2: &DMatrix<f64>,
) -> f64 {
    let z = 0.5 * (sigma1 + sigma2);
    let zeta = mu2 - mu1;

    // Z \ zeta (solve Z * x = zeta)
    let z_inv_zeta = match z.clone().cholesky() {
        Some(chol) => chol.solve(&zeta),
        None => return 1.0, // Maximum distance if singular
    };

    let det_sigma1 = sigma1.determinant();
    let det_sigma2 = sigma2.determinant();
    let det_z = z.determinant();

    if det_sigma1 <= 0.0 || det_sigma2 <= 0.0 || det_z <= 0.0 {
        return 1.0; // Maximum distance if non-positive definite
    }

    let exp_arg = -0.125 * zeta.dot(&z_inv_zeta)
        + 0.25 * det_sigma1.ln()
        + 0.25 * det_sigma2.ln()
        - 0.5 * det_z.ln();

    let h_val = 1.0 - exp_arg.exp();
    h_val.max(0.0).sqrt()
}

/// Compute Kullback-Leibler divergence between two probability distributions
///
/// KL(p || q) = sum(p * log(p / q))
///
/// # Arguments
/// * `p` - First probability distribution (reference)
/// * `q` - Second probability distribution (approximation)
///
/// # Returns
/// KL divergence (0 when p == q, larger when distributions differ)
///
/// # Notes
/// Matches MATLAB behavior: `logPQ = log(p./q); logPQ(isinf(logPQ)) = 0;`
/// - Handles zero probabilities gracefully (0 * log(0/q) = 0)
/// - Sets infinite log ratios to 0 (matches MATLAB's isinf() handling)
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "Distributions must have same length");

    // Matches MATLAB: logPQ = log(p./q); logPQ(isinf(logPQ)) = 0; kl = sum(p .* logPQ)
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi <= 0.0 {
                0.0 // 0 * log(0/q) = 0 by convention
            } else if qi <= 0.0 {
                0.0 // MATLAB: log(p/0) = Inf, then isinf() sets to 0
            } else {
                let log_ratio = (pi / qi).ln();
                if log_ratio.is_infinite() {
                    0.0 // MATLAB: isinf(logPQ) = 0
                } else {
                    pi * log_ratio
                }
            }
        })
        .sum()
}

/// Compute average KL divergence across multiple row-distributions
///
/// Each row of p and q is treated as a separate distribution.
///
/// # Arguments
/// * `p` - Matrix where each row is a distribution (reference)
/// * `q` - Matrix where each row is a distribution (approximation)
///
/// # Returns
/// Average KL divergence across all rows
pub fn average_kl_divergence(p: &DMatrix<f64>, q: &DMatrix<f64>) -> f64 {
    assert_eq!(p.nrows(), q.nrows(), "Matrices must have same number of rows");
    assert_eq!(p.ncols(), q.ncols(), "Matrices must have same number of columns");

    if p.nrows() == 0 {
        return 0.0;
    }

    let sum: f64 = (0..p.nrows())
        .map(|i| {
            let p_row: Vec<f64> = (0..p.ncols()).map(|j| p[(i, j)]).collect();
            let q_row: Vec<f64> = (0..q.ncols()).map(|j| q[(i, j)]).collect();
            kl_divergence(&p_row, &q_row)
        })
        .sum();

    sum / p.nrows() as f64
}

/// Compute Hellinger distance between two discrete probability distributions
///
/// H(p, q) = sqrt(1 - sum(sqrt(p * q)))
///
/// # Arguments
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
/// Hellinger distance (0 to 1, 0 when identical)
pub fn hellinger_distance_discrete(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "Distributions must have same length");

    let bc: f64 = p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi * qi).sqrt())
        .sum();

    (1.0 - bc).max(0.0).sqrt()
}

/// Compute average Hellinger distance across multiple row-distributions
///
/// # Arguments
/// * `p` - Matrix where each row is a distribution
/// * `q` - Matrix where each row is a distribution
///
/// # Returns
/// Average Hellinger distance across all rows
pub fn average_hellinger_distance(p: &DMatrix<f64>, q: &DMatrix<f64>) -> f64 {
    assert_eq!(p.nrows(), q.nrows(), "Matrices must have same number of rows");
    assert_eq!(p.ncols(), q.ncols(), "Matrices must have same number of columns");

    if p.nrows() == 0 {
        return 0.0;
    }

    let sum: f64 = (0..p.nrows())
        .map(|i| {
            let p_row: Vec<f64> = (0..p.ncols()).map(|j| p[(i, j)]).collect();
            let q_row: Vec<f64> = (0..q.ncols()).map(|j| q[(i, j)]).collect();
            hellinger_distance_discrete(&p_row, &q_row)
        })
        .sum();

    sum / p.nrows() as f64
}

/// Compute OSPA metric
///
/// Computes both Euclidean and Hellinger OSPA distances between ground truth
/// and filter estimates.
///
/// # Arguments
/// * `x` - Ground truth states
/// * `mu_gt` - Ground truth means (from Kalman filter)
/// * `sigma_gt` - Ground truth covariances
/// * `nu` - Filter state estimates
/// * `t` - Filter covariances
/// * `params` - OSPA parameters (cutoffs and orders)
///
/// # Returns
/// OspaMetrics with both Euclidean and Hellinger components
pub fn ospa(
    x: &[DVector<f64>],
    mu_gt: &[DVector<f64>],
    sigma_gt: &[DMatrix<f64>],
    nu: &[DVector<f64>],
    t: &[DMatrix<f64>],
    params: &OspaParameters,
) -> OspaMetrics {
    // Case 1: Both sets empty
    if mu_gt.is_empty() && nu.is_empty() {
        return OspaMetrics {
            e_ospa: OspaResult {
                total: 0.0,
                localization: 0.0,
                cardinality: 0.0,
            },
            h_ospa: OspaResult {
                total: 0.0,
                localization: 0.0,
                cardinality: 0.0,
            },
        };
    }

    // Case 2: One set empty
    if mu_gt.is_empty() || nu.is_empty() {
        return OspaMetrics {
            e_ospa: OspaResult {
                total: params.e_c,
                localization: 0.0,
                cardinality: params.e_c,
            },
            h_ospa: OspaResult {
                total: params.h_c,
                localization: 0.0,
                cardinality: params.h_c,
            },
        };
    }

    // Case 3: Non-empty sets
    let n = mu_gt.len();
    let m = nu.len();
    let ell = n.max(m);
    let q = (m as i32 - n as i32).abs() as usize;

    // Populate distance matrices
    let mut euclidean_distances = DMatrix::zeros(n, m);
    let mut hellinger_distances = DMatrix::zeros(n, m);

    for i in 0..m {
        for j in 0..n {
            let diff = &x[j] - &nu[i];
            euclidean_distances[(j, i)] = diff.norm();
            hellinger_distances[(j, i)] = compute_hellinger_distance(
                &mu_gt[j],
                &sigma_gt[j],
                &nu[i],
                &t[i],
            );
        }
    }

    // Apply cutoffs
    let mut euclidean_cutoff = DMatrix::zeros(n, m);
    let mut hellinger_cutoff = DMatrix::zeros(n, m);

    for i in 0..n {
        for j in 0..m {
            euclidean_cutoff[(i, j)] = euclidean_distances[(i, j)].min(params.e_c).powf(params.e_p);
            hellinger_cutoff[(i, j)] = hellinger_distances[(i, j)].min(params.h_c).powf(params.h_p);
        }
    }

    // Compute optimal assignments using Hungarian algorithm
    let hungarian_e = crate::common::association::hungarian::hungarian(&euclidean_cutoff);
    let hungarian_h = crate::common::association::hungarian::hungarian(&hellinger_cutoff);

    let euclidean_cost = hungarian_e.cost;
    let hellinger_cost = hungarian_h.cost;

    // Calculate final distances
    let e_total = ((1.0 / ell as f64)
        * (params.e_c.powf(params.e_p) * q as f64 + euclidean_cost))
        .powf(1.0 / params.e_p);
    let e_loc = ((1.0 / ell as f64) * euclidean_cost).powf(1.0 / params.e_p);
    let e_card = ((1.0 / ell as f64) * params.e_c.powf(params.e_p) * q as f64)
        .powf(1.0 / params.e_p);

    let h_total = ((1.0 / ell as f64)
        * (params.h_c.powf(params.h_p) * q as f64 + hellinger_cost))
        .powf(1.0 / params.h_p);
    let h_loc = ((1.0 / ell as f64) * hellinger_cost).powf(1.0 / params.h_p);
    let h_card = ((1.0 / ell as f64) * params.h_c.powf(params.h_p) * q as f64)
        .powf(1.0 / params.h_p);

    OspaMetrics {
        e_ospa: OspaResult {
            total: e_total,
            localization: e_loc,
            cardinality: e_card,
        },
        h_ospa: OspaResult {
            total: h_total,
            localization: h_loc,
            cardinality: h_card,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ospa_empty() {
        let params = OspaParameters {
            e_c: 5.0,
            e_p: 2.0,
            h_c: 0.5,
            h_p: 2.0,
        };

        let result = ospa(&[], &[], &[], &[], &[], &params);

        assert_eq!(result.e_ospa.total, 0.0);
        assert_eq!(result.h_ospa.total, 0.0);
    }

    #[test]
    fn test_ospa_one_empty() {
        let params = OspaParameters {
            e_c: 5.0,
            e_p: 2.0,
            h_c: 0.5,
            h_p: 2.0,
        };

        let x = vec![DVector::from_vec(vec![1.0, 2.0])];
        let mu = vec![DVector::from_vec(vec![1.0, 2.0])];
        let sigma = vec![DMatrix::identity(2, 2)];

        let result = ospa(&x, &mu, &sigma, &[], &[], &params);

        assert_eq!(result.e_ospa.total, params.e_c);
        assert_eq!(result.h_ospa.total, params.h_c);
    }

    #[test]
    fn test_ospa_perfect_match() {
        let params = OspaParameters {
            e_c: 5.0,
            e_p: 2.0,
            h_c: 0.5,
            h_p: 2.0,
        };

        let x = vec![DVector::from_vec(vec![1.0, 2.0])];
        let mu = vec![DVector::from_vec(vec![1.0, 2.0])];
        let sigma = vec![DMatrix::identity(2, 2)];
        let nu = vec![DVector::from_vec(vec![1.0, 2.0])];
        let t = vec![DMatrix::identity(2, 2)];

        let result = ospa(&x, &mu, &sigma, &nu, &t, &params);

        // Perfect match should have zero or near-zero distance
        assert!(result.e_ospa.total < 1e-10);
    }

    #[test]
    fn test_hellinger_distance_identical() {
        let mu = DVector::from_vec(vec![0.0, 0.0]);
        let sigma = DMatrix::identity(2, 2);

        let h = compute_hellinger_distance(&mu, &sigma, &mu, &sigma);

        assert!(h < 1e-10); // Should be zero for identical distributions
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![0.5, 0.3, 0.2];
        let kl = kl_divergence(&p, &p);
        assert!(kl < 1e-10, "KL divergence of identical distributions should be 0");
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = vec![0.5, 0.5];
        let q = vec![0.9, 0.1];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0, "KL divergence of different distributions should be > 0");
    }

    #[test]
    fn test_hellinger_distance_discrete_identical() {
        let p = vec![0.5, 0.3, 0.2];
        let h = hellinger_distance_discrete(&p, &p);
        assert!(h < 1e-10, "Hellinger distance of identical distributions should be 0");
    }

    #[test]
    fn test_hellinger_distance_discrete_different() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let h = hellinger_distance_discrete(&p, &q);
        assert!((h - 1.0).abs() < 1e-10, "Hellinger distance of disjoint distributions should be 1");
    }
}