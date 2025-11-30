//! Linear algebra utilities
//!
//! Mathematical functions for Gaussian operations, matrix manipulations,
//! and numerical computations required by the tracking algorithms.

use crate::common::types::{DMatrix, DVector};
use ndarray::Array1;
use ndarray_linalg::{Cholesky, Determinant, Inverse, Solve, UPLO};
use std::f64::consts::PI;

/// Compute multivariate Gaussian PDF
///
/// Computes the probability density function of a multivariate Gaussian
/// distribution at a given point.
///
/// # Arguments
/// * `x` - Point to evaluate (column vector)
/// * `mu` - Mean vector
/// * `sigma` - Covariance matrix
///
/// # Returns
/// Probability density value
pub fn gaussian_pdf(x: &DVector<f64>, mu: &DVector<f64>, sigma: &DMatrix<f64>) -> f64 {
    let n = x.len() as f64;
    let diff = x - mu;

    // Compute determinant
    let det = match sigma.det() {
        Ok(d) => d,
        Err(_) => return 0.0,
    };

    if det <= 0.0 {
        return 0.0; // Singular covariance
    }

    // Cholesky decomposition for numerical stability
    // TODO(faer): Cholesky decomposition in hot path.
    // faer's blocked algorithm is faster for matrices > 64x64.
    // See REFACTOR.md Phase 4 for benchmarks.
    match sigma.cholesky(UPLO::Lower) {
        Ok(chol) => {
            match chol.solve(&diff) {
                Ok(inv_sigma_diff) => {
                    let mahalanobis = diff.dot(&inv_sigma_diff);
                    let coeff = 1.0 / ((2.0 * PI).powf(n / 2.0) * det.sqrt());
                    coeff * (-0.5 * mahalanobis).exp()
                }
                Err(_) => 0.0,
            }
        }
        Err(_) => 0.0, // Failed Cholesky
    }
}

/// Compute Mahalanobis distance
///
/// # Arguments
/// * `x` - Point
/// * `mu` - Mean vector
/// * `sigma` - Covariance matrix
///
/// # Returns
/// Mahalanobis distance
pub fn mahalanobis_distance(x: &DVector<f64>, mu: &DVector<f64>, sigma: &DMatrix<f64>) -> f64 {
    let diff = x - mu;

    match sigma.cholesky(UPLO::Lower) {
        Ok(chol) => match chol.solve(&diff) {
            Ok(inv_sigma_diff) => diff.dot(&inv_sigma_diff).sqrt(),
            Err(_) => f64::INFINITY,
        },
        Err(_) => f64::INFINITY, // Singular covariance
    }
}

/// Compute log Gaussian PDF for numerical stability
///
/// # Arguments
/// * `x` - Point to evaluate
/// * `mu` - Mean vector
/// * `sigma` - Covariance matrix
///
/// # Returns
/// Log probability density
pub fn log_gaussian_pdf(x: &DVector<f64>, mu: &DVector<f64>, sigma: &DMatrix<f64>) -> f64 {
    let n = x.len() as f64;
    let diff = x - mu;

    let det = match sigma.det() {
        Ok(d) => d,
        Err(_) => return f64::NEG_INFINITY,
    };

    if det <= 0.0 {
        return f64::NEG_INFINITY;
    }

    match sigma.cholesky(UPLO::Lower) {
        Ok(chol) => match chol.solve(&diff) {
            Ok(inv_sigma_diff) => {
                let mahalanobis = diff.dot(&inv_sigma_diff);
                -0.5 * (n * (2.0 * PI).ln() + det.ln() + mahalanobis)
            }
            Err(_) => f64::NEG_INFINITY,
        },
        Err(_) => f64::NEG_INFINITY,
    }
}

/// Kalman filter update step
///
/// Performs a single Kalman filter measurement update
///
/// # Arguments
/// * `x_pred` - Predicted state mean
/// * `P_pred` - Predicted state covariance
/// * `z` - Measurement
/// * `H` - Measurement matrix
/// * `R` - Measurement noise covariance
///
/// # Returns
/// Tuple of (updated mean, updated covariance, innovation, innovation covariance, likelihood)
pub fn kalman_update(
    x_pred: &DVector<f64>,
    p_pred: &DMatrix<f64>,
    z: &DVector<f64>,
    h: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> (DVector<f64>, DMatrix<f64>, DVector<f64>, DMatrix<f64>, f64) {
    // Innovation
    let z_pred = h.dot(x_pred);
    let innovation = z - &z_pred;

    // Innovation covariance
    let s = h.dot(p_pred).dot(&h.t()) + r;

    // Kalman gain: K = P * H' * inv(S)
    // Use Cholesky solve or fallback to direct inverse
    let k = match s.inv() {
        Ok(s_inv) => p_pred.dot(&h.t()).dot(&s_inv),
        Err(_) => return (x_pred.clone(), p_pred.clone(), innovation.clone(), s, 0.0),
    };

    // Updated state
    let x_updated = x_pred + &k.dot(&innovation);

    // Updated covariance (Joseph form for numerical stability)
    let i_minus_kh = DMatrix::eye(x_pred.len()) - &k.dot(h);
    let p_updated = &i_minus_kh.dot(p_pred).dot(&i_minus_kh.t())
        + &k.dot(r).dot(&k.t());

    // Likelihood
    let likelihood = gaussian_pdf(&innovation, &Array1::zeros(innovation.len()), &s);

    (x_updated, p_updated, innovation, s, likelihood)
}

/// Compute log-sum-exp for numerical stability
///
/// Computes log(sum(exp(x))) in a numerically stable way
///
/// # Arguments
/// * `values` - Vector of log values
///
/// # Returns
/// Log of sum of exponentials
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() && max_val < 0.0 {
        return f64::NEG_INFINITY;
    }

    let sum: f64 = values.iter().map(|v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Normalize log weights
///
/// Convert log weights to normalized linear weights
///
/// # Arguments
/// * `log_weights` - Vector of log weights
///
/// # Returns
/// Normalized weights
pub fn normalize_log_weights(log_weights: &[f64]) -> Vec<f64> {
    let log_sum = log_sum_exp(log_weights);
    log_weights.iter().map(|w| (w - log_sum).exp()).collect()
}

/// Check if matrix is positive definite
///
/// # Arguments
/// * `matrix` - Matrix to check
///
/// # Returns
/// true if positive definite
pub fn is_positive_definite(matrix: &DMatrix<f64>) -> bool {
    matrix.cholesky(UPLO::Lower).is_ok()
}

/// Make matrix symmetric
///
/// Ensures a matrix is symmetric by averaging with its transpose
///
/// # Arguments
/// * `matrix` - Matrix to symmetrize
///
/// # Returns
/// Symmetric matrix
pub fn symmetrize(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (matrix + &matrix.t())
}
