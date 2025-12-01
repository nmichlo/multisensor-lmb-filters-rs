//! Linear algebra utilities
//!
//! Mathematical functions for Gaussian operations, matrix manipulations,
//! and numerical computations required by the tracking algorithms.

use crate::common::constants::SVD_TOLERANCE;
use nalgebra::{DMatrix, DVector};
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

    // Compute determinant and inverse
    let det = sigma.determinant();
    if det <= 0.0 {
        return 0.0; // Singular covariance
    }

    // Cholesky decomposition for numerical stability
    match sigma.clone().cholesky() {
        Some(chol) => {
            let inv_sigma_diff = chol.solve(&diff);
            let mahalanobis = diff.dot(&inv_sigma_diff);

            let coeff = 1.0 / ((2.0 * PI).powf(n / 2.0) * det.sqrt());
            coeff * (-0.5 * mahalanobis).exp()
        }
        None => 0.0, // Failed Cholesky
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

    match sigma.clone().cholesky() {
        Some(chol) => {
            let inv_sigma_diff = chol.solve(&diff);
            diff.dot(&inv_sigma_diff).sqrt()
        }
        None => f64::INFINITY, // Singular covariance
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

    let det = sigma.determinant();
    if det <= 0.0 {
        return f64::NEG_INFINITY;
    }

    match sigma.clone().cholesky() {
        Some(chol) => {
            let inv_sigma_diff = chol.solve(&diff);
            let mahalanobis = diff.dot(&inv_sigma_diff);

            -0.5 * (n * (2.0 * PI).ln() + det.ln() + mahalanobis)
        }
        None => f64::NEG_INFINITY,
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
    let z_pred = h * x_pred;
    let innovation = z - z_pred;

    // Innovation covariance
    let s = h * p_pred * h.transpose() + r;

    // Kalman gain
    let k = match s.clone().cholesky() {
        Some(chol) => {
            let temp = chol.solve(&(h * p_pred).transpose()).transpose();
            p_pred * h.transpose() * temp
        }
        None => {
            // Fallback to pseudo-inverse
            match s.clone().try_inverse() {
                Some(s_inv) => p_pred * h.transpose() * s_inv,
                None => return (x_pred.clone(), p_pred.clone(), innovation.clone(), s, 0.0),
            }
        }
    };

    // Updated state
    let x_updated = x_pred + &k * &innovation;

    // Updated covariance (Joseph form for numerical stability)
    let i_minus_kh = DMatrix::identity(x_pred.len(), x_pred.len()) - &k * h;
    let p_updated = &i_minus_kh * p_pred * i_minus_kh.transpose() + &k * r * k.transpose();

    // Likelihood
    let likelihood = gaussian_pdf(&innovation, &DVector::zeros(innovation.len()), &s);

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

/// Robustly compute matrix inverse with fallbacks
///
/// Attempts multiple methods to compute matrix inverse:
/// 1. Cholesky decomposition (fastest for positive definite matrices)
/// 2. LU decomposition via try_inverse
/// 3. SVD-based pseudo-inverse (last resort)
///
/// # Arguments
/// * `matrix` - Matrix to invert
///
/// # Returns
/// Some(inverse) if any method succeeds, None if all methods fail
pub fn robust_inverse(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    // Try Cholesky first (fastest for positive definite)
    if let Some(chol) = matrix.clone().cholesky() {
        return Some(chol.inverse());
    }

    // Fall back to LU decomposition
    if let Some(inv) = matrix.clone().try_inverse() {
        return Some(inv);
    }

    // Last resort: SVD pseudo-inverse
    let svd = matrix.clone().svd(true, true);
    svd.pseudo_inverse(SVD_TOLERANCE).ok()
}

/// Robustly invert matrix and compute log-determinant in single decomposition
///
/// This is more efficient than calling robust_inverse() + log_gaussian_normalizing_constant()
/// separately, as it avoids computing Cholesky twice for positive definite matrices.
///
/// Attempts multiple methods:
/// 1. Cholesky decomposition (fastest for positive definite matrices)
/// 2. LU decomposition
/// 3. SVD-based pseudo-inverse (last resort)
///
/// # Arguments
/// * `matrix` - Matrix to invert (must be square)
/// * `dimension` - Dimension of the matrix (for log normalizing constant computation)
///
/// # Returns
/// Some((inverse, log_normalizing_constant)) if any method succeeds, None if all fail
/// The log_normalizing_constant is -0.5 * (n*ln(2π) + ln|det(matrix)|)
///
/// Note: Takes ownership to avoid cloning. Callers should pass the matrix by value.
#[inline]
pub fn robust_inverse_with_log_det(matrix: DMatrix<f64>, dimension: usize) -> Option<(DMatrix<f64>, f64)> {
    let n = dimension as f64;
    let log_2pi = (2.0 * PI).ln();

    // Try Cholesky first (fastest for positive definite)
    // Clone only needed for fallback paths
    if let Some(chol) = matrix.clone().cholesky() {
        let inv = chol.inverse();
        // log(det(Z)) = 2 * sum(log(diag(L))) where Z = L*L'
        let log_det = 2.0 * chol.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
        let eta = -0.5 * (n * log_2pi + log_det);
        return Some((inv, eta));
    }

    // Fall back to LU decomposition
    if let Some(inv) = matrix.clone().try_inverse() {
        let log_det = matrix.determinant().ln();
        let eta = -0.5 * (n * log_2pi + log_det);
        return Some((inv, eta));
    }

    // Last resort: SVD pseudo-inverse
    let svd = matrix.svd(true, true);
    // Compute log-det from singular values before consuming svd
    let log_det = svd.singular_values.iter()
        .filter(|&&s| s > SVD_TOLERANCE)
        .map(|s| s.ln())
        .sum::<f64>();
    if let Ok(inv) = svd.pseudo_inverse(SVD_TOLERANCE) {
        let eta = -0.5 * (n * log_2pi + log_det);
        return Some((inv, eta));
    }

    None
}

/// Robustly solve linear system A*x = b with fallbacks
///
/// Attempts multiple methods to solve the system:
/// 1. Cholesky decomposition (fastest for positive definite A)
/// 2. LU decomposition via try_inverse
/// 3. SVD-based pseudo-inverse (last resort)
///
/// # Arguments
/// * `a` - System matrix
/// * `b` - Right-hand side (can be matrix for multiple RHS)
///
/// # Returns
/// Some(x) if any method succeeds, None if all methods fail
pub fn robust_solve(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    // Try Cholesky first (fastest for positive definite)
    if let Some(chol) = a.clone().cholesky() {
        return Some(chol.solve(b));
    }

    // Fall back to computing inverse and multiplying
    if let Some(inv) = robust_inverse(a) {
        return Some(inv * b);
    }

    None
}

/// Robustly solve linear system A*x = b for vector RHS
///
/// Convenience wrapper for robust_solve with vector right-hand side.
///
/// # Arguments
/// * `a` - System matrix
/// * `b` - Right-hand side vector
///
/// # Returns
/// Some(x) if any method succeeds, None if all methods fail
pub fn robust_solve_vec(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    // Try Cholesky first (fastest for positive definite)
    if let Some(chol) = a.clone().cholesky() {
        return Some(chol.solve(b));
    }

    // Fall back to computing inverse and multiplying
    if let Some(inv) = robust_inverse(a) {
        return Some(inv * b);
    }

    None
}

/// Check if matrix is positive definite
///
/// # Arguments
/// * `matrix` - Matrix to check
///
/// # Returns
/// true if positive definite
pub fn is_positive_definite(matrix: &DMatrix<f64>) -> bool {
    matrix.clone().cholesky().is_some()
}

// ============================================================================
// Likelihood computation helpers (for association matrix generation)
// ============================================================================

/// Compute predicted measurement and innovation covariance
///
/// Computes:
/// - mu_z = C * mu (predicted measurement)
/// - Z = C * Sigma * C' + Q (innovation covariance)
///
/// # Arguments
/// * `mu` - State mean vector
/// * `sigma` - State covariance matrix
/// * `c` - Observation matrix
/// * `q` - Measurement noise covariance
///
/// # Returns
/// Tuple of (predicted measurement, innovation covariance)
#[inline]
pub fn compute_innovation_params(
    mu: &DVector<f64>,
    sigma: &DMatrix<f64>,
    c: &DMatrix<f64>,
    q: &DMatrix<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let mu_z = c * mu;
    let z_cov = c * sigma * c.transpose() + q;
    (mu_z, z_cov)
}

/// Compute log Gaussian normalizing constant
///
/// Computes: -0.5 * (n * log(2*pi) + log(det(Z)))
///
/// Uses Cholesky decomposition for numerical stability when possible,
/// falls back to direct determinant computation.
///
/// # Arguments
/// * `z_cov` - Innovation covariance matrix
/// * `z_dimension` - Dimension of measurement space
///
/// # Returns
/// Log normalizing constant
#[inline]
pub fn log_gaussian_normalizing_constant(z_cov: &DMatrix<f64>, z_dimension: usize) -> f64 {
    let n = z_dimension as f64;
    let log_2pi = (2.0 * PI).ln();

    // Try Cholesky for more stable log-determinant computation
    if let Some(chol) = z_cov.clone().cholesky() {
        // log(det(Z)) = 2 * sum(log(diag(L))) where Z = L*L'
        let log_det = 2.0 * chol.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
        -0.5 * (n * log_2pi + log_det)
    } else {
        // Fallback to direct determinant
        -0.5 * (n * log_2pi + z_cov.determinant().ln())
    }
}

/// Compute Kalman gain and updated covariance
///
/// Computes:
/// - K = Sigma * C' * Z^{-1} (Kalman gain)
/// - Sigma_updated = (I - K*C) * Sigma (updated covariance)
///
/// # Arguments
/// * `sigma` - Prior state covariance
/// * `c` - Observation matrix
/// * `z_cov` - Innovation covariance
/// * `x_dimension` - State dimension
///
/// # Returns
/// Some((K, Sigma_updated, Z_inv)) if Z is invertible, None otherwise
#[inline]
pub fn compute_kalman_gain(
    sigma: &DMatrix<f64>,
    c: &DMatrix<f64>,
    z_cov: &DMatrix<f64>,
    x_dimension: usize,
) -> Option<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> {
    let z_inv = robust_inverse(z_cov)?;
    let k = sigma * c.transpose() * &z_inv;
    let sigma_updated = (DMatrix::identity(x_dimension, x_dimension) - &k * c) * sigma;
    Some((k, sigma_updated, z_inv))
}

/// Compute measurement log-likelihood (Gaussian)
///
/// Computes: log_norm - 0.5 * (z - mu_z)' * Z^{-1} * (z - mu_z)
///
/// # Arguments
/// * `measurement` - Measurement vector
/// * `mu_z` - Predicted measurement
/// * `z_inv` - Inverse of innovation covariance
/// * `log_norm` - Log normalizing constant from `log_gaussian_normalizing_constant`
///
/// # Returns
/// Log-likelihood value
#[inline]
pub fn compute_measurement_log_likelihood(
    measurement: &DVector<f64>,
    mu_z: &DVector<f64>,
    z_inv: &DMatrix<f64>,
    log_norm: f64,
) -> f64 {
    let nu = measurement - mu_z;
    log_norm - 0.5 * nu.dot(&(z_inv * &nu))
}

/// Compute updated mean given Kalman gain and innovation
///
/// Computes: mu_updated = mu + K * (z - mu_z)
///
/// # Arguments
/// * `mu` - Prior state mean
/// * `k` - Kalman gain
/// * `measurement` - Measurement vector
/// * `mu_z` - Predicted measurement
///
/// # Returns
/// Updated state mean
#[inline]
pub fn compute_kalman_updated_mean(
    mu: &DVector<f64>,
    k: &DMatrix<f64>,
    measurement: &DVector<f64>,
    mu_z: &DVector<f64>,
) -> DVector<f64> {
    let nu = measurement - mu_z;
    mu + k * &nu
}

// ============================================================================
// Prediction step helpers (for Chapman-Kolmogorov prediction)
// ============================================================================

/// Predict state mean using linear motion model
///
/// Computes: mu' = A * mu + u
///
/// # Arguments
/// * `mu` - Current state mean
/// * `a` - State transition matrix
/// * `u` - Control input / drift vector
///
/// # Returns
/// Predicted state mean
#[inline]
pub fn predict_mean(mu: &DVector<f64>, a: &DMatrix<f64>, u: &DVector<f64>) -> DVector<f64> {
    a * mu + u
}

/// Predict state covariance using linear motion model
///
/// Computes: Sigma' = A * Sigma * A' + R
///
/// # Arguments
/// * `sigma` - Current state covariance
/// * `a` - State transition matrix
/// * `r` - Process noise covariance
///
/// # Returns
/// Predicted state covariance
#[inline]
pub fn predict_covariance(
    sigma: &DMatrix<f64>,
    a: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> DMatrix<f64> {
    a * sigma * a.transpose() + r
}

/// Predict existence probability
///
/// Computes: r' = p_s * r
///
/// # Arguments
/// * `r` - Current existence probability
/// * `survival_probability` - Probability of survival
///
/// # Returns
/// Predicted existence probability
#[inline]
pub fn predict_existence(r: f64, survival_probability: f64) -> f64 {
    survival_probability * r
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
    0.5 * (matrix + matrix.transpose())
}

// ============================================================================
// Canonical form helpers (for Gaussian fusion in track merging)
// ============================================================================

/// Canonical form parameters for a Gaussian component
///
/// Represents a Gaussian in information/canonical form:
/// - K (precision matrix) = inv(Sigma)
/// - h (canonical mean) = K * mu
/// - g (log normalizing constant contribution)
#[derive(Debug, Clone)]
pub struct CanonicalGaussian {
    /// Precision matrix K = inv(Sigma)
    pub k: DMatrix<f64>,
    /// Canonical mean h = K * mu
    pub h: DVector<f64>,
    /// Log normalizing constant contribution
    pub g: f64,
}

/// Convert Gaussian from moment form to canonical form
///
/// Converts (mu, sigma, weight) to canonical parameters (K, h, g) where:
/// - K = inv(Sigma)
/// - h = K * mu
/// - g = -0.5 * mu' * K * mu - 0.5 * ln(det(2*pi*Sigma)) + ln(weight)
///
/// This is used in covariance intersection and information-form fusion.
///
/// # Arguments
/// * `mu` - State mean vector
/// * `sigma` - State covariance matrix
/// * `weight` - Mixture component weight (use 1.0 if not applicable)
///
/// # Returns
/// Some(CanonicalGaussian) if sigma is invertible, None otherwise
#[inline]
pub fn to_canonical_form(
    mu: &DVector<f64>,
    sigma: &DMatrix<f64>,
    weight: f64,
) -> Option<CanonicalGaussian> {
    let k = robust_inverse(sigma)?;
    let h = &k * mu;

    let det = sigma.determinant();
    let quad_term = -0.5 * mu.dot(&(&k * mu));
    let det_term = -0.5 * (2.0 * PI * det).ln();
    let weight_term = if weight > 0.0 { weight.ln() } else { f64::NEG_INFINITY };
    let g = quad_term + det_term + weight_term;

    Some(CanonicalGaussian { k, h, g })
}

/// Convert Gaussian from canonical form to moment form
///
/// Converts canonical parameters (K, h, g) back to moment form (mu, sigma):
/// - Sigma = inv(K)
/// - mu = Sigma * h
/// - g_out = g_in + 0.5 * mu' * K * mu + 0.5 * ln(det(2*pi*Sigma))
///
/// # Arguments
/// * `canonical` - Canonical form parameters
///
/// # Returns
/// Some((mu, sigma, g_out)) if K is invertible, None otherwise
/// where g_out can be used to compute the unnormalized weight
#[inline]
pub fn from_canonical_form(
    canonical: &CanonicalGaussian,
) -> Option<(DVector<f64>, DMatrix<f64>, f64)> {
    let sigma = robust_inverse(&canonical.k)?;
    let mu = &sigma * &canonical.h;

    let det = sigma.determinant();
    let g_out = canonical.g
        + 0.5 * mu.dot(&(&canonical.k * &mu))
        + 0.5 * (2.0 * PI * det).ln();

    Some((mu, sigma, g_out))
}

/// Convert Gaussian to weighted canonical form (for fusion with weights)
///
/// Converts (mu, sigma) to weighted canonical parameters where:
/// - K = weight * inv(Sigma)
/// - h = K * mu
/// - g = -0.5 * mu' * K * mu - 0.5 * weight * ln(det(2*pi*Sigma))
///
/// This is used in weighted geometric average (GA) fusion.
///
/// # Arguments
/// * `mu` - State mean vector
/// * `sigma` - State covariance matrix
/// * `weight` - Fusion weight (e.g., sensor weight for GA merging)
///
/// # Returns
/// Some(CanonicalGaussian) if sigma is invertible, None otherwise
#[inline]
pub fn to_weighted_canonical_form(
    mu: &DVector<f64>,
    sigma: &DMatrix<f64>,
    weight: f64,
) -> Option<CanonicalGaussian> {
    let sigma_inv = robust_inverse(sigma)?;
    let k = &sigma_inv * weight;
    let h = &k * mu;

    let det = sigma.determinant();
    let g = -0.5 * mu.dot(&(&k * mu)) - 0.5 * weight * (2.0 * PI * det).ln();

    Some(CanonicalGaussian { k, h, g })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robust_inverse_positive_definite() {
        // Positive definite matrix (should use Cholesky)
        let m = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let inv = robust_inverse(&m).expect("Should be invertible");

        // Check M * M^{-1} ≈ I
        let identity = &m * &inv;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((identity[(i, j)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_robust_inverse_non_positive_definite() {
        // Invertible but not positive definite
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let inv = robust_inverse(&m).expect("Should be invertible via LU");

        // Check M * M^{-1} ≈ I
        let identity = &m * &inv;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((identity[(i, j)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_robust_inverse_singular() {
        // Singular matrix
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        // SVD pseudo-inverse should still work
        let result = robust_inverse(&m);
        // Pseudo-inverse exists for singular matrices
        assert!(result.is_some());
    }

    #[test]
    fn test_robust_solve_positive_definite() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 2.0]);
        let x = robust_solve(&a, &b).expect("Should be solvable");

        // Check A*x ≈ b
        let result = &a * &x;
        assert!((result[(0, 0)] - b[(0, 0)]).abs() < 1e-10);
        assert!((result[(1, 0)] - b[(1, 0)]).abs() < 1e-10);
    }

    #[test]
    fn test_robust_solve_vec() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let x = robust_solve_vec(&a, &b).expect("Should be solvable");

        // Check A*x ≈ b
        let result = &a * &x;
        assert!((result[0] - b[0]).abs() < 1e-10);
        assert!((result[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_simple() {
        let values = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&values);
        // log(e^1 + e^2 + e^3) ≈ 3.4076
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_empty() {
        let values: Vec<f64> = vec![];
        let result = log_sum_exp(&values);
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_log_sum_exp_negative_infinity() {
        let values = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        let result = log_sum_exp(&values);
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_normalize_log_weights() {
        let log_weights = vec![0.0, 0.0, 0.0]; // Equal weights
        let normalized = normalize_log_weights(&log_weights);
        for w in &normalized {
            assert!((*w - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalize_log_weights_sum_to_one() {
        let log_weights = vec![-1.0, 0.0, 1.0, 2.0];
        let normalized = normalize_log_weights(&log_weights);
        let sum: f64 = normalized.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    // Tests for likelihood computation helpers

    #[test]
    fn test_compute_innovation_params() {
        let mu = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]); // 4D state
        let sigma = DMatrix::identity(4, 4);
        let c = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]); // 2D measurement
        let q = DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.1]);

        let (mu_z, z_cov) = compute_innovation_params(&mu, &sigma, &c, &q);

        // mu_z = C * mu = [1.0, 3.0]
        assert!((mu_z[0] - 1.0).abs() < 1e-10);
        assert!((mu_z[1] - 3.0).abs() < 1e-10);

        // z_cov = C * I * C' + Q = diag(1.1, 1.1)
        assert!((z_cov[(0, 0)] - 1.1).abs() < 1e-10);
        assert!((z_cov[(1, 1)] - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_log_gaussian_normalizing_constant() {
        // 2D identity covariance
        let z_cov = DMatrix::identity(2, 2);
        let log_norm = log_gaussian_normalizing_constant(&z_cov, 2);

        // -0.5 * (2 * log(2*pi) + log(1)) = -log(2*pi) ≈ -1.8379
        let expected = -0.5 * 2.0 * (2.0 * PI).ln();
        assert!((log_norm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_compute_kalman_gain() {
        let sigma = DMatrix::identity(4, 4);
        let c = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let z_cov = DMatrix::from_row_slice(2, 2, &[1.1, 0.0, 0.0, 1.1]);

        let result = compute_kalman_gain(&sigma, &c, &z_cov, 4);
        assert!(result.is_some());

        let (k, sigma_updated, z_inv) = result.unwrap();

        // K should be 4x2
        assert_eq!(k.nrows(), 4);
        assert_eq!(k.ncols(), 2);

        // Sigma_updated should be 4x4
        assert_eq!(sigma_updated.nrows(), 4);
        assert_eq!(sigma_updated.ncols(), 4);

        // Z_inv should be 2x2 and Z * Z_inv ≈ I
        let identity = &z_cov * &z_inv;
        assert!((identity[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((identity[(1, 1)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_measurement_log_likelihood() {
        let measurement = DVector::from_vec(vec![1.0, 2.0]);
        let mu_z = DVector::from_vec(vec![1.0, 2.0]); // Perfect prediction
        let z_inv = DMatrix::identity(2, 2);
        let log_norm = -1.0; // Arbitrary

        let log_lik = compute_measurement_log_likelihood(&measurement, &mu_z, &z_inv, log_norm);

        // With perfect prediction, nu = 0, so log_lik = log_norm
        assert!((log_lik - log_norm).abs() < 1e-10);
    }

    #[test]
    fn test_compute_kalman_updated_mean() {
        let mu = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let k = DMatrix::from_row_slice(4, 2, &[
            0.5, 0.0,
            0.0, 0.0,
            0.0, 0.5,
            0.0, 0.0,
        ]);
        let measurement = DVector::from_vec(vec![2.0, 4.0]);
        let mu_z = DVector::from_vec(vec![1.0, 3.0]);

        let mu_updated = compute_kalman_updated_mean(&mu, &k, &measurement, &mu_z);

        // nu = [2-1, 4-3] = [1, 1]
        // K * nu = [0.5, 0, 0.5, 0]
        // mu_updated = [1.5, 2, 3.5, 4]
        assert!((mu_updated[0] - 1.5).abs() < 1e-10);
        assert!((mu_updated[1] - 2.0).abs() < 1e-10);
        assert!((mu_updated[2] - 3.5).abs() < 1e-10);
        assert!((mu_updated[3] - 4.0).abs() < 1e-10);
    }

    // Tests for prediction helpers

    #[test]
    fn test_predict_mean() {
        let mu = DVector::from_vec(vec![1.0, 2.0]);
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]); // Near-constant velocity
        let u = DVector::from_vec(vec![0.0, 0.0]);

        let mu_pred = predict_mean(&mu, &a, &u);

        // mu' = A * mu = [1.0 + 0.1*2.0, 2.0] = [1.2, 2.0]
        assert!((mu_pred[0] - 1.2).abs() < 1e-10);
        assert!((mu_pred[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_covariance() {
        let sigma = DMatrix::identity(2, 2);
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]); // Identity
        let r = DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.1]); // Small process noise

        let sigma_pred = predict_covariance(&sigma, &a, &r);

        // sigma' = I * I * I' + R = I + R = diag(1.1, 1.1)
        assert!((sigma_pred[(0, 0)] - 1.1).abs() < 1e-10);
        assert!((sigma_pred[(1, 1)] - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_predict_existence() {
        let r = 0.8;
        let p_s = 0.99;

        let r_pred = predict_existence(r, p_s);

        assert!((r_pred - 0.792).abs() < 1e-10);
    }

    // Tests for canonical form helpers

    #[test]
    fn test_to_canonical_form() {
        let mu = DVector::from_vec(vec![1.0, 2.0]);
        let sigma = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
        let weight = 1.0;

        let canonical = to_canonical_form(&mu, &sigma, weight).expect("Should succeed");

        // K = inv(Sigma)
        let sigma_inv = robust_inverse(&sigma).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((canonical.k[(i, j)] - sigma_inv[(i, j)]).abs() < 1e-10);
            }
        }

        // h = K * mu
        let expected_h = &sigma_inv * &mu;
        for i in 0..2 {
            assert!((canonical.h[i] - expected_h[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_from_canonical_form() {
        let mu = DVector::from_vec(vec![1.0, 2.0]);
        let sigma = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
        let weight = 1.0;

        // Round trip: moment -> canonical -> moment
        let canonical = to_canonical_form(&mu, &sigma, weight).expect("Should succeed");
        let (mu_back, sigma_back, _g_out) =
            from_canonical_form(&canonical).expect("Should succeed");

        // mu should be recovered
        for i in 0..2 {
            assert!((mu_back[i] - mu[i]).abs() < 1e-10);
        }

        // sigma should be recovered
        for i in 0..2 {
            for j in 0..2 {
                assert!((sigma_back[(i, j)] - sigma[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_to_weighted_canonical_form() {
        let mu = DVector::from_vec(vec![1.0, 2.0]);
        let sigma = DMatrix::identity(2, 2);
        let weight = 0.5;

        let canonical = to_weighted_canonical_form(&mu, &sigma, weight).expect("Should succeed");

        // K = weight * inv(Sigma) = 0.5 * I
        assert!((canonical.k[(0, 0)] - 0.5).abs() < 1e-10);
        assert!((canonical.k[(1, 1)] - 0.5).abs() < 1e-10);

        // h = K * mu = [0.5, 1.0]
        assert!((canonical.h[0] - 0.5).abs() < 1e-10);
        assert!((canonical.h[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_canonical_form_fusion() {
        // Test that fusing two Gaussians in canonical form works correctly
        let mu1 = DVector::from_vec(vec![1.0, 0.0]);
        let sigma1 = DMatrix::identity(2, 2);

        let mu2 = DVector::from_vec(vec![0.0, 1.0]);
        let sigma2 = DMatrix::identity(2, 2);

        // Convert to canonical form with equal weights
        let c1 = to_weighted_canonical_form(&mu1, &sigma1, 0.5).unwrap();
        let c2 = to_weighted_canonical_form(&mu2, &sigma2, 0.5).unwrap();

        // Fuse by summing canonical parameters
        let fused = CanonicalGaussian {
            k: &c1.k + &c2.k,
            h: &c1.h + &c2.h,
            g: c1.g + c2.g,
        };

        // Convert back to moment form
        let (mu_fused, sigma_fused, _g) = from_canonical_form(&fused).unwrap();

        // With equal weights and equal covariances, mean should be average
        assert!((mu_fused[0] - 0.5).abs() < 1e-10);
        assert!((mu_fused[1] - 0.5).abs() < 1e-10);

        // Fused covariance should be I (since K_fused = 0.5*I + 0.5*I = I)
        assert!((sigma_fused[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((sigma_fused[(1, 1)] - 1.0).abs() < 1e-10);
    }
}
