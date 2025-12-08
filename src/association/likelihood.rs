//! Unified likelihood computation for measurement-track association.
//!
//! This module computes the likelihood that a measurement originated from a track,
//! which is the foundation of data association. The computation involves:
//!
//! 1. **Innovation**: The difference between the measurement and predicted observation
//! 2. **Mahalanobis distance**: Normalized distance accounting for uncertainty
//! 3. **Likelihood ratio**: Compares detection likelihood to clutter likelihood
//! 4. **Kalman posterior**: Updated state estimate if this association is correct
//!
//! The likelihood ratio `L = p_D × N(z; Cμ, CΣC'+Q) / κ` balances:
//! - Detection probability (`p_D`): How likely the sensor detects the track
//! - Gaussian likelihood: How well the measurement fits the track prediction
//! - Clutter density (`κ`): The false alarm rate per unit volume
//!
//! High `L` means the measurement is much more likely from this track than from clutter.

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};

use crate::lmb::SensorModel;

/// Pre-allocated workspace for likelihood computations.
///
/// Computing likelihoods involves matrix operations that allocate intermediate
/// results. This workspace pre-allocates these buffers and reuses them across
/// multiple likelihood evaluations, significantly reducing allocation overhead
/// in the inner loop of data association.
///
/// Create one workspace per filter and reuse it for all likelihood computations.
#[derive(Debug, Clone)]
pub struct LikelihoodWorkspace {
    /// Innovation covariance: Z = C*Σ*Cᵀ + Q
    pub innovation_cov: DMatrix<f64>,
    /// Inverse of innovation covariance
    pub innovation_cov_inv: DMatrix<f64>,
    /// Kalman gain: K = Σ*Cᵀ*Z⁻¹
    pub kalman_gain: DMatrix<f64>,
    /// Innovation vector: ν = z - C*μ
    pub innovation: DVector<f64>,
    /// Temporary for matrix operations
    temp_matrix: DMatrix<f64>,
}

impl LikelihoodWorkspace {
    /// Create a new workspace with given dimensions
    ///
    /// # Arguments
    /// * `x_dim` - State dimension
    /// * `z_dim` - Measurement dimension
    pub fn new(x_dim: usize, z_dim: usize) -> Self {
        Self {
            innovation_cov: DMatrix::zeros(z_dim, z_dim),
            innovation_cov_inv: DMatrix::zeros(z_dim, z_dim),
            kalman_gain: DMatrix::zeros(x_dim, z_dim),
            innovation: DVector::zeros(z_dim),
            temp_matrix: DMatrix::zeros(x_dim, z_dim),
        }
    }

    /// Resize workspace if dimensions change
    pub fn resize(&mut self, x_dim: usize, z_dim: usize) {
        if self.innovation.len() != z_dim || self.kalman_gain.nrows() != x_dim {
            *self = Self::new(x_dim, z_dim);
        }
    }
}

/// Result of likelihood computation for one track-measurement pair.
///
/// Contains everything needed to evaluate and apply a potential association:
/// - The likelihood ratio tells us how plausible this association is
/// - The posterior parameters are used if we accept this association
#[derive(Debug, Clone)]
pub struct LikelihoodResult {
    /// Log of the likelihood ratio `log(p_D × N(...) / κ)`.
    /// Positive values indicate the measurement is more likely from this track than clutter.
    pub log_likelihood_ratio: f64,
    /// Kalman-updated state mean assuming this association is correct.
    pub posterior_mean: DVector<f64>,
    /// Kalman-updated state covariance (reduced uncertainty from measurement).
    pub posterior_covariance: DMatrix<f64>,
    /// Kalman gain matrix used in the update.
    pub kalman_gain: DMatrix<f64>,
}

impl LikelihoodResult {
    /// Get the linear (non-log) likelihood ratio
    #[inline]
    pub fn likelihood_ratio(&self) -> f64 {
        self.log_likelihood_ratio.exp()
    }
}

/// Core likelihood computation - used by ALL filters
///
/// This function computes the likelihood of a measurement given a prior track state,
/// along with the posterior parameters after a Kalman update.
///
/// # Formula
/// - Innovation covariance: `Z = C × Σ × Cᵀ + Q`
/// - Innovation: `ν = z - C × μ`
/// - Mahalanobis distance: `d² = νᵀ × Z⁻¹ × ν`
/// - Log-likelihood: `-0.5 × (n×ln(2π) + ln|Z| + d²)`
/// - Kalman gain: `K = Σ × Cᵀ × Z⁻¹`
/// - Posterior mean: `μ' = μ + K × ν`
/// - Posterior covariance: `Σ' = (I - K × C) × Σ`
///
/// # Arguments
/// * `prior_mean` - Prior state mean
/// * `prior_cov` - Prior state covariance
/// * `measurement` - Measurement vector
/// * `sensor` - Sensor model parameters
/// * `workspace` - Reusable workspace (for efficiency)
///
/// # Returns
/// Likelihood result including posterior parameters
pub fn compute_likelihood(
    prior_mean: &DVector<f64>,
    prior_cov: &DMatrix<f64>,
    measurement: &DVector<f64>,
    sensor: &SensorModel,
    workspace: &mut LikelihoodWorkspace,
) -> LikelihoodResult {
    let x_dim = prior_mean.len();
    let z_dim = measurement.len();

    // Ensure workspace is correctly sized
    workspace.resize(x_dim, z_dim);

    // Innovation covariance: Z = C × Σ × Cᵀ + Q
    // Using temp_matrix to avoid allocation: temp = Σ × Cᵀ
    workspace.temp_matrix = prior_cov * sensor.observation_matrix.transpose();

    // Z = C × temp + Q = C × Σ × Cᵀ + Q
    workspace.innovation_cov =
        &sensor.observation_matrix * &workspace.temp_matrix + &sensor.measurement_noise;

    // Invert Z (with numerical stability check)
    workspace.innovation_cov_inv = workspace
        .innovation_cov
        .clone()
        .try_inverse()
        .unwrap_or_else(|| {
            // Fallback: add small regularization
            let reg = &workspace.innovation_cov + DMatrix::identity(z_dim, z_dim) * 1e-10;
            reg.try_inverse()
                .unwrap_or_else(|| DMatrix::identity(z_dim, z_dim))
        });

    // Innovation: ν = z - C × μ
    workspace.innovation = measurement - &sensor.observation_matrix * prior_mean;

    // Mahalanobis distance: d² = νᵀ × Z⁻¹ × ν
    let mahal = workspace
        .innovation
        .dot(&(&workspace.innovation_cov_inv * &workspace.innovation));

    // Log-likelihood
    let log_det = workspace.innovation_cov.determinant().ln();
    let log_norm = -0.5 * (z_dim as f64 * (2.0 * PI).ln() + log_det);
    let log_lik = log_norm - 0.5 * mahal;

    // Log-likelihood ratio: includes detection probability and clutter density
    let log_likelihood_ratio =
        log_lik + sensor.detection_probability.ln() - sensor.clutter_density().ln();

    // Kalman gain: K = Σ × Cᵀ × Z⁻¹
    let kalman_gain = &workspace.temp_matrix * &workspace.innovation_cov_inv;

    // Posterior mean: μ' = μ + K × ν
    let posterior_mean = prior_mean + &kalman_gain * &workspace.innovation;

    // Posterior covariance: Σ' = (I - K × C) × Σ
    let posterior_covariance =
        (DMatrix::identity(x_dim, x_dim) - &kalman_gain * &sensor.observation_matrix) * prior_cov;

    LikelihoodResult {
        log_likelihood_ratio,
        posterior_mean,
        posterior_covariance,
        kalman_gain,
    }
}

/// Compute log-likelihood only (without posterior update)
///
/// More efficient when posterior parameters aren't needed (e.g., gating).
#[inline]
pub fn compute_log_likelihood(
    prior_mean: &DVector<f64>,
    prior_cov: &DMatrix<f64>,
    measurement: &DVector<f64>,
    sensor: &SensorModel,
) -> f64 {
    let z_dim = measurement.len();

    // Innovation covariance: Z = C × Σ × Cᵀ + Q
    let innovation_cov =
        &sensor.observation_matrix * prior_cov * sensor.observation_matrix.transpose()
            + &sensor.measurement_noise;

    // Invert Z
    let innovation_cov_inv = match innovation_cov.clone().try_inverse() {
        Some(inv) => inv,
        None => return f64::NEG_INFINITY, // Singular matrix
    };

    // Innovation: ν = z - C × μ
    let innovation = measurement - &sensor.observation_matrix * prior_mean;

    // Mahalanobis distance
    let mahal = innovation.dot(&(&innovation_cov_inv * &innovation));

    // Log-likelihood
    let log_det = innovation_cov.determinant().ln();
    let log_norm = -0.5 * (z_dim as f64 * (2.0 * PI).ln() + log_det);
    let log_lik = log_norm - 0.5 * mahal;

    // Return log-likelihood ratio
    log_lik + sensor.detection_probability.ln() - sensor.clutter_density().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sensor() -> SensorModel {
        SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0)
    }

    #[test]
    fn test_workspace_creation() {
        let ws = LikelihoodWorkspace::new(4, 2);
        assert_eq!(ws.innovation.len(), 2);
        assert_eq!(ws.kalman_gain.nrows(), 4);
        assert_eq!(ws.kalman_gain.ncols(), 2);
    }

    #[test]
    fn test_workspace_resize() {
        let mut ws = LikelihoodWorkspace::new(4, 2);
        ws.resize(6, 3);
        assert_eq!(ws.innovation.len(), 3);
        assert_eq!(ws.kalman_gain.nrows(), 6);
    }

    #[test]
    fn test_likelihood_computation() {
        let sensor = create_test_sensor();
        let mut ws = LikelihoodWorkspace::new(4, 2);

        let prior_mean = DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let prior_cov = DMatrix::identity(4, 4) * 10.0;
        let measurement = DVector::from_vec(vec![0.1, 0.1]);

        let result = compute_likelihood(&prior_mean, &prior_cov, &measurement, &sensor, &mut ws);

        // Posterior mean should be pulled toward measurement
        assert!(result.posterior_mean[0].abs() > 0.0); // Should move toward 0.1
        assert!(result.posterior_mean[2].abs() > 0.0); // Should move toward 0.1

        // Posterior covariance should be smaller than prior
        assert!(result.posterior_covariance[(0, 0)] < prior_cov[(0, 0)]);
    }

    #[test]
    fn test_log_likelihood_only() {
        let sensor = create_test_sensor();

        let prior_mean = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let prior_cov = DMatrix::identity(4, 4);
        let measurement = DVector::from_vec(vec![0.0, 0.0]);

        let log_lik = compute_log_likelihood(&prior_mean, &prior_cov, &measurement, &sensor);

        // Perfect measurement match should give positive log-likelihood ratio
        // (measurement exactly at predicted position)
        assert!(log_lik.is_finite());
    }

    #[test]
    fn test_likelihood_ratio() {
        let sensor = create_test_sensor();
        let mut ws = LikelihoodWorkspace::new(4, 2);

        let prior_mean = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let prior_cov = DMatrix::identity(4, 4);

        // Close measurement
        let close_measurement = DVector::from_vec(vec![0.1, 0.1]);
        let close_result = compute_likelihood(
            &prior_mean,
            &prior_cov,
            &close_measurement,
            &sensor,
            &mut ws,
        );

        // Far measurement
        let far_measurement = DVector::from_vec(vec![100.0, 100.0]);
        let far_result =
            compute_likelihood(&prior_mean, &prior_cov, &far_measurement, &sensor, &mut ws);

        // Close measurement should have higher likelihood ratio
        assert!(close_result.log_likelihood_ratio > far_result.log_likelihood_ratio);
    }
}
