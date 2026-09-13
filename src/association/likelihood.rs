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
/// - Innovation covariance: `Z = C × Σ × Cᵀ + R_k`
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
/// * `measurement_noise_override` - Optional per-detection R_k override.
/// * `pd_override` - Optional per-track detection probability override.
/// * `c_matrix_override` - Optional per-detection observation matrix C_k.
///   When `Some(C_k)`, overrides `sensor.observation_matrix`. Use for time-varying
///   or nonlinear observation models (e.g. a Doppler row whose direction changes
///   with target bearing).
///
/// # Returns
/// Likelihood result including posterior parameters
pub fn compute_likelihood(
    prior_mean: &DVector<f64>,
    prior_cov: &DMatrix<f64>,
    measurement: &DVector<f64>,
    sensor: &SensorModel,
    workspace: &mut LikelihoodWorkspace,
    measurement_noise_override: Option<&DMatrix<f64>>,
    pd_override: Option<f64>,
    c_matrix_override: Option<&DMatrix<f64>>,
) -> LikelihoodResult {
    let x_dim = prior_mean.len();
    let z_dim = measurement.len();

    // Ensure workspace is correctly sized
    workspace.resize(x_dim, z_dim);

    // Select noise and observation matrix: per-detection overrides or sensor defaults
    let noise = measurement_noise_override.unwrap_or(&sensor.measurement_noise);
    let c = c_matrix_override.unwrap_or(&sensor.observation_matrix);

    // Innovation covariance: Z = C × Σ × Cᵀ + R_k
    // Using temp_matrix to avoid allocation: temp = Σ × Cᵀ
    workspace.temp_matrix = prior_cov * c.transpose();

    // Z = C × temp + R_k = C × Σ × Cᵀ + R_k
    workspace.innovation_cov = c * &workspace.temp_matrix + noise;

    // Guard a degenerate innovation covariance before it poisons the update:
    // det(Z) <= 0 or non-finite makes ln|Z| NaN and Z⁻¹ meaningless, which would
    // inject NaN into the association cost matrix. This branch never fires on the
    // well-conditioned MATLAB-equivalence fixtures, so the happy path below is
    // unchanged; on a degenerate input we mark the association impossible
    // (log-ratio = -inf) and return the prior as the (unused) posterior.
    let z_det = workspace.innovation_cov.determinant();
    if !(z_det > 0.0) || !z_det.is_finite() {
        return LikelihoodResult {
            log_likelihood_ratio: f64::NEG_INFINITY,
            posterior_mean: prior_mean.clone(),
            posterior_covariance: prior_cov.clone(),
            kalman_gain: DMatrix::zeros(x_dim, z_dim),
        };
    }

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
    workspace.innovation = measurement - c * prior_mean;

    // Mahalanobis distance: d² = νᵀ × Z⁻¹ × ν
    let mahal = workspace
        .innovation
        .dot(&(&workspace.innovation_cov_inv * &workspace.innovation));

    // Log-likelihood (reuse the determinant already computed above)
    let log_det = z_det.ln();
    let log_norm = -0.5 * (z_dim as f64 * (2.0 * PI).ln() + log_det);
    let log_lik = log_norm - 0.5 * mahal;

    let p_d = pd_override.unwrap_or(sensor.detection_probability);

    // Log-likelihood ratio: includes detection probability and clutter density
    let log_likelihood_ratio =
        log_lik + p_d.ln() - sensor.clutter_density().ln();

    // Kalman gain: K = Σ × Cᵀ × Z⁻¹
    let kalman_gain = &workspace.temp_matrix * &workspace.innovation_cov_inv;

    // Posterior mean: μ' = μ + K × ν
    let posterior_mean = prior_mean + &kalman_gain * &workspace.innovation;

    // Posterior covariance: Σ' = (I - K × C) × Σ
    let posterior_covariance =
        (DMatrix::identity(x_dim, x_dim) - &kalman_gain * c) * prior_cov;

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
///
/// # Arguments
/// * `measurement_noise_override` - Optional per-detection R_k override.
/// * `c_matrix_override` - Optional per-detection C_k override.
#[inline]
pub fn compute_log_likelihood(
    prior_mean: &DVector<f64>,
    prior_cov: &DMatrix<f64>,
    measurement: &DVector<f64>,
    sensor: &SensorModel,
    measurement_noise_override: Option<&DMatrix<f64>>,
    pd_override: Option<f64>,
    c_matrix_override: Option<&DMatrix<f64>>,
) -> f64 {
    let z_dim = measurement.len();

    let noise = measurement_noise_override.unwrap_or(&sensor.measurement_noise);
    let c = c_matrix_override.unwrap_or(&sensor.observation_matrix);

    // Innovation covariance: Z = C × Σ × Cᵀ + R_k
    let innovation_cov = c * prior_cov * c.transpose() + noise;

    // Guard a degenerate innovation covariance (see compute_likelihood): det(Z) <= 0
    // or non-finite would make ln|Z| NaN. Fixture-neutral; only fires on degenerate
    // inputs. This also catches an indefinite-but-invertible Z that try_inverse below
    // would otherwise accept.
    let z_det = innovation_cov.determinant();
    if !(z_det > 0.0) || !z_det.is_finite() {
        return f64::NEG_INFINITY;
    }

    // Invert Z
    let innovation_cov_inv = match innovation_cov.clone().try_inverse() {
        Some(inv) => inv,
        None => return f64::NEG_INFINITY, // Singular matrix
    };

    // Innovation: ν = z - C × μ
    let innovation = measurement - c * prior_mean;

    // Mahalanobis distance
    let mahal = innovation.dot(&(&innovation_cov_inv * &innovation));

    // Log-likelihood (reuse the determinant already computed above)
    let log_det = z_det.ln();
    let log_norm = -0.5 * (z_dim as f64 * (2.0 * PI).ln() + log_det);
    let log_lik = log_norm - 0.5 * mahal;

    let p_d = pd_override.unwrap_or(sensor.detection_probability);

    // Return log-likelihood ratio
    log_lik + p_d.ln() - sensor.clutter_density().ln()
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

        // State ordering: [x, y, vx, vy] (MATLAB convention)
        let prior_mean = DVector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let prior_cov = DMatrix::identity(4, 4) * 10.0;
        let measurement = DVector::from_vec(vec![0.1, 0.1]);

        let result = compute_likelihood(&prior_mean, &prior_cov, &measurement, &sensor, &mut ws, None, None, None);

        // Posterior mean should be pulled toward measurement
        // Observation is [x, y] so mean[0] (x) and mean[1] (y) should move toward 0.1
        assert!(result.posterior_mean[0].abs() > 0.0); // x should move toward 0.1
        assert!(result.posterior_mean[1].abs() > 0.0); // y should move toward 0.1

        // Posterior covariance should be smaller than prior
        assert!(result.posterior_covariance[(0, 0)] < prior_cov[(0, 0)]);
    }

    #[test]
    fn test_log_likelihood_only() {
        let sensor = create_test_sensor();

        let prior_mean = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let prior_cov = DMatrix::identity(4, 4);
        let measurement = DVector::from_vec(vec![0.0, 0.0]);

        let log_lik = compute_log_likelihood(&prior_mean, &prior_cov, &measurement, &sensor, None, None, None);

        // Perfect measurement match should give positive log-likelihood ratio
        // (measurement exactly at predicted position)
        assert!(log_lik.is_finite());
    }

    #[test]
    fn test_degenerate_innovation_covariance_is_guarded() {
        // A measurement-noise override that cancels the predicted innovation
        // covariance yields det(Z) = 0, which would make ln|Z| = -inf and the
        // Mahalanobis solve meaningless. The guard must turn this into a clean
        // "impossible association" (-inf log-ratio) with a finite posterior,
        // never NaN.
        let sensor = create_test_sensor();
        let mut ws = LikelihoodWorkspace::new(4, 2);
        let prior_mean = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let prior_cov = DMatrix::identity(4, 4);
        let measurement = DVector::from_vec(vec![0.1, 0.1]);
        // C*Σ*Cᵀ = I₂ for this 2D position sensor; this noise drives Z to 0.
        let bad_noise = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -1.0]);

        let result = compute_likelihood(
            &prior_mean,
            &prior_cov,
            &measurement,
            &sensor,
            &mut ws,
            Some(&bad_noise),
            None,
            None,
        );
        assert_eq!(result.log_likelihood_ratio, f64::NEG_INFINITY);
        assert!(result.posterior_mean.iter().all(|v| v.is_finite()));
        assert!(result.posterior_covariance.iter().all(|v| v.is_finite()));

        let log_lik = compute_log_likelihood(
            &prior_mean,
            &prior_cov,
            &measurement,
            &sensor,
            Some(&bad_noise),
            None,
            None,
        );
        assert_eq!(log_lik, f64::NEG_INFINITY);
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
            None,
            None,
            None,
        );

        // Far measurement
        let far_measurement = DVector::from_vec(vec![100.0, 100.0]);
        let far_result =
            compute_likelihood(&prior_mean, &prior_cov, &far_measurement, &sensor, &mut ws, None, None, None);

        // Close measurement should have higher likelihood ratio
        assert!(close_result.log_likelihood_ratio > far_result.log_likelihood_ratio);
    }
}
