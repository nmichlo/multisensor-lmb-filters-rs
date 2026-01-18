//! Multi-sensor track fusion strategies.
//!
//! This module contains implementations of the [`Merger`] trait for fusing
//! per-sensor track posteriors into unified estimates.
//!
//! # Available Strategies
//!
//! - [`MergerAverageArithmetic`] - Simple weighted average (fast, robust)
//! - [`MergerAverageGeometric`] - Covariance intersection (conservative)
//! - [`MergerParallelUpdate`] - Information-form fusion (optimal for independent sensors)
//! - [`MergerIteratedCorrector`] - Sequential sensor updates (order-dependent)

use nalgebra::{DMatrix, DVector};
use crate::utils::common_ops;
use crate::Merger;
use crate::types::{GaussianComponent, Track};

// ============================================================================
// Arithmetic Average Merger
// ============================================================================

/// Arithmetic Average (AA) track merger.
///
/// Fuses per-sensor track posteriors by computing weighted arithmetic means
/// of existence probabilities and concatenating weighted GM components.
///
/// This is the simplest fusion strategy:
/// - `r_fused = Σ(w_s × r_s)` for existence probabilities
/// - GM components are concatenated with sensor-weighted component weights
/// - Components are sorted by weight and truncated to max_components
///
/// AA fusion is fast and robust but doesn't account for correlation between
/// sensors and may produce sub-optimal covariances.
#[derive(Debug, Clone)]
pub struct MergerAverageArithmetic {
    /// Per-sensor fusion weights (should sum to 1.0)
    pub sensor_weights: Vec<f64>,
    /// Maximum GM components to keep after fusion
    pub max_components: usize,
}

impl MergerAverageArithmetic {
    /// Create with uniform weights for given number of sensors.
    pub fn uniform(num_sensors: usize, max_components: usize) -> Self {
        Self {
            sensor_weights: vec![1.0 / num_sensors as f64; num_sensors],
            max_components,
        }
    }

    /// Create with custom weights.
    pub fn with_weights(weights: Vec<f64>, max_components: usize) -> Self {
        Self {
            sensor_weights: weights,
            max_components,
        }
    }
}

impl Merger for MergerAverageArithmetic {
    fn merge(&self, per_sensor_tracks: &[Vec<Track>], weights: Option<&[f64]>) -> Vec<Track> {
        if per_sensor_tracks.is_empty() || per_sensor_tracks[0].is_empty() {
            return Vec::new();
        }

        let num_sensors = per_sensor_tracks.len();
        let num_tracks = per_sensor_tracks[0].len();
        let weights = weights.unwrap_or(&self.sensor_weights);

        let mut fused_tracks = per_sensor_tracks[0].clone();

        for i in 0..num_tracks {
            // Weighted sum of existence probabilities
            let mut r_sum = 0.0;
            for s in 0..num_sensors {
                r_sum += weights[s] * per_sensor_tracks[s][i].existence;
            }
            fused_tracks[i].existence = r_sum;

            // Concatenate weighted GM components
            let mut all_components: Vec<(f64, DVector<f64>, DMatrix<f64>)> = Vec::new();

            for s in 0..num_sensors {
                let track = &per_sensor_tracks[s][i];
                for comp in &track.components {
                    all_components.push((
                        weights[s] * comp.weight,
                        comp.mean.clone(),
                        comp.covariance.clone(),
                    ));
                }
            }

            // Truncate to max components and normalize (no Mahalanobis merging)
            fused_tracks[i].components =
                common_ops::truncate_components(all_components, self.max_components);
        }

        fused_tracks
    }

    fn name(&self) -> &'static str {
        "ArithmeticAverage"
    }
}

// ============================================================================
// Geometric Average Merger
// ============================================================================

/// Geometric Average (GA) track merger.
///
/// Fuses per-sensor track posteriors using weighted geometric average in
/// canonical (information) form. This implements covariance intersection.
///
/// The algorithm:
/// 1. Moment-match each sensor's GM to a single Gaussian (m-projection)
/// 2. Convert to canonical form: K = w_s × Σ⁻¹, h = K × μ
/// 3. Sum canonical parameters: K_fused = Σ K_s, h_fused = Σ h_s
/// 4. Convert back: Σ_fused = K_fused⁻¹, μ_fused = Σ_fused × h_fused
/// 5. Geometric mean of existence: r_fused ∝ Π(r_s^w_s)
///
/// GA produces conservative (larger) covariances and is robust to unknown
/// correlations between sensors.
#[derive(Debug, Clone)]
pub struct MergerAverageGeometric {
    /// Per-sensor fusion weights (should sum to 1.0)
    pub sensor_weights: Vec<f64>,
}

impl MergerAverageGeometric {
    /// Create with uniform weights for given number of sensors.
    pub fn uniform(num_sensors: usize) -> Self {
        Self {
            sensor_weights: vec![1.0 / num_sensors as f64; num_sensors],
        }
    }

    /// Create with custom weights.
    pub fn with_weights(weights: Vec<f64>) -> Self {
        Self {
            sensor_weights: weights,
        }
    }

    /// M-projection: collapse GM to single Gaussian by moment matching.
    fn moment_match(track: &Track) -> (DVector<f64>, DMatrix<f64>) {
        if track.components.is_empty() {
            let dim = 4; // fallback dimension
            return (DVector::zeros(dim), DMatrix::identity(dim, dim));
        }

        let dim = track.x_dim();

        // Weighted mean
        let mut nu = DVector::zeros(dim);
        for comp in &track.components {
            nu += &comp.mean * comp.weight;
        }

        // Weighted covariance
        let mut t = DMatrix::zeros(dim, dim);
        for comp in &track.components {
            let mu_diff = &comp.mean - &nu;
            t += (&comp.covariance + &mu_diff * mu_diff.transpose()) * comp.weight;
        }

        (nu, t)
    }
}

impl Merger for MergerAverageGeometric {
    fn merge(&self, per_sensor_tracks: &[Vec<Track>], weights: Option<&[f64]>) -> Vec<Track> {
        if per_sensor_tracks.is_empty() || per_sensor_tracks[0].is_empty() {
            return Vec::new();
        }

        let num_sensors = per_sensor_tracks.len();
        let num_tracks = per_sensor_tracks[0].len();
        let weights = weights.unwrap_or(&self.sensor_weights);

        let mut fused_tracks = per_sensor_tracks[0].clone();

        for i in 0..num_tracks {
            let dim = per_sensor_tracks[0][i].x_dim();

            // Accumulate canonical parameters
            let mut k_sum = DMatrix::zeros(dim, dim);
            let mut h_sum = DVector::zeros(dim);
            let mut g_sum = 0.0;

            for s in 0..num_sensors {
                let (nu, t) = Self::moment_match(&per_sensor_tracks[s][i]);

                // Convert to weighted canonical form
                if let Some(t_inv) = t.clone().try_inverse() {
                    let k_s = &t_inv * weights[s];
                    let h_s = &k_s * &nu;

                    // g = -0.5 * nu' * K * nu - 0.5 * w * log(det(2π*T))
                    let det_2pi_t = (2.0 * std::f64::consts::PI).powi(dim as i32) * t.determinant();
                    let g_s = -0.5 * nu.dot(&(&k_s * &nu)) - 0.5 * weights[s] * det_2pi_t.ln();

                    k_sum += k_s;
                    h_sum += h_s;
                    g_sum += g_s;
                }
            }

            // Convert back to moment form
            if let Some(sigma_fused) = k_sum.clone().try_inverse() {
                let mu_fused = &sigma_fused * &h_sum;

                // GA normalization: eta = exp(g + 0.5 * mu' * K * mu + 0.5 * log(det(2π*Σ)))
                // MATLAB: eta = exp(g + 0.5 * muGa' * K * muGa + 0.5 * log(det(2*pi*SigmaGa)));
                let det_2pi_sigma =
                    (2.0 * std::f64::consts::PI).powi(dim as i32) * sigma_fused.determinant();
                let eta =
                    (g_sum + 0.5 * mu_fused.dot(&(&k_sum * &mu_fused)) + 0.5 * det_2pi_sigma.ln())
                        .exp();

                // Geometric mean of existence
                let mut r_num = eta;
                let mut r_den = 1.0;
                for s in 0..num_sensors {
                    let r_s = per_sensor_tracks[s][i].existence;
                    r_num *= r_s.powf(weights[s]);
                    r_den *= (1.0 - r_s).powf(weights[s]);
                }
                let r_fused = r_num / (r_num + r_den);

                // Update track with single fused component
                fused_tracks[i].existence = r_fused;
                fused_tracks[i].components.clear();
                fused_tracks[i]
                    .components
                    .push(GaussianComponent::new(1.0, mu_fused, sigma_fused));
            }
        }

        fused_tracks
    }

    fn name(&self) -> &'static str {
        "GeometricAverage"
    }
}

// ============================================================================
// Parallel Update Merger
// ============================================================================

/// Parallel Update (PU) track merger.
///
/// Fuses per-sensor track posteriors using information-form fusion with
/// decorrelation. This is theoretically optimal for independent sensors.
///
/// The algorithm removes the utils prior contribution before fusion:
/// 1. Convert prior to canonical form
/// 2. K_fused = Σ K_sensor + (1-S) × K_prior  (decorrelation)
/// 3. Convert back and select max-weight component
///
/// PU requires access to the prior tracks and produces single-component results.
#[derive(Debug, Clone)]
pub struct MergerParallelUpdate {
    /// Prior tracks (before sensor updates)
    prior_tracks: Vec<Track>,
}

impl MergerParallelUpdate {
    /// Create with prior tracks for decorrelation.
    pub fn new(prior_tracks: Vec<Track>) -> Self {
        Self { prior_tracks }
    }

    /// Update prior tracks for next timestep.
    pub fn set_prior(&mut self, prior_tracks: Vec<Track>) {
        self.prior_tracks = prior_tracks;
    }
}

impl Merger for MergerParallelUpdate {
    fn set_prior(&mut self, prior_tracks: Vec<Track>) {
        self.prior_tracks = prior_tracks;
    }

    fn merge(&self, per_sensor_tracks: &[Vec<Track>], _weights: Option<&[f64]>) -> Vec<Track> {
        if per_sensor_tracks.is_empty() || per_sensor_tracks[0].is_empty() {
            return Vec::new();
        }

        let num_sensors = per_sensor_tracks.len();
        let num_tracks = per_sensor_tracks[0].len();

        let mut fused_tracks = self.prior_tracks.clone();
        let decorr_factor = (1 - num_sensors as i32) as f64;

        for i in 0..num_tracks {
            if i >= self.prior_tracks.len() || self.prior_tracks[i].components.is_empty() {
                continue;
            }

            let dim = self.prior_tracks[i].x_dim();

            // Get prior in canonical form (first component)
            let prior_comp = &self.prior_tracks[i].components[0];
            let prior_k = match prior_comp.covariance.clone().try_inverse() {
                Some(inv) => inv,
                None => continue,
            };
            let prior_h = &prior_k * &prior_comp.mean;

            // Prior's g contribution
            // MATLAB: gPrior = -0.5 * mu' * K * mu - 0.5 * log(det(2π*Σ))
            let det_2pi_prior =
                (2.0 * std::f64::consts::PI).powi(dim as i32) * prior_comp.covariance.determinant();
            let g_prior = -0.5 * prior_comp.mean.dot(&(&prior_k * &prior_comp.mean))
                - 0.5 * det_2pi_prior.ln();

            // Collect number of GM components per sensor for this track
            let num_components: Vec<usize> = per_sensor_tracks
                .iter()
                .map(|tracks| {
                    if i < tracks.len() {
                        tracks[i].components.len().max(1)
                    } else {
                        1
                    }
                })
                .collect();

            // Total number of Cartesian product combinations
            let total_combinations: usize = num_components.iter().product();

            // Build Cartesian product of sensor GM components
            // MATLAB: K = repmat({(1-S)*KPrior}, 1, prod(numberOfGmComponents))
            //         h = repmat({(1-S)*hPrior}, 1, prod(numberOfGmComponents))
            //         g = repmat((1-S)*gPrior, 1, prod(numberOfGmComponents))
            let mut k_vec: Vec<DMatrix<f64>> = vec![&prior_k * decorr_factor; total_combinations];
            let mut h_vec: Vec<DVector<f64>> = vec![&prior_h * decorr_factor; total_combinations];
            let mut g_vec: Vec<f64> = vec![decorr_factor * g_prior; total_combinations];

            // Iteratively build Cartesian product across sensors
            // MATLAB iterates: for s = 1:numberOfSensors, for j = 1:numberOfGmComponents(s)
            for s in 0..num_sensors {
                if i >= per_sensor_tracks[s].len() {
                    continue;
                }

                let sensor_track = &per_sensor_tracks[s][i];
                if sensor_track.components.is_empty() {
                    continue;
                }

                // Number of existing combinations before this sensor
                let current_mixture_size: usize = num_components[0..s].iter().product();

                // Clone current state to iterate from
                let k_prev = k_vec.clone();
                let h_prev = h_vec.clone();
                let g_prev = g_vec.clone();

                // For each component in this sensor
                let mut ell = 0;
                for j in 0..sensor_track.components.len() {
                    let comp = &sensor_track.components[j];

                    // Convert to canonical form
                    let k_c = match comp.covariance.clone().try_inverse() {
                        Some(inv) => inv,
                        None => continue,
                    };
                    let h_c = &k_c * &comp.mean;
                    let det_2pi_sigma = (2.0 * std::f64::consts::PI).powi(dim as i32)
                        * comp.covariance.determinant();
                    let g_c = -0.5 * comp.mean.dot(&(&k_c * &comp.mean)) - 0.5 * det_2pi_sigma.ln()
                        + comp.weight.ln();

                    // Add to each existing mixture from previous sensors
                    for k in 0..current_mixture_size {
                        if ell < total_combinations {
                            k_vec[ell] = &k_prev[k] + &k_c;
                            h_vec[ell] = &h_prev[k] + &h_c;
                            g_vec[ell] = g_prev[k] + g_c;
                            ell += 1;
                        }
                    }
                }
            }

            // Convert to covariance form and compute unnormalized weights
            let mut weights: Vec<f64> = Vec::with_capacity(total_combinations);
            let mut means: Vec<DVector<f64>> = Vec::with_capacity(total_combinations);
            let mut covariances: Vec<DMatrix<f64>> = Vec::with_capacity(total_combinations);

            for j in 0..total_combinations {
                if let Some(sigma) = k_vec[j].clone().try_inverse() {
                    let mu = &sigma * &h_vec[j];

                    // MATLAB: g(j) = g(j) + 0.5*h{j}'*T*h{j} + 0.5*log(det(2*pi*K{j}))
                    // where T = K (precision before inversion)
                    let det_2pi_sigma =
                        (2.0 * std::f64::consts::PI).powi(dim as i32) * sigma.determinant();
                    let g_final =
                        g_vec[j] + 0.5 * mu.dot(&(&k_vec[j] * &mu)) + 0.5 * det_2pi_sigma.ln();

                    weights.push(g_final.exp());
                    means.push(mu);
                    covariances.push(sigma);
                }
            }

            if weights.is_empty() {
                continue;
            }

            // Compute eta = sum(exp(g))
            // MATLAB: eta = sum(exp(g));
            let eta: f64 = weights.iter().sum();

            // Find max-weight component
            // MATLAB: [~, maxIndex] = max(w);
            let (max_idx, _) = weights
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let mu_fused = means[max_idx].clone();
            let sigma_fused = covariances[max_idx].clone();

            // Existence fusion with decorrelation
            // MATLAB: numerator = eta * (objects(i).r)^(1-S)
            //         partialDenominator = (1-objects(i).r)^(1-S)
            //         for s: numerator *= rS; partialDenominator *= (1-rS)
            //         r = numerator / (numerator + partialDenominator)
            let prior_r = self.prior_tracks[i].existence.clamp(1e-10, 1.0 - 1e-10);
            let mut r_num = eta * prior_r.powf(decorr_factor);
            let mut r_den = (1.0 - prior_r).powf(decorr_factor);

            for sensor_tracks in per_sensor_tracks.iter().take(num_sensors) {
                if i < sensor_tracks.len() {
                    let sensor_r = sensor_tracks[i].existence.clamp(1e-10, 1.0 - 1e-10);
                    r_num *= sensor_r;
                    r_den *= 1.0 - sensor_r;
                }
            }

            // Protect against NaN/inf from division
            let r_fused = if r_num.is_finite() && r_den.is_finite() && (r_num + r_den) > 0.0 {
                (r_num / (r_num + r_den)).clamp(0.0, 1.0)
            } else if r_num > r_den {
                1.0 - 1e-10
            } else {
                1e-10
            };

            // Update track with single fused component (max weight)
            fused_tracks[i].existence = r_fused;
            fused_tracks[i].components.clear();
            fused_tracks[i]
                .components
                .push(GaussianComponent::new(1.0, mu_fused, sigma_fused));
        }

        fused_tracks
    }

    fn name(&self) -> &'static str {
        "ParallelUpdate"
    }
}

// ============================================================================
// Iterated Corrector Merger
// ============================================================================

/// Iterated Corrector (IC) - sequential sensor updates.
///
/// Unlike true parallel fusion, IC processes sensors sequentially. Each sensor
/// update uses the output of the previous sensor as its prior.
///
/// This is simpler but order-dependent: sensor order affects the result.
/// IC is included for completeness and comparison with other strategies.
#[derive(Debug, Clone, Default)]
pub struct MergerIteratedCorrector;

impl MergerIteratedCorrector {
    /// Create a new iterated corrector merger.
    pub fn new() -> Self {
        Self
    }
}

impl Merger for MergerIteratedCorrector {
    fn merge(&self, per_sensor_tracks: &[Vec<Track>], _weights: Option<&[f64]>) -> Vec<Track> {
        // IC doesn't do parallel fusion - it processes sensors sequentially
        // Just return the last sensor's tracks (the final sequential result)
        if let Some(last) = per_sensor_tracks.last() {
            last.clone()
        } else {
            Vec::new()
        }
    }

    fn name(&self) -> &'static str {
        "IteratedCorrector"
    }

    fn is_sequential(&self) -> bool {
        true // IC-LMB processes sensors sequentially
    }
}
