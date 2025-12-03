//! Multi-sensor track fusion strategies.
//!
//! This module contains implementations of the [`Merger`] trait for fusing
//! per-sensor track posteriors into unified estimates.
//!
//! # Available Strategies
//!
//! - [`ArithmeticAverageMerger`] - Simple weighted average (fast, robust)
//! - [`GeometricAverageMerger`] - Covariance intersection (conservative)
//! - [`ParallelUpdateMerger`] - Information-form fusion (optimal for independent sensors)
//! - [`IteratedCorrectorMerger`] - Sequential sensor updates (order-dependent)

use nalgebra::{DMatrix, DVector};

use super::super::types::{GaussianComponent, Track};
use super::super::traits::Merger;

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
pub struct ArithmeticAverageMerger {
    /// Per-sensor fusion weights (should sum to 1.0)
    pub sensor_weights: Vec<f64>,
    /// Maximum GM components to keep after fusion
    pub max_components: usize,
}

impl ArithmeticAverageMerger {
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

impl Merger for ArithmeticAverageMerger {
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

            // Merge, truncate, and normalize using shared helper
            fused_tracks[i].components =
                super::super::common_ops::merge_and_truncate_components(all_components, self.max_components);
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
pub struct GeometricAverageMerger {
    /// Per-sensor fusion weights (should sum to 1.0)
    pub sensor_weights: Vec<f64>,
}

impl GeometricAverageMerger {
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

impl Merger for GeometricAverageMerger {
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
            if let Some(sigma_fused) = k_sum.try_inverse() {
                let mu_fused = &sigma_fused * &h_sum;
                let eta = g_sum.exp();

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
/// The algorithm removes the common prior contribution before fusion:
/// 1. Convert prior to canonical form
/// 2. K_fused = Σ K_sensor + (1-S) × K_prior  (decorrelation)
/// 3. Convert back and select max-weight component
///
/// PU requires access to the prior tracks and produces single-component results.
#[derive(Debug, Clone)]
pub struct ParallelUpdateMerger {
    /// Prior tracks (before sensor updates)
    prior_tracks: Vec<Track>,
}

impl ParallelUpdateMerger {
    /// Create with prior tracks for decorrelation.
    pub fn new(prior_tracks: Vec<Track>) -> Self {
        Self { prior_tracks }
    }

    /// Update prior tracks for next timestep.
    pub fn set_prior(&mut self, prior_tracks: Vec<Track>) {
        self.prior_tracks = prior_tracks;
    }
}

impl Merger for ParallelUpdateMerger {
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

            // Build Cartesian product of sensor components
            // For simplicity, use first component from each sensor
            let mut k_fused = &prior_k * decorr_factor;
            let mut h_fused = &prior_h * decorr_factor;
            let mut g_fused = 0.0;

            for s in 0..num_sensors {
                if i >= per_sensor_tracks[s].len() || per_sensor_tracks[s][i].components.is_empty()
                {
                    continue;
                }

                let comp = &per_sensor_tracks[s][i].components[0];
                if let Some(k_s) = comp.covariance.clone().try_inverse() {
                    let h_s = &k_s * &comp.mean;

                    // g for this component
                    let det_2pi_sigma =
                        (2.0 * std::f64::consts::PI).powi(dim as i32) * comp.covariance.determinant();
                    let g_s = -0.5 * comp.mean.dot(&(&k_s * &comp.mean))
                        - 0.5 * det_2pi_sigma.ln()
                        + comp.weight.ln();

                    k_fused += k_s;
                    h_fused += h_s;
                    g_fused += g_s;
                }
            }

            // Convert back to moment form
            if let Some(sigma_fused) = k_fused.try_inverse() {
                let mu_fused = &sigma_fused * &h_fused;
                let eta = g_fused.exp();

                // Existence fusion with decorrelation
                let prior_r = self.prior_tracks[i].existence;
                let mut r_num = eta * prior_r.powf(decorr_factor);
                let mut r_den = (1.0 - prior_r).powf(decorr_factor);

                for s in 0..num_sensors {
                    if i < per_sensor_tracks[s].len() {
                        r_num *= per_sensor_tracks[s][i].existence;
                        r_den *= 1.0 - per_sensor_tracks[s][i].existence;
                    }
                }
                let r_fused = r_num / (r_num + r_den);

                // Update track
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
pub struct IteratedCorrectorMerger;

impl IteratedCorrectorMerger {
    /// Create a new iterated corrector merger.
    pub fn new() -> Self {
        Self
    }
}

impl Merger for IteratedCorrectorMerger {
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
}
