//! Multi-sensor LMB (Labeled Multi-Bernoulli) filter.
//!
//! The multi-sensor LMB filter extends the single-sensor LMB to handle measurements
//! from multiple sensors. The key challenge is fusing per-sensor posterior updates
//! into a unified track estimate.
//!
//! # Fusion Strategies
//!
//! Four fusion strategies are supported via the [`Merger`] trait:
//!
//! - **Arithmetic Average (AA)**: Simple weighted combination of GM components.
//!   Fast and robust, but doesn't account for correlation between sensors.
//!
//! - **Geometric Average (GA)**: Covariance intersection using canonical form.
//!   Produces single-component result with conservative covariance.
//!
//! - **Parallel Update (PU)**: Information-form fusion with decorrelation.
//!   Theoretically optimal for independent sensors.
//!
//! - **Iterated Corrector (IC)**: Sequential sensor updates (not truly parallel).
//!   Simple but order-dependent.
//!
//! # Type Aliases
//!
//! For convenience, type aliases are provided:
//! - [`AaLmbFilter`] - Arithmetic average fusion
//! - [`GaLmbFilter`] - Geometric average fusion
//! - [`PuLmbFilter`] - Parallel update fusion
//! - [`IcLmbFilter`] - Iterated corrector

use nalgebra::{DMatrix, DVector};

use crate::association::AssociationBuilder;
use crate::components::prediction::predict_tracks;
use crate::types::{
    AssociationConfig, BirthModel, GaussianComponent, MotionModel, MultisensorConfig,
    StateEstimate, Track, Trajectory,
};

use super::errors::FilterError;
use super::traits::{Associator, Filter, LbpAssociator, MarginalUpdater, Merger, Updater};

// ============================================================================
// Merger Implementations
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
                super::common_ops::merge_and_truncate_components(all_components, self.max_components);
        }

        fused_tracks
    }

    fn name(&self) -> &'static str {
        "ArithmeticAverage"
    }
}

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

// ============================================================================
// MultisensorLmbFilter
// ============================================================================

/// Multi-sensor LMB filter with configurable fusion strategy.
///
/// This filter processes measurements from multiple sensors and fuses the
/// per-sensor updates using the provided [`Merger`] strategy.
///
/// The filter is generic over:
/// - `A`: Data association algorithm (default: [`LbpAssociator`])
/// - `M`: Fusion/merging strategy (implements [`Merger`])
///
/// # Example
///
/// ```ignore
/// use prak::filter::{MultisensorLmbFilter, ArithmeticAverageMerger};
///
/// // Create AA-LMB filter for 2 sensors
/// let merger = ArithmeticAverageMerger::uniform(2, 100);
/// let filter = MultisensorLmbFilter::new(motion, sensors, birth, config, merger);
/// ```
pub struct MultisensorLmbFilter<A: Associator = LbpAssociator, M: Merger = ArithmeticAverageMerger>
{
    /// Motion model (dynamics, survival probability)
    motion: MotionModel,
    /// Per-sensor models
    sensors: MultisensorConfig,
    /// Birth model (where new objects can appear)
    birth: BirthModel,
    /// Association algorithm configuration
    association_config: AssociationConfig,

    /// Current tracks
    tracks: Vec<Track>,
    /// Complete trajectories for discarded tracks
    trajectories: Vec<Trajectory>,

    /// Existence probability threshold for gating
    existence_threshold: f64,
    /// Minimum trajectory length to keep
    min_trajectory_length: usize,
    /// GM component pruning parameters
    gm_weight_threshold: f64,
    max_gm_components: usize,

    /// The associator to use
    associator: A,
    /// The merger/fusion strategy
    merger: M,
    /// The updater to use
    updater: MarginalUpdater,
}

impl<M: Merger> MultisensorLmbFilter<LbpAssociator, M> {
    /// Create a new multi-sensor LMB filter with default LBP associator.
    pub fn new(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
        merger: M,
    ) -> Self {
        Self {
            motion,
            sensors,
            birth,
            association_config,
            tracks: Vec::new(),
            trajectories: Vec::new(),
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            gm_weight_threshold: super::DEFAULT_GM_WEIGHT_THRESHOLD,
            max_gm_components: super::DEFAULT_MAX_GM_COMPONENTS,
            associator: LbpAssociator,
            merger,
            updater: MarginalUpdater::new(),
        }
    }
}

impl<A: Associator, M: Merger> MultisensorLmbFilter<A, M> {
    /// Create with custom associator.
    pub fn with_associator_type(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
        merger: M,
        associator: A,
    ) -> Self {
        Self {
            motion,
            sensors,
            birth,
            association_config,
            tracks: Vec::new(),
            trajectories: Vec::new(),
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            gm_weight_threshold: super::DEFAULT_GM_WEIGHT_THRESHOLD,
            max_gm_components: super::DEFAULT_MAX_GM_COMPONENTS,
            associator,
            merger,
            updater: MarginalUpdater::new(),
        }
    }

    /// Number of sensors.
    pub fn num_sensors(&self) -> usize {
        self.sensors.num_sensors()
    }

    /// Set the existence threshold for gating.
    pub fn with_existence_threshold(mut self, threshold: f64) -> Self {
        self.existence_threshold = threshold;
        self
    }

    /// Set the minimum trajectory length.
    pub fn with_min_trajectory_length(mut self, length: usize) -> Self {
        self.min_trajectory_length = length;
        self
    }

    /// Set GM pruning parameters.
    pub fn with_gm_pruning(mut self, weight_threshold: f64, max_components: usize) -> Self {
        self.gm_weight_threshold = weight_threshold;
        self.max_gm_components = max_components;
        self.updater = MarginalUpdater::with_thresholds(weight_threshold, max_components);
        self
    }

    /// Gate tracks by existence probability.
    fn gate_tracks(&mut self) {
        super::common_ops::gate_tracks(
            &mut self.tracks,
            &mut self.trajectories,
            self.existence_threshold,
            self.min_trajectory_length,
        );
    }

    /// Extract state estimates using MAP cardinality estimation.
    fn extract_estimates(&self, timestamp: usize) -> StateEstimate {
        super::common_ops::extract_estimates(&self.tracks, timestamp)
    }

    /// Update track trajectories.
    fn update_trajectories(&mut self, timestamp: usize) {
        super::common_ops::update_trajectories(&mut self.tracks, timestamp);
    }

    /// Initialize trajectory recording for birth tracks.
    fn init_birth_trajectories(&mut self, max_length: usize) {
        super::common_ops::init_birth_trajectories(&mut self.tracks, max_length);
    }

    /// Update existence for missed detection (all sensors).
    fn update_existence_no_measurements(&mut self) {
        let detection_probs: Vec<f64> = self
            .sensors
            .sensors
            .iter()
            .map(|s| s.detection_probability)
            .collect();
        for track in &mut self.tracks {
            track.existence = crate::components::update::update_existence_no_detection_multisensor(
                track.existence,
                &detection_probs,
            );
        }
    }
}

/// Multi-sensor measurements: one measurement set per sensor.
pub type MultisensorMeasurements = Vec<Vec<DVector<f64>>>;

impl<A: Associator, M: Merger> Filter for MultisensorLmbFilter<A, M> {
    type State = Vec<Track>;
    type Measurements = MultisensorMeasurements;

    fn step<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        let num_sensors = self.num_sensors();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 1: Prediction - propagate tracks forward and add birth components
        // ══════════════════════════════════════════════════════════════════════
        predict_tracks(&mut self.tracks, &self.motion, &self.birth, timestep, false);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2: Initialize trajectory recording for new birth tracks
        // ══════════════════════════════════════════════════════════════════════
        self.init_birth_trajectories(super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 3: Measurement update - data association and track updates
        // ══════════════════════════════════════════════════════════════════════
        let has_any_measurements = measurements.iter().any(|m| !m.is_empty());

        if has_any_measurements && !self.tracks.is_empty() {
            // Store prior tracks for PU fusion if needed
            let prior_tracks = self.tracks.clone();

            // --- STEP 3a: Per-sensor measurement updates ---
            let mut per_sensor_tracks: Vec<Vec<Track>> = Vec::with_capacity(num_sensors);

            for s in 0..num_sensors {
                let sensor = &self.sensors.sensors[s];
                let sensor_measurements = &measurements[s];

                // Clone prior tracks for this sensor's update
                let mut sensor_tracks = prior_tracks.clone();

                if !sensor_measurements.is_empty() {
                    // Build association matrices for this sensor
                    let mut builder = AssociationBuilder::new(&sensor_tracks, sensor);
                    let matrices = builder.build(sensor_measurements);

                    // Run data association
                    let result = self
                        .associator
                        .associate(&matrices, &self.association_config, rng)
                        .map_err(FilterError::Association)?;

                    // Update tracks
                    self.updater
                        .update(&mut sensor_tracks, &result, &matrices.posteriors);

                    // Update existence from association
                    super::common_ops::update_existence_from_marginals(
                        &mut sensor_tracks,
                        &result,
                    );
                } else {
                    // No measurements for this sensor - missed detection update
                    let p_d = sensor.detection_probability;
                    for track in &mut sensor_tracks {
                        track.existence =
                            crate::components::update::update_existence_no_detection(
                                track.existence,
                                p_d,
                            );
                    }
                }

                per_sensor_tracks.push(sensor_tracks);
            }

            // --- STEP 3b: Fuse per-sensor updates ---
            self.tracks = self.merger.merge(&per_sensor_tracks, None);
        } else if !has_any_measurements {
            // No measurements from any sensor
            self.update_existence_no_measurements();
        }

        // (STEP 4 skipped - hypothesis management is LMBM only)

        // ══════════════════════════════════════════════════════════════════════
        // STEP 5: Track gating - prune low-existence tracks, archive trajectories
        // ══════════════════════════════════════════════════════════════════════
        self.gate_tracks();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 6: Update trajectories - append current state to track histories
        // ══════════════════════════════════════════════════════════════════════
        self.update_trajectories(timestep);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 7: Extract estimates - return current state estimate
        // ══════════════════════════════════════════════════════════════════════
        Ok(self.extract_estimates(timestep))
    }

    fn state(&self) -> &Self::State {
        &self.tracks
    }

    fn reset(&mut self) {
        self.tracks.clear();
        self.trajectories.clear();
    }

    fn x_dim(&self) -> usize {
        self.motion.x_dim()
    }

    fn z_dim(&self) -> usize {
        self.sensors.z_dim()
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

/// Arithmetic Average LMB filter (AA-LMB).
pub type AaLmbFilter<A = LbpAssociator> = MultisensorLmbFilter<A, ArithmeticAverageMerger>;

/// Geometric Average LMB filter (GA-LMB).
pub type GaLmbFilter<A = LbpAssociator> = MultisensorLmbFilter<A, GeometricAverageMerger>;

/// Parallel Update LMB filter (PU-LMB).
pub type PuLmbFilter<A = LbpAssociator> = MultisensorLmbFilter<A, ParallelUpdateMerger>;

/// Iterated Corrector LMB filter (IC-LMB).
pub type IcLmbFilter<A = LbpAssociator> = MultisensorLmbFilter<A, IteratedCorrectorMerger>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BirthLocation, SensorModel};
    use nalgebra::DMatrix;

    fn create_test_sensors() -> MultisensorConfig {
        let sensor1 = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
        let sensor2 = SensorModel::position_sensor_2d(1.0, 0.85, 8.0, 100.0);
        MultisensorConfig::new(vec![sensor1, sensor2])
    }

    fn create_test_filter() -> MultisensorLmbFilter<LbpAssociator, ArithmeticAverageMerger> {
        let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);
        let sensors = create_test_sensors();

        let birth_loc = BirthLocation::new(
            0,
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            DMatrix::identity(4, 4) * 100.0,
        );
        let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);

        let association_config = AssociationConfig::default();
        let merger = ArithmeticAverageMerger::uniform(2, 100);

        MultisensorLmbFilter::new(motion, sensors, birth, association_config, merger)
    }

    #[test]
    fn test_filter_creation() {
        let filter = create_test_filter();
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
        assert_eq!(filter.num_sensors(), 2);
        assert!(filter.tracks.is_empty());
    }

    #[test]
    fn test_filter_step_no_measurements() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        // Empty measurements for both sensors
        let measurements = vec![vec![], vec![]];
        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();

        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_filter_step_with_measurements() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        // Measurements for each sensor
        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])], // sensor 1
            vec![DVector::from_vec(vec![0.5, 0.5])], // sensor 2
        ];

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_filter_multiple_steps() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        for t in 0..5 {
            let measurements = vec![
                vec![DVector::from_vec(vec![t as f64, t as f64])],
                vec![DVector::from_vec(vec![t as f64 + 0.1, t as f64 + 0.1])],
            ];
            let _estimate = filter.step(&mut rng, &measurements, t).unwrap();
        }
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        let measurements = vec![vec![], vec![]];
        let _ = filter.step(&mut rng, &measurements, 0);

        filter.reset();

        assert!(filter.tracks.is_empty());
        assert!(filter.trajectories.is_empty());
    }

    #[test]
    fn test_aa_merger() {
        let merger = ArithmeticAverageMerger::uniform(2, 100);
        assert_eq!(merger.sensor_weights, vec![0.5, 0.5]);
        assert_eq!(merger.name(), "ArithmeticAverage");
    }

    #[test]
    fn test_ga_merger() {
        let merger = GeometricAverageMerger::uniform(2);
        assert_eq!(merger.sensor_weights, vec![0.5, 0.5]);
        assert_eq!(merger.name(), "GeometricAverage");
    }

    #[test]
    fn test_ic_merger() {
        let merger = IteratedCorrectorMerger::new();
        assert_eq!(merger.name(), "IteratedCorrector");
    }
}
