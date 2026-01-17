//! Unified LMBM filter core.
//!
//! This module provides the [`LmbmFilterCore`] struct, a generic LMBM filter
//! implementation parameterized by the LMBM associator type. This unifies the
//! single-sensor and multi-sensor LMBM filter implementations.
//!
//! # Architecture
//!
//! Unlike LMB filters where single/multi-sensor use the same [`Associator`] trait
//! (with different schedulers), LMBM filters have fundamentally different association
//! patterns:
//! - Single-sensor: 2D cost matrix → sampled assignments
//! - Multi-sensor: N-dimensional Cartesian product tensor → joint samples
//!
//! This module abstracts over these differences via the [`LmbmAssociator`] trait,
//! which combines association AND posterior hypothesis generation into one operation.
//!
//! # Type Aliases
//!
//! - [`LmbmFilter`] - Single-sensor LMBM filter
//! - [`MultisensorLmbmFilter`] - Multi-sensor LMBM filter
//!
//! [`Associator`]: super::traits::Associator

use nalgebra::{DMatrix, DVector};

use crate::association::{AssociationBuilder, AssociationMatrices};
use crate::common::linalg::{log_gaussian_normalizing_constant, robust_inverse};

use super::builder::FilterBuilder;
use super::config::{
    AssociationConfig, BirthModel, FilterConfigSnapshot, LmbmConfig, MotionModel,
    MultisensorConfig, SensorSet,
};
use super::errors::FilterError;
use super::multisensor::traits::{MultisensorAssociator, MultisensorGibbsAssociator};
use super::multisensor::MultisensorMeasurements;
use super::output::{StateEstimate, Trajectory};
use super::traits::{
    AssociationResult, Associator, Filter, GibbsAssociator, HardAssignmentUpdater, Updater,
};
use super::types::{GaussianComponent, LmbmHypothesis, StepDetailedOutput, Track};

/// Log-likelihood floor to prevent underflow when computing ln(x) for very small x.
const LOG_UNDERFLOW: f64 = -700.0;

// ============================================================================
// LmbmAssociator trait - Unified association for LMBM filters
// ============================================================================

/// Unified LMBM association trait.
///
/// This trait abstracts over the different association patterns used by single-sensor
/// and multi-sensor LMBM filters. It combines association AND posterior hypothesis
/// generation into one operation, since the inputs and logic differ significantly.
///
/// # Implementation
///
/// Two implementations are provided:
/// - [`SingleSensorLmbmStrategy`] - Wraps any [`Associator`] for single-sensor use
/// - [`MultisensorLmbmStrategy`] - Wraps any [`MultisensorAssociator`] for multi-sensor use
///
/// # Type Parameters
///
/// Associated type `Measurements` determines the measurement format:
/// - Single-sensor: `Vec<DVector<f64>>`
/// - Multi-sensor: `Vec<Vec<DVector<f64>>>`
pub trait LmbmAssociator: Send + Sync {
    /// Type of measurements this associator accepts.
    type Measurements;

    /// Perform association and generate posterior hypotheses.
    ///
    /// This method handles the entire association pipeline:
    /// 1. Build association matrices/tensors from tracks and measurements
    /// 2. Run the underlying association algorithm
    /// 3. Generate posterior hypotheses from samples
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `hypotheses` - Current hypotheses (modified in place)
    /// * `measurements` - Sensor measurements
    /// * `sensor_config` - Sensor configuration
    /// * `motion` - Motion model (for state dimension)
    /// * `association_config` - Association algorithm configuration
    ///
    /// # Returns
    ///
    /// Ok(()) on success, with `hypotheses` updated to posterior hypotheses.
    fn associate_and_update<R: rand::Rng>(
        &self,
        rng: &mut R,
        hypotheses: &mut Vec<LmbmHypothesis>,
        measurements: &Self::Measurements,
        sensor_config: &SensorSet,
        motion: &MotionModel,
        association_config: &AssociationConfig,
    ) -> Result<Option<LmbmAssociationIntermediate>, FilterError>;

    /// Update existence probabilities when there are no measurements.
    fn update_existence_no_measurements(
        &self,
        hypotheses: &mut Vec<LmbmHypothesis>,
        sensor_config: &SensorSet,
    );

    /// Check if measurements are empty.
    fn measurements_empty(&self, measurements: &Self::Measurements) -> bool;

    /// Validate measurements match expected sensor count.
    fn validate_measurements(
        &self,
        measurements: &Self::Measurements,
        sensor_config: &SensorSet,
    ) -> Result<(), FilterError>;

    /// Algorithm name for logging and debugging.
    fn name(&self) -> &'static str;
}

/// Intermediate association data for detailed step output.
///
/// Used for fixture validation in single-sensor LMBM.
#[derive(Debug, Clone)]
pub struct LmbmAssociationIntermediate {
    /// Association matrices (single-sensor only).
    pub matrices: Option<AssociationMatrices>,
    /// Association result (single-sensor only).
    pub result: Option<AssociationResult>,
}

// ============================================================================
// SingleSensorLmbmStrategy - Wraps Associator for single-sensor LMBM
// ============================================================================

/// Single-sensor LMBM association strategy.
///
/// Wraps any [`Associator`] for use in single-sensor LMBM filters.
/// Uses the standard 2D cost matrix approach with sampled hard assignments.
#[derive(Debug, Clone, Default)]
pub struct SingleSensorLmbmStrategy<A: Associator = GibbsAssociator> {
    pub(crate) associator: A,
}

impl<A: Associator> SingleSensorLmbmStrategy<A> {
    /// Build log-likelihood matrix for computing hypothesis weights.
    ///
    /// The matrix is (n × (m+1)) where:
    /// - First column: log P(miss | track i)
    /// - Remaining columns: log P(measurement j | track i)
    fn build_log_likelihood_matrix(matrices: &AssociationMatrices) -> DMatrix<f64> {
        let n = matrices.eta.len();
        let m = matrices.psi.ncols();

        let mut log_likelihood = DMatrix::zeros(n, m + 1);

        for i in 0..n {
            // Miss column: log(eta_i)
            log_likelihood[(i, 0)] = if matrices.eta[i] > super::UNDERFLOW_THRESHOLD {
                matrices.eta[i].ln()
            } else {
                LOG_UNDERFLOW
            };

            // Measurement columns: log(eta_i * psi_ij)
            for j in 0..m {
                let likelihood_ij = matrices.eta[i] * matrices.psi[(i, j)];
                log_likelihood[(i, j + 1)] = if likelihood_ij > super::UNDERFLOW_THRESHOLD {
                    likelihood_ij.ln()
                } else {
                    LOG_UNDERFLOW
                };
            }
        }

        log_likelihood
    }

    /// Generate posterior hypotheses from sampled associations.
    fn generate_posterior_hypotheses(
        hypotheses: &mut Vec<LmbmHypothesis>,
        result: &AssociationResult,
        matrices: &AssociationMatrices,
        log_likelihoods: &DMatrix<f64>,
    ) {
        let samples = match &result.sampled_associations {
            Some(s) if !s.is_empty() => s,
            _ => return,
        };

        let miss_posterior_r = matrices.miss_posterior_existence();
        let mut new_hypotheses = Vec::new();

        for prior_hyp in hypotheses.iter() {
            for (sample_idx, assignments) in samples.iter().enumerate() {
                let mut new_hyp = prior_hyp.clone();

                // Set ALL tracks to miss-posterior existence first
                for (track_idx, track) in new_hyp.tracks.iter_mut().enumerate() {
                    if track_idx < miss_posterior_r.len() {
                        track.existence = miss_posterior_r[track_idx];
                    }
                }

                // Compute log weight contribution from this assignment
                let mut log_likelihood_sum = 0.0;
                for (track_idx, &meas_assignment) in assignments.iter().enumerate() {
                    if track_idx < new_hyp.tracks.len() {
                        let col_idx = if meas_assignment < 0 {
                            0
                        } else {
                            (meas_assignment + 1) as usize
                        };
                        if col_idx < log_likelihoods.ncols() {
                            log_likelihood_sum += log_likelihoods[(track_idx, col_idx)];
                        }
                    }
                }

                new_hyp.log_weight += log_likelihood_sum;

                // Apply hard assignment updates to tracks
                let updater = HardAssignmentUpdater::with_sample_index(sample_idx);
                updater.update(&mut new_hyp.tracks, result, &matrices.posteriors);

                // Update existence for detected tracks
                for (track_idx, &meas_assignment) in assignments.iter().enumerate() {
                    if track_idx < new_hyp.tracks.len() && meas_assignment >= 0 {
                        new_hyp.tracks[track_idx].existence = 1.0;
                    }
                }

                new_hypotheses.push(new_hyp);
            }
        }

        *hypotheses = new_hypotheses;
    }
}

impl<A: Associator> LmbmAssociator for SingleSensorLmbmStrategy<A> {
    type Measurements = Vec<DVector<f64>>;

    fn associate_and_update<R: rand::Rng>(
        &self,
        rng: &mut R,
        hypotheses: &mut Vec<LmbmHypothesis>,
        measurements: &Self::Measurements,
        sensor_config: &SensorSet,
        _motion: &MotionModel,
        association_config: &AssociationConfig,
    ) -> Result<Option<LmbmAssociationIntermediate>, FilterError> {
        if hypotheses.is_empty() || hypotheses[0].tracks.is_empty() {
            return Ok(None);
        }

        let sensor = sensor_config.single();
        let tracks = &hypotheses[0].tracks;

        let mut builder = AssociationBuilder::new(tracks, sensor);
        let matrices = builder.build(measurements);
        let log_likelihood = Self::build_log_likelihood_matrix(&matrices);

        let result = self
            .associator
            .associate(&matrices, association_config, rng)
            .map_err(FilterError::Association)?;

        Self::generate_posterior_hypotheses(hypotheses, &result, &matrices, &log_likelihood);

        Ok(Some(LmbmAssociationIntermediate {
            matrices: Some(matrices),
            result: Some(result),
        }))
    }

    fn update_existence_no_measurements(
        &self,
        hypotheses: &mut Vec<LmbmHypothesis>,
        sensor_config: &SensorSet,
    ) {
        let p_d = sensor_config.single().detection_probability;
        for hyp in hypotheses.iter_mut() {
            for track in &mut hyp.tracks {
                track.existence =
                    crate::components::update::update_existence_no_detection(track.existence, p_d);
            }
        }
    }

    fn measurements_empty(&self, measurements: &Self::Measurements) -> bool {
        measurements.is_empty()
    }

    fn validate_measurements(
        &self,
        _measurements: &Self::Measurements,
        sensor_config: &SensorSet,
    ) -> Result<(), FilterError> {
        if sensor_config.num_sensors() != 1 {
            return Err(FilterError::InvalidInput(format!(
                "SingleSensorLmbmStrategy requires 1 sensor, got {}",
                sensor_config.num_sensors()
            )));
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SingleSensorLmbm"
    }
}

// ============================================================================
// MultisensorLmbmStrategy - Wraps MultisensorAssociator for multi-sensor LMBM
// ============================================================================

/// Multi-sensor LMBM association strategy.
///
/// Wraps any [`MultisensorAssociator`] for use in multi-sensor LMBM filters.
/// Uses the Cartesian product tensor approach with joint association samples.
#[derive(Debug, Clone, Default)]
pub struct MultisensorLmbmStrategy<A: MultisensorAssociator = MultisensorGibbsAssociator> {
    pub(crate) associator: A,
}

/// Posterior parameters for a single entry in the flattened likelihood tensor.
#[derive(Clone)]
struct MultisensorPosterior {
    existence: f64,
    mean: DVector<f64>,
    covariance: DMatrix<f64>,
}

impl<A: MultisensorAssociator> MultisensorLmbmStrategy<A> {
    /// Convert linear index to Cartesian coordinates (MATLAB-style, 1-indexed).
    fn linear_to_cartesian(mut ell: usize, page_sizes: &[usize]) -> Vec<usize> {
        let m = page_sizes.len();
        let mut u = vec![0; m];

        for i in 0..m {
            let j = m - i - 1;
            let zeta = ell / page_sizes[j];
            let eta = ell % page_sizes[j];
            u[j] = zeta + if eta != 0 { 1 } else { 0 };
            ell -= page_sizes[j] * (zeta - if eta == 0 { 1 } else { 0 });
        }

        u
    }

    /// Convert Cartesian coordinates to linear index (MATLAB-style, 1-indexed).
    fn cartesian_to_linear(u: &[usize], dimensions: &[usize]) -> usize {
        let mut ell = u[0];
        let mut pi = 1;

        for i in 1..u.len() {
            pi *= dimensions[i - 1];
            ell += pi * (u[i] - 1);
        }

        ell - 1
    }

    /// Generate multi-sensor association matrices.
    fn generate_association_matrices(
        tracks: &[Track],
        measurements: &MultisensorMeasurements,
        sensors: &MultisensorConfig,
        motion: &MotionModel,
    ) -> (Vec<f64>, Vec<MultisensorPosterior>, Vec<usize>) {
        let num_sensors = sensors.num_sensors();
        let num_objects = tracks.len();
        let x_dim = motion.x_dim();

        // Dimensions: [m_1+1, ..., m_S+1, n]
        let mut dimensions = vec![0; num_sensors + 1];
        for s in 0..num_sensors {
            dimensions[s] = measurements[s].len() + 1;
        }
        dimensions[num_sensors] = num_objects;

        let num_entries: usize = dimensions.iter().product();

        // Page sizes for index conversion
        let mut page_sizes = vec![1; num_sensors + 1];
        for i in 1..=num_sensors {
            page_sizes[i] = page_sizes[i - 1] * dimensions[i - 1];
        }

        let mut log_likelihoods = vec![0.0; num_entries];
        let mut posteriors = vec![
            MultisensorPosterior {
                existence: 0.0,
                mean: DVector::zeros(x_dim),
                covariance: DMatrix::zeros(x_dim, x_dim),
            };
            num_entries
        ];

        for ell in 0..num_entries {
            let u = Self::linear_to_cartesian(ell + 1, &page_sizes);
            let obj_idx = u[num_sensors] - 1;
            let associations: Vec<usize> = u[0..num_sensors].iter().map(|&x| x - 1).collect();

            let (log_l, posterior) = Self::compute_log_likelihood(
                obj_idx,
                &associations,
                tracks,
                measurements,
                sensors,
                motion,
            );

            log_likelihoods[ell] = log_l;
            posteriors[ell] = posterior;
        }

        (log_likelihoods, posteriors, dimensions)
    }

    /// Compute log-likelihood and posterior for a single object-association pair.
    fn compute_log_likelihood(
        obj_idx: usize,
        associations: &[usize],
        tracks: &[Track],
        measurements: &MultisensorMeasurements,
        sensors: &MultisensorConfig,
        motion: &MotionModel,
    ) -> (f64, MultisensorPosterior) {
        let track = &tracks[obj_idx];
        let (prior_mean, prior_cov) = match (track.primary_mean(), track.primary_covariance()) {
            (Some(m), Some(c)) => (m.clone(), c.clone()),
            _ => {
                return (
                    f64::NEG_INFINITY,
                    MultisensorPosterior {
                        existence: 0.0,
                        mean: DVector::zeros(motion.x_dim()),
                        covariance: DMatrix::identity(motion.x_dim(), motion.x_dim()),
                    },
                );
            }
        };

        let num_sensors = associations.len();
        let detecting: Vec<bool> = associations.iter().map(|&a| a > 0).collect();
        let num_detections: usize = detecting.iter().filter(|&&x| x).count();

        if num_detections > 0 {
            let z_dim = sensors.sensors[0].z_dim();
            let z_dim_total = z_dim * num_detections;
            let x_dim = motion.x_dim();

            let mut z = DVector::zeros(z_dim_total);
            let mut c = DMatrix::zeros(z_dim_total, x_dim);
            let mut q_blocks = Vec::new();

            let mut counter = 0;
            for s in 0..num_sensors {
                if detecting[s] {
                    let sensor = &sensors.sensors[s];
                    let meas_idx = associations[s] - 1;
                    let start = z_dim * counter;

                    z.rows_mut(start, z_dim)
                        .copy_from(&measurements[s][meas_idx]);
                    c.view_mut((start, 0), (z_dim, x_dim))
                        .copy_from(&sensor.observation_matrix);
                    q_blocks.push(sensor.measurement_noise.clone());
                    counter += 1;
                }
            }

            let mut q = DMatrix::zeros(z_dim_total, z_dim_total);
            let mut offset = 0;
            for q_block in &q_blocks {
                q.view_mut((offset, offset), (z_dim, z_dim))
                    .copy_from(q_block);
                offset += z_dim;
            }

            let nu = &z - &c * &prior_mean;
            let s_mat = &c * &prior_cov * c.transpose() + &q;

            let s_inv = match robust_inverse(&s_mat) {
                Some(inv) => inv,
                None => {
                    return (
                        f64::NEG_INFINITY,
                        MultisensorPosterior {
                            existence: 0.0,
                            mean: prior_mean,
                            covariance: prior_cov,
                        },
                    );
                }
            };

            let k = &prior_cov * c.transpose() * &s_inv;
            let log_eta = log_gaussian_normalizing_constant(&s_mat, z_dim_total);

            let mut log_pd = 0.0;
            for (sensor, &is_detecting) in sensors.sensors.iter().zip(detecting.iter()) {
                let p_d = sensor.detection_probability;
                log_pd += if is_detecting {
                    p_d.ln()
                } else {
                    (1.0 - p_d).ln()
                };
            }

            let log_kappa: f64 = detecting
                .iter()
                .enumerate()
                .filter(|(_, &d)| d)
                .map(|(s, _)| sensors.sensors[s].clutter_density().ln())
                .sum();

            let log_l =
                track.existence.ln() + log_pd + log_eta - 0.5 * nu.dot(&(&s_inv * &nu)) - log_kappa;

            let post_mean = &prior_mean + &k * &nu;
            let post_cov = (DMatrix::identity(x_dim, x_dim) - &k * &c) * &prior_cov;

            (
                log_l,
                MultisensorPosterior {
                    existence: 1.0,
                    mean: post_mean,
                    covariance: post_cov,
                },
            )
        } else {
            let mut prob_no_detect = 1.0;
            for sensor in &sensors.sensors {
                prob_no_detect *= 1.0 - sensor.detection_probability;
            }

            let r = track.existence;
            let numerator = r * prob_no_detect;
            let denominator = 1.0 - r + numerator;

            let log_l = denominator.ln();
            let post_r = numerator / denominator;

            (
                log_l,
                MultisensorPosterior {
                    existence: post_r,
                    mean: prior_mean,
                    covariance: prior_cov,
                },
            )
        }
    }

    /// Generate posterior hypotheses from association samples.
    fn generate_posterior_hypotheses(
        hypotheses: &mut Vec<LmbmHypothesis>,
        samples: &[Vec<usize>],
        log_likelihoods: &[f64],
        posteriors: &[MultisensorPosterior],
        dimensions: &[usize],
    ) {
        if samples.is_empty() {
            return;
        }

        let num_sensors = dimensions.len() - 1;
        let num_objects = dimensions[num_sensors];
        let mut new_hypotheses = Vec::new();

        for prior_hyp in hypotheses.iter() {
            for sample in samples {
                let mut new_hyp = prior_hyp.clone();
                let mut log_likelihood_sum = 0.0;

                for i in 0..num_objects.min(new_hyp.tracks.len()) {
                    let mut u: Vec<usize> = Vec::with_capacity(num_sensors + 1);
                    for s in 0..num_sensors {
                        let v_is = sample[s * num_objects + i];
                        u.push(v_is + 1);
                    }
                    u.push(i + 1);

                    let ell = Self::cartesian_to_linear(&u, dimensions);
                    log_likelihood_sum += log_likelihoods[ell];

                    let posterior = &posteriors[ell];
                    new_hyp.tracks[i].existence = posterior.existence;
                    new_hyp.tracks[i].components.clear();
                    new_hyp.tracks[i].components.push(GaussianComponent {
                        weight: 1.0,
                        mean: posterior.mean.clone(),
                        covariance: posterior.covariance.clone(),
                    });
                }

                new_hyp.log_weight += log_likelihood_sum;
                new_hypotheses.push(new_hyp);
            }
        }

        *hypotheses = new_hypotheses;
    }
}

impl<A: MultisensorAssociator> LmbmAssociator for MultisensorLmbmStrategy<A> {
    type Measurements = MultisensorMeasurements;

    fn associate_and_update<R: rand::Rng>(
        &self,
        rng: &mut R,
        hypotheses: &mut Vec<LmbmHypothesis>,
        measurements: &Self::Measurements,
        sensor_config: &SensorSet,
        motion: &MotionModel,
        association_config: &AssociationConfig,
    ) -> Result<Option<LmbmAssociationIntermediate>, FilterError> {
        if hypotheses.is_empty() {
            return Ok(None);
        }

        let sensors = sensor_config.multi();
        let tracks = &hypotheses[0].tracks;

        let (log_likelihoods, posteriors, dimensions) =
            Self::generate_association_matrices(tracks, measurements, sensors, motion);

        let association_result = self
            .associator
            .associate(rng, &log_likelihoods, &dimensions, association_config)
            .map_err(FilterError::Association)?;

        Self::generate_posterior_hypotheses(
            hypotheses,
            &association_result.samples,
            &log_likelihoods,
            &posteriors,
            &dimensions,
        );

        Ok(None) // Multi-sensor doesn't expose intermediate data
    }

    fn update_existence_no_measurements(
        &self,
        hypotheses: &mut Vec<LmbmHypothesis>,
        sensor_config: &SensorSet,
    ) {
        let detection_probs = sensor_config.detection_probabilities();
        for hyp in hypotheses.iter_mut() {
            for track in &mut hyp.tracks {
                track.existence =
                    crate::components::update::update_existence_no_detection_multisensor(
                        track.existence,
                        &detection_probs,
                    );
            }
        }
    }

    fn measurements_empty(&self, measurements: &Self::Measurements) -> bool {
        measurements.iter().all(|m| m.is_empty())
    }

    fn validate_measurements(
        &self,
        measurements: &Self::Measurements,
        sensor_config: &SensorSet,
    ) -> Result<(), FilterError> {
        let expected = sensor_config.num_sensors();
        if measurements.len() != expected {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                expected,
                measurements.len()
            )));
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "MultisensorLmbm"
    }
}

// ============================================================================
// LmbmFilterCore
// ============================================================================

/// Generic LMBM filter core parameterized by association strategy.
///
/// This struct unifies single-sensor and multi-sensor LMBM filter implementations.
/// The association strategy determines whether it operates in single or multi-sensor mode.
///
/// # Type Parameters
///
/// * `S` - The LMBM association strategy (determines measurement type and association logic)
///
/// # Example
///
/// ```ignore
/// use multisensor_lmb_filters_rs::lmb::{LmbmFilter, MultisensorLmbmFilter};
///
/// // Single-sensor LMBM (type alias)
/// let filter = LmbmFilter::new(motion, sensor, birth, assoc_config, lmbm_config);
///
/// // Multi-sensor LMBM (type alias)
/// let filter = MultisensorLmbmFilter::new(motion, sensors, birth, assoc_config, lmbm_config);
/// ```
pub struct LmbmFilterCore<S: LmbmAssociator> {
    /// Motion model (dynamics, survival probability)
    motion: MotionModel,
    /// Sensor configuration
    sensors: SensorSet,
    /// Birth model (where new objects can appear)
    birth: BirthModel,
    /// Association algorithm configuration
    association_config: AssociationConfig,
    /// LMBM-specific configuration
    lmbm_config: LmbmConfig,

    /// Current hypotheses (weighted mixture of track sets)
    hypotheses: Vec<LmbmHypothesis>,
    /// Complete trajectories for all discarded long-lived tracks
    trajectories: Vec<Trajectory>,

    /// Existence probability threshold for gating tracks
    existence_threshold: f64,
    /// Minimum trajectory length to keep when pruning
    min_trajectory_length: usize,

    /// The LMBM association strategy
    strategy: S,
}

// ============================================================================
// Generic implementation
// ============================================================================

impl<S: LmbmAssociator> LmbmFilterCore<S> {
    /// Create a filter with explicit strategy.
    pub fn with_strategy(
        motion: MotionModel,
        sensors: SensorSet,
        birth: BirthModel,
        association_config: AssociationConfig,
        lmbm_config: LmbmConfig,
        strategy: S,
    ) -> Self {
        let initial_hypothesis = LmbmHypothesis::new(0.0, Vec::new());

        Self {
            motion,
            sensors,
            birth,
            association_config,
            lmbm_config,
            hypotheses: vec![initial_hypothesis],
            trajectories: Vec::new(),
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            strategy,
        }
    }

    /// Number of sensors.
    pub fn num_sensors(&self) -> usize {
        self.sensors.num_sensors()
    }

    // ========================================================================
    // Internal helper methods
    // ========================================================================

    /// Predict all hypotheses forward in time.
    fn predict_hypotheses(&mut self, timestep: usize) {
        super::common_ops::predict_all_hypotheses(
            &mut self.hypotheses,
            &self.motion,
            &self.birth,
            timestep,
        );
    }

    /// Normalize, gate hypotheses, and prune tracks.
    fn normalize_gate_and_prune_tracks(&mut self) -> Vec<bool> {
        super::common_ops::normalize_gate_and_prune_tracks(
            &mut self.hypotheses,
            &mut self.trajectories,
            self.lmbm_config.hypothesis_weight_threshold,
            self.lmbm_config.max_hypotheses,
            self.existence_threshold,
            self.min_trajectory_length,
        )
    }

    /// Normalize and gate hypotheses (without track pruning).
    fn normalize_and_gate_hypotheses(&mut self) {
        super::common_ops::normalize_and_gate_hypotheses(
            &mut self.hypotheses,
            self.lmbm_config.hypothesis_weight_threshold,
            self.lmbm_config.max_hypotheses,
        );
    }

    /// Gate tracks by existence probability across all hypotheses.
    fn gate_tracks(&mut self) {
        super::common_ops::gate_hypothesis_tracks(
            &mut self.hypotheses,
            &mut self.trajectories,
            self.existence_threshold,
            self.min_trajectory_length,
        );
    }

    /// Extract state estimates from the hypothesis mixture.
    fn extract_estimates(&self, timestamp: usize) -> StateEstimate {
        super::common_ops::extract_hypothesis_estimates(
            &self.hypotheses,
            timestamp,
            self.lmbm_config.use_eap,
        )
    }

    /// Update track trajectories after measurement update.
    fn update_trajectories(&mut self, timestamp: usize) {
        super::common_ops::update_hypothesis_trajectories(&mut self.hypotheses, timestamp);
    }

    /// Initialize trajectory recording for new birth tracks.
    fn init_birth_trajectories(&mut self, max_length: usize) {
        super::common_ops::init_hypothesis_birth_trajectories(&mut self.hypotheses, max_length);
    }

    // ========================================================================
    // Testing/Fixture Validation Methods
    // ========================================================================

    /// Set the internal hypotheses directly (for fixture testing).
    pub fn set_hypotheses(&mut self, hypotheses: Vec<LmbmHypothesis>) {
        self.hypotheses = hypotheses;
    }

    /// Get the current hypotheses (for fixture testing).
    pub fn get_hypotheses(&self) -> Vec<LmbmHypothesis> {
        self.hypotheses.clone()
    }

    /// Get tracks from highest-weight hypothesis (for fixture testing).
    pub fn get_tracks(&self) -> Vec<Track> {
        if self.hypotheses.is_empty() {
            return Vec::new();
        }
        self.hypotheses
            .iter()
            .max_by(|a, b| a.log_weight.partial_cmp(&b.log_weight).unwrap())
            .map(|h| h.tracks.clone())
            .unwrap_or_default()
    }
}

// ============================================================================
// Single-sensor Filter trait implementation
// ============================================================================

impl<A: Associator> LmbmFilterCore<SingleSensorLmbmStrategy<A>> {
    /// Get configuration snapshot for debugging.
    pub fn get_config(&self) -> FilterConfigSnapshot {
        FilterConfigSnapshot::single_sensor_lmbm(
            "LmbmFilter",
            &self.motion,
            self.sensors.single(),
            &self.birth,
            &self.association_config,
            self.existence_threshold,
            self.min_trajectory_length,
            &self.lmbm_config,
        )
    }

    /// Detailed step for fixture validation (single-sensor).
    pub fn step_detailed<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &[DVector<f64>],
        timestep: usize,
    ) -> Result<StepDetailedOutput, FilterError> {
        // Prediction
        self.predict_hypotheses(timestep);
        let predicted_hypotheses = self.hypotheses.clone();
        let predicted_tracks = self.get_tracks();

        // Association and update
        let (association_matrices, association_result) = if !measurements.is_empty()
            && !self.hypotheses.is_empty()
            && !self.hypotheses[0].tracks.is_empty()
        {
            // Convert slice to Vec for trait interface
            let measurements_vec: Vec<DVector<f64>> = measurements.to_vec();
            let intermediate = self.strategy.associate_and_update(
                rng,
                &mut self.hypotheses,
                &measurements_vec,
                &self.sensors,
                &self.motion,
                &self.association_config,
            )?;

            match intermediate {
                Some(i) => (i.matrices, i.result),
                None => (None, None),
            }
        } else {
            if measurements.is_empty() {
                self.strategy
                    .update_existence_no_measurements(&mut self.hypotheses, &self.sensors);
            }
            (None, None)
        };

        let pre_normalization_hypotheses = self.hypotheses.clone();
        let objects_likely_to_exist = self.normalize_gate_and_prune_tracks();
        let normalized_hypotheses = self.hypotheses.clone();
        let updated_tracks = self.get_tracks();

        let cardinality = super::common_ops::compute_hypothesis_cardinality(
            &self.hypotheses,
            self.lmbm_config.use_eap,
        );

        let final_estimate = self.extract_estimates(timestep);

        Ok(StepDetailedOutput {
            predicted_tracks,
            association_matrices,
            association_result,
            updated_tracks,
            cardinality,
            final_estimate,
            sensor_updates: None,
            predicted_hypotheses: Some(predicted_hypotheses),
            pre_normalization_hypotheses: Some(pre_normalization_hypotheses),
            normalized_hypotheses: Some(normalized_hypotheses),
            objects_likely_to_exist: Some(objects_likely_to_exist),
        })
    }
}

impl<A: Associator> Filter for LmbmFilterCore<SingleSensorLmbmStrategy<A>> {
    type State = Vec<LmbmHypothesis>;
    type Measurements = Vec<DVector<f64>>;

    fn step<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        // Prediction
        self.predict_hypotheses(timestep);
        self.init_birth_trajectories(super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        // Association and update
        if !measurements.is_empty() {
            if !self.hypotheses.is_empty() && !self.hypotheses[0].tracks.is_empty() {
                self.strategy.associate_and_update(
                    rng,
                    &mut self.hypotheses,
                    measurements,
                    &self.sensors,
                    &self.motion,
                    &self.association_config,
                )?;
            }
        } else {
            self.strategy
                .update_existence_no_measurements(&mut self.hypotheses, &self.sensors);
        }

        // Hypothesis management and extraction
        self.normalize_gate_and_prune_tracks();
        self.update_trajectories(timestep);
        Ok(self.extract_estimates(timestep))
    }

    fn state(&self) -> &Self::State {
        &self.hypotheses
    }

    fn reset(&mut self) {
        self.hypotheses.clear();
        self.hypotheses.push(LmbmHypothesis::new(0.0, Vec::new()));
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
// Multi-sensor Filter trait implementation
// ============================================================================

impl<A: MultisensorAssociator> LmbmFilterCore<MultisensorLmbmStrategy<A>> {
    /// Get configuration snapshot for debugging.
    pub fn get_config(&self) -> FilterConfigSnapshot {
        FilterConfigSnapshot::multi_sensor_lmbm(
            "MultisensorLmbmFilter",
            &self.motion,
            self.sensors.multi(),
            &self.birth,
            &self.association_config,
            self.existence_threshold,
            self.min_trajectory_length,
            &self.lmbm_config,
        )
    }

    /// Detailed step for fixture validation (multi-sensor).
    pub fn step_detailed<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &MultisensorMeasurements,
        timestep: usize,
    ) -> Result<StepDetailedOutput, FilterError> {
        self.strategy
            .validate_measurements(measurements, &self.sensors)?;

        // Prediction
        self.predict_hypotheses(timestep);
        let predicted_hypotheses = Some(self.hypotheses.clone());
        let predicted_tracks = self.get_tracks();

        // Association and update
        let has_measurements = !self.strategy.measurements_empty(measurements);

        if has_measurements && !self.hypotheses.is_empty() {
            self.strategy.associate_and_update(
                rng,
                &mut self.hypotheses,
                measurements,
                &self.sensors,
                &self.motion,
                &self.association_config,
            )?;
        } else if !has_measurements {
            self.strategy
                .update_existence_no_measurements(&mut self.hypotheses, &self.sensors);
        }

        let pre_normalization_hypotheses = Some(self.hypotheses.clone());
        self.normalize_and_gate_hypotheses();
        let normalized_hypotheses = Some(self.hypotheses.clone());

        let objects_likely_to_exist = Some(super::common_ops::compute_objects_likely_to_exist(
            &self.hypotheses,
            self.existence_threshold,
        ));

        let updated_tracks = self.get_tracks();
        let cardinality = super::common_ops::compute_cardinality(&updated_tracks);

        self.gate_tracks();
        let final_estimate = self.extract_estimates(timestep);

        Ok(StepDetailedOutput {
            predicted_tracks,
            association_matrices: None,
            association_result: None,
            updated_tracks,
            cardinality,
            final_estimate,
            sensor_updates: None,
            predicted_hypotheses,
            pre_normalization_hypotheses,
            normalized_hypotheses,
            objects_likely_to_exist,
        })
    }
}

impl<A: MultisensorAssociator> Filter for LmbmFilterCore<MultisensorLmbmStrategy<A>> {
    type State = Vec<LmbmHypothesis>;
    type Measurements = MultisensorMeasurements;

    fn step<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.strategy
            .validate_measurements(measurements, &self.sensors)?;

        // Prediction
        self.predict_hypotheses(timestep);
        self.init_birth_trajectories(super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        // Association and update
        let has_measurements = !self.strategy.measurements_empty(measurements);

        if has_measurements && !self.hypotheses.is_empty() {
            self.strategy.associate_and_update(
                rng,
                &mut self.hypotheses,
                measurements,
                &self.sensors,
                &self.motion,
                &self.association_config,
            )?;
        } else if !has_measurements {
            self.strategy
                .update_existence_no_measurements(&mut self.hypotheses, &self.sensors);
        }

        // Hypothesis management
        self.normalize_and_gate_hypotheses();
        self.gate_tracks();
        self.update_trajectories(timestep);

        Ok(self.extract_estimates(timestep))
    }

    fn state(&self) -> &Self::State {
        &self.hypotheses
    }

    fn reset(&mut self) {
        self.hypotheses.clear();
        self.hypotheses.push(LmbmHypothesis::new(0.0, Vec::new()));
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
// Builder Trait Implementation
// ============================================================================

impl<S: LmbmAssociator> FilterBuilder for LmbmFilterCore<S> {
    fn existence_threshold_mut(&mut self) -> &mut f64 {
        &mut self.existence_threshold
    }

    fn min_trajectory_length_mut(&mut self) -> &mut usize {
        &mut self.min_trajectory_length
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

/// Single-sensor LMBM filter with Gibbs associator.
///
/// This is the standard LMBM filter for tracking with a single sensor.
/// For multi-sensor tracking, use [`MultisensorLmbmFilter`].
///
/// For custom associators, use [`LmbmFilterCore`] directly.
///
/// # Example
///
/// ```ignore
/// use multisensor_lmb_filters_rs::lmb::{lmbm_filter, LmbmFilter};
///
/// // Using factory function (recommended)
/// let filter = lmbm_filter(motion, sensor, birth, assoc_config, lmbm_config);
///
/// // Or using type directly
/// let filter: LmbmFilter = LmbmFilterCore::with_strategy(...);
/// ```
pub type LmbmFilter = LmbmFilterCore<SingleSensorLmbmStrategy<GibbsAssociator>>;

/// Multi-sensor LMBM filter with Gibbs associator.
///
/// Performs joint data association across all sensors using a Cartesian
/// product likelihood tensor.
///
/// For custom associators, use [`LmbmFilterCore`] directly.
///
/// # Warning
///
/// Memory usage is O(∏(m_s + 1) × n) where m_s is measurements from sensor s.
pub type MultisensorLmbmFilter =
    LmbmFilterCore<MultisensorLmbmStrategy<MultisensorGibbsAssociator>>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmb::config::{BirthLocation, SensorModel};
    use crate::lmb::factory::{lmbm_filter, multisensor_lmbm_filter};

    fn create_motion() -> MotionModel {
        MotionModel::constant_velocity_2d(1.0, 0.1, 0.99)
    }

    fn create_sensor() -> SensorModel {
        SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0)
    }

    fn create_multi_sensor() -> MultisensorConfig {
        let sensor1 = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
        let sensor2 = SensorModel::position_sensor_2d(1.5, 0.85, 12.0, 100.0);
        MultisensorConfig::new(vec![sensor1, sensor2])
    }

    fn create_birth() -> BirthModel {
        let birth_loc = BirthLocation::new(
            0,
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            DMatrix::identity(4, 4) * 100.0,
        );
        BirthModel::new(vec![birth_loc], 0.1, 0.01)
    }

    #[test]
    fn test_lmbm_filter_creation() {
        let filter = lmbm_filter(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
        assert_eq!(filter.num_sensors(), 1);
        assert_eq!(filter.hypotheses.len(), 1);
    }

    #[test]
    fn test_lmbm_filter_step_no_measurements() {
        let mut filter = lmbm_filter(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        let mut rng = rand::thread_rng();

        let estimate = filter.step(&mut rng, &vec![], 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_lmbm_filter_step_with_measurements() {
        let mut filter = lmbm_filter(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        let mut rng = rand::thread_rng();

        let measurements = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![5.0, 5.0]),
        ];

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_lmbm_filter_multiple_steps() {
        let mut filter = lmbm_filter(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        let mut rng = rand::thread_rng();

        for t in 0..5 {
            let measurements = vec![DVector::from_vec(vec![t as f64, t as f64])];
            let _estimate = filter.step(&mut rng, &measurements, t).unwrap();
        }

        assert!(!filter.hypotheses.is_empty());
    }

    #[test]
    fn test_lmbm_filter_reset() {
        let mut filter = lmbm_filter(
            create_motion(),
            create_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        let mut rng = rand::thread_rng();

        let _ = filter.step(&mut rng, &vec![], 0);
        filter.reset();

        assert_eq!(filter.hypotheses.len(), 1);
        assert!(filter.hypotheses[0].tracks.is_empty());
        assert!(filter.trajectories.is_empty());
    }

    #[test]
    fn test_multisensor_lmbm_filter_creation() {
        let filter = multisensor_lmbm_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
        assert_eq!(filter.num_sensors(), 2);
        assert_eq!(filter.hypotheses.len(), 1);
    }

    #[test]
    fn test_multisensor_lmbm_filter_step_no_measurements() {
        let mut filter = multisensor_lmbm_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        let mut rng = rand::thread_rng();

        let measurements = vec![vec![], vec![]];
        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_multisensor_lmbm_filter_step_with_measurements() {
        let mut filter = multisensor_lmbm_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        let mut rng = rand::thread_rng();

        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])],
            vec![DVector::from_vec(vec![0.5, 0.5])],
        ];

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_multisensor_lmbm_filter_multiple_steps() {
        let mut filter = multisensor_lmbm_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        let mut rng = rand::thread_rng();

        for t in 0..5 {
            let measurements = vec![
                vec![DVector::from_vec(vec![t as f64, t as f64])],
                vec![DVector::from_vec(vec![t as f64 + 0.1, t as f64 + 0.1])],
            ];
            let _estimate = filter.step(&mut rng, &measurements, t).unwrap();
        }

        assert!(!filter.hypotheses.is_empty());
    }

    #[test]
    fn test_multisensor_lmbm_filter_reset() {
        let mut filter = multisensor_lmbm_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        let mut rng = rand::thread_rng();

        let measurements = vec![vec![], vec![]];
        let _ = filter.step(&mut rng, &measurements, 0);
        filter.reset();

        assert_eq!(filter.hypotheses.len(), 1);
        assert!(filter.hypotheses[0].tracks.is_empty());
        assert!(filter.trajectories.is_empty());
    }

    #[test]
    fn test_multisensor_lmbm_filter_wrong_sensor_count() {
        let mut filter = multisensor_lmbm_filter(
            create_motion(),
            create_multi_sensor(),
            create_birth(),
            AssociationConfig::default(),
            LmbmConfig::default(),
        );
        let mut rng = rand::thread_rng();

        // Only 1 sensor instead of 2
        let measurements = vec![vec![DVector::from_vec(vec![0.0, 0.0])]];
        let result = filter.step(&mut rng, &measurements, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lmbm_sensor_set_single() {
        let sensor = create_sensor();
        let set: SensorSet = sensor.into();
        assert_eq!(set.num_sensors(), 1);
        assert_eq!(set.z_dim(), 2);
    }

    #[test]
    fn test_lmbm_sensor_set_multi() {
        let sensors = create_multi_sensor();
        let set: SensorSet = sensors.into();
        assert_eq!(set.num_sensors(), 2);
        assert_eq!(set.z_dim(), 2);
    }
}
