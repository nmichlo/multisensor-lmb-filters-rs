//! Multi-sensor LMBM (Labeled Multi-Bernoulli Mixture) filter.
//!
//! The multi-sensor LMBM filter extends the LMBM filter to handle multiple sensors
//! simultaneously. It performs joint data association across all sensors using
//! a Cartesian product likelihood matrix.
//!
//! Key features:
//! - **Joint association**: Considers all sensor measurements simultaneously
//! - **Hypothesis management**: Maintains weighted hypotheses with hard assignments
//! - **Multi-sensor updates**: Properly handles per-sensor detection probabilities
//! - **Pluggable association**: Uses [`MultisensorAssociator`] trait for swappable algorithms
//!
//! # Warning
//!
//! This implementation can be very memory intensive for large numbers of objects
//! and sensors, as the likelihood matrix grows as O(∏(m_s + 1) × n) where m_s is
//! the number of measurements from sensor s and n is the number of objects.

use nalgebra::{DMatrix, DVector};

use crate::common::linalg::{log_gaussian_normalizing_constant, robust_inverse};

use super::super::builder::FilterBuilder;
use super::super::config::{
    AssociationConfig, BirthModel, FilterConfigSnapshot, FilterParams, LmbmConfig, MotionModel,
    MultisensorConfig,
};
use super::super::errors::FilterError;
use super::super::output::{StateEstimate, Trajectory};
use super::super::traits::Filter;
use super::super::types::{
    GaussianComponent, LmbmHypothesis, StepDetailedOutput, Track, TrackLabel,
};
use super::lmb::MultisensorMeasurements;
use super::traits::{
    MultisensorAssociationResult, MultisensorAssociator, MultisensorGibbsAssociator,
};

/// Per-measurement covariance matrices, grouped by sensor and measurement.
///
/// The shape must match [`MultisensorMeasurements`]. Each covariance must be
/// square and match the measurement dimension for its sensor.
pub type MultisensorMeasurementCovariances = Vec<Vec<DMatrix<f64>>>;

/// Result of a covariance-aware multisensor LMBM step.
#[derive(Debug, Clone)]
pub struct MultisensorLmbmStepResult {
    /// Filtered state estimate.
    pub estimate: StateEstimate,
    /// Labels in the object order used by the native association sample.
    pub predicted_track_labels: Vec<TrackLabel>,
    /// Highest-likelihood native joint association sample.
    ///
    /// The sample is column-major by sensor: `sample[s * n + i]` is zero for
    /// a miss or a one-based measurement index for track `i` and sensor `s`.
    pub best_association: Option<Vec<usize>>,
    /// Number of sensors represented in `best_association`.
    pub num_sensors: usize,
}

/// Multi-sensor LMBM filter.
///
/// This filter extends the LMBM approach to multiple sensors by performing joint
/// data association across all sensors. The association likelihood is computed
/// in the Cartesian product space of all sensor measurements.
///
/// The filter is generic over the association algorithm via [`MultisensorAssociator`].
/// The default is [`MultisensorGibbsAssociator`] which uses Gibbs sampling.
///
/// # Type Parameters
///
/// * `A` - Multi-sensor associator type (default: [`MultisensorGibbsAssociator`])
///
/// # Performance Warning
///
/// Memory usage is O(∏(m_s + 1) × n) where:
/// - m_s is the number of measurements from sensor s
/// - n is the number of objects
///
/// For example, with 2 sensors, 10 measurements each, and 5 objects:
/// (10+1) × (10+1) × 5 = 605 entries
///
/// With 3 sensors: (10+1)³ × 5 = 6,655 entries
pub struct MultisensorLmbmFilter<A: MultisensorAssociator = MultisensorGibbsAssociator> {
    /// Motion model (dynamics, survival probability)
    motion: MotionModel,
    /// Multi-sensor configuration
    sensors: MultisensorConfig,
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

    /// The multi-sensor associator instance
    associator: A,
}

/// Posterior parameters for a single entry in the flattened likelihood tensor.
#[derive(Clone)]
struct MultisensorPosterior {
    /// Posterior existence probability
    existence: f64,
    /// Posterior mean
    mean: DVector<f64>,
    /// Posterior covariance
    covariance: DMatrix<f64>,
}

impl MultisensorLmbmFilter<MultisensorGibbsAssociator> {
    /// Create a new multi-sensor LMBM filter with default Gibbs associator.
    pub fn new(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
        lmbm_config: LmbmConfig,
    ) -> Self {
        Self::with_associator_type(
            motion,
            sensors,
            birth,
            association_config,
            lmbm_config,
            MultisensorGibbsAssociator,
        )
    }

    /// Create from FilterParams with default Gibbs associator.
    pub fn from_params(params: &FilterParams) -> Self {
        Self::new(
            params.motion.clone(),
            params.sensor.multi().clone(),
            params.birth.clone(),
            params.association.clone(),
            params.lmbm.clone(),
        )
    }
}

impl<A: MultisensorAssociator> MultisensorLmbmFilter<A> {
    /// Create a new multi-sensor LMBM filter with a custom associator type.
    pub fn with_associator_type(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
        lmbm_config: LmbmConfig,
        associator: A,
    ) -> Self {
        // Start with a single empty hypothesis with weight 1.0
        let initial_hypothesis = LmbmHypothesis::new(0.0, Vec::new()); // log(1.0) = 0.0

        Self {
            motion,
            sensors,
            birth,
            association_config,
            lmbm_config,
            hypotheses: vec![initial_hypothesis],
            trajectories: Vec::new(),
            existence_threshold: super::super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            associator,
        }
    }

    /// Number of sensors.
    #[inline]
    pub fn num_sensors(&self) -> usize {
        self.sensors.num_sensors()
    }

    /// Replaces the locations used for births on the next prediction step.
    ///
    /// Existence probabilities remain unchanged. This supports applications
    /// that derive birth hypotheses from the current sensor field of view
    /// rather than freezing births to the first few frames of a replay.
    pub fn replace_birth_locations(&mut self, locations: Vec<super::super::config::BirthLocation>) {
        self.birth.locations = locations;
    }

    /// Predict all hypotheses forward in time.
    fn predict_hypotheses(&mut self, timestep: usize) {
        super::super::common_ops::predict_all_hypotheses(
            &mut self.hypotheses,
            &self.motion,
            &self.birth,
            timestep,
        );
    }

    /// Generate multi-sensor association matrices.
    ///
    /// Returns (log_likelihoods, posteriors, dimensions) where:
    /// - log_likelihoods: Flattened tensor of log-likelihood values
    /// - posteriors: Flattened tensor of posterior parameters
    /// - dimensions: [m_1+1, m_2+1, ..., m_S+1, n]
    fn generate_association_matrices(
        &self,
        tracks: &[Track],
        measurements: &MultisensorMeasurements,
        measurement_covariances: Option<&MultisensorMeasurementCovariances>,
    ) -> (Vec<f64>, Vec<MultisensorPosterior>, Vec<usize>) {
        let num_sensors = self.num_sensors();
        let num_objects = tracks.len();

        // Dimensions: [m_1+1, ..., m_S+1, n]
        let mut dimensions = vec![0; num_sensors + 1];
        for s in 0..num_sensors {
            dimensions[s] = measurements[s].len() + 1; // +1 for miss
        }
        dimensions[num_sensors] = num_objects;

        // Total entries
        let num_entries: usize = dimensions.iter().product();

        // Page sizes for index conversion
        let mut page_sizes = vec![1; num_sensors + 1];
        for i in 1..=num_sensors {
            page_sizes[i] = page_sizes[i - 1] * dimensions[i - 1];
        }

        // Allocate output
        let x_dim = self.motion.x_dim();
        let mut log_likelihoods = vec![0.0; num_entries];
        let mut posteriors = vec![
            MultisensorPosterior {
                existence: 0.0,
                mean: DVector::zeros(x_dim),
                covariance: DMatrix::zeros(x_dim, x_dim),
            };
            num_entries
        ];

        // Compute each entry
        for ell in 0..num_entries {
            // Convert linear index to Cartesian coordinates
            let u = self.linear_to_cartesian(ell + 1, &page_sizes); // 1-indexed internally

            // Object index (0-indexed)
            let obj_idx = u[num_sensors] - 1;

            // Association vector: a[s] = 0 for miss, 1..m_s for measurement index (1-indexed)
            let associations: Vec<usize> = u[0..num_sensors].iter().map(|&x| x - 1).collect();

            // Compute log-likelihood and posterior
            let (log_l, posterior) = self.compute_log_likelihood(
                obj_idx,
                &associations,
                tracks,
                measurements,
                measurement_covariances,
            );

            log_likelihoods[ell] = log_l;
            posteriors[ell] = posterior;
        }

        (log_likelihoods, posteriors, dimensions)
    }

    /// Convert linear index to Cartesian coordinates (MATLAB-style, 1-indexed).
    fn linear_to_cartesian(&self, mut ell: usize, page_sizes: &[usize]) -> Vec<usize> {
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
    fn cartesian_to_linear(&self, u: &[usize], dimensions: &[usize]) -> usize {
        let mut ell = u[0];
        let mut pi = 1;

        for i in 1..u.len() {
            pi *= dimensions[i - 1];
            ell += pi * (u[i] - 1);
        }

        ell - 1 // Convert to 0-indexed
    }

    /// Compute log-likelihood and posterior for a single object-association pair.
    fn compute_log_likelihood(
        &self,
        obj_idx: usize,
        associations: &[usize], // 0 = miss, 1..m = measurement index (0-indexed measurement)
        tracks: &[Track],
        measurements: &MultisensorMeasurements,
        measurement_covariances: Option<&MultisensorMeasurementCovariances>,
    ) -> (f64, MultisensorPosterior) {
        let track = &tracks[obj_idx];
        let (prior_mean, prior_cov) = match (track.primary_mean(), track.primary_covariance()) {
            (Some(m), Some(c)) => (m.clone(), c.clone()),
            _ => {
                return (
                    f64::NEG_INFINITY,
                    MultisensorPosterior {
                        existence: 0.0,
                        mean: DVector::zeros(self.motion.x_dim()),
                        covariance: DMatrix::identity(self.motion.x_dim(), self.motion.x_dim()),
                    },
                );
            }
        };

        // Check which sensors have detections
        let num_sensors = associations.len();
        let detecting: Vec<bool> = associations.iter().map(|&a| a > 0).collect();
        let num_detections: usize = detecting.iter().filter(|&&x| x).count();

        if num_detections > 0 {
            // Build stacked measurement vector and observation model
            let z_dim_total: usize = detecting
                .iter()
                .enumerate()
                .filter(|(_, &is_detecting)| is_detecting)
                .map(|(sensor_idx, _)| self.sensors.sensors[sensor_idx].z_dim())
                .sum();
            let x_dim = self.motion.x_dim();

            let mut z = DVector::zeros(z_dim_total);
            let mut c = DMatrix::zeros(z_dim_total, x_dim);
            let mut q_blocks = Vec::new();

            let mut counter = 0;
            for s in 0..num_sensors {
                if detecting[s] {
                    let sensor = &self.sensors.sensors[s];
                    let meas_idx = associations[s] - 1; // Convert to 0-indexed
                    let z_dim = sensor.z_dim();
                    let start = counter;

                    if measurements[s][meas_idx].len() != z_dim {
                        return (
                            f64::NEG_INFINITY,
                            MultisensorPosterior {
                                existence: 0.0,
                                mean: prior_mean,
                                covariance: prior_cov,
                            },
                        );
                    }

                    // Copy measurement
                    z.rows_mut(start, z_dim)
                        .copy_from(&measurements[s][meas_idx]);

                    // Copy observation matrix
                    c.view_mut((start, 0), (z_dim, x_dim))
                        .copy_from(&sensor.observation_matrix);

                    // Collect noise covariance
                    q_blocks.push(
                        measurement_covariances
                            .map(|covariances| covariances[s][meas_idx].clone())
                            .unwrap_or_else(|| sensor.measurement_noise.clone()),
                    );

                    counter += z_dim;
                }
            }

            // Build block-diagonal Q
            let mut q = DMatrix::zeros(z_dim_total, z_dim_total);
            let mut offset = 0;
            for q_block in &q_blocks {
                let z_dim = q_block.nrows();
                q.view_mut((offset, offset), (z_dim, z_dim))
                    .copy_from(q_block);
                offset += z_dim;
            }

            // Innovation and covariance
            let nu = &z - &c * &prior_mean;
            let s_mat = &c * &prior_cov * c.transpose() + &q;

            // Compute Kalman gain
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

            // Log Gaussian constant
            let log_eta = log_gaussian_normalizing_constant(&s_mat, z_dim_total);

            // Detection probability product (log)
            let mut log_pd = 0.0;
            for (sensor_idx, (sensor, &is_detecting)) in self
                .sensors
                .sensors
                .iter()
                .zip(detecting.iter())
                .enumerate()
            {
                let p_d = sensor.detection_probability;
                log_pd += if is_detecting {
                    p_d.ln()
                } else if measurements[sensor_idx].is_empty() {
                    // Sparse asynchronous sensors should not be treated as
                    // active scans that missed when no measurement packet was
                    // present for this frame.
                    0.0
                } else {
                    (1.0 - p_d).ln()
                };
            }

            // Clutter density product (log)
            let log_kappa: f64 = detecting
                .iter()
                .enumerate()
                .filter(|(_, &d)| d)
                .map(|(s, _)| self.sensors.sensors[s].clutter_density().ln())
                .sum();

            // Log-likelihood
            let log_l =
                track.existence.ln() + log_pd + log_eta - 0.5 * nu.dot(&(&s_inv * &nu)) - log_kappa;

            // Posterior parameters
            let post_mean = &prior_mean + &k * &nu;
            let post_cov = (DMatrix::identity(x_dim, x_dim) - &k * &c) * &prior_cov;

            (
                log_l,
                MultisensorPosterior {
                    existence: 1.0, // Detected
                    mean: post_mean,
                    covariance: post_cov,
                },
            )
        } else {
            // All missed detections
            let mut prob_no_detect = 1.0;
            for (sensor_idx, sensor) in self.sensors.sensors.iter().enumerate() {
                if !measurements[sensor_idx].is_empty() {
                    prob_no_detect *= 1.0 - sensor.detection_probability;
                }
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
        &mut self,
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

        // The previous implementation cloned every prior hypothesis for every
        // association sample and truncated the Cartesian product afterward.
        // Rank the scalar combinations first so we clone only the exact top-K
        // posterior hypotheses that normalization could retain.
        let sample_scores = samples
            .iter()
            .map(|sample| self.association_sample_score(sample, log_likelihoods, dimensions))
            .collect::<Vec<_>>();

        let mut combinations = self
            .hypotheses
            .iter()
            .enumerate()
            .flat_map(|(prior_index, prior_hyp)| {
                sample_scores
                    .iter()
                    .enumerate()
                    .map(move |(sample_index, sample_score)| {
                        (
                            prior_hyp.log_weight + sample_score,
                            prior_index,
                            sample_index,
                        )
                    })
            })
            .collect::<Vec<_>>();

        combinations.sort_by(|left, right| right.0.total_cmp(&left.0));
        combinations.truncate(self.lmbm_config.max_hypotheses);

        let mut new_hypotheses = Vec::with_capacity(combinations.len());

        for (_score, prior_index, sample_index) in combinations {
            let prior_hyp = &self.hypotheses[prior_index];
            let sample = &samples[sample_index];
            // Create new hypothesis from prior
            let mut new_hyp = prior_hyp.clone();

            // Decode sample: sample is flattened [v_{1,1}, v_{1,2}, ..., v_{n,S}] column-major
            // So sample[s * num_objects + i] = v[i, s]
            for i in 0..num_objects.min(new_hyp.tracks.len()) {
                // Build association vector for object i
                let mut u: Vec<usize> = Vec::with_capacity(num_sensors + 1);
                for s in 0..num_sensors {
                    let v_is = sample[s * num_objects + i];
                    u.push(v_is + 1); // Convert to 1-indexed
                }
                u.push(i + 1); // Object index (1-indexed)

                // Get linear index
                let ell = self.cartesian_to_linear(&u, dimensions);

                // Update track with posterior
                let posterior = &posteriors[ell];
                new_hyp.tracks[i].existence = posterior.existence;

                // Replace GM components with single posterior
                new_hyp.tracks[i].components.clear();
                new_hyp.tracks[i].components.push(GaussianComponent {
                    weight: 1.0,
                    mean: posterior.mean.clone(),
                    covariance: posterior.covariance.clone(),
                });
            }

            new_hyp.log_weight += sample_scores[sample_index];
            new_hypotheses.push(new_hyp);
        }

        self.hypotheses = new_hypotheses;
    }

    /// Normalize and gate hypotheses.
    fn normalize_and_gate_hypotheses(&mut self) {
        super::super::common_ops::normalize_and_gate_hypotheses(
            &mut self.hypotheses,
            self.lmbm_config.hypothesis_weight_threshold,
            self.lmbm_config.max_hypotheses,
        );
    }

    /// Gate tracks by existence probability across all hypotheses.
    fn gate_tracks(&mut self) {
        super::super::common_ops::gate_hypothesis_tracks(
            &mut self.hypotheses,
            &mut self.trajectories,
            self.existence_threshold,
            self.min_trajectory_length,
        );
    }

    /// Extract state estimates from the hypothesis mixture.
    fn extract_estimates(&self, timestamp: usize) -> StateEstimate {
        super::super::common_ops::extract_hypothesis_estimates(
            &self.hypotheses,
            timestamp,
            self.lmbm_config.use_eap,
        )
    }

    /// Update track trajectories.
    fn update_trajectories(&mut self, timestamp: usize) {
        super::super::common_ops::update_hypothesis_trajectories(&mut self.hypotheses, timestamp);
    }

    /// Initialize trajectory recording for birth tracks.
    fn init_birth_trajectories(&mut self, max_length: usize) {
        super::super::common_ops::init_hypothesis_birth_trajectories(
            &mut self.hypotheses,
            max_length,
        );
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

    /// Get a snapshot of the filter's configuration for debugging.
    ///
    /// Returns all initialization parameters as a serializable struct,
    /// useful for comparing configurations across implementations.
    pub fn get_config(&self) -> FilterConfigSnapshot {
        FilterConfigSnapshot::multi_sensor_lmbm(
            "MultisensorLmbmFilter",
            &self.motion,
            &self.sensors,
            &self.birth,
            &self.association_config,
            self.existence_threshold,
            self.min_trajectory_length,
            &self.lmbm_config,
        )
    }

    /// Detailed step for fixture validation.
    ///
    /// Exposes intermediate hypothesis states for testing against MATLAB fixtures.
    pub fn step_detailed<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &MultisensorMeasurements,
        timestep: usize,
    ) -> Result<StepDetailedOutput, FilterError> {
        // Validate measurements
        if measurements.len() != self.num_sensors() {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                self.num_sensors(),
                measurements.len()
            )));
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 1: Prediction
        // ══════════════════════════════════════════════════════════════════════
        self.predict_hypotheses(timestep);

        // Capture predicted hypotheses for fixture validation (MATLAB step1_prediction)
        let predicted_hypotheses = Some(self.hypotheses.clone());
        let predicted_tracks = self.get_tracks();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2-3: Association and posterior hypothesis generation
        // ══════════════════════════════════════════════════════════════════════
        let has_measurements = measurements.iter().any(|m| !m.is_empty());

        // CRITICAL: Always run association when measurements are available, even with 0 tracks
        // (births can generate new hypotheses). Matches MATLAB behavior (runMultisensorLmbmFilter.m:51).
        if has_measurements && !self.hypotheses.is_empty() {
            let (log_likelihoods, posteriors, dimensions) =
                self.generate_association_matrices(&self.hypotheses[0].tracks, measurements, None);

            let association_result = self
                .associator
                .associate(rng, &log_likelihoods, &dimensions, &self.association_config)
                .map_err(FilterError::Association)?;

            self.generate_posterior_hypotheses(
                &association_result.samples,
                &log_likelihoods,
                &posteriors,
                &dimensions,
            );
        } else if !has_measurements {
            // No sensor produced a measurement packet for this frame. In the
            // asynchronous wrapper this means no active scan was available, so
            // existence should carry through prediction without a miss penalty.
        }

        // Capture pre-normalization hypotheses (step4 in MATLAB)
        let pre_normalization_hypotheses = Some(self.hypotheses.clone());

        // ══════════════════════════════════════════════════════════════════════
        // STEP 4: Hypothesis management
        // ══════════════════════════════════════════════════════════════════════
        self.normalize_and_gate_hypotheses();

        // Capture normalized hypotheses (step5 in MATLAB)
        let normalized_hypotheses = Some(self.hypotheses.clone());

        // Compute objects_likely_to_exist (existence > threshold check)
        let objects_likely_to_exist =
            Some(super::super::common_ops::compute_objects_likely_to_exist(
                &self.hypotheses,
                self.existence_threshold,
            ));

        let updated_tracks = self.get_tracks();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 5: Cardinality extraction
        // ══════════════════════════════════════════════════════════════════════
        let cardinality = super::super::common_ops::compute_cardinality(&updated_tracks);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 6: Track gating
        // ══════════════════════════════════════════════════════════════════════
        self.gate_tracks();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 7: Extract final estimate
        // ══════════════════════════════════════════════════════════════════════
        let final_estimate = self.extract_estimates(timestep);

        Ok(StepDetailedOutput {
            predicted_tracks,
            association_matrices: None, // Too complex for multisensor (Cartesian product tensor)
            association_result: None,   // Association samples are internal to hypothesis generation
            updated_tracks,
            cardinality,
            final_estimate,
            // LMBM uses joint association - per-sensor data not directly available
            sensor_updates: None,
            predicted_hypotheses,
            pre_normalization_hypotheses,
            normalized_hypotheses,
            objects_likely_to_exist,
        })
    }

    /// Advances the filter using covariance supplied for every measurement.
    ///
    /// This is an additive API: callers that only have one covariance model
    /// per sensor can continue to use [`Filter::step`].
    pub fn step_with_covariances<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &MultisensorMeasurements,
        measurement_covariances: &MultisensorMeasurementCovariances,
        timestep: usize,
    ) -> Result<MultisensorLmbmStepResult, FilterError> {
        self.step_internal(rng, measurements, Some(measurement_covariances), timestep)
    }

    fn step_internal<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &MultisensorMeasurements,
        measurement_covariances: Option<&MultisensorMeasurementCovariances>,
        timestep: usize,
    ) -> Result<MultisensorLmbmStepResult, FilterError> {
        if measurements.len() != self.num_sensors() {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                self.num_sensors(),
                measurements.len()
            )));
        }

        if let Some(covariances) = measurement_covariances {
            self.validate_measurement_covariances(measurements, covariances)?;
        }

        self.predict_hypotheses(timestep);
        self.init_birth_trajectories(super::super::DEFAULT_MAX_TRAJECTORY_LENGTH);

        let predicted_track_labels = self
            .hypotheses
            .first()
            .map(|hypothesis| hypothesis.tracks.iter().map(|track| track.label).collect())
            .unwrap_or_default();
        let has_measurements = measurements.iter().any(|sensor| !sensor.is_empty());
        let mut best_association = None;

        if has_measurements && !self.hypotheses.is_empty() {
            let (log_likelihoods, posteriors, dimensions) = self.generate_association_matrices(
                &self.hypotheses[0].tracks,
                measurements,
                measurement_covariances,
            );

            let association_result = self
                .associator
                .associate(rng, &log_likelihoods, &dimensions, &self.association_config)
                .map_err(FilterError::Association)?;

            best_association =
                self.best_association_sample(&association_result, &log_likelihoods, &dimensions);

            self.generate_posterior_hypotheses(
                &association_result.samples,
                &log_likelihoods,
                &posteriors,
                &dimensions,
            );
        }

        self.normalize_and_gate_hypotheses();
        self.gate_tracks();
        self.update_trajectories(timestep);

        Ok(MultisensorLmbmStepResult {
            estimate: self.extract_estimates(timestep),
            predicted_track_labels,
            best_association,
            num_sensors: self.num_sensors(),
        })
    }

    fn validate_measurement_covariances(
        &self,
        measurements: &MultisensorMeasurements,
        covariances: &MultisensorMeasurementCovariances,
    ) -> Result<(), FilterError> {
        if covariances.len() != measurements.len() {
            return Err(FilterError::InvalidInput(format!(
                "Expected covariance groups for {} sensors, got {}",
                measurements.len(),
                covariances.len()
            )));
        }

        for (sensor_index, (sensor_measurements, sensor_covariances)) in
            measurements.iter().zip(covariances).enumerate()
        {
            if sensor_measurements.len() != sensor_covariances.len() {
                return Err(FilterError::InvalidInput(format!(
                    "Sensor {sensor_index} has {} measurements but {} covariances",
                    sensor_measurements.len(),
                    sensor_covariances.len()
                )));
            }

            for (measurement_index, (measurement, covariance)) in sensor_measurements
                .iter()
                .zip(sensor_covariances)
                .enumerate()
            {
                let dimension = measurement.len();
                let valid_shape =
                    covariance.nrows() == dimension && covariance.ncols() == dimension;
                let finite = covariance.iter().all(|value| value.is_finite());
                let symmetric = (0..dimension).all(|row| {
                    (0..dimension).all(|col| {
                        (covariance[(row, col)] - covariance[(col, row)]).abs() <= 1.0e-8
                    })
                });
                let positive_definite = valid_shape && covariance.clone().cholesky().is_some();

                if !valid_shape || !finite || !symmetric || !positive_definite {
                    return Err(FilterError::InvalidInput(format!(
                        "Invalid covariance for sensor {sensor_index} measurement {measurement_index}"
                    )));
                }
            }
        }

        Ok(())
    }

    fn best_association_sample(
        &self,
        result: &MultisensorAssociationResult,
        log_likelihoods: &[f64],
        dimensions: &[usize],
    ) -> Option<Vec<usize>> {
        let num_sensors = dimensions.len().checked_sub(1)?;
        let num_objects = *dimensions.get(num_sensors)?;

        result
            .samples
            .iter()
            .filter(|sample| sample.len() >= num_sensors * num_objects)
            .max_by(|left, right| {
                self.association_sample_score(left, log_likelihoods, dimensions)
                    .total_cmp(&self.association_sample_score(right, log_likelihoods, dimensions))
            })
            .cloned()
    }

    fn association_sample_score(
        &self,
        sample: &[usize],
        log_likelihoods: &[f64],
        dimensions: &[usize],
    ) -> f64 {
        let num_sensors = dimensions.len() - 1;
        let num_objects = dimensions[num_sensors];

        (0..num_objects)
            .map(|object_index| {
                let mut coordinate = (0..num_sensors)
                    .map(|sensor_index| sample[sensor_index * num_objects + object_index] + 1)
                    .collect::<Vec<_>>();
                coordinate.push(object_index + 1);
                log_likelihoods[self.cartesian_to_linear(&coordinate, dimensions)]
            })
            .sum()
    }
}

impl<A: MultisensorAssociator> Filter for MultisensorLmbmFilter<A> {
    type State = Vec<LmbmHypothesis>;
    type Measurements = MultisensorMeasurements;

    fn step<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        timestep: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.step_internal(rng, measurements, None, timestep)
            .map(|result| result.estimate)
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
// Builder Trait Implementations
// ============================================================================

impl<A: MultisensorAssociator> FilterBuilder for MultisensorLmbmFilter<A> {
    fn existence_threshold_mut(&mut self) -> &mut f64 {
        &mut self.existence_threshold
    }

    fn min_trajectory_length_mut(&mut self) -> &mut usize {
        &mut self.min_trajectory_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmb::{BirthLocation, SensorModel};
    use rand::SeedableRng;

    fn create_test_filter() -> MultisensorLmbmFilter {
        let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);

        // Two sensors
        let sensor1 = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
        let sensor2 = SensorModel::position_sensor_2d(1.5, 0.85, 12.0, 100.0);
        let sensors = MultisensorConfig::new(vec![sensor1, sensor2]);

        let birth_loc = BirthLocation::new(
            0,
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            DMatrix::identity(4, 4) * 100.0,
        );
        let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);

        let association_config = AssociationConfig::default();
        let lmbm_config = LmbmConfig::default();

        MultisensorLmbmFilter::new(motion, sensors, birth, association_config, lmbm_config)
    }

    #[test]
    fn test_filter_creation() {
        let filter = create_test_filter();
        assert_eq!(filter.x_dim(), 4);
        assert_eq!(filter.z_dim(), 2);
        assert_eq!(filter.num_sensors(), 2);
        assert_eq!(filter.hypotheses.len(), 1);
        assert!(filter.hypotheses[0].tracks.is_empty());
    }

    #[test]
    fn test_filter_step_no_measurements() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        let measurements = vec![vec![], vec![]]; // No measurements from either sensor

        let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
        assert_eq!(estimate.timestamp, 0);
    }

    #[test]
    fn test_filter_step_with_measurements() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])], // Sensor 1
            vec![DVector::from_vec(vec![0.5, 0.5])], // Sensor 2
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

        assert!(!filter.hypotheses.is_empty());
    }

    #[test]
    fn test_measurement_specific_covariance_changes_posterior() {
        let mut filter = create_test_filter();
        filter.predict_hypotheses(0);
        let tracks = filter.hypotheses[0].tracks.clone();
        let measurements = vec![vec![DVector::from_vec(vec![25.0, 25.0])], vec![]];
        let small = vec![vec![DMatrix::identity(2, 2)], vec![]];
        let large = vec![vec![DMatrix::identity(2, 2) * 10_000.0], vec![]];

        let (_, small_posterior) =
            filter.compute_log_likelihood(0, &[1, 0], &tracks, &measurements, Some(&small));
        let (_, large_posterior) =
            filter.compute_log_likelihood(0, &[1, 0], &tracks, &measurements, Some(&large));

        assert!(small_posterior.mean[0] > large_posterior.mean[0]);
        assert!(small_posterior.covariance[(0, 0)] < large_posterior.covariance[(0, 0)]);
    }

    #[test]
    fn test_covariance_aware_step_reports_native_association() {
        let mut filter = create_test_filter();
        filter.association_config = AssociationConfig::gibbs(25);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])],
            vec![DVector::from_vec(vec![0.2, 0.2])],
        ];
        let covariances = vec![
            vec![DMatrix::identity(2, 2)],
            vec![DMatrix::identity(2, 2) * 2.0],
        ];

        let _initial = filter
            .step_with_covariances(&mut rng, &measurements, &covariances, 0)
            .unwrap();
        let result = filter
            .step_with_covariances(&mut rng, &measurements, &covariances, 1)
            .unwrap();

        assert_eq!(result.num_sensors, 2);
        assert!(!result.predicted_track_labels.is_empty());
        assert_eq!(
            result.best_association.unwrap().len(),
            result.num_sensors * result.predicted_track_labels.len()
        );
    }

    #[test]
    fn test_birth_locations_can_be_replaced_between_steps() {
        let mut filter = create_test_filter();
        filter.replace_birth_locations(vec![BirthLocation::new(
            9,
            DVector::from_vec(vec![100.0, 0.0, 200.0, 0.0]),
            DMatrix::identity(4, 4) * 50.0,
        )]);

        filter.predict_hypotheses(3);

        assert!(filter.hypotheses[0]
            .tracks
            .iter()
            .any(|track| track.label.birth_time == 3 && track.label.birth_location == 9));
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        let measurements = vec![vec![], vec![]];
        let _ = filter.step(&mut rng, &measurements, 0);

        filter.reset();

        assert_eq!(filter.hypotheses.len(), 1);
        assert!(filter.hypotheses[0].tracks.is_empty());
        assert!(filter.trajectories.is_empty());
    }

    #[test]
    fn test_filter_wrong_sensor_count() {
        let mut filter = create_test_filter();
        let mut rng = rand::thread_rng();

        // Only 1 sensor instead of 2
        let measurements = vec![vec![DVector::from_vec(vec![0.0, 0.0])]];

        let result = filter.step(&mut rng, &measurements, 0);
        assert!(result.is_err());
    }
}
