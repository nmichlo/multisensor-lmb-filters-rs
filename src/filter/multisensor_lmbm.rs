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

use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector};

use crate::common::linalg::{log_gaussian_normalizing_constant, robust_inverse};
use crate::components::prediction::predict_tracks;
use crate::types::{
    AssociationConfig, BirthModel, FilterParams, GaussianComponent, LmbmConfig, LmbmHypothesis,
    MotionModel, MultisensorConfig, StateEstimate, Track, Trajectory,
};

use super::errors::FilterError;
use super::multisensor_lmb::MultisensorMeasurements;
use super::multisensor_traits::{MultisensorAssociator, MultisensorGibbsAssociator};
use super::traits::Filter;

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

    /// Phantom data for associator type
    _associator: PhantomData<A>,
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
        Self::with_associator(motion, sensors, birth, association_config, lmbm_config)
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
    pub fn with_associator(
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association_config: AssociationConfig,
        lmbm_config: LmbmConfig,
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
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            _associator: PhantomData,
        }
    }

    /// Set the existence threshold for gating.
    pub fn with_existence_threshold(mut self, threshold: f64) -> Self {
        self.existence_threshold = threshold;
        self
    }

    /// Set the minimum trajectory length for keeping discarded tracks.
    pub fn with_min_trajectory_length(mut self, length: usize) -> Self {
        self.min_trajectory_length = length;
        self
    }

    /// Number of sensors.
    #[inline]
    pub fn num_sensors(&self) -> usize {
        self.sensors.num_sensors()
    }

    /// Predict all hypotheses forward in time.
    fn predict_hypotheses(&mut self, timestep: usize) {
        for hyp in &mut self.hypotheses {
            predict_tracks(&mut hyp.tracks, &self.motion, &self.birth, timestep, true);
        }
    }

    /// Update existence probabilities when there are no measurements from any sensor.
    fn update_existence_no_measurements(&mut self) {
        let detection_probs: Vec<f64> = self
            .sensors
            .sensors
            .iter()
            .map(|s| s.detection_probability)
            .collect();
        for hyp in &mut self.hypotheses {
            for track in &mut hyp.tracks {
                track.existence =
                    crate::components::update::update_existence_no_detection_multisensor(
                        track.existence,
                        &detection_probs,
                    );
            }
        }
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
            let (log_l, posterior) =
                self.compute_log_likelihood(obj_idx, &associations, tracks, measurements);

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
            ell = ell - page_sizes[j] * (zeta - if eta == 0 { 1 } else { 0 });
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
            let z_dim = self.sensors.sensors[0].z_dim(); // Assume uniform z_dim
            let z_dim_total = z_dim * num_detections;
            let x_dim = self.motion.x_dim();

            let mut z = DVector::zeros(z_dim_total);
            let mut c = DMatrix::zeros(z_dim_total, x_dim);
            let mut q_blocks = Vec::new();

            let mut counter = 0;
            for s in 0..num_sensors {
                if detecting[s] {
                    let sensor = &self.sensors.sensors[s];
                    let meas_idx = associations[s] - 1; // Convert to 0-indexed
                    let start = z_dim * counter;

                    // Copy measurement
                    z.rows_mut(start, z_dim)
                        .copy_from(&measurements[s][meas_idx]);

                    // Copy observation matrix
                    c.view_mut((start, 0), (z_dim, x_dim))
                        .copy_from(&sensor.observation_matrix);

                    // Collect noise covariance
                    q_blocks.push(sensor.measurement_noise.clone());

                    counter += 1;
                }
            }

            // Build block-diagonal Q
            let mut q = DMatrix::zeros(z_dim_total, z_dim_total);
            let mut offset = 0;
            for q_block in &q_blocks {
                q.view_mut((offset, offset), (z_dim, z_dim)).copy_from(q_block);
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
            for s in 0..num_sensors {
                let p_d = self.sensors.sensors[s].detection_probability;
                log_pd += if detecting[s] {
                    p_d.ln()
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
            for sensor in &self.sensors.sensors {
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

        let mut new_hypotheses = Vec::new();

        for prior_hyp in &self.hypotheses {
            for sample in samples {
                // Create new hypothesis from prior
                let mut new_hyp = prior_hyp.clone();

                // Decode sample: sample is flattened [v_{1,1}, v_{1,2}, ..., v_{n,S}] column-major
                // So sample[s * num_objects + i] = v[i, s]
                let mut log_likelihood_sum = 0.0;

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

                    // Accumulate log-likelihood
                    log_likelihood_sum += log_likelihoods[ell];

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

                new_hyp.log_weight += log_likelihood_sum;
                new_hypotheses.push(new_hyp);
            }
        }

        self.hypotheses = new_hypotheses;
    }

    /// Normalize and gate hypotheses.
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

    /// Update track trajectories.
    fn update_trajectories(&mut self, timestamp: usize) {
        super::common_ops::update_hypothesis_trajectories(&mut self.hypotheses, timestamp);
    }

    /// Initialize trajectory recording for birth tracks.
    fn init_birth_trajectories(&mut self, max_length: usize) {
        super::common_ops::init_hypothesis_birth_trajectories(&mut self.hypotheses, max_length);
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
        // Validate measurements
        if measurements.len() != self.num_sensors() {
            return Err(FilterError::InvalidInput(format!(
                "Expected {} sensors, got {}",
                self.num_sensors(),
                measurements.len()
            )));
        }

        // 1. Prediction
        self.predict_hypotheses(timestep);
        self.init_birth_trajectories(1000);

        // 2. Measurement update
        let has_measurements = measurements.iter().any(|m| !m.is_empty());

        if has_measurements && !self.hypotheses.is_empty() && !self.hypotheses[0].tracks.is_empty()
        {
            // Generate joint association matrices
            let (log_likelihoods, posteriors, dimensions) =
                self.generate_association_matrices(&self.hypotheses[0].tracks, measurements);

            // Run multi-sensor association using the associator trait
            let associator = A::default();
            let association_result = associator
                .associate(rng, &log_likelihoods, &dimensions, &self.association_config)
                .map_err(|e| FilterError::Association(e))?;

            // Generate posterior hypotheses
            self.generate_posterior_hypotheses(
                &association_result.samples,
                &log_likelihoods,
                &posteriors,
                &dimensions,
            );
        } else if !has_measurements {
            // No measurements from any sensor
            self.update_existence_no_measurements();
        }

        // 3. Normalize and gate hypotheses
        self.normalize_and_gate_hypotheses();

        // 4. Gate tracks
        self.gate_tracks();

        // 5. Update trajectories
        self.update_trajectories(timestep);

        // 6. Extract estimates
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
        self.sensors.sensors.get(0).map_or(0, |s| s.z_dim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BirthLocation, SensorModel};

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
            vec![DVector::from_vec(vec![0.0, 0.0])],  // Sensor 1
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
