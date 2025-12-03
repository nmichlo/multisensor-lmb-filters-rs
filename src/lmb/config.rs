//! Configuration types for filters
//!
//! This module provides decomposed configuration types that replace
//! the monolithic Model struct with focused, purpose-specific configs.

use nalgebra::{DMatrix, DVector};

/// Motion model parameters for prediction
#[derive(Debug, Clone)]
pub struct MotionModel {
    /// State transition matrix (A)
    pub transition_matrix: DMatrix<f64>,
    /// Process noise covariance (R)
    pub process_noise: DMatrix<f64>,
    /// Control input vector (u)
    pub control_input: DVector<f64>,
    /// Survival probability (probability target persists)
    pub survival_probability: f64,
}

impl MotionModel {
    /// Create a new motion model
    pub fn new(
        transition_matrix: DMatrix<f64>,
        process_noise: DMatrix<f64>,
        control_input: DVector<f64>,
        survival_probability: f64,
    ) -> Self {
        Self {
            transition_matrix,
            process_noise,
            control_input,
            survival_probability,
        }
    }

    /// Get state dimension
    #[inline]
    pub fn x_dim(&self) -> usize {
        self.transition_matrix.nrows()
    }

    /// Create a constant velocity motion model in 2D
    /// State: [x, vx, y, vy]
    pub fn constant_velocity_2d(dt: f64, process_noise_std: f64, survival_prob: f64) -> Self {
        let a = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, dt, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, dt,
                0.0, 0.0, 0.0, 1.0,
            ],
        );

        // Process noise (continuous white noise acceleration)
        let q = process_noise_std * process_noise_std;
        let r = DMatrix::from_row_slice(
            4,
            4,
            &[
                q * dt.powi(3) / 3.0, q * dt.powi(2) / 2.0, 0.0, 0.0,
                q * dt.powi(2) / 2.0, q * dt, 0.0, 0.0,
                0.0, 0.0, q * dt.powi(3) / 3.0, q * dt.powi(2) / 2.0,
                0.0, 0.0, q * dt.powi(2) / 2.0, q * dt,
            ],
        );

        let u = DVector::zeros(4);

        Self::new(a, r, u, survival_prob)
    }
}

/// Sensor observation model
#[derive(Debug, Clone)]
pub struct SensorModel {
    /// Observation matrix (C)
    pub observation_matrix: DMatrix<f64>,
    /// Measurement noise covariance (Q)
    pub measurement_noise: DMatrix<f64>,
    /// Detection probability
    pub detection_probability: f64,
    /// Clutter rate (expected false alarms per timestep)
    pub clutter_rate: f64,
    /// Observation space volume
    pub observation_volume: f64,
}

impl SensorModel {
    /// Create a new sensor model
    pub fn new(
        observation_matrix: DMatrix<f64>,
        measurement_noise: DMatrix<f64>,
        detection_probability: f64,
        clutter_rate: f64,
        observation_volume: f64,
    ) -> Self {
        Self {
            observation_matrix,
            measurement_noise,
            detection_probability,
            clutter_rate,
            observation_volume,
        }
    }

    /// Get measurement dimension
    #[inline]
    pub fn z_dim(&self) -> usize {
        self.observation_matrix.nrows()
    }

    /// Get state dimension
    #[inline]
    pub fn x_dim(&self) -> usize {
        self.observation_matrix.ncols()
    }

    /// Clutter density (clutter per unit volume)
    #[inline]
    pub fn clutter_density(&self) -> f64 {
        self.clutter_rate / self.observation_volume
    }

    /// Create a position-only sensor for 4D state [x, vx, y, vy]
    /// Measures [x, y]
    pub fn position_sensor_2d(
        measurement_noise_std: f64,
        detection_prob: f64,
        clutter_rate: f64,
        obs_volume: f64,
    ) -> Self {
        let c = DMatrix::from_row_slice(
            2,
            4,
            &[
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        );

        let q_var = measurement_noise_std * measurement_noise_std;
        let q = DMatrix::from_diagonal(&DVector::from_vec(vec![q_var, q_var]));

        Self::new(c, q, detection_prob, clutter_rate, obs_volume)
    }
}

/// Multi-sensor configuration
#[derive(Debug, Clone)]
pub struct MultisensorConfig {
    /// Individual sensor models
    pub sensors: Vec<SensorModel>,
}

impl MultisensorConfig {
    /// Create a new multi-sensor configuration
    pub fn new(sensors: Vec<SensorModel>) -> Self {
        Self { sensors }
    }

    /// Number of sensors
    #[inline]
    pub fn num_sensors(&self) -> usize {
        self.sensors.len()
    }

    /// Get sensor by index
    pub fn sensor(&self, idx: usize) -> Option<&SensorModel> {
        self.sensors.get(idx)
    }

    /// Measurement dimension (from first sensor).
    ///
    /// Returns 0 if no sensors are configured.
    ///
    /// # Note on standardization
    ///
    /// This method was extracted to standardize inconsistent implementations:
    /// - `MultisensorLmbFilter` originally used `unwrap_or(2)` (assumed 2D position sensor)
    /// - `MultisensorLmbmFilter` originally used `map_or(0, ...)` (explicit zero)
    ///
    /// We standardized on `0` because it makes misconfiguration explicit rather than
    /// silently assuming a 2D sensor. If you need a default, check `num_sensors() > 0`
    /// before calling this method.
    #[inline]
    pub fn z_dim(&self) -> usize {
        self.sensors.first().map_or(0, |s| s.z_dim())
    }
}

/// Birth location parameters
#[derive(Debug, Clone)]
pub struct BirthLocation {
    /// Location label/index
    pub label: usize,
    /// Birth mean state
    pub mean: DVector<f64>,
    /// Birth covariance
    pub covariance: DMatrix<f64>,
}

impl BirthLocation {
    /// Create a new birth location
    pub fn new(label: usize, mean: DVector<f64>, covariance: DMatrix<f64>) -> Self {
        Self {
            label,
            mean,
            covariance,
        }
    }
}

/// Birth model parameters
#[derive(Debug, Clone)]
pub struct BirthModel {
    /// Birth locations
    pub locations: Vec<BirthLocation>,
    /// Birth existence probability for LMB
    pub lmb_existence: f64,
    /// Birth existence probability for LMBM
    pub lmbm_existence: f64,
}

impl BirthModel {
    /// Create a new birth model
    pub fn new(locations: Vec<BirthLocation>, lmb_existence: f64, lmbm_existence: f64) -> Self {
        Self {
            locations,
            lmb_existence,
            lmbm_existence,
        }
    }

    /// Number of birth locations
    #[inline]
    pub fn num_locations(&self) -> usize {
        self.locations.len()
    }
}

/// Data association method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataAssociationMethod {
    /// Loopy Belief Propagation
    Lbp,
    /// Loopy Belief Propagation with fixed iterations
    LbpFixed,
    /// Gibbs sampling
    Gibbs,
    /// Murty's k-best algorithm
    Murty,
}

/// Data association configuration
#[derive(Debug, Clone)]
pub struct AssociationConfig {
    /// Association method to use
    pub method: DataAssociationMethod,
    /// Maximum LBP iterations
    pub lbp_max_iterations: usize,
    /// LBP convergence tolerance
    pub lbp_tolerance: f64,
    /// Number of Gibbs samples
    pub gibbs_samples: usize,
    /// Number of Murty assignments
    pub murty_assignments: usize,
}

impl AssociationConfig {
    /// Create LBP configuration
    pub fn lbp(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            method: DataAssociationMethod::Lbp,
            lbp_max_iterations: max_iterations,
            lbp_tolerance: tolerance,
            gibbs_samples: 0,
            murty_assignments: 0,
        }
    }

    /// Create Gibbs configuration
    pub fn gibbs(samples: usize) -> Self {
        Self {
            method: DataAssociationMethod::Gibbs,
            lbp_max_iterations: 0,
            lbp_tolerance: 0.0,
            gibbs_samples: samples,
            murty_assignments: 0,
        }
    }

    /// Create Murty configuration
    pub fn murty(assignments: usize) -> Self {
        Self {
            method: DataAssociationMethod::Murty,
            lbp_max_iterations: 0,
            lbp_tolerance: 0.0,
            gibbs_samples: 0,
            murty_assignments: assignments,
        }
    }
}

impl Default for AssociationConfig {
    fn default() -> Self {
        Self::lbp(50, 1e-6)
    }
}

/// Filter threshold configuration
#[derive(Debug, Clone)]
pub struct FilterThresholds {
    /// Existence probability threshold for track confirmation
    pub existence_threshold: f64,
    /// GM component weight threshold for pruning
    pub gm_weight_threshold: f64,
    /// Maximum number of GM components to keep
    pub max_gm_components: usize,
    /// Minimum trajectory length for output
    pub min_trajectory_length: usize,
}

impl FilterThresholds {
    /// Create new filter thresholds
    pub fn new(
        existence_threshold: f64,
        gm_weight_threshold: f64,
        max_gm_components: usize,
        min_trajectory_length: usize,
    ) -> Self {
        Self {
            existence_threshold,
            gm_weight_threshold,
            max_gm_components,
            min_trajectory_length,
        }
    }
}

impl Default for FilterThresholds {
    fn default() -> Self {
        Self {
            existence_threshold: 0.5,
            gm_weight_threshold: 1e-4,
            max_gm_components: 100,
            min_trajectory_length: 3,
        }
    }
}

/// LMBM-specific configuration
#[derive(Debug, Clone)]
pub struct LmbmConfig {
    /// Maximum number of posterior hypotheses
    pub max_hypotheses: usize,
    /// Hypothesis weight threshold
    pub hypothesis_weight_threshold: f64,
    /// Use EAP (Expected A Posteriori) for state extraction
    pub use_eap: bool,
}

impl Default for LmbmConfig {
    fn default() -> Self {
        Self {
            max_hypotheses: 1000,
            hypothesis_weight_threshold: 1e-6,
            use_eap: false,
        }
    }
}

/// Sensor configuration variant (single or multi-sensor)
#[derive(Debug, Clone)]
pub enum SensorVariant {
    /// Single sensor
    Single(SensorModel),
    /// Multiple sensors
    Multi(MultisensorConfig),
}

impl SensorVariant {
    /// Check if this is a multi-sensor configuration
    pub fn is_multisensor(&self) -> bool {
        matches!(self, SensorVariant::Multi(_))
    }

    /// Get number of sensors
    pub fn num_sensors(&self) -> usize {
        match self {
            SensorVariant::Single(_) => 1,
            SensorVariant::Multi(m) => m.num_sensors(),
        }
    }

    /// Get single sensor (panics if multi-sensor)
    pub fn single(&self) -> &SensorModel {
        match self {
            SensorVariant::Single(s) => s,
            SensorVariant::Multi(_) => panic!("Expected single sensor, got multi-sensor"),
        }
    }

    /// Get multi-sensor config (panics if single)
    pub fn multi(&self) -> &MultisensorConfig {
        match self {
            SensorVariant::Single(_) => panic!("Expected multi-sensor, got single sensor"),
            SensorVariant::Multi(m) => m,
        }
    }
}

/// Complete filter parameters
#[derive(Debug, Clone)]
pub struct FilterParams {
    /// Motion model
    pub motion: MotionModel,
    /// Sensor configuration
    pub sensor: SensorVariant,
    /// Birth model
    pub birth: BirthModel,
    /// Association configuration
    pub association: AssociationConfig,
    /// Filter thresholds
    pub thresholds: FilterThresholds,
    /// LMBM-specific config (only used by LMBM filters)
    pub lmbm: LmbmConfig,
}

impl FilterParams {
    /// Create a new filter params builder
    pub fn builder() -> FilterParamsBuilder {
        FilterParamsBuilder::new()
    }

    /// Get state dimension
    #[inline]
    pub fn x_dim(&self) -> usize {
        self.motion.x_dim()
    }

    /// Check if multi-sensor
    #[inline]
    pub fn is_multisensor(&self) -> bool {
        self.sensor.is_multisensor()
    }
}

/// Builder for FilterParams
#[derive(Debug, Default)]
pub struct FilterParamsBuilder {
    motion: Option<MotionModel>,
    sensor: Option<SensorVariant>,
    birth: Option<BirthModel>,
    association: Option<AssociationConfig>,
    thresholds: Option<FilterThresholds>,
    lmbm: Option<LmbmConfig>,
}

impl FilterParamsBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set motion model
    pub fn motion(mut self, motion: MotionModel) -> Self {
        self.motion = Some(motion);
        self
    }

    /// Set single sensor
    pub fn sensor(mut self, sensor: SensorModel) -> Self {
        self.sensor = Some(SensorVariant::Single(sensor));
        self
    }

    /// Set multiple sensors
    pub fn sensors(mut self, sensors: Vec<SensorModel>) -> Self {
        self.sensor = Some(SensorVariant::Multi(MultisensorConfig::new(sensors)));
        self
    }

    /// Set birth model
    pub fn birth(mut self, birth: BirthModel) -> Self {
        self.birth = Some(birth);
        self
    }

    /// Set association config
    pub fn association(mut self, association: AssociationConfig) -> Self {
        self.association = Some(association);
        self
    }

    /// Set filter thresholds
    pub fn thresholds(mut self, thresholds: FilterThresholds) -> Self {
        self.thresholds = Some(thresholds);
        self
    }

    /// Set LMBM config
    pub fn lmbm(mut self, lmbm: LmbmConfig) -> Self {
        self.lmbm = Some(lmbm);
        self
    }

    /// Build the filter params
    pub fn build(self) -> Result<FilterParams, &'static str> {
        Ok(FilterParams {
            motion: self.motion.ok_or("Motion model is required")?,
            sensor: self.sensor.ok_or("Sensor model is required")?,
            birth: self.birth.ok_or("Birth model is required")?,
            association: self.association.unwrap_or_default(),
            thresholds: self.thresholds.unwrap_or_default(),
            lmbm: self.lmbm.unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_model_cv_2d() {
        let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);
        assert_eq!(motion.x_dim(), 4);
        assert_eq!(motion.survival_probability, 0.99);
    }

    #[test]
    fn test_sensor_model_position() {
        let sensor = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
        assert_eq!(sensor.z_dim(), 2);
        assert_eq!(sensor.x_dim(), 4);
        assert!((sensor.clutter_density() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_filter_params_builder() {
        let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);
        let sensor = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
        let birth = BirthModel::new(vec![], 0.1, 0.01);

        let params = FilterParams::builder()
            .motion(motion)
            .sensor(sensor)
            .birth(birth)
            .association(AssociationConfig::lbp(50, 1e-6))
            .build()
            .unwrap();

        assert_eq!(params.x_dim(), 4);
        assert!(!params.is_multisensor());
    }
}
