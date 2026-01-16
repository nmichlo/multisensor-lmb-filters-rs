//! Configuration types for filters
//!
//! This module provides decomposed configuration types that replace
//! the monolithic Model struct with focused, purpose-specific configs.

use nalgebra::{DMatrix, DVector};
use serde::Serialize;

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
    /// State: [x, y, vx, vy] (matches MATLAB convention)
    ///
    /// Transition matrix A = [I, dt*I; 0, I] where I is 2x2 identity
    /// Process noise R = q * [(dt³/3)*I, (dt²/2)*I; (dt²/2)*I, dt*I]
    pub fn constant_velocity_2d(dt: f64, process_noise_std: f64, survival_prob: f64) -> Self {
        // A = [eye(2), dt*eye(2); zeros(2), eye(2)]
        // State ordering: [x, y, vx, vy]
        #[rustfmt::skip]
        let a = DMatrix::from_row_slice(4, 4, &[
            1.0, 0.0, dt,  0.0,   // x' = x + dt*vx
            0.0, 1.0, 0.0, dt,    // y' = y + dt*vy
            0.0, 0.0, 1.0, 0.0,   // vx' = vx
            0.0, 0.0, 0.0, 1.0,   // vy' = vy
        ]);

        // Process noise (continuous white noise acceleration)
        // R = q * [(1/3)*dt^3*eye(2), 0.5*dt^2*eye(2); 0.5*dt^2*eye(2), dt*eye(2)]
        let q = process_noise_std * process_noise_std;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        #[rustfmt::skip]
        let r = DMatrix::from_row_slice(4, 4, &[
            q * dt3 / 3.0,  0.0,            q * dt2 / 2.0,  0.0,
            0.0,            q * dt3 / 3.0,  0.0,            q * dt2 / 2.0,
            q * dt2 / 2.0,  0.0,            q * dt,         0.0,
            0.0,            q * dt2 / 2.0,  0.0,            q * dt,
        ]);

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

    /// Create a position-only sensor for 4D state [x, y, vx, vy]
    /// Measures [x, y] (matches MATLAB convention)
    ///
    /// Observation matrix C = [eye(2), zeros(2)]
    pub fn position_sensor_2d(
        measurement_noise_std: f64,
        detection_prob: f64,
        clutter_rate: f64,
        obs_volume: f64,
    ) -> Self {
        // C = [eye(2), zeros(2)] for state [x, y, vx, vy]
        #[rustfmt::skip]
        let c = DMatrix::from_row_slice(2, 4, &[
            1.0, 0.0, 0.0, 0.0,   // z[0] = x
            0.0, 1.0, 0.0, 0.0,   // z[1] = y
        ]);

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
    /// Mahalanobis distance threshold for GM component merging (MATLAB-compatible).
    /// Set to `f64::INFINITY` to disable merging (faster but not MATLAB-equivalent).
    pub gm_merge_threshold: f64,
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
            gm_merge_threshold: super::DEFAULT_GM_MERGE_THRESHOLD,
        }
    }

    /// Create with custom merge threshold
    pub fn with_merge_threshold(
        existence_threshold: f64,
        gm_weight_threshold: f64,
        max_gm_components: usize,
        min_trajectory_length: usize,
        gm_merge_threshold: f64,
    ) -> Self {
        Self {
            existence_threshold,
            gm_weight_threshold,
            max_gm_components,
            min_trajectory_length,
            gm_merge_threshold,
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
            gm_merge_threshold: super::DEFAULT_GM_MERGE_THRESHOLD,
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

// ============================================================================
// Type-Safe Filter Configurations
// ============================================================================
//
// These configuration types make illegal states unrepresentable:
// - LmbFilterConfig: Used ONLY for LMB filters (has GM-specific fields)
// - LmbmFilterConfig: Used ONLY for LMBM filters (has hypothesis-specific fields)
//
// You cannot accidentally set `max_hypotheses` on an LMB filter because the
// field doesn't exist in LmbFilterConfig.

/// Common configuration shared by all filter types.
///
/// These settings apply to both LMB and LMBM filters and control
/// track lifecycle management.
#[derive(Debug, Clone)]
pub struct CommonConfig {
    /// Existence probability threshold for track gating.
    /// Tracks with existence below this threshold are pruned.
    pub existence_threshold: f64,

    /// Minimum trajectory length to keep when pruning.
    /// Short-lived tracks are discarded without saving their trajectory.
    pub min_trajectory_length: usize,

    /// Maximum trajectory length for track history recording.
    /// Older history entries are discarded when this limit is exceeded.
    pub max_trajectory_length: usize,
}

impl CommonConfig {
    /// Create a new common configuration.
    pub fn new(
        existence_threshold: f64,
        min_trajectory_length: usize,
        max_trajectory_length: usize,
    ) -> Self {
        Self {
            existence_threshold,
            min_trajectory_length,
            max_trajectory_length,
        }
    }

    /// Create with builder pattern.
    pub fn builder() -> CommonConfigBuilder {
        CommonConfigBuilder::default()
    }
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self {
            existence_threshold: super::DEFAULT_EXISTENCE_THRESHOLD,
            min_trajectory_length: super::DEFAULT_MIN_TRAJECTORY_LENGTH,
            max_trajectory_length: super::DEFAULT_MAX_TRAJECTORY_LENGTH,
        }
    }
}

/// Builder for CommonConfig.
#[derive(Debug, Default)]
pub struct CommonConfigBuilder {
    existence_threshold: Option<f64>,
    min_trajectory_length: Option<usize>,
    max_trajectory_length: Option<usize>,
}

impl CommonConfigBuilder {
    /// Set existence probability threshold.
    pub fn existence_threshold(mut self, threshold: f64) -> Self {
        self.existence_threshold = Some(threshold);
        self
    }

    /// Set minimum trajectory length.
    pub fn min_trajectory_length(mut self, length: usize) -> Self {
        self.min_trajectory_length = Some(length);
        self
    }

    /// Set maximum trajectory length.
    pub fn max_trajectory_length(mut self, length: usize) -> Self {
        self.max_trajectory_length = Some(length);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> CommonConfig {
        CommonConfig {
            existence_threshold: self
                .existence_threshold
                .unwrap_or(super::DEFAULT_EXISTENCE_THRESHOLD),
            min_trajectory_length: self
                .min_trajectory_length
                .unwrap_or(super::DEFAULT_MIN_TRAJECTORY_LENGTH),
            max_trajectory_length: self
                .max_trajectory_length
                .unwrap_or(super::DEFAULT_MAX_TRAJECTORY_LENGTH),
        }
    }
}

/// LMB filter configuration (Gaussian mixture posteriors).
///
/// This configuration type is used exclusively by LMB-family filters
/// (not LMBM). It contains settings for Gaussian mixture component
/// management that don't apply to LMBM filters.
///
/// # Example
///
/// ```ignore
/// use multisensor_lmb_filters_rs::lmb::LmbFilterConfig;
///
/// let config = LmbFilterConfig::builder()
///     .existence_threshold(0.001)
///     .max_gm_components(50)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct LmbFilterConfig {
    /// Common settings (existence threshold, trajectory lengths).
    pub common: CommonConfig,

    /// GM component weight threshold for pruning.
    /// Components with weight below this threshold are pruned.
    pub gm_weight_threshold: f64,

    /// Maximum number of GM components per track.
    pub max_gm_components: usize,

    /// Mahalanobis distance threshold for GM component merging.
    /// Components closer than this threshold are merged.
    /// Set to `f64::INFINITY` to disable merging.
    pub gm_merge_threshold: f64,
}

impl LmbFilterConfig {
    /// Create a new LMB filter configuration.
    pub fn new(
        common: CommonConfig,
        gm_weight_threshold: f64,
        max_gm_components: usize,
        gm_merge_threshold: f64,
    ) -> Self {
        Self {
            common,
            gm_weight_threshold,
            max_gm_components,
            gm_merge_threshold,
        }
    }

    /// Create with builder pattern.
    pub fn builder() -> LmbFilterConfigBuilder {
        LmbFilterConfigBuilder::default()
    }

    /// Convenience accessor for existence threshold.
    #[inline]
    pub fn existence_threshold(&self) -> f64 {
        self.common.existence_threshold
    }

    /// Convenience accessor for min trajectory length.
    #[inline]
    pub fn min_trajectory_length(&self) -> usize {
        self.common.min_trajectory_length
    }

    /// Convenience accessor for max trajectory length.
    #[inline]
    pub fn max_trajectory_length(&self) -> usize {
        self.common.max_trajectory_length
    }
}

impl Default for LmbFilterConfig {
    fn default() -> Self {
        Self {
            common: CommonConfig::default(),
            gm_weight_threshold: super::DEFAULT_GM_WEIGHT_THRESHOLD,
            max_gm_components: super::DEFAULT_MAX_GM_COMPONENTS,
            gm_merge_threshold: super::DEFAULT_GM_MERGE_THRESHOLD,
        }
    }
}

/// Builder for LmbFilterConfig.
#[derive(Debug, Default)]
pub struct LmbFilterConfigBuilder {
    common: Option<CommonConfig>,
    existence_threshold: Option<f64>,
    min_trajectory_length: Option<usize>,
    max_trajectory_length: Option<usize>,
    gm_weight_threshold: Option<f64>,
    max_gm_components: Option<usize>,
    gm_merge_threshold: Option<f64>,
}

impl LmbFilterConfigBuilder {
    /// Set the common configuration (overrides individual common fields).
    pub fn common(mut self, common: CommonConfig) -> Self {
        self.common = Some(common);
        self
    }

    /// Set existence probability threshold.
    pub fn existence_threshold(mut self, threshold: f64) -> Self {
        self.existence_threshold = Some(threshold);
        self
    }

    /// Set minimum trajectory length.
    pub fn min_trajectory_length(mut self, length: usize) -> Self {
        self.min_trajectory_length = Some(length);
        self
    }

    /// Set maximum trajectory length.
    pub fn max_trajectory_length(mut self, length: usize) -> Self {
        self.max_trajectory_length = Some(length);
        self
    }

    /// Set GM component weight threshold.
    pub fn gm_weight_threshold(mut self, threshold: f64) -> Self {
        self.gm_weight_threshold = Some(threshold);
        self
    }

    /// Set maximum GM components per track.
    pub fn max_gm_components(mut self, max: usize) -> Self {
        self.max_gm_components = Some(max);
        self
    }

    /// Set GM merge threshold (Mahalanobis distance).
    /// Use `f64::INFINITY` to disable merging.
    pub fn gm_merge_threshold(mut self, threshold: f64) -> Self {
        self.gm_merge_threshold = Some(threshold);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> LmbFilterConfig {
        let common = self.common.unwrap_or_else(|| CommonConfig {
            existence_threshold: self
                .existence_threshold
                .unwrap_or(super::DEFAULT_EXISTENCE_THRESHOLD),
            min_trajectory_length: self
                .min_trajectory_length
                .unwrap_or(super::DEFAULT_MIN_TRAJECTORY_LENGTH),
            max_trajectory_length: self
                .max_trajectory_length
                .unwrap_or(super::DEFAULT_MAX_TRAJECTORY_LENGTH),
        });

        LmbFilterConfig {
            common,
            gm_weight_threshold: self
                .gm_weight_threshold
                .unwrap_or(super::DEFAULT_GM_WEIGHT_THRESHOLD),
            max_gm_components: self
                .max_gm_components
                .unwrap_or(super::DEFAULT_MAX_GM_COMPONENTS),
            gm_merge_threshold: self
                .gm_merge_threshold
                .unwrap_or(super::DEFAULT_GM_MERGE_THRESHOLD),
        }
    }
}

/// LMBM filter configuration (hypothesis mixture posteriors).
///
/// This configuration type is used exclusively by LMBM-family filters
/// (not LMB). It contains settings for hypothesis management that
/// don't apply to LMB filters.
///
/// # Example
///
/// ```ignore
/// use multisensor_lmb_filters_rs::lmb::LmbmFilterConfig;
///
/// let config = LmbmFilterConfig::builder()
///     .existence_threshold(0.001)
///     .max_hypotheses(500)
///     .use_eap(true)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct LmbmFilterConfig {
    /// Common settings (existence threshold, trajectory lengths).
    pub common: CommonConfig,

    /// Maximum number of posterior hypotheses to maintain.
    pub max_hypotheses: usize,

    /// Hypothesis weight threshold for pruning.
    /// Hypotheses with weight below this threshold are pruned.
    pub hypothesis_weight_threshold: f64,

    /// Use EAP (Expected A Posteriori) for state extraction.
    /// If false, uses MAP (Maximum A Posteriori).
    pub use_eap: bool,
}

impl LmbmFilterConfig {
    /// Create a new LMBM filter configuration.
    pub fn new(
        common: CommonConfig,
        max_hypotheses: usize,
        hypothesis_weight_threshold: f64,
        use_eap: bool,
    ) -> Self {
        Self {
            common,
            max_hypotheses,
            hypothesis_weight_threshold,
            use_eap,
        }
    }

    /// Create with builder pattern.
    pub fn builder() -> LmbmFilterConfigBuilder {
        LmbmFilterConfigBuilder::default()
    }

    /// Convenience accessor for existence threshold.
    #[inline]
    pub fn existence_threshold(&self) -> f64 {
        self.common.existence_threshold
    }

    /// Convenience accessor for min trajectory length.
    #[inline]
    pub fn min_trajectory_length(&self) -> usize {
        self.common.min_trajectory_length
    }

    /// Convenience accessor for max trajectory length.
    #[inline]
    pub fn max_trajectory_length(&self) -> usize {
        self.common.max_trajectory_length
    }

    /// Convert to the legacy LmbmConfig for backward compatibility.
    pub fn to_legacy_lmbm_config(&self) -> LmbmConfig {
        LmbmConfig {
            max_hypotheses: self.max_hypotheses,
            hypothesis_weight_threshold: self.hypothesis_weight_threshold,
            use_eap: self.use_eap,
        }
    }
}

impl Default for LmbmFilterConfig {
    fn default() -> Self {
        Self {
            common: CommonConfig::default(),
            max_hypotheses: super::DEFAULT_LMBM_MAX_HYPOTHESES,
            hypothesis_weight_threshold: super::DEFAULT_LMBM_WEIGHT_THRESHOLD,
            use_eap: false,
        }
    }
}

/// Builder for LmbmFilterConfig.
#[derive(Debug, Default)]
pub struct LmbmFilterConfigBuilder {
    common: Option<CommonConfig>,
    existence_threshold: Option<f64>,
    min_trajectory_length: Option<usize>,
    max_trajectory_length: Option<usize>,
    max_hypotheses: Option<usize>,
    hypothesis_weight_threshold: Option<f64>,
    use_eap: Option<bool>,
}

impl LmbmFilterConfigBuilder {
    /// Set the common configuration (overrides individual common fields).
    pub fn common(mut self, common: CommonConfig) -> Self {
        self.common = Some(common);
        self
    }

    /// Set existence probability threshold.
    pub fn existence_threshold(mut self, threshold: f64) -> Self {
        self.existence_threshold = Some(threshold);
        self
    }

    /// Set minimum trajectory length.
    pub fn min_trajectory_length(mut self, length: usize) -> Self {
        self.min_trajectory_length = Some(length);
        self
    }

    /// Set maximum trajectory length.
    pub fn max_trajectory_length(mut self, length: usize) -> Self {
        self.max_trajectory_length = Some(length);
        self
    }

    /// Set maximum number of hypotheses.
    pub fn max_hypotheses(mut self, max: usize) -> Self {
        self.max_hypotheses = Some(max);
        self
    }

    /// Set hypothesis weight threshold.
    pub fn hypothesis_weight_threshold(mut self, threshold: f64) -> Self {
        self.hypothesis_weight_threshold = Some(threshold);
        self
    }

    /// Set whether to use EAP for state extraction.
    pub fn use_eap(mut self, use_eap: bool) -> Self {
        self.use_eap = Some(use_eap);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> LmbmFilterConfig {
        let common = self.common.unwrap_or_else(|| CommonConfig {
            existence_threshold: self
                .existence_threshold
                .unwrap_or(super::DEFAULT_EXISTENCE_THRESHOLD),
            min_trajectory_length: self
                .min_trajectory_length
                .unwrap_or(super::DEFAULT_MIN_TRAJECTORY_LENGTH),
            max_trajectory_length: self
                .max_trajectory_length
                .unwrap_or(super::DEFAULT_MAX_TRAJECTORY_LENGTH),
        });

        LmbmFilterConfig {
            common,
            max_hypotheses: self
                .max_hypotheses
                .unwrap_or(super::DEFAULT_LMBM_MAX_HYPOTHESES),
            hypothesis_weight_threshold: self
                .hypothesis_weight_threshold
                .unwrap_or(super::DEFAULT_LMBM_WEIGHT_THRESHOLD),
            use_eap: self.use_eap.unwrap_or(false),
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

// ============================================================================
// Configuration Snapshots (for debugging/comparison)
// ============================================================================

/// Snapshot of motion model configuration for debugging.
#[derive(Debug, Clone, Serialize)]
pub struct MotionModelSnapshot {
    /// State dimension
    pub x_dim: usize,
    /// Survival probability
    pub survival_probability: f64,
    /// State transition matrix A (flattened row-major)
    pub transition_matrix: Vec<f64>,
    /// Process noise covariance Q (flattened row-major)
    pub process_noise: Vec<f64>,
}

impl From<&MotionModel> for MotionModelSnapshot {
    fn from(m: &MotionModel) -> Self {
        Self {
            x_dim: m.x_dim(),
            survival_probability: m.survival_probability,
            transition_matrix: m.transition_matrix.iter().copied().collect(),
            process_noise: m.process_noise.iter().copied().collect(),
        }
    }
}

/// Snapshot of sensor model configuration for debugging.
#[derive(Debug, Clone, Serialize)]
pub struct SensorModelSnapshot {
    /// Measurement dimension
    pub z_dim: usize,
    /// State dimension
    pub x_dim: usize,
    /// Detection probability
    pub detection_probability: f64,
    /// Clutter rate (expected false alarms per timestep)
    pub clutter_rate: f64,
    /// Observation space volume
    pub observation_volume: f64,
    /// Clutter density (derived: clutter_rate / observation_volume)
    pub clutter_density: f64,
    /// Observation matrix C (flattened row-major)
    pub observation_matrix: Vec<f64>,
    /// Measurement noise covariance R (flattened row-major)
    pub measurement_noise: Vec<f64>,
}

impl From<&SensorModel> for SensorModelSnapshot {
    fn from(s: &SensorModel) -> Self {
        Self {
            z_dim: s.z_dim(),
            x_dim: s.x_dim(),
            detection_probability: s.detection_probability,
            clutter_rate: s.clutter_rate,
            observation_volume: s.observation_volume,
            clutter_density: s.clutter_density(),
            observation_matrix: s.observation_matrix.iter().copied().collect(),
            measurement_noise: s.measurement_noise.iter().copied().collect(),
        }
    }
}

/// Snapshot of a single birth location for debugging.
#[derive(Debug, Clone, Serialize)]
pub struct BirthLocationSnapshot {
    /// Location label/index
    pub label: usize,
    /// Birth mean state (flattened)
    pub mean: Vec<f64>,
    /// Birth covariance diagonal (for readability - full matrix available separately)
    pub covariance_diag: Vec<f64>,
}

impl From<&BirthLocation> for BirthLocationSnapshot {
    fn from(loc: &BirthLocation) -> Self {
        Self {
            label: loc.label,
            mean: loc.mean.iter().copied().collect(),
            covariance_diag: loc.covariance.diagonal().iter().copied().collect(),
        }
    }
}

/// Snapshot of birth model configuration for debugging.
#[derive(Debug, Clone, Serialize)]
pub struct BirthModelSnapshot {
    /// Number of birth locations
    pub num_locations: usize,
    /// Birth existence probability for LMB filters
    pub lmb_existence: f64,
    /// Birth existence probability for LMBM filters
    pub lmbm_existence: f64,
    /// Details of each birth location
    pub locations: Vec<BirthLocationSnapshot>,
}

impl From<&BirthModel> for BirthModelSnapshot {
    fn from(b: &BirthModel) -> Self {
        Self {
            num_locations: b.num_locations(),
            lmb_existence: b.lmb_existence,
            lmbm_existence: b.lmbm_existence,
            locations: b.locations.iter().map(|l| l.into()).collect(),
        }
    }
}

/// Snapshot of association configuration for debugging.
#[derive(Debug, Clone, Serialize)]
pub struct AssociationConfigSnapshot {
    /// Association method name
    pub method: String,
    /// LBP max iterations (if applicable)
    pub lbp_max_iterations: usize,
    /// LBP convergence tolerance (if applicable)
    pub lbp_tolerance: f64,
    /// Gibbs samples (if applicable)
    pub gibbs_samples: usize,
    /// Murty assignments (if applicable)
    pub murty_assignments: usize,
}

impl From<&AssociationConfig> for AssociationConfigSnapshot {
    fn from(a: &AssociationConfig) -> Self {
        let method = match a.method {
            DataAssociationMethod::Lbp => "LBP",
            DataAssociationMethod::LbpFixed => "LBP-Fixed",
            DataAssociationMethod::Gibbs => "Gibbs",
            DataAssociationMethod::Murty => "Murty",
        }
        .to_string();
        Self {
            method,
            lbp_max_iterations: a.lbp_max_iterations,
            lbp_tolerance: a.lbp_tolerance,
            gibbs_samples: a.gibbs_samples,
            murty_assignments: a.murty_assignments,
        }
    }
}

/// Snapshot of filter thresholds for debugging.
#[derive(Debug, Clone, Serialize)]
pub struct ThresholdsSnapshot {
    /// Existence probability threshold
    pub existence_threshold: f64,
    /// GM component weight threshold
    pub gm_weight_threshold: f64,
    /// Maximum GM components per track
    pub max_gm_components: usize,
    /// Minimum trajectory length
    pub min_trajectory_length: usize,
    /// GM merge threshold (Mahalanobis distance), None means infinity (disabled)
    #[serde(serialize_with = "serialize_option_f64")]
    pub gm_merge_threshold: Option<f64>,
}

/// Custom serializer for Option<f64> that outputs null for None
fn serialize_option_f64<S>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match value {
        Some(v) => serializer.serialize_f64(*v),
        None => serializer.serialize_none(),
    }
}

/// Snapshot of LMBM-specific configuration for debugging.
#[derive(Debug, Clone, Serialize)]
pub struct LmbmConfigSnapshot {
    /// Maximum number of hypotheses
    pub max_hypotheses: usize,
    /// Hypothesis weight threshold
    pub hypothesis_weight_threshold: f64,
    /// Use EAP for state extraction
    pub use_eap: bool,
}

impl From<&LmbmConfig> for LmbmConfigSnapshot {
    fn from(c: &LmbmConfig) -> Self {
        Self {
            max_hypotheses: c.max_hypotheses,
            hypothesis_weight_threshold: c.hypothesis_weight_threshold,
            use_eap: c.use_eap,
        }
    }
}

/// Complete filter configuration snapshot for debugging and comparison.
///
/// This struct captures all configuration parameters used to initialize a filter,
/// making it easy to compare configurations across implementations (Rust, Python, Octave).
///
/// # Example
///
/// ```rust,ignore
/// use multisensor_lmb_filters_rs::lmb::*;
///
/// let filter = LmbFilter::new(motion, sensor, birth, assoc_config);
/// let config = filter.get_config();
/// println!("{}", serde_json::to_string_pretty(&config).unwrap());
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct FilterConfigSnapshot {
    /// Filter type identifier
    pub filter_type: String,
    /// Motion model configuration
    pub motion: MotionModelSnapshot,
    /// Sensor configuration (single sensor or summary for multi-sensor)
    pub sensor: SensorModelSnapshot,
    /// Number of sensors (1 for single-sensor filters)
    pub num_sensors: usize,
    /// Birth model configuration
    pub birth: BirthModelSnapshot,
    /// Association configuration
    pub association: AssociationConfigSnapshot,
    /// Filter thresholds
    pub thresholds: ThresholdsSnapshot,
    /// LMBM-specific config (None for LMB filters)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lmbm_config: Option<LmbmConfigSnapshot>,
}

impl FilterConfigSnapshot {
    /// Create a snapshot for a single-sensor LMB-style filter.
    #[allow(clippy::too_many_arguments)]
    pub fn single_sensor_lmb(
        filter_type: &str,
        motion: &MotionModel,
        sensor: &SensorModel,
        birth: &BirthModel,
        association: &AssociationConfig,
        existence_threshold: f64,
        gm_weight_threshold: f64,
        max_gm_components: usize,
        min_trajectory_length: usize,
        gm_merge_threshold: f64,
    ) -> Self {
        Self {
            filter_type: filter_type.to_string(),
            motion: motion.into(),
            sensor: sensor.into(),
            num_sensors: 1,
            birth: birth.into(),
            association: association.into(),
            thresholds: ThresholdsSnapshot {
                existence_threshold,
                gm_weight_threshold,
                max_gm_components,
                min_trajectory_length,
                gm_merge_threshold: if gm_merge_threshold.is_infinite() {
                    None
                } else {
                    Some(gm_merge_threshold)
                },
            },
            lmbm_config: None,
        }
    }

    /// Create a snapshot for a single-sensor LMBM-style filter.
    #[allow(clippy::too_many_arguments)]
    pub fn single_sensor_lmbm(
        filter_type: &str,
        motion: &MotionModel,
        sensor: &SensorModel,
        birth: &BirthModel,
        association: &AssociationConfig,
        existence_threshold: f64,
        min_trajectory_length: usize,
        lmbm_config: &LmbmConfig,
    ) -> Self {
        Self {
            filter_type: filter_type.to_string(),
            motion: motion.into(),
            sensor: sensor.into(),
            num_sensors: 1,
            birth: birth.into(),
            association: association.into(),
            thresholds: ThresholdsSnapshot {
                existence_threshold,
                gm_weight_threshold: 0.0, // Not used by LMBM
                max_gm_components: 0,     // Not used by LMBM
                min_trajectory_length,
                gm_merge_threshold: None, // Not used by LMBM
            },
            lmbm_config: Some(lmbm_config.into()),
        }
    }

    /// Create a snapshot for a multi-sensor LMB-style filter.
    #[allow(clippy::too_many_arguments)]
    pub fn multi_sensor_lmb(
        filter_type: &str,
        motion: &MotionModel,
        sensors: &MultisensorConfig,
        birth: &BirthModel,
        association: &AssociationConfig,
        existence_threshold: f64,
        gm_weight_threshold: f64,
        max_gm_components: usize,
        min_trajectory_length: usize,
        gm_merge_threshold: f64,
    ) -> Self {
        // Use first sensor as representative (all sensors typically identical in benchmarks)
        let sensor_snapshot =
            sensors
                .sensors
                .first()
                .map(|s| s.into())
                .unwrap_or(SensorModelSnapshot {
                    z_dim: 0,
                    x_dim: 0,
                    detection_probability: 0.0,
                    clutter_rate: 0.0,
                    observation_volume: 0.0,
                    clutter_density: 0.0,
                    observation_matrix: vec![],
                    measurement_noise: vec![],
                });

        Self {
            filter_type: filter_type.to_string(),
            motion: motion.into(),
            sensor: sensor_snapshot,
            num_sensors: sensors.num_sensors(),
            birth: birth.into(),
            association: association.into(),
            thresholds: ThresholdsSnapshot {
                existence_threshold,
                gm_weight_threshold,
                max_gm_components,
                min_trajectory_length,
                gm_merge_threshold: if gm_merge_threshold.is_infinite() {
                    None
                } else {
                    Some(gm_merge_threshold)
                },
            },
            lmbm_config: None,
        }
    }

    /// Create a snapshot for a multi-sensor LMBM-style filter.
    #[allow(clippy::too_many_arguments)]
    pub fn multi_sensor_lmbm(
        filter_type: &str,
        motion: &MotionModel,
        sensors: &MultisensorConfig,
        birth: &BirthModel,
        association: &AssociationConfig,
        existence_threshold: f64,
        min_trajectory_length: usize,
        lmbm_config: &LmbmConfig,
    ) -> Self {
        let sensor_snapshot =
            sensors
                .sensors
                .first()
                .map(|s| s.into())
                .unwrap_or(SensorModelSnapshot {
                    z_dim: 0,
                    x_dim: 0,
                    detection_probability: 0.0,
                    clutter_rate: 0.0,
                    observation_volume: 0.0,
                    clutter_density: 0.0,
                    observation_matrix: vec![],
                    measurement_noise: vec![],
                });

        Self {
            filter_type: filter_type.to_string(),
            motion: motion.into(),
            sensor: sensor_snapshot,
            num_sensors: sensors.num_sensors(),
            birth: birth.into(),
            association: association.into(),
            thresholds: ThresholdsSnapshot {
                existence_threshold,
                gm_weight_threshold: 0.0, // Not used by LMBM
                max_gm_components: 0,     // Not used by LMBM
                min_trajectory_length,
                gm_merge_threshold: None, // Not used by LMBM
            },
            lmbm_config: Some(lmbm_config.into()),
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Serialize to pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
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

    // ============================================================================
    // Type-Safe Configuration Tests (Phase 2)
    // ============================================================================

    #[test]
    fn test_common_config_default() {
        let config = CommonConfig::default();
        assert_eq!(
            config.existence_threshold,
            super::super::DEFAULT_EXISTENCE_THRESHOLD
        );
        assert_eq!(
            config.min_trajectory_length,
            super::super::DEFAULT_MIN_TRAJECTORY_LENGTH
        );
        assert_eq!(
            config.max_trajectory_length,
            super::super::DEFAULT_MAX_TRAJECTORY_LENGTH
        );
    }

    #[test]
    fn test_common_config_builder() {
        let config = CommonConfig::builder()
            .existence_threshold(0.01)
            .min_trajectory_length(5)
            .max_trajectory_length(500)
            .build();

        assert!((config.existence_threshold - 0.01).abs() < 1e-10);
        assert_eq!(config.min_trajectory_length, 5);
        assert_eq!(config.max_trajectory_length, 500);
    }

    #[test]
    fn test_lmb_filter_config_default() {
        let config = LmbFilterConfig::default();
        assert_eq!(
            config.existence_threshold(),
            super::super::DEFAULT_EXISTENCE_THRESHOLD
        );
        assert_eq!(
            config.gm_weight_threshold,
            super::super::DEFAULT_GM_WEIGHT_THRESHOLD
        );
        assert_eq!(
            config.max_gm_components,
            super::super::DEFAULT_MAX_GM_COMPONENTS
        );
        assert!(config.gm_merge_threshold.is_infinite());
    }

    #[test]
    fn test_lmb_filter_config_builder() {
        let config = LmbFilterConfig::builder()
            .existence_threshold(0.005)
            .gm_weight_threshold(1e-5)
            .max_gm_components(50)
            .gm_merge_threshold(4.0)
            .build();

        assert!((config.existence_threshold() - 0.005).abs() < 1e-10);
        assert!((config.gm_weight_threshold - 1e-5).abs() < 1e-15);
        assert_eq!(config.max_gm_components, 50);
        assert!((config.gm_merge_threshold - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_lmb_filter_config_with_common() {
        let common = CommonConfig::new(0.02, 10, 200);
        let config = LmbFilterConfig::builder()
            .common(common.clone())
            .max_gm_components(75)
            .build();

        // Common fields should come from the CommonConfig
        assert!((config.existence_threshold() - 0.02).abs() < 1e-10);
        assert_eq!(config.min_trajectory_length(), 10);
        assert_eq!(config.max_trajectory_length(), 200);
        // GM-specific field set explicitly
        assert_eq!(config.max_gm_components, 75);
        // GM-specific field from default
        assert_eq!(
            config.gm_weight_threshold,
            super::super::DEFAULT_GM_WEIGHT_THRESHOLD
        );
    }

    #[test]
    fn test_lmbm_filter_config_default() {
        let config = LmbmFilterConfig::default();
        assert_eq!(
            config.existence_threshold(),
            super::super::DEFAULT_EXISTENCE_THRESHOLD
        );
        assert_eq!(
            config.max_hypotheses,
            super::super::DEFAULT_LMBM_MAX_HYPOTHESES
        );
        assert_eq!(
            config.hypothesis_weight_threshold,
            super::super::DEFAULT_LMBM_WEIGHT_THRESHOLD
        );
        assert!(!config.use_eap);
    }

    #[test]
    fn test_lmbm_filter_config_builder() {
        let config = LmbmFilterConfig::builder()
            .existence_threshold(0.002)
            .max_hypotheses(500)
            .hypothesis_weight_threshold(1e-7)
            .use_eap(true)
            .build();

        assert!((config.existence_threshold() - 0.002).abs() < 1e-10);
        assert_eq!(config.max_hypotheses, 500);
        assert!((config.hypothesis_weight_threshold - 1e-7).abs() < 1e-15);
        assert!(config.use_eap);
    }

    #[test]
    fn test_lmbm_filter_config_to_legacy() {
        let config = LmbmFilterConfig::builder()
            .max_hypotheses(200)
            .hypothesis_weight_threshold(1e-4)
            .use_eap(true)
            .build();

        let legacy = config.to_legacy_lmbm_config();
        assert_eq!(legacy.max_hypotheses, 200);
        assert!((legacy.hypothesis_weight_threshold - 1e-4).abs() < 1e-15);
        assert!(legacy.use_eap);
    }

    #[test]
    fn test_type_safety_lmb_has_no_hypothesis_fields() {
        // This test documents that LmbFilterConfig has NO max_hypotheses field.
        // The following would not compile:
        // let config = LmbFilterConfig::builder().max_hypotheses(100).build();
        //
        // We verify type safety by checking LmbFilterConfig has GM-specific fields
        let config = LmbFilterConfig::default();
        let _ = config.max_gm_components; // This exists
        let _ = config.gm_merge_threshold; // This exists
                                           // config.max_hypotheses would NOT exist - that's the point!
    }

    #[test]
    fn test_type_safety_lmbm_has_no_gm_fields() {
        // This test documents that LmbmFilterConfig has NO GM-specific fields.
        // The following would not compile:
        // let config = LmbmFilterConfig::builder().max_gm_components(100).build();
        //
        // We verify type safety by checking LmbmFilterConfig has hypothesis-specific fields
        let config = LmbmFilterConfig::default();
        let _ = config.max_hypotheses; // This exists
        let _ = config.use_eap; // This exists
                                // config.max_gm_components would NOT exist - that's the point!
    }
}
