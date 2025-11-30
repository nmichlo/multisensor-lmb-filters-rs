//! Core data structures for tracking
//!
//! This module defines the fundamental data structures used throughout
//! the tracking library, matching the MATLAB implementation exactly.

use nalgebra::{DMatrix, DVector};

// Re-export ParallelUpdateMode for use in Model
pub use crate::multisensor_lmb::parallel_update::ParallelUpdateMode;

/// Data association method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataAssociationMethod {
    /// Loopy Belief Propagation
    LBP,
    /// Loopy Belief Propagation with fixed iterations
    LBPFixed,
    /// Gibbs sampling
    Gibbs,
    /// Murty's algorithm
    Murty,
}

/// Scenario type for birth locations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioType {
    /// Four fixed birth locations
    Fixed,
    /// Randomly generated birth locations
    Random,
    /// Coalescence scenario
    Coalescence,
}

/// OSPA (Optimal Sub-Pattern Assignment) parameters
#[derive(Debug, Clone)]
pub struct OspaParameters {
    /// Euclidean cut-off
    pub e_c: f64,
    /// Euclidean order parameter
    pub e_p: f64,
    /// Hellinger cut-off
    pub h_c: f64,
    /// Hellinger order parameter
    pub h_p: f64,
}

/// Gaussian mixture component
#[derive(Debug, Clone)]
pub struct GaussianComponent {
    /// Weight
    pub w: f64,
    /// Mean vector
    pub mu: DVector<f64>,
    /// Covariance matrix
    pub sigma: DMatrix<f64>,
}

/// Tracking object (target/track)
///
/// Corresponds to MATLAB's object struct
#[derive(Debug, Clone)]
pub struct Object {
    /// Birth location label
    pub birth_location: usize,
    /// Birth time step
    pub birth_time: usize,
    /// Existence probability
    pub r: f64,
    /// Number of Gaussian mixture components
    pub number_of_gm_components: usize,
    /// GM component weights
    pub w: Vec<f64>,
    /// GM component means (each is a state vector)
    pub mu: Vec<DVector<f64>>,
    /// GM component covariances
    pub sigma: Vec<DMatrix<f64>>,
    /// Length of stored trajectory
    pub trajectory_length: usize,
    /// Historical state estimates (columns are time steps)
    pub trajectory: DMatrix<f64>,
    /// Time indices for trajectory
    pub timestamps: Vec<usize>,
}

impl Object {
    /// Create empty object
    pub fn empty(x_dim: usize) -> Self {
        Self {
            birth_location: 0,
            birth_time: 0,
            r: 0.0,
            number_of_gm_components: 0,
            w: Vec::new(),
            mu: Vec::new(),
            sigma: Vec::new(),
            trajectory_length: 0,
            trajectory: DMatrix::zeros(x_dim, 100), // Pre-allocate 100 timesteps
            timestamps: Vec::new(),
        }
    }
}

/// LMBM hypothesis
///
/// Represents a single hypothesis in the LMBM filter
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Array of birth locations for objects in this hypothesis
    pub birth_location: Vec<usize>,
    /// Array of birth times
    pub birth_time: Vec<usize>,
    /// Hypothesis weight (lowercase sigma in theory)
    pub w: f64,
    /// Array of existence probabilities
    pub r: Vec<f64>,
    /// Array of means (one per object)
    pub mu: Vec<DVector<f64>>,
    /// Array of covariances (one per object)
    pub sigma: Vec<DMatrix<f64>>,
}

impl Hypothesis {
    /// Create empty hypothesis
    pub fn empty() -> Self {
        Self {
            birth_location: Vec::new(),
            birth_time: Vec::new(),
            w: 1.0,
            r: Vec::new(),
            mu: Vec::new(),
            sigma: Vec::new(),
        }
    }
}

/// Trajectory for LMBM
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Birth location label
    pub birth_location: usize,
    /// Birth time
    pub birth_time: usize,
    /// Historical state estimates
    pub trajectory: DMatrix<f64>,
    /// Length of trajectory
    pub trajectory_length: usize,
    /// Time indices
    pub timestamps: Vec<usize>,
}

impl Trajectory {
    /// Create empty trajectory
    pub fn empty(x_dim: usize) -> Self {
        Self {
            birth_location: 0,
            birth_time: 0,
            trajectory: DMatrix::zeros(x_dim, 100),
            trajectory_length: 0,
            timestamps: Vec::new(),
        }
    }
}

/// Complete tracking model
///
/// Contains all parameters and configuration for the tracking system
#[derive(Debug, Clone)]
pub struct Model {
    /// Scenario type
    pub scenario_type: ScenarioType,

    // Dimensions
    /// State space dimension (4D: [x, vx, y, vy])
    pub x_dimension: usize,
    /// Measurement space dimension (2D: [x, y])
    pub z_dimension: usize,

    // Time
    /// Sampling period
    pub t: f64,

    // Motion model (linear Gaussian)
    /// Survival probability
    pub survival_probability: f64,
    /// Existence threshold for pruning
    pub existence_threshold: f64,
    /// State transition matrix A
    pub a: DMatrix<f64>,
    /// Control input u
    pub u: DVector<f64>,
    /// Process noise covariance R
    pub r: DMatrix<f64>,

    // Observation model (linear Gaussian)
    /// Observation matrix C
    pub c: DMatrix<f64>,
    /// Measurement noise covariance Q
    pub q: DMatrix<f64>,

    // Detection
    /// Detection probability
    pub detection_probability: f64,

    // Observation space
    /// Limits of observation space [[x_min, x_max], [y_min, y_max]]
    pub observation_space_limits: DMatrix<f64>,
    /// Volume of observation space
    pub observation_space_volume: f64,
    /// Expected clutter rate (number per timestep)
    pub clutter_rate: f64,
    /// Clutter per unit volume
    pub clutter_per_unit_volume: f64,

    // Birth parameters
    /// Number of birth locations
    pub number_of_birth_locations: usize,
    /// Birth location labels
    pub birth_location_labels: Vec<usize>,
    /// Birth existence probabilities for LMB
    pub r_b: Vec<f64>,
    /// Birth existence probabilities for LMBM
    pub r_b_lmbm: Vec<f64>,
    /// Birth means (one per location)
    pub mu_b: Vec<DVector<f64>>,
    /// Birth covariances (one per location)
    pub sigma_b: Vec<DMatrix<f64>>,

    // Template objects
    /// Template object structure
    pub object: Vec<Object>,
    /// Birth parameters as objects
    pub birth_parameters: Vec<Object>,
    /// Template hypothesis
    pub hypotheses: Hypothesis,

    // LMBM trajectories
    /// Template trajectory
    pub trajectory: Vec<Trajectory>,
    /// Birth trajectories
    pub birth_trajectory: Vec<Trajectory>,

    // GM parameters
    /// Weight threshold for pruning GM components
    pub gm_weight_threshold: f64,
    /// Maximum number of GM components to keep
    pub maximum_number_of_gm_components: usize,

    // Track parameters
    /// Minimum trajectory length for output
    pub minimum_trajectory_length: usize,

    // Data association
    /// Data association method to use
    pub data_association_method: DataAssociationMethod,

    // LBP parameters
    /// Maximum iterations for LBP
    pub maximum_number_of_lbp_iterations: usize,
    /// Convergence tolerance for LBP
    pub lbp_convergence_tolerance: f64,

    // Gibbs parameters
    /// Number of samples for Gibbs sampling
    pub number_of_samples: usize,

    // Murty parameters
    /// Number of assignments for Murty's algorithm
    pub number_of_assignments: usize,

    // LMBM parameters
    /// Maximum number of posterior hypotheses
    pub maximum_number_of_posterior_hypotheses: usize,
    /// Posterior hypothesis weight threshold
    pub posterior_hypothesis_weight_threshold: f64,
    /// Use EAP (Expected A Posteriori) for LMBM state extraction
    pub use_eap_on_lmbm: bool,

    // OSPA parameters
    /// OSPA metric parameters
    pub ospa_parameters: OspaParameters,

    // Multisensor parameters (None for single-sensor systems)
    /// Number of sensors (None for single-sensor)
    pub number_of_sensors: Option<usize>,
    /// Observation matrices per sensor (for multisensor)
    pub c_multisensor: Option<Vec<DMatrix<f64>>>,
    /// Measurement noise covariances per sensor (for multisensor)
    pub q_multisensor: Option<Vec<DMatrix<f64>>>,
    /// Detection probabilities per sensor (for multisensor)
    pub detection_probability_multisensor: Option<Vec<f64>>,
    /// Clutter rates per sensor (for multisensor)
    pub clutter_rate_multisensor: Option<Vec<f64>>,
    /// Clutter per unit volume per sensor (for multisensor)
    pub clutter_per_unit_volume_multisensor: Option<Vec<f64>>,
    /// LMB parallel update mode (for multisensor LMB)
    pub lmb_parallel_update_mode: Option<ParallelUpdateMode>,
    /// AA-LMB sensor weights (for multisensor LMB with AA fusion)
    pub aa_sensor_weights: Option<Vec<f64>>,
    /// GA-LMB sensor weights (for multisensor LMB with GA fusion)
    pub ga_sensor_weights: Option<Vec<f64>>,
}

impl Model {
    /// Get detection probability for sensor (or default for single-sensor)
    ///
    /// Centralizes the Option handling pattern previously repeated in:
    /// - multisensor_lmb/association.rs
    /// - multisensor_lmbm/association.rs
    /// - multisensor_lmb/utils.rs
    #[inline]
    pub fn get_detection_probability(&self, sensor_idx: Option<usize>) -> f64 {
        match (sensor_idx, &self.detection_probability_multisensor) {
            (Some(s), Some(vec)) => vec[s],
            _ => self.detection_probability,
        }
    }

    /// Get observation matrix for sensor (or default for single-sensor)
    ///
    /// Originally: manual Option handling in association files
    #[inline]
    pub fn get_observation_matrix(&self, sensor_idx: Option<usize>) -> &DMatrix<f64> {
        match (sensor_idx, &self.c_multisensor) {
            (Some(s), Some(vec)) => &vec[s],
            _ => &self.c,
        }
    }

    /// Get measurement noise covariance for sensor (or default for single-sensor)
    ///
    /// Originally: manual Option handling in association files
    #[inline]
    pub fn get_measurement_noise(&self, sensor_idx: Option<usize>) -> &DMatrix<f64> {
        match (sensor_idx, &self.q_multisensor) {
            (Some(s), Some(vec)) => &vec[s],
            _ => &self.q,
        }
    }

    /// Get clutter rate for sensor (or default for single-sensor)
    ///
    /// Originally: manual Option handling in association files
    #[inline]
    pub fn get_clutter_rate(&self, sensor_idx: Option<usize>) -> f64 {
        match (sensor_idx, &self.clutter_rate_multisensor) {
            (Some(s), Some(vec)) => vec[s],
            _ => self.clutter_rate,
        }
    }

    /// Get clutter per unit volume for sensor (or default for single-sensor)
    ///
    /// Originally: manual Option handling in association files
    #[inline]
    pub fn get_clutter_per_unit_volume(&self, sensor_idx: Option<usize>) -> f64 {
        match (sensor_idx, &self.clutter_per_unit_volume_multisensor) {
            (Some(s), Some(vec)) => vec[s],
            _ => self.clutter_per_unit_volume,
        }
    }

    /// Check if this is a multi-sensor model
    #[inline]
    pub fn is_multisensor(&self) -> bool {
        self.number_of_sensors.map(|n| n > 1).unwrap_or(false)
    }
}

/// Measurement data
///
/// Represents sensor measurements at a single time step
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Measurement vectors (each column is a measurement)
    pub z: DMatrix<f64>,
    /// Number of measurements
    pub count: usize,
}

impl Measurement {
    /// Create empty measurement set
    pub fn empty(z_dim: usize) -> Self {
        Self {
            z: DMatrix::zeros(z_dim, 0),
            count: 0,
        }
    }
}

/// Ground truth data
///
/// Contains true object states for simulation and evaluation
#[derive(Debug, Clone)]
pub struct GroundTruth {
    /// True object states at each time step
    pub states: Vec<DMatrix<f64>>,
    /// Number of objects at each time step
    pub cardinality: Vec<usize>,
    /// Birth times for each object
    pub birth_times: Vec<usize>,
    /// Death times for each object
    pub death_times: Vec<usize>,
}
