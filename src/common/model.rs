//! Model generation
//!
//! Functions to generate tracking model configurations

use crate::common::types::*;
use crate::multisensor_lmb::parallel_update::ParallelUpdateMode;
use nalgebra::{DMatrix, DVector};
use rand::Rng;

/// Generate a tracking model
///
/// Creates a complete model structure with all parameters for simulation and filtering.
/// This matches the MATLAB generateModel function exactly.
///
/// # Arguments
/// * `clutter_rate` - Expected number of clutter measurements per time step
/// * `detection_probability` - Probability of detecting a target
/// * `data_association_method` - Method to use for data association
/// * `scenario_type` - Type of scenario ('Fixed' or 'Random')
/// * `number_of_birth_locations` - Number of birth locations (only for 'Random' scenario)
///
/// # Returns
/// Complete Model struct
pub fn generate_model(
    clutter_rate: f64,
    detection_probability: f64,
    data_association_method: DataAssociationMethod,
    scenario_type: ScenarioType,
    number_of_birth_locations: Option<usize>,
) -> Model {
    // State and measurement space dimensions
    let x_dimension = 4; // [x, vx, y, vy]
    let z_dimension = 2; // [x, y]

    // Sampling period
    let t = 1.0;

    // Survival and existence parameters
    let survival_probability = 0.95;
    let existence_threshold = 1e-2;

    // State transition matrix A = [I T*I; 0 I]
    let mut a = DMatrix::zeros(x_dimension, x_dimension);
    a.view_mut((0, 0), (2, 2)).copy_from(&DMatrix::identity(2, 2));
    a.view_mut((0, 2), (2, 2)).copy_from(&(t * DMatrix::identity(2, 2)));
    a.view_mut((2, 2), (2, 2)).copy_from(&DMatrix::identity(2, 2));

    // Control input (zero)
    let u = DVector::zeros(x_dimension);

    // Process noise covariance R
    let r0 = 1.0;
    let t_sq = t * t;
    let t_cb = t * t * t;
    let mut r = DMatrix::zeros(x_dimension, x_dimension);
    r.view_mut((0, 0), (2, 2)).copy_from(&((t_cb / 3.0) * r0 * DMatrix::identity(2, 2)));
    r.view_mut((0, 2), (2, 2)).copy_from(&((t_sq / 2.0) * r0 * DMatrix::identity(2, 2)));
    r.view_mut((2, 0), (2, 2)).copy_from(&((t_sq / 2.0) * r0 * DMatrix::identity(2, 2)));
    r.view_mut((2, 2), (2, 2)).copy_from(&(t * r0 * DMatrix::identity(2, 2)));

    // Observation matrix C = [I 0]
    let mut c = DMatrix::zeros(z_dimension, x_dimension);
    c.view_mut((0, 0), (2, 2)).copy_from(&DMatrix::identity(2, 2));

    // Measurement noise covariance Q
    let q0 = 3.0_f64.powi(2);
    let q = q0 * DMatrix::identity(z_dimension, z_dimension);

    // Observation space
    let mut observation_space_limits = DMatrix::zeros(2, 2);
    observation_space_limits[(0, 0)] = -100.0;
    observation_space_limits[(0, 1)] = 100.0;
    observation_space_limits[(1, 0)] = -100.0;
    observation_space_limits[(1, 1)] = 100.0;

    let observation_space_volume = (observation_space_limits[(0, 1)] - observation_space_limits[(0, 0)])
        * (observation_space_limits[(1, 1)] - observation_space_limits[(1, 0)]);

    let clutter_per_unit_volume = clutter_rate / observation_space_volume;

    // Birth parameters
    let (num_birth_locs, birth_locations) = match scenario_type {
        ScenarioType::Fixed => {
            // Four fixed birth locations
            // Each column is [x, y, vx, vy] for one birth location (matches MATLAB)
            let mut locs = DMatrix::zeros(x_dimension, 4);
            // Location 1: [-80, -20, 0, 0] -> x=-80, y=-20, vx=0, vy=0
            locs[(0, 0)] = -80.0;  // x
            locs[(1, 0)] = -20.0;  // y
            locs[(2, 0)] = 0.0;    // vx
            locs[(3, 0)] = 0.0;    // vy
            // Location 2: [-20, 80, 0, 0] -> x=-20, y=80, vx=0, vy=0
            locs[(0, 1)] = -20.0;
            locs[(1, 1)] = 80.0;
            locs[(2, 1)] = 0.0;
            locs[(3, 1)] = 0.0;
            // Location 3: [0, 0, 0, 0]
            locs[(0, 2)] = 0.0;
            locs[(1, 2)] = 0.0;
            locs[(2, 2)] = 0.0;
            locs[(3, 2)] = 0.0;
            // Location 4: [40, -60, 0, 0] -> x=40, y=-60, vx=0, vy=0
            locs[(0, 3)] = 40.0;
            locs[(1, 3)] = -60.0;
            locs[(2, 3)] = 0.0;
            locs[(3, 3)] = 0.0;
            (4, locs)
        }
        ScenarioType::Random => {
            let n = number_of_birth_locations.expect("Must specify number of birth locations for Random scenario");
            let mut locs = DMatrix::zeros(x_dimension, n);
            let mut rng = rand::thread_rng();

            for i in 0..n {
                locs[(0, i)] = rng.gen_range(observation_space_limits[(0, 0)]..observation_space_limits[(0, 1)]) * 0.5;
                locs[(1, i)] = rng.gen_range(observation_space_limits[(1, 0)]..observation_space_limits[(1, 1)]) * 0.5;
            }
            (n, locs)
        }
        ScenarioType::Coalescence => {
            panic!("Coalescence scenario not yet implemented");
        }
    };

    let birth_location_labels: Vec<usize> = (1..=num_birth_locs).collect();

    let r_b = vec![0.03; num_birth_locs];
    let r_b_lmbm = vec![0.045; num_birth_locs];

    let mut mu_b: Vec<DVector<f64>> = Vec::new();
    let mut sigma_b: Vec<DMatrix<f64>> = Vec::new();

    for i in 0..num_birth_locs {
        mu_b.push(birth_locations.column(i).into());
        sigma_b.push(DMatrix::from_diagonal(&DVector::from_element(x_dimension, 0.5_f64.powi(2))));
    }

    // Create birth parameters as objects
    let mut birth_parameters = Vec::new();
    let mut birth_trajectory = Vec::new();

    for i in 0..num_birth_locs {
        let mut obj = Object::empty(x_dimension);
        obj.birth_location = birth_location_labels[i];
        obj.birth_time = 0;
        obj.r = r_b[i];
        obj.number_of_gm_components = 1;
        obj.w = vec![1.0];
        obj.mu = vec![mu_b[i].clone()];
        obj.sigma = vec![sigma_b[i].clone()];
        obj.trajectory_length = 0;
        obj.trajectory = DMatrix::from_element(x_dimension, 100, 80.0);
        obj.timestamps = Vec::new();

        birth_parameters.push(obj);

        let mut traj = Trajectory::empty(x_dimension);
        traj.birth_location = birth_location_labels[i];
        traj.birth_time = 0;
        traj.trajectory_length = 0;
        traj.trajectory = DMatrix::from_element(x_dimension, 100, 80.0);
        traj.timestamps = Vec::new();

        birth_trajectory.push(traj);
    }

    // OSPA parameters
    let ospa_parameters = OspaParameters {
        e_c: 5.0,  // Euclidean cut-off
        e_p: 2.0,  // Euclidean order parameter
        h_c: 0.5,  // Hellinger cut-off
        h_p: 2.0,  // Hellinger order parameter
    };

    Model {
        scenario_type,
        x_dimension,
        z_dimension,
        t,
        survival_probability,
        existence_threshold,
        a,
        u,
        r,
        c,
        q,
        detection_probability,
        observation_space_limits,
        observation_space_volume,
        clutter_rate,
        clutter_per_unit_volume,
        number_of_birth_locations: num_birth_locs,
        birth_location_labels,
        r_b,
        r_b_lmbm,
        mu_b,
        sigma_b,
        object: Vec::new(), // Empty template
        birth_parameters,
        hypotheses: Hypothesis::empty(),
        trajectory: Vec::new(),
        birth_trajectory,
        gm_weight_threshold: 1e-6,
        maximum_number_of_gm_components: 5,
        minimum_trajectory_length: 20,
        data_association_method,
        maximum_number_of_lbp_iterations: 1000,
        lbp_convergence_tolerance: 1e-6,
        number_of_samples: 2500,
        number_of_assignments: 25,
        maximum_number_of_posterior_hypotheses: 25,
        posterior_hypothesis_weight_threshold: 1e-3,
        use_eap_on_lmbm: false,
        ospa_parameters,
        // Single-sensor: multisensor parameters are None
        number_of_sensors: None,
        c_multisensor: None,
        q_multisensor: None,
        detection_probability_multisensor: None,
        clutter_rate_multisensor: None,
        clutter_per_unit_volume_multisensor: None,
        lmb_parallel_update_mode: None,
        aa_sensor_weights: None,
        ga_sensor_weights: None,
    }
}

/// Generate a multisensor tracking model
///
/// Creates a complete model structure with all parameters for multisensor simulation and filtering.
/// This matches the MATLAB generateMultisensorModel function exactly.
///
/// # Arguments
/// * `number_of_sensors` - Number of sensors
/// * `clutter_rates` - Expected clutter per time step for each sensor
/// * `detection_probabilities` - Detection probability for each sensor
/// * `q_values` - Measurement noise standard deviations for each sensor
/// * `lmb_parallel_update_mode` - Fusion mode for multisensor LMB ('PU', 'AA', or 'GA')
/// * `data_association_method` - Method to use for data association
/// * `scenario_type` - Type of scenario ('Fixed' or 'Random')
/// * `number_of_birth_locations` - Number of birth locations (only for 'Random' scenario)
///
/// # Returns
/// Complete Model struct configured for multisensor tracking
pub fn generate_multisensor_model(
    number_of_sensors: usize,
    clutter_rates: Vec<f64>,
    detection_probabilities: Vec<f64>,
    q_values: Vec<f64>,
    lmb_parallel_update_mode: ParallelUpdateMode,
    data_association_method: DataAssociationMethod,
    scenario_type: ScenarioType,
    number_of_birth_locations: Option<usize>,
) -> Model {
    // State and measurement space dimensions
    let x_dimension = 4; // [x, vx, y, vy]
    let z_dimension = 2; // [x, y]

    // Sampling period
    let t = 1.0;

    // Survival and existence parameters
    let survival_probability = 0.95;
    let existence_threshold = 1e-2;

    // State transition matrix A = [I T*I; 0 I]
    let mut a = DMatrix::zeros(x_dimension, x_dimension);
    a.view_mut((0, 0), (2, 2)).copy_from(&DMatrix::identity(2, 2));
    a.view_mut((0, 2), (2, 2)).copy_from(&(t * DMatrix::identity(2, 2)));
    a.view_mut((2, 2), (2, 2)).copy_from(&DMatrix::identity(2, 2));

    // Control input (zero)
    let u = DVector::zeros(x_dimension);

    // Process noise covariance R
    let r0 = 1.0;
    let t_sq = t * t;
    let t_cb = t * t * t;
    let mut r = DMatrix::zeros(x_dimension, x_dimension);
    r.view_mut((0, 0), (2, 2)).copy_from(&((t_cb / 3.0) * r0 * DMatrix::identity(2, 2)));
    r.view_mut((0, 2), (2, 2)).copy_from(&((t_sq / 2.0) * r0 * DMatrix::identity(2, 2)));
    r.view_mut((2, 0), (2, 2)).copy_from(&((t_sq / 2.0) * r0 * DMatrix::identity(2, 2)));
    r.view_mut((2, 2), (2, 2)).copy_from(&(t * r0 * DMatrix::identity(2, 2)));

    // Multisensor observation parameters
    // Observation matrices C (one per sensor, typically identical)
    let mut c_multisensor = Vec::new();
    for _ in 0..number_of_sensors {
        let mut c = DMatrix::zeros(z_dimension, x_dimension);
        c.view_mut((0, 0), (2, 2)).copy_from(&DMatrix::identity(2, 2));
        c_multisensor.push(c);
    }

    // Measurement noise covariances Q (different per sensor)
    let mut q_multisensor = Vec::new();
    for &q_val in &q_values {
        let q = (q_val * q_val) * DMatrix::identity(z_dimension, z_dimension);
        q_multisensor.push(q);
    }

    // Default single-sensor values (use first sensor's values)
    let c = c_multisensor[0].clone();
    let q = q_multisensor[0].clone();
    let detection_probability = detection_probabilities[0];

    // Observation space
    let mut observation_space_limits = DMatrix::zeros(2, 2);
    observation_space_limits[(0, 0)] = -100.0;
    observation_space_limits[(0, 1)] = 100.0;
    observation_space_limits[(1, 0)] = -100.0;
    observation_space_limits[(1, 1)] = 100.0;

    let observation_space_volume = (observation_space_limits[(0, 1)] - observation_space_limits[(0, 0)])
        * (observation_space_limits[(1, 1)] - observation_space_limits[(1, 0)]);

    // Clutter parameters per sensor
    let clutter_per_unit_volume_multisensor: Vec<f64> = clutter_rates
        .iter()
        .map(|&rate| rate / observation_space_volume)
        .collect();

    // Default single-sensor clutter (use first sensor)
    let clutter_rate = clutter_rates[0];
    let clutter_per_unit_volume = clutter_per_unit_volume_multisensor[0];

    // Birth parameters
    let (num_birth_locs, birth_locations) = match scenario_type {
        ScenarioType::Fixed => {
            // Four fixed birth locations
            let mut locs = DMatrix::zeros(x_dimension, 4);
            locs[(0, 0)] = -80.0;
            locs[(1, 0)] = -20.0;
            locs[(0, 1)] = -20.0;
            locs[(1, 1)] = 80.0;
            locs[(0, 2)] = 0.0;
            locs[(1, 2)] = 0.0;
            locs[(0, 3)] = 40.0;
            locs[(1, 3)] = -60.0;
            (4, locs)
        }
        ScenarioType::Random => {
            let n = number_of_birth_locations.expect("Must specify number of birth locations for Random scenario");
            let mut locs = DMatrix::zeros(x_dimension, n);
            let mut rng = rand::thread_rng();

            for i in 0..n {
                locs[(0, i)] = rng.gen_range(observation_space_limits[(0, 0)]..observation_space_limits[(0, 1)]) * 0.5;
                locs[(1, i)] = rng.gen_range(observation_space_limits[(1, 0)]..observation_space_limits[(1, 1)]) * 0.5;
            }
            (n, locs)
        }
        ScenarioType::Coalescence => {
            panic!("Coalescence scenario not yet implemented");
        }
    };

    let birth_location_labels: Vec<usize> = (1..=num_birth_locs).collect();

    let r_b = vec![0.03; num_birth_locs];
    let r_b_lmbm = vec![0.06; num_birth_locs]; // Higher for multisensor

    let mut mu_b: Vec<DVector<f64>> = Vec::new();
    let mut sigma_b: Vec<DMatrix<f64>> = Vec::new();

    for i in 0..num_birth_locs {
        mu_b.push(birth_locations.column(i).into());
        sigma_b.push(DMatrix::from_diagonal(&DVector::from_element(x_dimension, 10.0_f64.powi(2))));
    }

    // Create birth parameters as objects
    let mut birth_parameters = Vec::new();
    let mut birth_trajectory = Vec::new();

    for i in 0..num_birth_locs {
        let mut obj = Object::empty(x_dimension);
        obj.birth_location = birth_location_labels[i];
        obj.birth_time = 0;
        obj.r = r_b[i];
        obj.number_of_gm_components = 1;
        obj.w = vec![1.0];
        obj.mu = vec![mu_b[i].clone()];
        obj.sigma = vec![sigma_b[i].clone()];
        obj.trajectory_length = 0;
        obj.trajectory = DMatrix::from_element(x_dimension, 100, 80.0);
        obj.timestamps = Vec::new();

        birth_parameters.push(obj);

        let mut traj = Trajectory::empty(x_dimension);
        traj.birth_location = birth_location_labels[i];
        traj.birth_time = 0;
        traj.trajectory_length = 0;
        traj.trajectory = DMatrix::from_element(x_dimension, 100, 80.0);
        traj.timestamps = Vec::new();

        birth_trajectory.push(traj);
    }

    // OSPA parameters
    let ospa_parameters = OspaParameters {
        e_c: 5.0,  // Euclidean cut-off
        e_p: 2.0,  // Euclidean order parameter
        h_c: 0.5,  // Hellinger cut-off
        h_p: 2.0,  // Hellinger order parameter
    };

    // Sensor weights for AA and GA fusion
    let aa_sensor_weights = vec![1.0 / number_of_sensors as f64; number_of_sensors];
    let ga_sensor_weights = vec![1.0 / number_of_sensors as f64; number_of_sensors];

    Model {
        scenario_type,
        x_dimension,
        z_dimension,
        t,
        survival_probability,
        existence_threshold,
        a,
        u,
        r,
        c,
        q,
        detection_probability,
        observation_space_limits,
        observation_space_volume,
        clutter_rate,
        clutter_per_unit_volume,
        number_of_birth_locations: num_birth_locs,
        birth_location_labels,
        r_b,
        r_b_lmbm,
        mu_b,
        sigma_b,
        object: Vec::new(), // Empty template
        birth_parameters,
        hypotheses: Hypothesis::empty(),
        trajectory: Vec::new(),
        birth_trajectory,
        gm_weight_threshold: 1e-6,
        maximum_number_of_gm_components: 20, // Higher for multisensor
        minimum_trajectory_length: 20,
        data_association_method,
        maximum_number_of_lbp_iterations: 1000,
        lbp_convergence_tolerance: 1e-6,
        number_of_samples: 1000, // Lower for multisensor (per sensor)
        number_of_assignments: 25,
        maximum_number_of_posterior_hypotheses: 10, // Lower for multisensor
        posterior_hypothesis_weight_threshold: 1e-3,
        use_eap_on_lmbm: false,
        ospa_parameters,
        // Multisensor-specific fields
        number_of_sensors: Some(number_of_sensors),
        c_multisensor: Some(c_multisensor),
        q_multisensor: Some(q_multisensor),
        detection_probability_multisensor: Some(detection_probabilities),
        clutter_rate_multisensor: Some(clutter_rates),
        clutter_per_unit_volume_multisensor: Some(clutter_per_unit_volume_multisensor),
        lmb_parallel_update_mode: Some(lmb_parallel_update_mode),
        aa_sensor_weights: Some(aa_sensor_weights),
        ga_sensor_weights: Some(ga_sensor_weights),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_fixed_model() {
        let model = generate_model(
            10.0,
            0.95,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        assert_eq!(model.x_dimension, 4);
        assert_eq!(model.z_dimension, 2);
        assert_eq!(model.number_of_birth_locations, 4);
        assert_eq!(model.birth_parameters.len(), 4);
        assert!((model.clutter_rate - 10.0).abs() < 1e-10);
        assert!((model.detection_probability - 0.95).abs() < 1e-10);
    }
}
