//! Model generation
//!
//! Functions to generate tracking model configurations

use crate::common::types::*;
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
