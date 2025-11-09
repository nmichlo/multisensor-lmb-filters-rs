//! Ground truth and measurement generation
//!
//! Functions to simulate object trajectories and sensor measurements for testing

use crate::common::types::*;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, Normal, Poisson};

/// Ground truth trajectory for a single object
#[derive(Debug, Clone)]
pub struct ObjectTrajectory {
    /// Birth time
    pub birth_time: usize,
    /// Death time
    pub death_time: usize,
    /// Birth location index
    pub birth_location_index: usize,
    /// States over time: each column is [t; x; vx; y; vy]
    pub states: DMatrix<f64>,
}

/// RFS (Random Finite Set) ground truth representation
#[derive(Debug, Clone)]
pub struct GroundTruthRfs {
    /// True states at each timestep
    pub x: Vec<Vec<DVector<f64>>>,
    /// Kalman filter means at each timestep
    pub mu: Vec<Vec<DVector<f64>>>,
    /// Kalman filter covariances at each timestep
    pub sigma: Vec<Vec<DMatrix<f64>>>,
    /// Number of objects at each timestep
    pub cardinality: Vec<usize>,
}

/// Complete ground truth output
pub struct GroundTruthOutput {
    /// Individual object trajectories
    pub ground_truth: Vec<ObjectTrajectory>,
    /// Measurements at each timestep (including clutter)
    pub measurements: Vec<Vec<DVector<f64>>>,
    /// RFS representation
    pub ground_truth_rfs: GroundTruthRfs,
}

/// Generate ground truth trajectories and measurements
///
/// Creates a simulated scenario with object trajectories and sensor measurements.
/// Matches the MATLAB generateGroundTruth function exactly.
///
/// # Arguments
/// * `model` - The tracking model
/// * `number_of_objects` - Number of objects (only for Random scenario)
///
/// # Returns
/// Complete ground truth with trajectories, measurements, and RFS representation
pub fn generate_ground_truth(
    model: &Model,
    number_of_objects: Option<usize>,
) -> GroundTruthOutput {
    let mut rng = rand::thread_rng();

    let (simulation_length, num_objects, birth_times, death_times, birth_location_indices, prior_locations) =
        match model.scenario_type {
            ScenarioType::Fixed => {
                // Fixed scenario with 10 objects
                let sim_len = 100;
                let n_objs = 10;
                let births = vec![1, 1, 20, 20, 40, 40, 60, 60, 60, 60];
                let deaths = vec![70, 70, 80, 80, 90, 90, 100, 100, 100, 100];
                let indices = vec![1, 2, 3, 4, 1, 4, 1, 2, 3, 4];

                // Prior locations matrix (4 x 10)
                let mut locs = DMatrix::zeros(4, n_objs);
                // Object 1: birth location 1
                locs[(0, 0)] = -80.0; locs[(1, 0)] = -20.0; locs[(2, 0)] = 0.75; locs[(3, 0)] = 1.5;
                // Object 2: birth location 2
                locs[(0, 1)] = -20.0; locs[(1, 1)] = 80.0; locs[(2, 1)] = -1.0; locs[(3, 1)] = -2.0;
                // Object 3: birth location 3
                locs[(0, 2)] = 0.0; locs[(1, 2)] = 0.0; locs[(2, 2)] = -0.5; locs[(3, 2)] = -1.0;
                // Object 4: birth location 4
                locs[(0, 3)] = 40.0; locs[(1, 3)] = -60.0; locs[(2, 3)] = -0.25; locs[(3, 3)] = -0.5;
                // Object 5: birth location 1
                locs[(0, 4)] = -80.0; locs[(1, 4)] = -20.0; locs[(2, 4)] = 1.0; locs[(3, 4)] = 1.0;
                // Object 6: birth location 4
                locs[(0, 5)] = 40.0; locs[(1, 5)] = -60.0; locs[(2, 5)] = -1.0; locs[(3, 5)] = 2.0;
                // Object 7: birth location 1
                locs[(0, 6)] = -80.0; locs[(1, 6)] = -20.0; locs[(2, 6)] = 1.0; locs[(3, 6)] = -0.5;
                // Object 8: birth location 2
                locs[(0, 7)] = -20.0; locs[(1, 7)] = 80.0; locs[(2, 7)] = 1.0; locs[(3, 7)] = -1.0;
                // Object 9: birth location 3
                locs[(0, 8)] = 0.0; locs[(1, 8)] = 0.0; locs[(2, 8)] = 1.0; locs[(3, 8)] = -1.0;
                // Object 10: birth location 4
                locs[(0, 9)] = 40.0; locs[(1, 9)] = -60.0; locs[(2, 9)] = -1.0; locs[(3, 9)] = 0.5;

                (sim_len, n_objs, births, deaths, indices, locs)
            }
            ScenarioType::Random => {
                let n_objs = number_of_objects.expect("Must specify number of objects for Random scenario");
                let sim_len = 100;
                let births = vec![1; n_objs];
                let deaths = vec![sim_len; n_objs];
                // Cycle through birth locations for each object
                let indices: Vec<usize> = (0..n_objs)
                    .map(|i| (i % model.number_of_birth_locations) + 1)
                    .collect();

                // Prior locations from birth means
                let mut locs = DMatrix::zeros(4, n_objs);
                for i in 0..n_objs {
                    let birth_idx = indices[i] - 1;
                    locs[(0, i)] = model.mu_b[birth_idx][0];
                    locs[(1, i)] = model.mu_b[birth_idx][1];
                    // Random velocities
                    let normal = Normal::new(0.0, 3.0).unwrap();
                    locs[(2, i)] = normal.sample(&mut rng);
                    locs[(3, i)] = normal.sample(&mut rng);
                }

                (sim_len, n_objs, births, deaths, indices, locs)
            }
            ScenarioType::Coalescence => {
                panic!("Coalescence scenario not yet implemented");
            }
        };

    // Allocate outputs
    let mut measurements: Vec<Vec<DVector<f64>>> = vec![Vec::new(); simulation_length];
    let mut ground_truth: Vec<ObjectTrajectory> = Vec::with_capacity(num_objects);
    let mut ground_truth_rfs = GroundTruthRfs {
        x: vec![Vec::new(); simulation_length],
        mu: vec![Vec::new(); simulation_length],
        sigma: vec![Vec::new(); simulation_length],
        cardinality: vec![0; simulation_length],
    };

    // Generate clutter measurements
    let poisson = Poisson::new(model.clutter_rate).unwrap();
    let normal = Normal::new(0.0, 1.0).unwrap();

    for t in 0..simulation_length {
        let num_clutter = poisson.sample(&mut rng) as usize;
        for _ in 0..num_clutter {
            let mut z = DVector::zeros(model.z_dimension);
            for d in 0..model.z_dimension {
                let range = model.observation_space_limits[(d, 1)] - model.observation_space_limits[(d, 0)];
                z[d] = model.observation_space_limits[(d, 0)] + range * rng.gen::<f64>();
            }
            measurements[t].push(z);
        }
    }

    // Cholesky decomposition of measurement noise
    let q_chol = model.q.clone().cholesky().expect("Q must be positive definite");

    // Simulate each object
    for obj_idx in 0..num_objects {
        let birth_time = birth_times[obj_idx];
        let death_time = death_times[obj_idx];
        let trajectory_length = death_time - birth_time + 1;
        let birth_loc_idx = birth_location_indices[obj_idx] - 1; // Convert to 0-indexed

        // Initialize state
        let mut x = DVector::from_column_slice(&[
            prior_locations[(0, obj_idx)],
            prior_locations[(1, obj_idx)],
            prior_locations[(2, obj_idx)],
            prior_locations[(3, obj_idx)],
        ]);

        let mut mu = model.mu_b[birth_loc_idx].clone();
        let mut sigma = model.sigma_b[birth_loc_idx].clone();

        // Allocate trajectory storage: [t; x; vx; y; vy]
        let mut states = DMatrix::zeros(5, trajectory_length);

        // Simulate trajectory
        for j in 0..trajectory_length {
            let t = birth_time + j;

            // Prediction step (skip for j=0)
            if j > 0 {
                x = &model.a * &x + &model.u;
                mu = &model.a * &mu + &model.u;
                sigma = &model.a * &sigma * model.a.transpose() + &model.r;
            }

            // Store state with timestamp
            states[(0, j)] = t as f64;
            states.view_mut((1, j), (4, 1)).copy_from(&x);

            // Generate measurement with detection probability
            if rng.gen::<f64>() < model.detection_probability {
                // Measurement = C * x + noise
                let mut z = &model.c * &x;
                let noise = DVector::from_fn(model.z_dimension, |_, _| normal.sample(&mut rng));
                z += q_chol.l() * noise;

                measurements[t - 1].push(z.clone());

                // Kalman filter update
                let z_pred = &model.c * &mu;
                let innovation = &z - &z_pred;
                let s = &model.c * &sigma * model.c.transpose() + &model.q;

                // Kalman gain
                let s_inv = s.try_inverse().expect("Innovation covariance must be invertible");
                let k = &sigma * model.c.transpose() * s_inv;

                mu = &mu + &k * innovation;
                let i_minus_kc = DMatrix::identity(model.x_dimension, model.x_dimension) - &k * &model.c;
                sigma = i_minus_kc * sigma;
            }

            // Add to RFS (t-1 for 0-indexing)
            ground_truth_rfs.x[t - 1].push(x.clone());
            ground_truth_rfs.mu[t - 1].push(mu.clone());
            ground_truth_rfs.sigma[t - 1].push(sigma.clone());
            ground_truth_rfs.cardinality[t - 1] += 1;
        }

        ground_truth.push(ObjectTrajectory {
            birth_time,
            death_time,
            birth_location_index: birth_location_indices[obj_idx],
            states,
        });
    }

    GroundTruthOutput {
        ground_truth,
        measurements,
        ground_truth_rfs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;

    #[test]
    fn test_generate_fixed_ground_truth() {
        let model = generate_model(
            10.0,
            0.95,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let output = generate_ground_truth(&model, None);

        // Check basic properties
        assert_eq!(output.ground_truth.len(), 10);
        assert_eq!(output.measurements.len(), 100);
        assert_eq!(output.ground_truth_rfs.cardinality.len(), 100);

        // Check first object properties
        assert_eq!(output.ground_truth[0].birth_time, 1);
        assert_eq!(output.ground_truth[0].death_time, 70);
        assert_eq!(output.ground_truth[0].states.ncols(), 70);
        assert_eq!(output.ground_truth[0].states.nrows(), 5); // [t; x; vx; y; vy]
    }

    #[test]
    fn test_generate_random_ground_truth() {
        let model = generate_model(
            10.0,
            0.95,
            DataAssociationMethod::LBP,
            ScenarioType::Random,
            Some(4),
        );

        let output = generate_ground_truth(&model, Some(5));

        assert_eq!(output.ground_truth.len(), 5);
        assert_eq!(output.measurements.len(), 100);
    }
}