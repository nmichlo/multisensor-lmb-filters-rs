//! Ground truth and measurement generation
//!
//! Functions to simulate object trajectories and sensor measurements for testing

use crate::common::types::*;
use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
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
    /// States over time: each column is [t; x; y; vx; vy] (matches MATLAB exactly)
    /// where state vector is [x, y, vx, vy] representing position and velocity in 2D
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
/// * `seed` - Optional RNG seed for reproducibility (use 42 to match MATLAB fixtures)
///
/// # Returns
/// Complete ground truth with trajectories, measurements, and RFS representation
pub fn generate_ground_truth(
    model: &Model,
    number_of_objects: Option<usize>,
    seed: Option<u64>,
) -> GroundTruthOutput {
    // Use StdRng for deterministic ground truth generation
    let seed_value = seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
    });
    let mut rng = StdRng::seed_from_u64(seed_value);

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
                    // Random velocities (Normal with mean=0, std=3)
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
    for t in 0..simulation_length {
        let num_clutter = Poisson::new(model.clutter_rate).unwrap().sample(&mut rng) as usize;
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
        // Prior locations are [x, y, vx, vy] - use directly
        let mut x = DVector::from_column_slice(&[
            prior_locations[(0, obj_idx)],
            prior_locations[(1, obj_idx)],
            prior_locations[(2, obj_idx)],
            prior_locations[(3, obj_idx)],
        ]);

        let mut mu = model.mu_b[birth_loc_idx].clone();
        let mut sigma = model.sigma_b[birth_loc_idx].clone();

        // Allocate trajectory storage: [t; x; y; vx; vy] - matches MATLAB exactly
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
                let standard_normal = Normal::new(0.0, 1.0).unwrap();
                let noise = DVector::from_fn(model.z_dimension, |_, _| standard_normal.sample(&mut rng));
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

/// Multisensor ground truth output structure
#[derive(Debug, Clone)]
pub struct MultisensorGroundTruthOutput {
    /// Individual object trajectories
    pub ground_truth: Vec<ObjectTrajectory>,
    /// Measurements at each timestep for each sensor [sensor][time][measurements]
    pub measurements: Vec<Vec<Vec<DVector<f64>>>>,
    /// RFS representation
    pub ground_truth_rfs: GroundTruthRfs,
}

/// Generate multisensor ground truth trajectories and measurements
///
/// Creates a simulated scenario with object trajectories and multi-sensor measurements.
/// Matches the MATLAB generateMultisensorGroundTruth function exactly.
///
/// # Arguments
/// * `model` - The multisensor tracking model
/// * `number_of_objects` - Number of objects for Random scenario (None for Fixed)
/// * `seed` - Optional RNG seed for reproducibility (use 42 to match MATLAB fixtures)
///
/// # Returns
/// MultisensorGroundTruthOutput with trajectories, measurements, and RFS representation
pub fn generate_multisensor_ground_truth(
    model: &Model,
    number_of_objects: Option<usize>,
    seed: Option<u64>,
) -> MultisensorGroundTruthOutput {
    // Use StdRng for deterministic ground truth generation
    let seed_value = seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
    });
    let mut rng = StdRng::seed_from_u64(seed_value);

    let number_of_sensors = model.number_of_sensors.expect("Model must be configured for multisensor");

    let (simulation_length, num_objects, birth_times, death_times, birth_location_indices, prior_locations) =
        match model.scenario_type {
            ScenarioType::Fixed => {
                // Fixed scenario with 10 objects
                let sim_len = 100;
                let n_obj = 10;
                let births = vec![1, 1, 20, 20, 40, 40, 60, 60, 60, 60];
                let deaths = vec![70, 70, 80, 80, 90, 90, 100, 100, 100, 100];
                let birth_locs = vec![1, 2, 3, 4, 1, 4, 1, 2, 3, 4];

                // Prior locations: [x, vx, y, vy] for each object
                let mut locs = DMatrix::zeros(4, n_obj);
                // Object 1
                locs[(0, 0)] = -80.0; locs[(1, 0)] = 0.75;
                locs[(2, 0)] = -20.0; locs[(3, 0)] = 1.5;
                // Object 2
                locs[(0, 1)] = -20.0; locs[(1, 1)] = -1.0;
                locs[(2, 1)] = 80.0; locs[(3, 1)] = -2.0;
                // Object 3
                locs[(0, 2)] = 0.0; locs[(1, 2)] = -0.5;
                locs[(2, 2)] = 0.0; locs[(3, 2)] = -1.0;
                // Object 4
                locs[(0, 3)] = 40.0; locs[(1, 3)] = -0.25;
                locs[(2, 3)] = -60.0; locs[(3, 3)] = -0.5;
                // Object 5
                locs[(0, 4)] = -80.0; locs[(1, 4)] = 1.0;
                locs[(2, 4)] = -20.0; locs[(3, 4)] = 1.0;
                // Object 6
                locs[(0, 5)] = 40.0; locs[(1, 5)] = -1.0;
                locs[(2, 5)] = -60.0; locs[(3, 5)] = 2.0;
                // Object 7
                locs[(0, 6)] = -80.0; locs[(1, 6)] = 1.0;
                locs[(2, 6)] = -20.0; locs[(3, 6)] = -0.5;
                // Object 8
                locs[(0, 7)] = -20.0; locs[(1, 7)] = 1.0;
                locs[(2, 7)] = 80.0; locs[(3, 7)] = -1.0;
                // Object 9
                locs[(0, 8)] = 0.0; locs[(1, 8)] = 1.0;
                locs[(2, 8)] = 0.0; locs[(3, 8)] = -1.0;
                // Object 10
                locs[(0, 9)] = 40.0; locs[(1, 9)] = -1.0;
                locs[(2, 9)] = -60.0; locs[(3, 9)] = 0.5;

                (sim_len, n_obj, births, deaths, birth_locs, locs)
            }
            ScenarioType::Random => {
                let n_obj = number_of_objects.expect("Must specify number of objects for Random scenario");
                let sim_len = 100;
                let births = vec![1; n_obj];
                let deaths = vec![sim_len; n_obj];
                let birth_locs: Vec<usize> = (1..=model.number_of_birth_locations).collect();

                let mut locs = DMatrix::zeros(4, n_obj);
                for i in 0..n_obj {
                    locs[(0, i)] = model.mu_b[birth_locs[i] - 1][0];
                    locs[(2, i)] = model.mu_b[birth_locs[i] - 1][2];
                    let normal = Normal::new(0.0, 3.0).unwrap();
                    locs[(1, i)] = normal.sample(&mut rng);
                    locs[(3, i)] = normal.sample(&mut rng);
                }

                (sim_len, n_obj, births, deaths, birth_locs, locs)
            }
            ScenarioType::Coalescence => {
                panic!("Coalescence scenario not implemented");
            }
        };

    // Allocate outputs
    // measurements[sensor][time][measurement_idx]
    let mut measurements: Vec<Vec<Vec<DVector<f64>>>> = vec![vec![Vec::new(); simulation_length]; number_of_sensors];
    let mut ground_truth = Vec::new();
    let mut ground_truth_rfs = GroundTruthRfs {
        x: vec![Vec::new(); simulation_length],
        mu: vec![Vec::new(); simulation_length],
        sigma: vec![Vec::new(); simulation_length],
        cardinality: vec![0; simulation_length],
    };

    // Add clutter measurements for each sensor
    for t in 0..simulation_length {
        for s in 0..number_of_sensors {
            let clutter_rate = model.clutter_rate_multisensor.as_ref().unwrap()[s];
            let num_clutter = Poisson::new(clutter_rate).unwrap().sample(&mut rng) as usize;

            for _ in 0..num_clutter {
                let mut clutter = DVector::zeros(model.z_dimension);
                clutter[0] = model.observation_space_limits[(0, 0)]
                    + (model.observation_space_limits[(0, 1)] - model.observation_space_limits[(0, 0)]) * rng.gen::<f64>();
                clutter[1] = model.observation_space_limits[(1, 0)]
                    + (model.observation_space_limits[(1, 1)] - model.observation_space_limits[(1, 0)]) * rng.gen::<f64>();
                measurements[s][t].push(clutter);
            }
        }
    }

    let c_multisensor = model.c_multisensor.as_ref().unwrap();
    let q_multisensor = model.q_multisensor.as_ref().unwrap();
    let detection_probabilities = model.detection_probability_multisensor.as_ref().unwrap();

    // Generate each object's trajectory and measurements
    for obj_idx in 0..num_objects {
        let birth_time = birth_times[obj_idx];
        let death_time = death_times[obj_idx];
        let trajectory_length = death_time - birth_time + 1;

        let mut states = DMatrix::zeros(5, trajectory_length); // [t, x, y, vx, vy]
        // Prior locations are [x, y, vx, vy] - use directly
        let mut x = prior_locations.column(obj_idx).into_owned();
        let mut mu = model.mu_b[birth_location_indices[obj_idx] - 1].clone();
        let mut sigma = model.sigma_b[birth_location_indices[obj_idx] - 1].clone();

        // Initial state
        states[(0, 0)] = birth_time as f64;
        states.view_mut((1, 0), (4, 1)).copy_from(&x);

        // Simulate trajectory
        for j in 0..trajectory_length {
            let t = birth_time + j;

            // Prediction (except for first timestep)
            if j > 0 {
                // Point estimate
                x = &model.a * &x + &model.u;
                // Kalman filter prediction
                mu = &model.a * &mu + &model.u;
                sigma = &model.a * &sigma * model.a.transpose() + &model.r;

                // Store state
                states[(0, j)] = t as f64;
                states.view_mut((1, j), (4, 1)).copy_from(&x);
            }

            // Generate measurements for each sensor
            let mut generated_measurement = vec![false; number_of_sensors];
            for s in 0..number_of_sensors {
                generated_measurement[s] = rng.gen::<f64>() < detection_probabilities[s];
            }

            let num_detections: usize = generated_measurement.iter().filter(|&&x| x).count();

            if num_detections > 0 {
                // Generate measurements and stack for multi-sensor update
                let mut z = DVector::zeros(model.z_dimension * num_detections);
                let mut c_stacked = DMatrix::zeros(model.z_dimension * num_detections, model.x_dimension);
                let mut q_blocks = Vec::new();

                let mut counter = 0;
                for s in 0..number_of_sensors {
                    if generated_measurement[s] {
                        // Generate measurement with sensor-specific noise
                        let standard_normal = Normal::new(0.0, 1.0).unwrap();
                        let noise = q_multisensor[s].clone().cholesky().unwrap().l() * DVector::from_fn(model.z_dimension, |_, _| standard_normal.sample(&mut rng));
                        let y = &c_multisensor[s] * &x + noise;
                        measurements[s][t - 1].push(y.clone());

                        // Stack for Kalman update
                        let start = model.z_dimension * counter;
                        z.view_mut((start, 0), (model.z_dimension, 1)).copy_from(&y);
                        c_stacked.view_mut((start, 0), (model.z_dimension, model.x_dimension))
                            .copy_from(&c_multisensor[s]);
                        q_blocks.push(q_multisensor[s].clone());
                        counter += 1;
                    }
                }

                // Multi-sensor Kalman filter update using block diagonal Q
                let q_block_diag = create_block_diagonal(&q_blocks);
                let innovation = &z - &c_stacked * &mu;
                let s_mat = &c_stacked * &sigma * c_stacked.transpose() + q_block_diag;
                let k = &sigma * c_stacked.transpose() * s_mat.clone().try_inverse().unwrap();
                mu = &mu + &k * innovation;
                let i_minus_kc = DMatrix::identity(model.x_dimension, model.x_dimension) - &k * &c_stacked;
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

    MultisensorGroundTruthOutput {
        ground_truth,
        measurements,
        ground_truth_rfs,
    }
}

/// Helper function to create block diagonal matrix from a vector of matrices
fn create_block_diagonal(blocks: &[DMatrix<f64>]) -> DMatrix<f64> {
    if blocks.is_empty() {
        return DMatrix::zeros(0, 0);
    }

    let block_rows = blocks[0].nrows();
    let block_cols = blocks[0].ncols();
    let n = blocks.len();
    let total_rows = block_rows * n;
    let total_cols = block_cols * n;

    let mut result = DMatrix::zeros(total_rows, total_cols);

    for (i, block) in blocks.iter().enumerate() {
        let row_start = i * block_rows;
        let col_start = i * block_cols;
        result
            .view_mut((row_start, col_start), (block_rows, block_cols))
            .copy_from(block);
    }

    result
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

        let output = generate_ground_truth(&model, None, None);

        // Check basic properties
        assert_eq!(output.ground_truth.len(), 10);
        assert_eq!(output.measurements.len(), 100);
        assert_eq!(output.ground_truth_rfs.cardinality.len(), 100);

        // Check first object properties
        assert_eq!(output.ground_truth[0].birth_time, 1);
        assert_eq!(output.ground_truth[0].death_time, 70);
        assert_eq!(output.ground_truth[0].states.ncols(), 70);
        assert_eq!(output.ground_truth[0].states.nrows(), 5); // [t; x; y; vx; vy]
    }

    #[test]
    fn test_state_vector_ordering_matches_matlab() {
        // Verify state vector ordering matches MATLAB exactly: [x, y, vx, vy]
        // MATLAB priorLocations first column (after transpose): [-80.0, -20.0, 0.75, 1.5]'

        let model = generate_model(
            10.0,
            0.95,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        // Use fixed seed for determinism
        let output = generate_ground_truth(&model, None, Some(42));

        // First object should have birth_location_index = 0 (first prior location)
        let first_obj = &output.ground_truth[0];

        // Verify trajectory storage format is [t; x; y; vx; vy]
        assert_eq!(first_obj.states.nrows(), 5);

        // Get initial state (first column, skip time row)
        let initial_state = first_obj.states.column(0);
        let t = initial_state[0];
        let x = initial_state[1];
        let y = initial_state[2];
        let vx = initial_state[3];
        let vy = initial_state[4];

        // Time should match birth time
        assert_eq!(t as usize, first_obj.birth_time);

        // Initial position and velocity should match MATLAB's first prior location
        // MATLAB: priorLocations(:, 1) = [-80.0; -20.0; 0.75; 1.5] (after transpose)
        // which represents [x; y; vx; vy]
        assert_eq!(x, -80.0);
        assert_eq!(y, -20.0);
        assert_eq!(vx, 0.75);
        assert_eq!(vy, 1.5);
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

        let output = generate_ground_truth(&model, Some(5), None);

        assert_eq!(output.ground_truth.len(), 5);
        assert_eq!(output.measurements.len(), 100);
    }
}