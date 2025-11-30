//! Shared utility functions for multi-sensor LMB filters
//!
//! These functions are used by both parallel_update.rs and iterated_corrector.rs
//! to reduce code duplication while maintaining MATLAB equivalence.

use crate::common::types::{Model, Object, Trajectory};
use crate::common::utils::update_existence_missed_detection;
use crate::lmb::cardinality::lmb_map_cardinality_estimate;
use nalgebra::{DMatrix, DVector};

/// Update existence probabilities when a sensor has no measurements
///
/// Uses the missed detection formula: r' = r*(1-p_d) / (1 - r*p_d)
///
/// # Arguments
/// * `objects` - Objects to update
/// * `sensor_index` - Sensor index for getting per-sensor detection probability
/// * `model` - Model parameters
#[inline]
pub fn update_existence_no_measurements_sensor(
    objects: &mut [Object],
    sensor_index: usize,
    model: &Model,
) {
    // Use Model accessor method for detection probability
    let p_d = model.get_detection_probability(Some(sensor_index));

    for obj in objects {
        obj.r = update_existence_missed_detection(obj.r, p_d);
    }
}

/// Result of gating and exporting tracks
pub struct GateAndExportResult {
    /// Objects that passed the existence threshold
    pub objects: Vec<Object>,
    /// Long trajectories that were exported
    pub exported_count: usize,
}

/// Gate tracks by existence probability and export long discarded trajectories
///
/// # Arguments
/// * `objects` - Current objects
/// * `all_objects` - Vector to append exported trajectories to
/// * `t` - Current time step (0-indexed)
/// * `model` - Model parameters
///
/// # Returns
/// Objects that passed the existence threshold
pub fn gate_and_export_tracks(
    objects: Vec<Object>,
    all_objects: &mut Vec<Trajectory>,
    t: usize,
    model: &Model,
) -> Vec<Object> {
    let objects_likely_to_exist: Vec<bool> = objects
        .iter()
        .map(|obj| obj.r > model.existence_threshold)
        .collect();

    // Export long discarded trajectories
    for (i, obj) in objects.iter().enumerate() {
        if !objects_likely_to_exist[i] && obj.trajectory_length > model.minimum_trajectory_length {
            let traj = Trajectory {
                birth_location: obj.birth_location,
                birth_time: obj.birth_time,
                trajectory: DMatrix::from_columns(
                    &obj.mu.iter().map(|m| m.clone()).collect::<Vec<_>>(),
                ),
                trajectory_length: obj.trajectory_length,
                timestamps: (0..obj.trajectory_length)
                    .map(|i| t - obj.trajectory_length + i + 1)
                    .collect(),
            };
            all_objects.push(traj);
        }
    }

    // Keep objects with high existence probabilities
    objects
        .into_iter()
        .zip(objects_likely_to_exist.iter())
        .filter_map(|(obj, &keep)| if keep { Some(obj) } else { None })
        .collect()
}

/// MAP state estimates extracted from LMB
pub struct MapStateEstimates {
    /// Labels matrix (2 x n_map): [birth_time; birth_location]
    pub labels: DMatrix<usize>,
    /// Mean estimates for MAP objects
    pub mu: Vec<DVector<f64>>,
    /// Covariance estimates for MAP objects
    pub sigma: Vec<DMatrix<f64>>,
}

/// Extract MAP state estimates from objects
///
/// Computes MAP cardinality estimate and extracts state estimates for the
/// most likely objects.
///
/// # Arguments
/// * `objects` - Current objects with existence probabilities
///
/// # Returns
/// MAP state estimates (labels, means, covariances)
pub fn extract_map_state_estimates(objects: &[Object]) -> MapStateEstimates {
    let r_vec: Vec<f64> = objects.iter().map(|obj| obj.r).collect();
    let (n_map, map_indices) = lmb_map_cardinality_estimate(&r_vec);

    let mut labels = DMatrix::zeros(2, n_map);
    let mut mu = Vec::with_capacity(n_map);
    let mut sigma = Vec::with_capacity(n_map);

    for (i, &j) in map_indices.iter().enumerate() {
        if j < objects.len() && !objects[j].mu.is_empty() {
            labels[(0, i)] = objects[j].birth_time;
            labels[(1, i)] = objects[j].birth_location;
            mu.push(objects[j].mu[0].clone());
            sigma.push(objects[j].sigma[0].clone());
        }
    }

    MapStateEstimates { labels, mu, sigma }
}

/// Update object trajectories after a time step
///
/// Increments trajectory length and adds current timestamp
///
/// # Arguments
/// * `objects` - Objects to update
/// * `t` - Current time step (0-indexed, will be stored as t+1)
#[inline]
pub fn update_object_trajectories(objects: &mut [Object], t: usize) {
    for obj in objects {
        if !obj.mu.is_empty() {
            obj.trajectory_length += 1;
            obj.timestamps.push(t + 1);
        }
    }
}

/// Export remaining long trajectories at end of filter run
///
/// # Arguments
/// * `objects` - Final objects
/// * `all_objects` - Vector to append exported trajectories to
/// * `model` - Model parameters
pub fn export_remaining_trajectories(
    objects: &[Object],
    all_objects: &mut Vec<Trajectory>,
    model: &Model,
) {
    for obj in objects {
        if obj.trajectory_length > model.minimum_trajectory_length {
            let traj = Trajectory {
                birth_location: obj.birth_location,
                birth_time: obj.birth_time,
                trajectory: DMatrix::from_columns(
                    &obj.mu.iter().map(|m| m.clone()).collect::<Vec<_>>(),
                ),
                trajectory_length: obj.trajectory_length,
                timestamps: obj.timestamps.clone(),
            };
            all_objects.push(traj);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::rng::SimpleRng;
    use crate::common::types::{DataAssociationMethod, ScenarioType};

    fn make_test_object(r: f64, birth_time: usize, birth_location: usize) -> Object {
        Object {
            r,
            birth_time,
            birth_location,
            number_of_gm_components: 1,
            w: vec![1.0],
            mu: vec![DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0])],
            sigma: vec![DMatrix::identity(4, 4)],
            trajectory: DMatrix::zeros(4, 1),
            trajectory_length: 0,
            timestamps: vec![],
        }
    }

    #[test]
    fn test_update_existence_no_measurements_sensor() {
        let mut objects = vec![
            make_test_object(0.8, 1, 1),
            make_test_object(0.5, 1, 2),
        ];

        let mut rng = SimpleRng::new(42);
        let mut model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );
        model.detection_probability_multisensor = Some(vec![0.9, 0.85]);

        update_existence_no_measurements_sensor(&mut objects, 1, &model);

        // r' = r*(1-p_d) / (1 - r*p_d)
        // For r=0.8, p_d=0.85: 0.8*0.15 / (1 - 0.8*0.85) = 0.12/0.32 = 0.375
        let expected_r0 = (0.8 * 0.15) / (1.0 - 0.8 * 0.85);
        assert!((objects[0].r - expected_r0).abs() < 1e-10);

        // For r=0.5, p_d=0.85: 0.5*0.15 / (1 - 0.5*0.85) = 0.075/0.575 ≈ 0.1304
        let expected_r1 = (0.5 * 0.15) / (1.0 - 0.5 * 0.85);
        assert!((objects[1].r - expected_r1).abs() < 1e-10);
    }

    #[test]
    fn test_gate_and_export_tracks() {
        let mut obj1 = make_test_object(0.8, 1, 1);
        obj1.trajectory_length = 10;

        let mut obj2 = make_test_object(0.001, 2, 1);
        obj2.trajectory_length = 10;

        let mut obj3 = make_test_object(0.002, 3, 1);
        obj3.trajectory_length = 2; // Too short to export

        let objects = vec![obj1, obj2, obj3];
        let mut all_objects = Vec::new();

        let mut rng = SimpleRng::new(42);
        let mut model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );
        model.existence_threshold = 0.01;
        model.minimum_trajectory_length = 5;

        let remaining = gate_and_export_tracks(objects, &mut all_objects, 20, &model);

        // Only obj1 should remain (r > 0.01)
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].birth_time, 1);

        // Only obj2 should be exported (r <= 0.01 and trajectory_length > 5)
        assert_eq!(all_objects.len(), 1);
        assert_eq!(all_objects[0].birth_time, 2);
    }

    #[test]
    fn test_extract_map_state_estimates() {
        let objects = vec![
            make_test_object(0.9, 1, 1),
            make_test_object(0.8, 2, 2),
            make_test_object(0.1, 3, 3),
        ];

        let estimates = extract_map_state_estimates(&objects);

        // MAP should select objects with high existence probabilities
        assert!(estimates.labels.ncols() > 0);
        assert_eq!(estimates.mu.len(), estimates.labels.ncols());
        assert_eq!(estimates.sigma.len(), estimates.labels.ncols());
    }

    #[test]
    fn test_update_object_trajectories() {
        let mut objects = vec![
            make_test_object(0.9, 1, 1),
            make_test_object(0.8, 2, 2),
        ];

        update_object_trajectories(&mut objects, 5);

        assert_eq!(objects[0].trajectory_length, 1);
        assert_eq!(objects[0].timestamps, vec![6]); // t+1

        assert_eq!(objects[1].trajectory_length, 1);
        assert_eq!(objects[1].timestamps, vec![6]);
    }
}
