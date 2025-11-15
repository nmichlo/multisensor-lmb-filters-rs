//! Main LMB filter implementation
//!
//! Implements the complete LMB filter for multi-object tracking.
//! Matches MATLAB runLmbFilter.m exactly.

use crate::common::types::{DataAssociationMethod, Model, Object};
use crate::common::utils::gate_objects_by_existence;
use crate::lmb::association::generate_lmb_association_matrices;
use crate::lmb::cardinality::lmb_map_cardinality_estimate;
use crate::lmb::data_association::{lmb_gibbs, lmb_lbp, lmb_lbp_fixed, lmb_murtys};
use crate::lmb::prediction::lmb_prediction_step;
use crate::lmb::update::{compute_posterior_lmb_spatial_distributions, update_no_measurements};
use nalgebra::{DMatrix, DVector};

/// State estimates output from LMB filter
#[derive(Debug, Clone)]
pub struct LmbStateEstimates {
    /// Labels for each time-step (birth_time, birth_location)
    pub labels: Vec<DMatrix<usize>>,
    /// Mean estimates for each time-step
    pub mu: Vec<Vec<DVector<f64>>>,
    /// Covariance estimates for each time-step
    pub sigma: Vec<Vec<DMatrix<f64>>>,
    /// All objects (including discarded long trajectories)
    pub objects: Vec<Object>,
}

/// Run the LMB filter
///
/// Determines the objects' state estimates using the LMB filter.
///
/// # Arguments
/// * `model` - Model parameters
/// * `measurements` - Measurements for each time-step
///
/// # Returns
/// LmbStateEstimates containing MAP estimates and trajectories
///
/// # Implementation Notes
/// Matches MATLAB runLmbFilter.m exactly:
/// 1. Prediction step
/// 2. Measurement update (data association + posterior computation)
/// 3. Gate tracks by existence probability
/// 4. MAP cardinality extraction
/// 5. Update trajectories
pub fn run_lmb_filter(
    rng: &mut impl crate::common::rng::Rng,
    model: &Model,
    measurements: &[Vec<DVector<f64>>],
) -> LmbStateEstimates {
    let simulation_length = measurements.len();

    // Initialize
    let mut objects = model.object.clone();
    let mut labels = Vec::with_capacity(simulation_length);
    let mut mu_estimates = Vec::with_capacity(simulation_length);
    let mut sigma_estimates = Vec::with_capacity(simulation_length);
    let mut all_objects = Vec::new();

    // Run filter
    for t in 0..simulation_length {
        // Prediction
        objects = lmb_prediction_step(objects, model, t + 1);

        // Measurement update
        if !measurements[t].is_empty() {
            // Generate association matrices
            let association_result =
                generate_lmb_association_matrices(&objects, &measurements[t], model);

            // Data association
            let (r, w) = match model.data_association_method {
                DataAssociationMethod::LBP => {
                    lmb_lbp(&association_result, model.lbp_convergence_tolerance, model.maximum_number_of_lbp_iterations)
                }
                DataAssociationMethod::LBPFixed => {
                    lmb_lbp_fixed(&association_result, model.maximum_number_of_lbp_iterations)
                }
                DataAssociationMethod::Gibbs => {
                    lmb_gibbs(rng, &association_result, model.number_of_samples)
                }
                DataAssociationMethod::Murty => {
                    let (r, w, _v) = lmb_murtys(&association_result, model.number_of_assignments);
                    (r, w)
                }
            };

            // Compute posterior spatial distributions
            objects = compute_posterior_lmb_spatial_distributions(
                objects,
                &r,
                &w,
                &association_result.posterior_parameters,
                model,
            );
        } else {
            // No measurements
            objects = update_no_measurements(objects, model.detection_probability);
        }

        // Gate tracks
        let objects_likely_to_exist = gate_objects_by_existence(
            &objects.iter().map(|obj| obj.r).collect::<Vec<_>>(),
            model.existence_threshold,
        );

        // Extract discarded objects with long trajectories
        for (i, obj) in objects.iter().enumerate() {
            if !objects_likely_to_exist[i]
                && obj.trajectory_length > model.minimum_trajectory_length
            {
                all_objects.push(obj.clone());
            }
        }

        // Keep only likely objects
        objects = objects
            .into_iter()
            .enumerate()
            .filter_map(|(i, obj)| {
                if objects_likely_to_exist[i] {
                    Some(obj)
                } else {
                    None
                }
            })
            .collect();

        // MAP cardinality extraction
        let existence_probs: Vec<f64> = objects.iter().map(|obj| obj.r).collect();
        let (n_map, map_indices) = lmb_map_cardinality_estimate(&existence_probs);

        // Extract RFS state estimate
        let mut labels_t = DMatrix::zeros(2, n_map);
        let mut mu_t = Vec::with_capacity(n_map);
        let mut sigma_t = Vec::with_capacity(n_map);

        for (i, &j) in map_indices.iter().enumerate() {
            labels_t[(0, i)] = objects[j].birth_time;
            labels_t[(1, i)] = objects[j].birth_location;
            mu_t.push(objects[j].mu[0].clone());
            sigma_t.push(objects[j].sigma[0].clone());
        }

        labels.push(labels_t);
        mu_estimates.push(mu_t);
        sigma_estimates.push(sigma_t);

        // Update trajectories
        for obj in &mut objects {
            let j = obj.trajectory_length;
            obj.trajectory_length = j + 1;

            // Resize trajectory if needed
            if obj.trajectory.ncols() < j + 1 {
                let mut new_traj = DMatrix::zeros(obj.mu[0].len(), j + 2);
                new_traj.view_mut((0, 0), (obj.mu[0].len(), obj.trajectory.ncols()))
                    .copy_from(&obj.trajectory);
                obj.trajectory = new_traj;
            }

            obj.trajectory.column_mut(j).copy_from(&obj.mu[0]);

            if obj.timestamps.len() <= j {
                obj.timestamps.resize(j + 1, 0);
            }
            obj.timestamps[j] = t + 1;
        }
    }

    // Get any long trajectories that weren't extracted
    for obj in &objects {
        if obj.trajectory_length > model.minimum_trajectory_length {
            all_objects.push(obj.clone());
        }
    }

    LmbStateEstimates {
        labels,
        mu: mu_estimates,
        sigma: sigma_estimates,
        objects: all_objects,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};

    #[test]
    fn test_run_lmb_filter_no_measurements() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        // Run filter with no measurements
        let measurements = vec![vec![]; 5];
        let estimates = run_lmb_filter(&model, &measurements);

        assert_eq!(estimates.labels.len(), 5);
        assert_eq!(estimates.mu.len(), 5);
        assert_eq!(estimates.sigma.len(), 5);
    }

    #[test]
    fn test_run_lmb_filter_with_measurements() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        // Run filter with some measurements
        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])],
            vec![DVector::from_vec(vec![1.0, 1.0])],
            vec![],
        ];

        let estimates = run_lmb_filter(&model, &measurements);

        assert_eq!(estimates.labels.len(), 3);
        assert_eq!(estimates.mu.len(), 3);
        assert_eq!(estimates.sigma.len(), 3);
    }
}
