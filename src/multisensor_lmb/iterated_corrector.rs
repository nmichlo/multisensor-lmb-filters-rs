//! Iterated Corrector (IC) LMB Filter
//!
//! Implements the iterated corrector multi-sensor LMB filter.
//! Matches MATLAB runIcLmbFilter.m exactly.

use crate::common::association::gibbs::{lmb_gibbs_sampling, GibbsAssociationMatrices};
use crate::common::association::lbp::{loopy_belief_propagation, AssociationMatrices};
use crate::common::association::murtys::murtys_algorithm_wrapper;
use crate::common::types::{DataAssociationMethod, Model};
use crate::lmb::prediction::lmb_prediction_step;
use crate::multisensor_lmb::association::generate_lmb_sensor_association_matrices;
use crate::multisensor_lmb::parallel_update::{
    compute_posterior_lmb_spatial_distributions_multisensor, ParallelUpdateStateEstimates,
};
use crate::multisensor_lmb::utils::{
    extract_map_state_estimates, export_remaining_trajectories, gate_and_export_tracks,
    update_existence_no_measurements_sensor, update_object_trajectories,
};
use nalgebra::{DMatrix, DVector};

/// Run the iterated corrector (IC) multi-sensor LMB filter
///
/// Determines the objects' state estimates using IC-LMB with sequential sensor updates.
///
/// # Arguments
/// * `model` - Model parameters
/// * `measurements` - Measurements for each sensor and time-step [sensor][time][measurements]
/// * `number_of_sensors` - Number of sensors
///
/// # Returns
/// ParallelUpdateStateEstimates containing MAP estimates and trajectories
///
/// # Implementation Notes
/// Matches MATLAB runIcLmbFilter.m exactly:
/// 1. For each time step:
///    - Prediction
///    - For each sensor sequentially: measurement update (iterated correction)
///    - Gate tracks
///    - MAP cardinality extraction
///    - Update trajectories
///
/// Key difference from PU-LMB: Sequential updates instead of parallel + fusion
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn run_ic_lmb_filter(
    rng: &mut impl crate::common::rng::Rng,
    model: &Model,
    measurements: &[Vec<Vec<DVector<f64>>>], // [sensor][time][measurements]
    number_of_sensors: usize,
) -> ParallelUpdateStateEstimates {
    let simulation_length = measurements[0].len();

    // Initialize with empty objects - prediction will add births each timestep
    let mut objects = Vec::new();
    let mut all_objects = model.trajectory.clone();
    let mut labels = Vec::with_capacity(simulation_length);
    let mut mu_estimates = Vec::with_capacity(simulation_length);
    let mut sigma_estimates = Vec::with_capacity(simulation_length);

    // Run filter
    for t in 0..simulation_length {
        // Prediction
        objects = lmb_prediction_step(objects, model, t + 1);

        // Measurement update - sequential (iterated corrector)
        for s in 0..number_of_sensors {
            if !measurements[s][t].is_empty() {
                // Generate association matrices
                let (association_matrices, posterior_parameters) =
                    generate_lmb_sensor_association_matrices(&objects, &measurements[s][t], model, s);

                // Data association
                let (r, w) = match model.data_association_method {
                    DataAssociationMethod::LBP => {
                        let lbp_matrices = AssociationMatrices {
                            psi: association_matrices.psi.clone(),
                            phi: association_matrices.phi.clone(),
                            eta: association_matrices.eta.clone(),
                        };
                        let result = loopy_belief_propagation(
                            &lbp_matrices,
                            model.lbp_convergence_tolerance,
                            model.maximum_number_of_lbp_iterations,
                        );
                        (result.r.as_slice().to_vec(), result.w)
                    }
                    DataAssociationMethod::Gibbs | DataAssociationMethod::LBPFixed => {
                        let gibbs_matrices = GibbsAssociationMatrices {
                            p: association_matrices.p.clone(),
                            l: association_matrices.l.clone(),
                            r: association_matrices.gibbs_r.clone(),
                            c: association_matrices.cost.clone(),
                        };
                        let result = lmb_gibbs_sampling(rng, &gibbs_matrices, model.number_of_samples);
                        (result.r.as_slice().to_vec(), result.w)
                    }
                    DataAssociationMethod::Murty => {
                        let murty_result = murtys_algorithm_wrapper(
                            &association_matrices.cost,
                            model.number_of_assignments,
                        );

                        // Convert costs to weights
                        let mut assignment_weights = Vec::with_capacity(murty_result.costs.len());
                        for &cost in &murty_result.costs {
                            assignment_weights.push((-cost).exp());
                        }
                        let sum_weights: f64 = assignment_weights.iter().sum();
                        for w in &mut assignment_weights {
                            *w /= sum_weights;
                        }

                        // Marginalize
                        let n = objects.len();
                        let m = measurements[s][t].len();
                        let mut r_vec = vec![0.0; n];
                        let mut w_matrix = DMatrix::zeros(n, m + 1);

                        for (assignment_idx, assignment) in murty_result.assignments.row_iter().enumerate() {
                            let weight = assignment_weights[assignment_idx];
                            for (obj_idx, &meas_assign) in assignment.iter().enumerate() {
                                w_matrix[(obj_idx, meas_assign)] += weight;
                                if meas_assign > 0 {
                                    r_vec[obj_idx] += weight;
                                }
                            }
                        }

                        // Normalize and compute existence
                        for obj_idx in 0..n {
                            let row_sum: f64 = w_matrix.row(obj_idx).sum();
                            if row_sum > 1e-15 {
                                for meas_idx in 0..=m {
                                    w_matrix[(obj_idx, meas_idx)] /= row_sum;
                                }
                                r_vec[obj_idx] = 1.0 - w_matrix[(obj_idx, 0)];
                            }
                        }

                        (r_vec, w_matrix)
                    }
                };

                // Compute posterior spatial distributions
                objects = compute_posterior_lmb_spatial_distributions_multisensor(
                    objects,
                    &r,
                    &w,
                    &posterior_parameters,
                    model,
                );
            } else {
                // No measurements - update existence probabilities
                update_existence_no_measurements_sensor(&mut objects, s, model);
            }
        }

        // Gate tracks and export long discarded trajectories
        objects = gate_and_export_tracks(objects, &mut all_objects, t, model);

        // MAP cardinality extraction and RFS state estimate
        let map_estimates = extract_map_state_estimates(&objects);
        labels.push(map_estimates.labels);
        mu_estimates.push(map_estimates.mu);
        sigma_estimates.push(map_estimates.sigma);

        // Update each object's trajectory
        update_object_trajectories(&mut objects, t);
    }

    // Get any long trajectories that weren't extracted
    export_remaining_trajectories(&objects, &mut all_objects, model);

    ParallelUpdateStateEstimates {
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
    use crate::common::types::ScenarioType;

    #[test]
    fn test_run_ic_lmb_filter() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        // 2 sensors, 3 time steps
        let measurements = vec![
            vec![vec![], vec![], vec![]], // Sensor 1
            vec![vec![], vec![], vec![]], // Sensor 2
        ];

        let mut rng2 = crate::common::rng::SimpleRng::new(42);
        let estimates = run_ic_lmb_filter(&mut rng2, &model, &measurements, 2);

        assert_eq!(estimates.labels.len(), 3);
        assert_eq!(estimates.mu.len(), 3);
        assert_eq!(estimates.sigma.len(), 3);
    }
}