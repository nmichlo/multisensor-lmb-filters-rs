//! Parallel Update LMB Filter
//!
//! Implements the parallel update multi-sensor LMB filter with track merging.
//! Matches MATLAB runParallelUpdateLmbFilter.m and computePosteriorLmbSpatialDistributions.m exactly.

use crate::common::association::gibbs::{lmb_gibbs_sampling, GibbsAssociationMatrices};
use crate::common::association::lbp::{loopy_belief_propagation, AssociationMatrices};
use crate::common::association::murtys::murtys_algorithm_wrapper;
use crate::common::types::{DataAssociationMethod, Model, Object, Trajectory};
use crate::common::utils::prune_gaussian_mixture;
use crate::lmb::prediction::lmb_prediction_step;
use crate::multisensor_lmb::association::{generate_lmb_sensor_association_matrices, LmbPosteriorParameters};
use crate::multisensor_lmb::merging::{aa_lmb_track_merging, ga_lmb_track_merging, pu_lmb_track_merging};
use crate::multisensor_lmb::utils::{
    extract_map_state_estimates, export_remaining_trajectories, gate_and_export_tracks,
    update_existence_no_measurements_sensor, update_object_trajectories,
};
use nalgebra::{DMatrix, DVector};

/// Parallel update mode for track merging
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelUpdateMode {
    /// Arithmetic Average
    AA,
    /// Geometric Average
    GA,
    /// Parallel Update (information form)
    PU,
}

/// State estimates output from parallel update LMB filter
#[derive(Debug, Clone)]
pub struct ParallelUpdateStateEstimates {
    /// Labels for each time-step (birth_time, birth_location)
    pub labels: Vec<DMatrix<usize>>,
    /// Mean estimates for each time-step
    pub mu: Vec<Vec<DVector<f64>>>,
    /// Covariance estimates for each time-step
    pub sigma: Vec<Vec<DMatrix<f64>>>,
    /// All trajectories (including discarded long trajectories)
    pub objects: Vec<Trajectory>,
}

/// Compute posterior LMB spatial distributions for multi-sensor
///
/// Completes the measurement update by computing each object's posterior spatial distribution.
///
/// # Arguments
/// * `objects` - Prior LMB Bernoulli components
/// * `r` - Posterior existence probabilities (n x 1)
/// * `w` - Marginal association probabilities (n x (m+1))
/// * `posterior_parameters` - Posterior parameters from multi-sensor association step
/// * `model` - Model parameters
///
/// # Returns
/// Updated objects with posterior spatial distributions
///
/// # Implementation Notes
/// Matches MATLAB computePosteriorLmbSpatialDistributions.m exactly:
/// 1. Update existence probability
/// 2. Reweight measurement-updated GMs using marginal association probabilities
/// 3. Apply crude mixture reduction (sort, discard, cap)
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn compute_posterior_lmb_spatial_distributions_multisensor(
    mut objects: Vec<Object>,
    r: &[f64],
    w: &DMatrix<f64>,
    posterior_parameters: &LmbPosteriorParameters,
    model: &Model,
) -> Vec<Object> {
    for i in 0..objects.len() {
        // Update posterior existence probability
        objects[i].r = r[i];

        // Reweight measurement-updated Gaussian mixtures
        // W(i, :)' .* posteriorParameters(i).w
        let num_measurements = w.ncols() - 1; // First column is miss
        let num_gm = if !posterior_parameters.w[i].is_empty() {
            posterior_parameters.w[i][0].len()
        } else {
            0
        };

        let mut posterior_weights = Vec::new();
        for meas_idx in 0..=num_measurements {
            for gm_idx in 0..num_gm {
                posterior_weights.push(w[(i, meas_idx)] * posterior_parameters.w[i][meas_idx][gm_idx]);
            }
        }

        // Normalize
        let sum: f64 = posterior_weights.iter().sum();
        if sum > 1e-15 {
            for weight in &mut posterior_weights {
                *weight /= sum;
            }
        }

        // Crude mixture reduction
        let pruned = prune_gaussian_mixture(
            &posterior_weights,
            model.gm_weight_threshold,
            model.maximum_number_of_gm_components,
        );

        objects[i].number_of_gm_components = pruned.num_components;
        objects[i].w = pruned.weights.clone();

        // Extract corresponding mu and sigma using sorted indices
        objects[i].mu = Vec::with_capacity(pruned.num_components);
        objects[i].sigma = Vec::with_capacity(pruned.num_components);

        for &original_idx in &pruned.indices {
            // Convert flat index to (meas, gm)
            let meas_idx = original_idx / num_gm;
            let gm_idx = original_idx % num_gm;

            objects[i].mu.push(posterior_parameters.mu[i][meas_idx][gm_idx].clone());
            objects[i].sigma.push(posterior_parameters.sigma[i][meas_idx][gm_idx].clone());
        }
    }

    objects
}

/// Run the parallel update multi-sensor LMB filter
///
/// Determines the objects' state estimates using parallel update LMB with track merging.
///
/// # Arguments
/// * `model` - Model parameters
/// * `measurements` - Measurements for each sensor and time-step [sensor][time][measurements]
/// * `number_of_sensors` - Number of sensors
/// * `update_mode` - Track merging mode (AA, GA, or PU)
///
/// # Returns
/// ParallelUpdateStateEstimates containing MAP estimates and trajectories
///
/// # Implementation Notes
/// Matches MATLAB runParallelUpdateLmbFilter.m exactly:
/// 1. For each time step:
///    - Prediction
///    - For each sensor: measurement update
///    - Track merging (AA/GA/PU)
///    - Gate tracks
///    - MAP cardinality extraction
///    - Update trajectories
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn run_parallel_update_lmb_filter(
    rng: &mut impl crate::common::rng::Rng,
    model: &Model,
    measurements: &[Vec<Vec<DVector<f64>>>], // [sensor][time][measurements]
    number_of_sensors: usize,
    update_mode: ParallelUpdateMode,
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

        // Save predicted objects for PU merging (need prior before sensor updates)
        let predicted_objects = objects.clone();


        // Measurement update for each sensor
        let mut measurement_updated_distributions = Vec::with_capacity(number_of_sensors);

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
                        // Matches MATLAB lmbMurtysAlgorithm.m and single-sensor Rust implementation
                        let n = objects.len();
                        let m = measurements[s][t].len();

                        let murty_result = murtys_algorithm_wrapper(
                            &association_matrices.cost,
                            model.number_of_assignments,
                        );
                        let v = murty_result.assignments; // (k x n)

                        // Determine marginal distributions
                        // W = repmat(V, 1, 1, m+1) == reshape(0:m, 1, 1, m+1)
                        let k = v.nrows();
                        let mut w_indicator = vec![DMatrix::zeros(k, n); m + 1];

                        for meas_idx in 0..=m {
                            for i in 0..k {
                                for j in 0..n {
                                    if v[(i, j)] == meas_idx {
                                        w_indicator[meas_idx][(i, j)] = 1.0;
                                    }
                                }
                            }
                        }

                        // J = reshape(associationMatrices.L(n * V + (1:n)), size(V, 1), n)
                        let mut j_matrix = DMatrix::zeros(k, n);
                        for i in 0..k {
                            for obj_idx in 0..n {
                                let meas_idx = v[(i, obj_idx)];
                                j_matrix[(i, obj_idx)] = association_matrices.l[(obj_idx, meas_idx)];
                            }
                        }

                        // L = permute(sum(prod(J, 2) .* W, 1), [2 1 3])
                        let mut l_marg = Vec::with_capacity(m + 1);
                        for meas_idx in 0..=m {
                            let mut l_col = vec![0.0; n];
                            for obj_idx in 0..n {
                                let mut sum = 0.0;
                                for event_idx in 0..k {
                                    // prod(J, 2): product across objects
                                    let mut prod = 1.0;
                                    for j in 0..n {
                                        prod *= j_matrix[(event_idx, j)];
                                    }
                                    sum += prod * w_indicator[meas_idx][(event_idx, obj_idx)];
                                }
                                l_col[obj_idx] = sum;
                            }
                            l_marg.push(l_col);
                        }

                        // Sigma = reshape(L, n, m+1)
                        let mut sigma = DMatrix::zeros(n, m + 1);
                        for obj_idx in 0..n {
                            for meas_idx in 0..=m {
                                sigma[(obj_idx, meas_idx)] = l_marg[meas_idx][obj_idx];
                            }
                        }

                        // Tau = (Sigma .* R) ./ sum(Sigma, 2)
                        // CRITICAL: This was missing in the old implementation!
                        let mut tau = DMatrix::zeros(n, m + 1);
                        for obj_idx in 0..n {
                            let row_sum: f64 = sigma.row(obj_idx).sum();
                            if row_sum > 1e-15 {
                                for meas_idx in 0..=m {
                                    tau[(obj_idx, meas_idx)] =
                                        (sigma[(obj_idx, meas_idx)] * association_matrices.gibbs_r[(obj_idx, meas_idx)])
                                        / row_sum;
                                }
                            }
                        }

                        // r = sum(Tau, 2)
                        let mut r_vec = vec![0.0; n];
                        for obj_idx in 0..n {
                            r_vec[obj_idx] = tau.row(obj_idx).sum();
                        }

                        // W = Tau ./ r
                        let mut w_matrix = DMatrix::zeros(n, m + 1);
                        for obj_idx in 0..n {
                            if r_vec[obj_idx] > 1e-15 {
                                for meas_idx in 0..=m {
                                    w_matrix[(obj_idx, meas_idx)] = tau[(obj_idx, meas_idx)] / r_vec[obj_idx];
                                }
                            }
                        }

                        (r_vec, w_matrix)
                    }
                };

                // DEBUG: Print r values from data association at t=1
                if t == 1 && update_mode == ParallelUpdateMode::AA {
                    eprintln!("\n=== DEBUG t={} Sensor {} AFTER DATA ASSOCIATION ===", t, s);
                    eprintln!("r from assoc: {:?}", r.iter().map(|&ri| format!("{:.10e}", ri)).collect::<Vec<_>>());
                }

                // Compute posterior spatial distributions
                let updated = compute_posterior_lmb_spatial_distributions_multisensor(
                    objects.clone(),
                    &r,
                    &w,
                    &posterior_parameters,
                    model,
                );

                // DEBUG: Print r values after spatial update at t=1
                if t == 1 && update_mode == ParallelUpdateMode::AA {
                    eprintln!("r after spatial: {:?}", updated.iter().map(|obj| format!("{:.10e}", obj.r)).collect::<Vec<_>>());
                }

                measurement_updated_distributions.push(updated.clone());
            } else {
                // No measurements - update existence probabilities
                let mut updated = objects.clone();
                update_existence_no_measurements_sensor(&mut updated, s, model);
                measurement_updated_distributions.push(updated);
            }
        }

        // Track merging

        objects = match update_mode {
            ParallelUpdateMode::AA => {
                aa_lmb_track_merging(&measurement_updated_distributions, model)
            }
            ParallelUpdateMode::GA => {
                ga_lmb_track_merging(&measurement_updated_distributions, model)
            }
            ParallelUpdateMode::PU => {
                // For PU, we need the prior (predicted) objects before sensor updates
                pu_lmb_track_merging(&measurement_updated_distributions, &predicted_objects, number_of_sensors)
            }
        };

        // DEBUG: Print existence probabilities after merging at t=1 or t=2 or t=82
        if (t == 1 || t == 2 || t == 82) && update_mode == ParallelUpdateMode::AA {
            eprintln!("\n=== DEBUG t={} AFTER AA MERGING ===", t);
            eprintln!("Merged objects: {}", objects.len());
            for (i, obj) in objects.iter().enumerate() {
                // For t=82, also print mu[0] for the target object
                if t == 82 && obj.birth_time == 20 && obj.birth_location == 3 {
                    eprintln!("  Object {} (t=20,loc=3): r={:.10e}, n_gm={}, mu[0]=({:.4},{:.4}), w[0]={:.6e}",
                        i, obj.r, obj.number_of_gm_components, obj.mu[0][0], obj.mu[0][1], obj.w[0]);
                    // Print all GM components for this object
                    for j in 0..obj.number_of_gm_components.min(5) {
                        eprintln!("    GM[{}]: w={:.6e}, mu=({:.4},{:.4})", j, obj.w[j], obj.mu[j][0], obj.mu[j][1]);
                    }
                } else if t != 82 {
                    eprintln!("  Object {}: r={:.10e}", i, obj.r);
                }
            }
        }

        // DEBUG: Print gating results at t=1 or t=2
        if (t == 1 || t == 2) && update_mode == ParallelUpdateMode::AA {
            eprintln!("\n=== DEBUG t={} AFTER GATING (threshold={}) ===", t, model.existence_threshold);
            for (i, obj) in objects.iter().enumerate() {
                let keep = obj.r > model.existence_threshold;
                eprintln!("  Object {}: r={:.10e} -> {}", i, obj.r, if keep { "KEEP" } else { "DROP" });
            }
            let kept_count = objects.iter().filter(|obj| obj.r > model.existence_threshold).count();
            eprintln!("Kept {} / {} objects", kept_count, objects.len());
        }

        // Gate tracks and export long discarded trajectories
        objects = gate_and_export_tracks(objects, &mut all_objects, t, model);

        // DEBUG: Print MAP cardinality inputs/outputs at t=1 or t=2
        if (t == 1 || t == 2) && update_mode == ParallelUpdateMode::AA {
            let r_vec: Vec<f64> = objects.iter().map(|obj| obj.r).collect();
            eprintln!("\n=== DEBUG t={} MAP CARDINALITY ESTIMATION ===", t);
            eprintln!("Input r_vec ({}): {:?}", r_vec.len(), r_vec.iter().map(|&r| format!("{:.6e}", r)).collect::<Vec<_>>());
            let (n_map, map_indices) = crate::lmb::cardinality::lmb_map_cardinality_estimate(&r_vec);
            eprintln!("Output n_map={}, map_indices={:?}", n_map, map_indices);
        }

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
    fn test_run_parallel_update_lmb_filter_aa() {
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

        let estimates = run_parallel_update_lmb_filter(&mut rng, &model, &measurements, 2, ParallelUpdateMode::AA);

        assert_eq!(estimates.labels.len(), 3);
        assert_eq!(estimates.mu.len(), 3);
        assert_eq!(estimates.sigma.len(), 3);
    }
}