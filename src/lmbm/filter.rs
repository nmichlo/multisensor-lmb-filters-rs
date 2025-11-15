//! Main LMBM filter implementation
//!
//! Implements the complete LMBM filter for multi-object tracking with hypothesis management.
//! Matches MATLAB runLmbmFilter.m exactly.

use crate::common::association::murtys::murtys_algorithm_wrapper;
use crate::common::types::{DataAssociationMethod, Model, Trajectory};
use crate::lmbm::association::{generate_lmbm_association_matrices, lmbm_gibbs_sampling};
use crate::lmbm::hypothesis::{
    determine_posterior_hypothesis_parameters, lmbm_normalisation_and_gating,
    lmbm_state_extraction,
};
use crate::lmbm::prediction::lmbm_prediction_step;
use nalgebra::{DMatrix, DVector};

/// State estimates output from LMBM filter
#[derive(Debug, Clone)]
pub struct LmbmStateEstimates {
    /// Labels for each time-step (birth_time, birth_location)
    pub labels: Vec<DMatrix<usize>>,
    /// Mean estimates for each time-step
    pub mu: Vec<Vec<DVector<f64>>>,
    /// Covariance estimates for each time-step
    pub sigma: Vec<Vec<DMatrix<f64>>>,
    /// All trajectories (including discarded long trajectories)
    pub objects: Vec<Trajectory>,
}

/// Run the LMBM filter
///
/// Determines the objects' state estimates using the LMBM filter.
///
/// # Arguments
/// * `rng` - Random number generator
/// * `model` - Model parameters
/// * `measurements` - Measurements for each time-step
///
/// # Returns
/// LmbmStateEstimates containing MAP estimates and trajectories
///
/// # Implementation Notes
/// Matches MATLAB runLmbmFilter.m exactly:
/// 1. For each prior hypothesis:
///    - Prediction step
///    - Measurement update (data association + posterior computation)
///    - If no measurements, update existence probabilities
/// 2. Normalize and gate posterior hypotheses
/// 3. Gate trajectories by existence probability
/// 4. State extraction using heuristic MAP
/// 5. Update trajectories
pub fn run_lmbm_filter(
    rng: &mut impl crate::common::rng::Rng,
    model: &Model,
    measurements: &[Vec<DVector<f64>>],
) -> LmbmStateEstimates {
    let simulation_length = measurements.len();

    // Initialize
    let mut hypotheses = vec![model.hypotheses.clone()];
    let mut objects = model.trajectory.clone();
    let mut labels = Vec::with_capacity(simulation_length);
    let mut mu_estimates = Vec::with_capacity(simulation_length);
    let mut sigma_estimates = Vec::with_capacity(simulation_length);
    let mut all_objects = Vec::new();

    // Run filter
    for t in 0..simulation_length {
        // Add new trajectory structs for birth locations
        let mut new_birth_objects = model.birth_trajectory.clone();
        for obj in &mut new_birth_objects {
            obj.birth_time = t + 1;
        }
        objects.extend(new_birth_objects);

        // Preallocate posterior hypotheses
        let mut posterior_hypotheses = Vec::new();

        // Generate posterior hypotheses for each prior hypothesis
        for prior_hyp in &hypotheses {
            // Prediction step
            let prior_hypothesis = lmbm_prediction_step(prior_hyp.clone(), model, t + 1);

            // Measurement update
            if !measurements[t].is_empty() {
                // Generate association matrices and posterior parameters
                let association_result =
                    generate_lmbm_association_matrices(&prior_hypothesis, &measurements[t], model);

                // Generate posterior hypotheses using Gibbs sampling or Murty's
                let v = match model.data_association_method {
                    DataAssociationMethod::Murty => {
                        let murtys_result = murtys_algorithm_wrapper(
                            &association_result.association_matrices.c,
                            model.number_of_assignments,
                        );
                        murtys_result.assignments
                    }
                    _ => {
                        // Gibbs, LBP, LBPFixed all use Gibbs sampling for LMBM
                        lmbm_gibbs_sampling(
                            rng,
                            &association_result.association_matrices.p,
                            &association_result.association_matrices.c,
                            model.number_of_samples,
                        )
                    }
                };

                // Determine each posterior hypothesis' parameters
                let new_hypotheses = determine_posterior_hypothesis_parameters(
                    &v,
                    &association_result.association_matrices.l,
                    &association_result.posterior_parameters,
                    &prior_hypothesis,
                );

                // Add posterior hypotheses to the pile
                posterior_hypotheses.extend(new_hypotheses);
            } else {
                // No measurements - update existence probabilities
                let mut updated_hypothesis = prior_hypothesis.clone();
                for i in 0..updated_hypothesis.r.len() {
                    let r = updated_hypothesis.r[i];
                    updated_hypothesis.r[i] = ((1.0 - model.detection_probability) * r)
                        / (1.0 - model.detection_probability * r);
                }
                posterior_hypotheses.push(updated_hypothesis);
            }
        }

        // Normalize posterior hypothesis weights and discard unlikely hypotheses
        let (gated_hypotheses, objects_likely_to_exist) =
            lmbm_normalisation_and_gating(posterior_hypotheses, model);
        hypotheses = gated_hypotheses;

        // Gate trajectories
        // Objects with low existence probabilities and long trajectories are worth exporting
        for (i, obj) in objects.iter().enumerate() {
            if i < objects_likely_to_exist.len() {
                if !objects_likely_to_exist[i]
                    && obj.trajectory_length > model.minimum_trajectory_length
                {
                    all_objects.push(obj.clone());
                }
            }
        }

        // Keep objects with high existence probabilities
        let mut kept_objects = Vec::new();
        for (i, obj) in objects.iter().enumerate() {
            if i < objects_likely_to_exist.len() && objects_likely_to_exist[i] {
                kept_objects.push(obj.clone());
            }
        }
        objects = kept_objects;

        // State extraction (using heuristic MAP)
        let (cardinality_estimate, extraction_indices) =
            lmbm_state_extraction(&hypotheses, false);

        // Extract RFS state estimate
        let mut labels_t = DMatrix::zeros(2, cardinality_estimate);
        let mut mu_t = Vec::with_capacity(cardinality_estimate);
        let mut sigma_t = Vec::with_capacity(cardinality_estimate);

        if !hypotheses.is_empty() {
            for (i, &j) in extraction_indices.iter().enumerate() {
                // Hypotheses in the posterior LMBM are sorted according to weight
                if j < hypotheses[0].birth_time.len() {
                    labels_t[(0, i)] = hypotheses[0].birth_time[j];
                    labels_t[(1, i)] = hypotheses[0].birth_location[j];
                    mu_t.push(hypotheses[0].mu[j].clone());
                    sigma_t.push(hypotheses[0].sigma[j].clone());
                }
            }
        }

        labels.push(labels_t);
        mu_estimates.push(mu_t);
        sigma_estimates.push(sigma_t);

        // Update each object's trajectory
        if !hypotheses.is_empty() {
            for (i, obj) in objects.iter_mut().enumerate() {
                if i < hypotheses[0].mu.len() {
                    let j = obj.trajectory_length;
                    obj.trajectory_length = j + 1;

                    // Resize trajectory if needed
                    if obj.trajectory.ncols() < j + 1 {
                        let mut new_traj = DMatrix::zeros(hypotheses[0].mu[i].len(), j + 2);
                        new_traj
                            .view_mut((0, 0), (hypotheses[0].mu[i].len(), obj.trajectory.ncols()))
                            .copy_from(&obj.trajectory);
                        obj.trajectory = new_traj;
                    }

                    obj.trajectory.column_mut(j).copy_from(&hypotheses[0].mu[i]);

                    if obj.timestamps.len() <= j {
                        obj.timestamps.resize(j + 1, 0);
                    }
                    obj.timestamps[j] = t + 1;
                }
            }
        }
    }

    // Get any long trajectories that weren't extracted
    for obj in &objects {
        if obj.trajectory_length > model.minimum_trajectory_length {
            all_objects.push(obj.clone());
        }
    }

    LmbmStateEstimates {
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
    fn test_run_lmbm_filter_no_measurements() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        // Run filter with no measurements
        let measurements = vec![vec![]; 5];
        let estimates = run_lmbm_filter(&model, &measurements);

        assert_eq!(estimates.labels.len(), 5);
        assert_eq!(estimates.mu.len(), 5);
        assert_eq!(estimates.sigma.len(), 5);
    }

    #[test]
    fn test_run_lmbm_filter_with_measurements() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        // Run filter with some measurements
        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])],
            vec![DVector::from_vec(vec![1.0, 1.0])],
            vec![],
        ];

        let estimates = run_lmbm_filter(&model, &measurements);

        assert_eq!(estimates.labels.len(), 3);
        assert_eq!(estimates.mu.len(), 3);
        assert_eq!(estimates.sigma.len(), 3);
    }
}
