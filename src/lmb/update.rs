//! LMB posterior computation
//!
//! Computes posterior spatial distributions for LMB filter.
//! Matches MATLAB computePosteriorLmbSpatialDistributions.m exactly.

use crate::common::types::{DMatrix, DVector, Model, Object};
use crate::common::utils::prune_gaussian_mixture;
use crate::lmb::association::PosteriorParameters;
use ndarray::{Array1, Array2};

/// Compute posterior LMB spatial distributions
///
/// Completes the LMB filter's measurement update by computing each object's
/// posterior spatial distribution.
///
/// # Arguments
/// * `objects` - Prior LMB Bernoulli components
/// * `r` - Posterior existence probabilities (n x 1)
/// * `w` - Marginal association probabilities (n x (m+1))
/// * `posterior_parameters` - Posterior parameters from association step
/// * `model` - Model parameters
///
/// # Returns
/// Updated objects with posterior spatial distributions
///
/// # Implementation Notes
/// Matches MATLAB computePosteriorLmbSpatialDistributions.m exactly:
/// 1. Update existence probability: r'
/// 2. Reweight measurement-updated GMs using marginal association probabilities
/// 3. Apply crude mixture reduction:
///    - Sort weights descending
///    - Discard components below threshold
///    - Cap to maximum number of components
pub fn compute_posterior_lmb_spatial_distributions(
    mut objects: Vec<Object>,
    r: &DVector<f64>,
    w: &DMatrix<f64>,
    posterior_parameters: &[PosteriorParameters],
    model: &Model,
) -> Vec<Object> {
    for i in 0..objects.len() {
        // Update posterior existence probability
        objects[i].r = r[i];

        // Reweight measurement-updated Gaussian mixtures
        let num_posterior_components = posterior_parameters[i].w.ncols();
        let num_meas_plus_one = posterior_parameters[i].w.nrows();
        let mut posterior_weights = Vec::with_capacity(num_meas_plus_one * num_posterior_components);

        // Flatten: W(i, :)' .* posteriorParameters(i).w
        // IMPORTANT: MATLAB uses COLUMN-MAJOR ordering when reshaping!
        // Column-major: iterate columns first, then rows
        for comp_idx in 0..num_posterior_components {
            for meas_idx in 0..num_meas_plus_one {
                posterior_weights.push(
                    w[(i, meas_idx)] * posterior_parameters[i].w[(meas_idx, comp_idx)],
                );
            }
        }

        // Normalize
        let sum: f64 = posterior_weights.iter().sum();
        if sum > 1e-15 {
            for weight in &mut posterior_weights {
                *weight /= sum;
            }
        }

        // Crude mixture reduction algorithm
        let pruned =
            prune_gaussian_mixture(&posterior_weights, model.gm_weight_threshold, model.maximum_number_of_gm_components);

        objects[i].number_of_gm_components = pruned.num_components;
        objects[i].w = pruned.weights.clone();

        // Extract corresponding mu and sigma using sorted indices
        objects[i].mu = Vec::with_capacity(pruned.num_components);
        objects[i].sigma = Vec::with_capacity(pruned.num_components);

        for &original_idx in &pruned.indices {
            // Convert flat index to (meas_idx, comp_idx) using column-major ordering
            // Column-major: flat_idx = meas_idx + comp_idx * num_rows
            let comp_idx = original_idx / num_meas_plus_one;
            let meas_idx = original_idx % num_meas_plus_one;

            objects[i].mu.push(posterior_parameters[i].mu[meas_idx][comp_idx].clone());
            objects[i].sigma.push(posterior_parameters[i].sigma[meas_idx][comp_idx].clone());
        }
    }

    objects
}

/// Update existence probability when no measurements are received
///
/// Computes posterior existence probability for missed detection.
///
/// # Arguments
/// * `objects` - LMB Bernoulli components
/// * `detection_probability` - Detection probability p_D
///
/// # Returns
/// Updated objects with adjusted existence probabilities
///
/// # Implementation Notes
/// Matches MATLAB runLmbFilter.m lines 54-56:
/// r' = r * (1 - p_D) / (1 - r * p_D)
pub fn update_no_measurements(mut objects: Vec<Object>, detection_probability: f64) -> Vec<Object> {
    for obj in &mut objects {
        obj.r = (obj.r * (1.0 - detection_probability)) / (1.0 - obj.r * detection_probability);
    }
    objects
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};
    use crate::lmb::association::generate_lmb_association_matrices;

    #[test]
    fn test_update_no_measurements() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let mut objects = model.birth_parameters.clone();
        let initial_r = objects[0].r;

        objects = update_no_measurements(objects, model.detection_probability);

        // Check existence probability updated correctly
        let expected_r = (initial_r * (1.0 - model.detection_probability))
            / (1.0 - initial_r * model.detection_probability);
        assert!((objects[0].r - expected_r).abs() < 1e-10);
    }

    #[test]
    fn test_compute_posterior_weights_normalized() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let objects = model.birth_parameters.clone();
        let measurements = vec![DVector::from_vec(vec![0.0, 0.0])];

        let association_result =
            generate_lmb_association_matrices(&objects, &measurements, &model);

        // Create dummy r and W
        let r = Array1::from_vec(vec![0.8; objects.len()]);
        let w = Array2::from_elem((objects.len(), 2), 0.5); // [miss, meas1]

        let updated = compute_posterior_lmb_spatial_distributions(
            objects.clone(),
            &r,
            &w,
            &association_result.posterior_parameters,
            &model,
        );

        // Check weights are normalized
        for obj in &updated {
            let sum: f64 = obj.w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Weight sum: {}", sum);
        }
    }

    #[test]
    fn test_compute_posterior_existence_updated() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let objects = model.birth_parameters.clone();
        let measurements = vec![DVector::from_vec(vec![0.0, 0.0])];

        let association_result =
            generate_lmb_association_matrices(&objects, &measurements, &model);

        let r = Array1::from_vec(vec![0.75; objects.len()]);
        let w = Array2::from_elem((objects.len(), 2), 0.5);

        let updated = compute_posterior_lmb_spatial_distributions(
            objects.clone(),
            &r,
            &w,
            &association_result.posterior_parameters,
            &model,
        );

        // Check existence probability updated
        for obj in &updated {
            assert!((obj.r - 0.75).abs() < 1e-10);
        }
    }
}
