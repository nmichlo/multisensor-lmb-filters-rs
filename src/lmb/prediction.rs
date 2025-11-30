//! LMB prediction step
//!
//! Implements the LMB filter's prediction step using the Chapman-Kolmogorov equation.
//! Matches MATLAB lmbPredictionStep.m exactly.

use crate::common::types::{Model, Object};
use ndarray::Array2;
use ndarray_linalg::Norm;

/// LMB prediction step
///
/// Computes predicted prior for the current time-step using the Chapman-Kolmogorov
/// equation, assuming an LMB prior and the standard multi-object motion model.
///
/// # Arguments
/// * `objects` - Vector of posterior LMB Bernoulli components from previous time
/// * `model` - Model parameters
/// * `t` - Current time-step
///
/// # Returns
/// Vector of prior LMB Bernoulli components (surviving + newly born objects)
///
/// # Implementation Notes
/// Matches MATLAB lmbPredictionStep.m exactly:
/// 1. Put existing Bernoulli components through the motion model
///    - Predict existence: r' = p_S * r
///    - Predict means: mu' = A * mu + u
///    - Predict covariances: Sigma' = A * Sigma * A' + R
/// 2. Add Bernoulli components for newly appearing objects
pub fn lmb_prediction_step(mut objects: Vec<Object>, model: &Model, t: usize) -> Vec<Object> {
    // Put existing Bernoulli components through the motion model
    for obj in &mut objects {
        // Predict existence probability
        obj.r = model.survival_probability * obj.r;

        // Predict each GM component
        for j in 0..obj.number_of_gm_components {
            // Predict mean: mu' = A * mu + u
            obj.mu[j] = model.a.dot(&obj.mu[j]) + &model.u;

            // Predict covariance: Sigma' = A * Sigma * A' + R
            obj.sigma[j] = model.a.dot(&obj.sigma[j]).dot(&model.a.t()) + &model.r;
        }
    }

    // Add Bernoulli components for newly appearing objects
    for birth_obj in &model.birth_parameters {
        let mut new_obj = birth_obj.clone();
        new_obj.birth_time = t;
        objects.push(new_obj);
    }

    objects
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, DMatrix, DVector, ScenarioType};

    #[test]
    fn test_lmb_prediction_empty() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let objects = vec![];
        let predicted = lmb_prediction_step(objects, &model, 1);

        // Should have birth objects
        assert_eq!(predicted.len(), model.number_of_birth_locations);
    }

    #[test]
    fn test_lmb_prediction_survival() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        // Create a test object
        let obj = Object {
            birth_location: 1,
            birth_time: 1,
            r: 0.8,
            number_of_gm_components: 1,
            w: vec![1.0],
            mu: vec![DVector::from_vec(vec![10.0, 1.0, 20.0, 2.0])],
            sigma: vec![Array2::eye(4)],
            trajectory_length: 0,
            trajectory: Array2::zeros((4, 0)),
            timestamps: vec![],
        };

        let objects = vec![obj.clone()];
        let predicted = lmb_prediction_step(objects, &model, 2);

        // Should have surviving object + birth objects
        assert_eq!(
            predicted.len(),
            1 + model.number_of_birth_locations
        );

        // Check existence probability updated
        assert!((predicted[0].r - model.survival_probability * 0.8).abs() < 1e-10);

        // Check mean predicted
        let expected_mu = &model.a * &obj.mu[0] + &model.u;
        let diff = &predicted[0].mu[0] - &expected_mu;
        assert!(diff.norm() < 1e-10);

        // Check covariance predicted
        let expected_sigma =
            &model.a * &obj.sigma[0] * model.a.t() + &model.r;
        let diff_sigma = &predicted[0].sigma[0] - &expected_sigma;
        assert!(diff_sigma.norm() < 1e-10);
    }

    #[test]
    fn test_lmb_prediction_birth_time() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let objects = vec![];
        let predicted = lmb_prediction_step(objects, &model, 5);

        // All new objects should have birth_time = 5
        for i in 0..model.number_of_birth_locations {
            assert_eq!(predicted[i].birth_time, 5);
        }
    }
}
