//! LMBM prediction step
//!
//! Implements the LMBM filter's prediction step using the Chapman-Kolmogorov equation.
//! Matches MATLAB lmbmPredictionStep.m exactly.

use crate::common::types::{Hypothesis, Model};

/// LMBM prediction step
///
/// Computes predicted prior for the current time-step using the Chapman-Kolmogorov
/// equation, assuming an LMBM prior and the standard multi-object motion model.
///
/// # Arguments
/// * `hypothesis` - Posterior LMBM hypothesis from previous time
/// * `model` - Model parameters
/// * `t` - Current time-step
///
/// # Returns
/// Prior LMBM hypothesis
///
/// # Implementation Notes
/// Matches MATLAB lmbmPredictionStep.m exactly:
/// 1. Put existing Bernoulli components through the motion model
/// 2. Add Bernoulli components for newly appearing objects
pub fn lmbm_prediction_step(mut hypothesis: Hypothesis, model: &Model, t: usize) -> Hypothesis {
    let number_of_objects = hypothesis.r.len();

    // Put existing Bernoulli components through the motion model
    for i in 0..number_of_objects {
        // Predict existence probability
        hypothesis.r[i] = model.survival_probability * hypothesis.r[i];

        // Predict mean: mu' = A * mu + u
        hypothesis.mu[i] = &model.a * &hypothesis.mu[i] + &model.u;

        // Predict covariance: Sigma' = A * Sigma * A' + R
        hypothesis.sigma[i] = &model.a * &hypothesis.sigma[i] * model.a.transpose() + &model.r;
    }

    // Add Bernoulli components for newly appearing objects
    let stride_start = number_of_objects;
    let stride_end = number_of_objects + model.number_of_birth_locations;

    for (idx, birth_loc) in model.birth_location_labels.iter().enumerate() {
        let global_idx = stride_start + idx;

        // Resize vectors if needed
        if hypothesis.birth_location.len() <= global_idx {
            hypothesis.birth_location.resize(stride_end, 0);
            hypothesis.birth_time.resize(stride_end, 0);
            hypothesis.r.resize(stride_end, 0.0);
            hypothesis.mu.resize(stride_end, model.mu_b[0].clone());
            hypothesis.sigma.resize(stride_end, model.sigma_b[0].clone());
        }

        hypothesis.birth_location[global_idx] = *birth_loc;
        hypothesis.birth_time[global_idx] = t;
        hypothesis.r[global_idx] = model.r_b_lmbm[idx];
        hypothesis.mu[global_idx] = model.mu_b[idx].clone();
        hypothesis.sigma[global_idx] = model.sigma_b[idx].clone();
    }

    hypothesis
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_lmbm_prediction_birth() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let mut hypothesis = model.hypotheses.clone();
        let initial_count = hypothesis.r.len();

        hypothesis = lmbm_prediction_step(hypothesis, &model, 2);

        // Should have initial + birth objects
        assert_eq!(
            hypothesis.r.len(),
            initial_count + model.number_of_birth_locations
        );

        // Check birth times set correctly
        for i in initial_count..hypothesis.birth_time.len() {
            assert_eq!(hypothesis.birth_time[i], 2);
        }
    }

    #[test]
    fn test_lmbm_prediction_survival() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let mut hypothesis = Hypothesis {
            w: 1.0,
            birth_location: vec![1],
            birth_time: vec![1],
            r: vec![0.8],
            mu: vec![DVector::from_vec(vec![10.0, 1.0, 20.0, 2.0])],
            sigma: vec![DMatrix::identity(4, 4)],
        };

        let initial_r = hypothesis.r[0];
        let initial_mu = hypothesis.mu[0].clone();

        hypothesis = lmbm_prediction_step(hypothesis, &model, 2);

        // Check existence probability updated
        assert!((hypothesis.r[0] - model.survival_probability * initial_r).abs() < 1e-10);

        // Check mean predicted
        let expected_mu = &model.a * &initial_mu + &model.u;
        let diff = &hypothesis.mu[0] - &expected_mu;
        assert!(diff.norm() < 1e-10);
    }
}
