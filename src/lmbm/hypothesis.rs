//! LMBM hypothesis management
//!
//! Implements hypothesis parameter determination, normalization, gating, and state extraction for LMBM filter.
//! Matches MATLAB determinePosteriorHypothesisParameters.m, lmbmNormalisationAndGating.m, and lmbmStateExtraction.m exactly.

use crate::common::types::{Hypothesis, Model};
use crate::lmb::cardinality::lmb_map_cardinality_estimate;
use crate::lmbm::association::LmbmPosteriorParameters;
use nalgebra::DMatrix;

/// Determine parameters for a new set of posterior LMBM hypotheses
///
/// Creates posterior hypotheses from association events with unnormalized weights.
///
/// # Arguments
/// * `v` - Association events matrix (k x n), where each row is an association event
/// * `l` - Log likelihood matrix (n x (m+1)) from association matrices
/// * `posterior_parameters` - Posterior parameters from association matrices
/// * `prior_hypothesis` - Prior LMBM hypothesis
///
/// # Returns
/// Vector of posterior hypotheses with unnormalized weights
///
/// # Implementation Notes
/// Matches MATLAB determinePosteriorHypothesisParameters.m exactly:
/// 1. For each association event, create a new hypothesis
/// 2. Compute hypothesis weight from log likelihoods
/// 3. Set existence probabilities to 1 for detected objects
/// 4. Select appropriate means and covariances from posterior parameters
pub fn determine_posterior_hypothesis_parameters(
    v: &DMatrix<usize>,
    l: &DMatrix<f64>,
    posterior_parameters: &LmbmPosteriorParameters,
    prior_hypothesis: &Hypothesis,
) -> Vec<Hypothesis> {
    let number_of_objects = prior_hypothesis.r.len();
    let number_of_posterior_hypotheses = v.nrows();

    // Eta = 1:numberOfObjects (object indices, 0-indexed in Rust)
    let _eta: Vec<usize> = (0..number_of_objects).collect();

    // Create posterior hypotheses
    let mut posterior_hypotheses = Vec::with_capacity(number_of_posterior_hypotheses);

    for i in 0..number_of_posterior_hypotheses {
        // Association event
        let v_row: Vec<usize> = v.row(i).iter().copied().collect();

        // Linear indices: ell = numberOfObjects * v + eta
        // In MATLAB this is 1-indexed, but we're 0-indexed
        // For matrix L which is (n x (m+1)), we need L(obj_idx, meas_idx)
        let mut log_likelihood_sum = 0.0;
        for obj_idx in 0..number_of_objects {
            let meas_idx = v_row[obj_idx]; // 0 = miss, 1..m = measurements
            log_likelihood_sum += l[(obj_idx, meas_idx)];
        }

        // Hypothesis weight
        let w = prior_hypothesis.w.ln() + log_likelihood_sum;

        // Existence probabilities - set to 1 for detected objects
        let mut r = posterior_parameters.r.clone();
        for obj_idx in 0..number_of_objects {
            if v_row[obj_idx] > 0 {
                // Generated a measurement
                r[obj_idx] = 1.0;
            }
        }

        // Means - select based on association event
        let mut mu = Vec::with_capacity(number_of_objects);
        for obj_idx in 0..number_of_objects {
            let meas_idx = v_row[obj_idx];
            mu.push(posterior_parameters.mu[obj_idx][meas_idx].clone());
        }

        // Covariances - updated for detected objects
        let mut sigma = posterior_parameters.sigma.clone();
        for obj_idx in 0..number_of_objects {
            if v_row[obj_idx] > 0 {
                // Keep the updated covariance
            } else {
                // Use prior covariance for missed detections
                sigma[obj_idx] = prior_hypothesis.sigma[obj_idx].clone();
            }
        }

        posterior_hypotheses.push(Hypothesis {
            w,
            birth_location: prior_hypothesis.birth_location.clone(),
            birth_time: prior_hypothesis.birth_time.clone(),
            r,
            mu,
            sigma,
        });
    }

    posterior_hypotheses
}

/// Normalize and gate posterior LMBM hypotheses
///
/// Discards unlikely hypotheses and Bernoulli components with low existence probabilities.
///
/// # Arguments
/// * `posterior_hypotheses` - Posterior hypotheses with unnormalized weights
/// * `model` - Model parameters
///
/// # Returns
/// Tuple of (gated hypotheses with normalized weights, boolean mask of kept objects)
///
/// # Implementation Notes
/// Matches MATLAB lmbmNormalisationAndGating.m exactly:
/// 1. Normalize hypothesis weights using log-sum-exp
/// 2. Gate hypotheses by weight threshold
/// 3. Sort by descending weight
/// 4. Cap to maximum number of hypotheses
/// 5. Compute total existence probability per object
/// 6. Discard unlikely objects from all hypotheses
pub fn lmbm_normalisation_and_gating(
    posterior_hypotheses: Vec<Hypothesis>,
    model: &Model,
) -> (Vec<Hypothesis>, Vec<bool>) {
    if posterior_hypotheses.is_empty() {
        return (vec![model.hypotheses.clone()], vec![false; 0]);
    }

    let number_of_objects = posterior_hypotheses[0].r.len();

    // Normalize posterior hypothesis weights
    let log_w: Vec<f64> = posterior_hypotheses.iter().map(|h| h.w).collect();
    let max_w = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_w: Vec<f64> = log_w.iter().map(|&lw| (lw - max_w).exp()).collect();
    let sum_exp_w: f64 = exp_w.iter().sum();
    let mut w: Vec<f64> = exp_w.iter().map(|&ew| ew / sum_exp_w).collect();

    // Gate posterior hypotheses
    let likely_indices: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(i, &weight)| {
            if weight > model.posterior_hypothesis_weight_threshold {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if likely_indices.is_empty() {
        return (vec![model.hypotheses.clone()], vec![true; number_of_objects]);
    }

    let mut hypotheses: Vec<Hypothesis> = likely_indices
        .iter()
        .map(|&i| posterior_hypotheses[i].clone())
        .collect();

    w = likely_indices.iter().map(|&i| w[i]).collect();
    let sum_w: f64 = w.iter().sum();
    w = w.iter().map(|&weight| weight / sum_w).collect();

    // Sort hypotheses by descending weight
    let mut indices: Vec<usize> = (0..w.len()).collect();
    indices.sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap());

    let sorted_w: Vec<f64> = indices.iter().map(|&i| w[i]).collect();
    let sorted_hypotheses: Vec<Hypothesis> = indices.iter().map(|&i| hypotheses[i].clone()).collect();

    w = sorted_w;
    hypotheses = sorted_hypotheses;

    // Cap to maximum number of hypotheses
    let mut number_of_hypotheses = w.len();
    if number_of_hypotheses > model.maximum_number_of_posterior_hypotheses {
        w = w[..model.maximum_number_of_posterior_hypotheses].to_vec();
        let sum_w: f64 = w.iter().sum();
        w = w.iter().map(|&weight| weight / sum_w).collect();
        hypotheses = hypotheses[..model.maximum_number_of_posterior_hypotheses].to_vec();
        number_of_hypotheses = model.maximum_number_of_posterior_hypotheses;
    }

    // Determine total existence probability for each object
    let mut r_total = vec![0.0; number_of_objects];
    for (i, hyp) in hypotheses.iter().enumerate() {
        for obj_idx in 0..number_of_objects {
            r_total[obj_idx] += w[i] * hyp.r[obj_idx];
        }
    }

    let objects_likely_to_exist: Vec<bool> = r_total
        .iter()
        .map(|&r| r > model.existence_threshold)
        .collect();

    // Discard unlikely components from each hypothesis
    for i in 0..number_of_hypotheses {
        let mut new_birth_location = Vec::new();
        let mut new_birth_time = Vec::new();
        let mut new_r = Vec::new();
        let mut new_mu = Vec::new();
        let mut new_sigma = Vec::new();

        for (obj_idx, &keep) in objects_likely_to_exist.iter().enumerate() {
            if keep {
                new_birth_location.push(hypotheses[i].birth_location[obj_idx]);
                new_birth_time.push(hypotheses[i].birth_time[obj_idx]);
                new_r.push(hypotheses[i].r[obj_idx]);
                new_mu.push(hypotheses[i].mu[obj_idx].clone());
                new_sigma.push(hypotheses[i].sigma[obj_idx].clone());
            }
        }

        hypotheses[i].w = w[i];
        hypotheses[i].birth_location = new_birth_location;
        hypotheses[i].birth_time = new_birth_time;
        hypotheses[i].r = new_r;
        hypotheses[i].mu = new_mu;
        hypotheses[i].sigma = new_sigma;
    }

    // If no hypotheses remain, return initial hypotheses
    if hypotheses.is_empty() {
        (vec![model.hypotheses.clone()], vec![true; number_of_objects])
    } else {
        (hypotheses, objects_likely_to_exist)
    }
}

/// Extract state estimate from LMBM hypotheses
///
/// Heuristically determines the number of objects present and their indices.
///
/// # Arguments
/// * `hypotheses` - Posterior LMBM hypotheses with normalized weights
/// * `use_eap_on_lmbm` - If true, use EAP estimate; if false, use MAP estimate
///
/// # Returns
/// Tuple of (cardinality estimate, extraction indices)
///
/// # Implementation Notes
/// Matches MATLAB lmbmStateExtraction.m exactly:
/// - EAP: floor(sum(r_total)), then select top-k objects by existence probability from first hypothesis
/// - MAP: Use LMB MAP cardinality estimator on total existence probabilities
pub fn lmbm_state_extraction(
    hypotheses: &[Hypothesis],
    use_eap_on_lmbm: bool,
) -> (usize, Vec<usize>) {
    if hypotheses.is_empty() {
        return (0, vec![]);
    }

    let number_of_objects = hypotheses[0].r.len();

    // Compute total existence probability: r_total = sum(w .* r, 2)
    let mut r_total = vec![0.0; number_of_objects];
    for hyp in hypotheses {
        for obj_idx in 0..number_of_objects {
            r_total[obj_idx] += hyp.w * hyp.r[obj_idx];
        }
    }

    if use_eap_on_lmbm {
        // Heuristic EAP estimate
        let cardinality_estimate = r_total.iter().sum::<f64>().floor() as usize;

        // Select top-k objects by existence probability from first hypothesis
        let mut indexed_r: Vec<(usize, f64)> = hypotheses[0]
            .r
            .iter()
            .enumerate()
            .map(|(i, &r)| (i, r))
            .collect();

        // Sort by descending existence probability
        indexed_r.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let extraction_indices: Vec<usize> = indexed_r
            .iter()
            .take(cardinality_estimate)
            .map(|&(i, _)| i)
            .collect();

        (cardinality_estimate, extraction_indices)
    } else {
        // Very heuristic MAP estimate
        lmb_map_cardinality_estimate(&r_total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_determine_posterior_hypothesis_parameters() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let prior_hypothesis = model.hypotheses.clone();
        let n = prior_hypothesis.r.len();

        // Create dummy association events (2 events)
        let v = DMatrix::from_row_slice(2, n, &vec![0; 2 * n]);

        // Create dummy L matrix
        let l = DMatrix::from_element(n, 2, -1.0);

        // Create dummy posterior parameters
        let posterior_params = LmbmPosteriorParameters {
            r: vec![0.5; n],
            mu: vec![vec![DVector::zeros(4); 2]; n],
            sigma: vec![DMatrix::identity(4, 4); n],
        };

        let posterior = determine_posterior_hypothesis_parameters(&v, &l, &posterior_params, &prior_hypothesis);

        assert_eq!(posterior.len(), 2);
        assert_eq!(posterior[0].r.len(), n);
    }

    #[test]
    fn test_lmbm_normalisation_and_gating() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let mut hypotheses = vec![model.hypotheses.clone(), model.hypotheses.clone()];
        hypotheses[0].w = 0.6;
        hypotheses[1].w = 0.4;

        let (gated, mask) = lmbm_normalisation_and_gating(hypotheses, &model);

        assert!(!gated.is_empty());
        // Mask can be empty if no objects exist in the hypotheses
        if !gated[0].r.is_empty() {
            assert!(!mask.is_empty());
        }

        // Check weights sum to 1
        let sum_w: f64 = gated.iter().map(|h| h.w).sum();
        assert!((sum_w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lmbm_state_extraction_eap() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let mut hypothesis = model.hypotheses.clone();
        hypothesis.w = 1.0;
        hypothesis.r = vec![0.9, 0.8, 0.1, 0.05];

        let hypotheses = vec![hypothesis];

        let (n, indices) = lmbm_state_extraction(&hypotheses, true);

        // Expected cardinality: floor(0.9 + 0.8 + 0.1 + 0.05) = 1
        assert_eq!(n, 1);
        assert_eq!(indices.len(), 1);
        // Should select the object with highest r (index 0 with r=0.9)
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_lmbm_state_extraction_map() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let mut hypothesis = model.hypotheses.clone();
        hypothesis.w = 1.0;
        hypothesis.r = vec![0.9, 0.8, 0.1, 0.05];

        let hypotheses = vec![hypothesis];

        let (n, indices) = lmbm_state_extraction(&hypotheses, false);

        // Should use MAP estimate
        assert!(n <= hypotheses[0].r.len());
        assert_eq!(indices.len(), n);
    }
}
