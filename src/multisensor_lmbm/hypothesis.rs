//! Multi-sensor LMBM hypothesis management
//!
//! Implements posterior hypothesis parameter determination for multi-sensor LMBM.
//! Matches MATLAB determineMultisensorPosteriorHypothesisParameters.m exactly.

use super::determine_linear_index;
use crate::common::types::{Hypothesis, Model};
use crate::multisensor_lmbm::association::{
    compute_posterior_params_for_indices, MultisensorLmbmPosteriorParameters,
};
use nalgebra::{DMatrix, DVector};
use std::collections::HashSet;

/// Determine parameters for a new set of posterior LMBM hypotheses
///
/// Generates posterior hypotheses with unnormalized hypothesis weights
/// from association events sampled via Gibbs.
///
/// # Arguments
/// * `a` - Association events matrix (rows = events, cols = flattened n*S associations)
/// * `l` - Flattened log likelihood matrix
/// * `dimensions` - Dimensions [m1+1, m2+1, ..., ms+1, n]
/// * `posterior_parameters` - Posterior parameters from association step
/// * `prior_hypothesis` - Prior LMBM hypothesis
///
/// # Returns
/// Vector of posterior hypotheses with unnormalized weights
///
/// # Implementation Notes
/// Matches MATLAB determineMultisensorPosteriorHypothesisParameters.m exactly:
/// 1. For each association event in A:
///    - Convert to Cartesian coordinates (U matrix)
///    - Determine linear indices for all objects
///    - Compute hypothesis weight: log(prior.w) + sum(L(ell))
///    - Extract posterior parameters using linear indices
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn determine_multisensor_posterior_hypothesis_parameters(
    a: &DMatrix<usize>,
    l: &[f64],
    dimensions: &[usize],
    posterior_parameters: &MultisensorLmbmPosteriorParameters,
    prior_hypothesis: &Hypothesis,
) -> Vec<Hypothesis> {
    let number_of_sensors = dimensions.len() - 1;
    let number_of_objects = dimensions[number_of_sensors];
    let number_of_posterior_hypotheses = a.nrows();

    // Initialize output
    let mut posterior_hypotheses = Vec::with_capacity(number_of_posterior_hypotheses);

    // For each association event
    for i in 0..number_of_posterior_hypotheses {
        // Build association matrix U (n x (S+1))
        // Columns: [sensor1_assoc, sensor2_assoc, ..., sensorS_assoc, object_index]
        let mut u = DMatrix::zeros(number_of_objects, number_of_sensors + 1);

        // Fill sensor associations (convert from flattened row)
        for obj_idx in 0..number_of_objects {
            for s in 0..number_of_sensors {
                // A is stored column-major: [s0_obj0, s0_obj1, ..., s0_objN, s1_obj0, ...]
                // This matches MATLAB's reshape(V, 1, n * S) which is column-major
                let col = s * number_of_objects + obj_idx;
                // Add 1 to convert from 0-indexed to MATLAB 1-indexed
                u[(obj_idx, s)] = a[(i, col)] + 1;
            }
        }

        // Fill object indices (1-indexed for MATLAB compatibility)
        for obj_idx in 0..number_of_objects {
            u[(obj_idx, number_of_sensors)] = obj_idx + 1;
        }

        // Determine linear indices for all objects
        let mut ell_indices = Vec::with_capacity(number_of_objects);
        for obj_idx in 0..number_of_objects {
            let u_row: Vec<usize> = u.row(obj_idx).iter().copied().collect();
            let ell = determine_linear_index(&u_row, dimensions);
            ell_indices.push(ell);
        }

        // Compute hypothesis weight: log(prior.w) + sum(L(ell))
        let log_weight_sum: f64 = ell_indices.iter().map(|&idx| l[idx]).sum();
        let hypothesis_weight = prior_hypothesis.w.ln() + log_weight_sum;

        // Extract posterior parameters using linear indices
        let r: Vec<f64> = ell_indices.iter().map(|&idx| posterior_parameters.r[idx]).collect();
        let mu: Vec<DVector<f64>> = ell_indices
            .iter()
            .map(|&idx| posterior_parameters.mu[idx].clone())
            .collect();
        let sigma: Vec<DMatrix<f64>> = ell_indices
            .iter()
            .map(|&idx| posterior_parameters.sigma[idx].clone())
            .collect();

        // Create posterior hypothesis
        posterior_hypotheses.push(Hypothesis {
            birth_location: prior_hypothesis.birth_location.clone(),
            birth_time: prior_hypothesis.birth_time.clone(),
            w: hypothesis_weight, // Store in log space (will be normalized later)
            r,
            mu,
            sigma,
        });
    }

    posterior_hypotheses
}

/// Determine posterior hypothesis parameters with DEFERRED param computation
///
/// Phase C optimization: Instead of receiving precomputed posterior_parameters for all
/// 10.7M entries, this function:
/// 1. Collects unique linear indices from the association matrix `a`
/// 2. Computes posterior params (r, mu, sigma) ONLY for those ~1000 indices
/// 3. Builds hypotheses using the sparse params
///
/// This provides ~10000x reduction in posterior param computation.
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn determine_multisensor_posterior_hypothesis_parameters_deferred(
    a: &DMatrix<usize>,
    l: &[f64],
    dimensions: &[usize],
    prior_hypothesis: &Hypothesis,
    measurements: &[&[DVector<f64>]],
    model: &Model,
    number_of_sensors: usize,
) -> Vec<Hypothesis> {
    let number_of_objects = dimensions[number_of_sensors];
    let number_of_posterior_hypotheses = a.nrows();

    // Step 1: Collect ALL unique linear indices from association matrix
    let mut all_ell_indices = Vec::with_capacity(number_of_posterior_hypotheses * number_of_objects);

    for i in 0..number_of_posterior_hypotheses {
        // Build association matrix U for this event
        let mut u = DMatrix::zeros(number_of_objects, number_of_sensors + 1);

        for obj_idx in 0..number_of_objects {
            for s in 0..number_of_sensors {
                let col = s * number_of_objects + obj_idx;
                u[(obj_idx, s)] = a[(i, col)] + 1;
            }
            u[(obj_idx, number_of_sensors)] = obj_idx + 1;
        }

        // Compute linear indices for all objects
        for obj_idx in 0..number_of_objects {
            let u_row: Vec<usize> = u.row(obj_idx).iter().copied().collect();
            let ell = determine_linear_index(&u_row, dimensions);
            all_ell_indices.push(ell);
        }
    }

    // Step 2: Get unique indices
    let unique_indices: HashSet<usize> = all_ell_indices.iter().copied().collect();
    let unique_indices_vec: Vec<usize> = unique_indices.into_iter().collect();

    // Step 3: Compute posterior params ONLY for unique indices (~1000 vs 10.7M)
    let sparse_params = compute_posterior_params_for_indices(
        &unique_indices_vec,
        prior_hypothesis,
        measurements,
        model,
        number_of_sensors,
        dimensions,
    );

    // Step 4: Build hypotheses using sparse params
    let mut posterior_hypotheses = Vec::with_capacity(number_of_posterior_hypotheses);

    let mut idx = 0;
    for _i in 0..number_of_posterior_hypotheses {
        let ell_start = idx;
        let ell_end = idx + number_of_objects;
        let ell_indices = &all_ell_indices[ell_start..ell_end];
        idx = ell_end;

        // Compute hypothesis weight: log(prior.w) + sum(L(ell))
        let log_weight_sum: f64 = ell_indices.iter().map(|&ell| l[ell]).sum();
        let hypothesis_weight = prior_hypothesis.w.ln() + log_weight_sum;

        // Extract posterior parameters using sparse params
        let r: Vec<f64> = ell_indices
            .iter()
            .map(|&ell| sparse_params.get(&ell).map(|(r, _, _)| *r).unwrap_or(0.0))
            .collect();
        let mu: Vec<DVector<f64>> = ell_indices
            .iter()
            .map(|&ell| {
                sparse_params
                    .get(&ell)
                    .map(|(_, mu, _)| mu.clone())
                    .unwrap_or_else(|| prior_hypothesis.mu[0].clone())
            })
            .collect();
        let sigma: Vec<DMatrix<f64>> = ell_indices
            .iter()
            .map(|&ell| {
                sparse_params
                    .get(&ell)
                    .map(|(_, _, sigma)| sigma.clone())
                    .unwrap_or_else(|| prior_hypothesis.sigma[0].clone())
            })
            .collect();

        posterior_hypotheses.push(Hypothesis {
            birth_location: prior_hypothesis.birth_location.clone(),
            birth_time: prior_hypothesis.birth_time.clone(),
            w: hypothesis_weight,
            r,
            mu,
            sigma,
        });
    }

    posterior_hypotheses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::Hypothesis;

    #[test]
    fn test_determine_multisensor_posterior_hypothesis_parameters() {
        // Simple test with 2 sensors, 2 objects
        // Dimensions: [2+1, 2+1, 2] = [3, 3, 2]
        let dimensions = vec![3, 3, 2];
        let total_size = 3 * 3 * 2;

        // Create association matrix: 1 event, 2 objects * 2 sensors = 4 entries
        // Event: all misses (0-indexed)
        let mut a = DMatrix::zeros(1, 4);

        // Create simple likelihood matrix
        let l = vec![0.0; total_size];

        // Create simple posterior parameters
        let posterior_parameters = MultisensorLmbmPosteriorParameters {
            r: vec![0.5; total_size],
            mu: vec![DVector::from_vec(vec![0.0, 0.0]); total_size],
            sigma: vec![DMatrix::identity(2, 2); total_size],
        };

        // Create prior hypothesis
        let prior_hypothesis = Hypothesis {
            birth_location: vec![0, 1],
            birth_time: vec![1, 1],
            w: 1.0,
            r: vec![0.5, 0.5],
            mu: vec![
                DVector::from_vec(vec![0.0, 0.0]),
                DVector::from_vec(vec![1.0, 1.0]),
            ],
            sigma: vec![DMatrix::identity(2, 2), DMatrix::identity(2, 2)],
        };

        let posterior = determine_multisensor_posterior_hypothesis_parameters(
            &a,
            &l,
            &dimensions,
            &posterior_parameters,
            &prior_hypothesis,
        );

        assert_eq!(posterior.len(), 1);
        assert_eq!(posterior[0].r.len(), 2);
    }
}
