//! Multi-sensor LMBM hypothesis management
//!
//! Implements posterior hypothesis parameter determination for multi-sensor LMBM.
//! Matches MATLAB determineMultisensorPosteriorHypothesisParameters.m exactly.

use crate::common::types::Hypothesis;
use crate::multisensor_lmbm::association::MultisensorLmbmPosteriorParameters;
use nalgebra::{DMatrix, DVector};

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
                // A is stored as [obj0_s0, obj0_s1, ..., obj1_s0, obj1_s1, ...]
                let col = obj_idx * number_of_sensors + s;
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
            w: hypothesis_weight.exp(), // Store as linear weight (will be normalized later)
            r,
            mu,
            sigma,
        });
    }

    posterior_hypotheses
}

/// Determine linear index from Cartesian coordinates
///
/// Converts multi-dimensional indices to linear index for the flattened
/// likelihood matrix.
///
/// # Arguments
/// * `u` - Cartesian coordinates [u1, u2, ..., us, i] (1-indexed, MATLAB style)
/// * `dimensions` - Dimensions [m1+1, m2+1, ..., ms+1, n]
///
/// # Returns
/// Linear index into flattened array (0-indexed)
///
/// # Implementation Notes
/// Matches MATLAB determineLinearIndex exactly:
/// ell = u(1) + d(1)*(u(2)-1) + d(1)*d(2)*(u(3)-1) + ...
fn determine_linear_index(u: &[usize], dimensions: &[usize]) -> usize {
    let mut ell = u[0];
    let mut pi = 1;

    for i in 1..u.len() {
        pi *= dimensions[i - 1];
        ell += pi * (u[i] - 1);
    }

    // Convert from 1-indexed to 0-indexed
    ell - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::Hypothesis;

    #[test]
    fn test_determine_linear_index() {
        // Test with dimensions [3, 4, 5] (2+1, 3+1, 4+1)
        let dimensions = vec![3, 4, 5];

        // First element: u = [1, 1, 1]
        let idx = determine_linear_index(&[1, 1, 1], &dimensions);
        assert_eq!(idx, 0);

        // u = [2, 1, 1] should be idx 1
        let idx = determine_linear_index(&[2, 1, 1], &dimensions);
        assert_eq!(idx, 1);

        // u = [1, 2, 1] should be idx 3
        let idx = determine_linear_index(&[1, 2, 1], &dimensions);
        assert_eq!(idx, 3);
    }

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
