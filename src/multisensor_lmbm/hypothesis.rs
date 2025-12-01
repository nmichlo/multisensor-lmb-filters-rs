//! Multi-sensor LMBM hypothesis management
//!
//! Implements posterior hypothesis parameter determination for multi-sensor LMBM.
//! Matches MATLAB determineMultisensorPosteriorHypothesisParameters.m exactly.

use super::determine_linear_index;
use super::lazy::LazyLikelihood;
use crate::common::types::Hypothesis;
use nalgebra::{DMatrix, DVector};

/// Determine parameters for a new set of posterior LMBM hypotheses
///
/// Generates posterior hypotheses with unnormalized hypothesis weights
/// from association events sampled via Gibbs.
///
/// # Arguments
/// * `a` - Association events matrix (rows = events, cols = flattened n*S associations)
/// * `lazy` - Lazy likelihood computer (computes values on-demand)
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
    lazy: &LazyLikelihood,
    prior_hypothesis: &Hypothesis,
) -> Vec<Hypothesis> {
    let dimensions = lazy.dimensions();
    let number_of_sensors = lazy.number_of_sensors();
    let number_of_objects = lazy.number_of_objects();
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
        // Use lazy likelihood - only computes values that are actually accessed
        let log_weight_sum: f64 = ell_indices.iter().map(|&idx| lazy.get_l(idx)).sum();
        let hypothesis_weight = prior_hypothesis.w.ln() + log_weight_sum;

        // Extract posterior parameters using linear indices
        // These are also computed lazily and cached
        let r: Vec<f64> = ell_indices.iter().map(|&idx| lazy.get_r(idx)).collect();
        let mu: Vec<DVector<f64>> = ell_indices
            .iter()
            .map(|&idx| lazy.get_mu(idx))
            .collect();
        let sigma: Vec<DMatrix<f64>> = ell_indices
            .iter()
            .map(|&idx| lazy.get_sigma(idx))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, Hypothesis, ScenarioType};

    #[test]
    fn test_determine_multisensor_posterior_hypothesis_parameters() {
        use crate::common::rng::SimpleRng;

        let mut rng = SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        // Create a hypothesis with 2 objects
        let hypothesis = Hypothesis {
            birth_location: vec![0, 1],
            birth_time: vec![1, 1],
            w: 1.0,
            r: vec![0.5, 0.5],
            mu: vec![
                DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
                DVector::from_vec(vec![1.0, 1.0, 0.0, 0.0]),
            ],
            sigma: vec![
                DMatrix::identity(4, 4) * 10.0,
                DMatrix::identity(4, 4) * 10.0,
            ],
        };

        // 2 sensors, 2 measurements each
        let measurements = vec![
            vec![
                DVector::from_vec(vec![0.0, 0.0]),
                DVector::from_vec(vec![1.0, 1.0]),
            ],
            vec![
                DVector::from_vec(vec![2.0, 2.0]),
                DVector::from_vec(vec![3.0, 3.0]),
            ],
        ];

        // Create lazy likelihood
        let lazy = LazyLikelihood::new(&hypothesis, &measurements, &model, 2);

        // Create association matrix: 1 event, 2 objects * 2 sensors = 4 entries
        // Event: all misses (0-indexed)
        let a = DMatrix::zeros(1, 4);

        let posterior = determine_multisensor_posterior_hypothesis_parameters(
            &a,
            &lazy,
            &hypothesis,
        );

        assert_eq!(posterior.len(), 1);
        assert_eq!(posterior[0].r.len(), 2);
    }
}
