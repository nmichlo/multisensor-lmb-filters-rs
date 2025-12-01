/*!
Multi-sensor LMBM filter implementation.

Exact solution for multi-sensor tracking, memory-intensive but accurate.
*/

pub mod association;
pub mod filter;
pub mod gibbs;
pub mod hypothesis;
pub mod lazy;
pub mod workspace;

// Re-export the main public API
pub use filter::{run_multisensor_lmbm_filter, MultisensorLmbmStateEstimates};
pub use gibbs::multisensor_lmbm_gibbs_sampling;
#[cfg(feature = "gibbs-trace")]
pub use gibbs::{reset_access_trace, get_access_stats, print_access_report};
pub use hypothesis::determine_multisensor_posterior_hypothesis_parameters;
pub use lazy::LazyLikelihood;
pub use workspace::LmbmLikelihoodWorkspace;

// Note: generate_multisensor_lmbm_association_matrices is kept in association.rs
// for internal testing/equivalence checks but not re-exported as public API.
// Use LazyLikelihood instead for on-demand computation.

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
pub fn determine_linear_index(u: &[usize], dimensions: &[usize]) -> usize {
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

    #[test]
    fn test_determine_linear_index() {
        // Test with dimensions [3, 4, 5] (2+1, 3+1, 4+1)
        // Total size: 3 * 4 * 5 = 60
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
}
