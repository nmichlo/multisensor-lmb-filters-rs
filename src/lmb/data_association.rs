//! Data association wrappers for LMB filter
//!
//! Implements wrappers for different data association algorithms.
//! Matches MATLAB lmbMurtysAlgorithm.m, and uses loopyBeliefPropagation and lmbGibbsSampling.

use crate::common::association::gibbs::lmb_gibbs_sampling;
use crate::common::association::lbp::{fixed_loopy_belief_propagation, loopy_belief_propagation};
use crate::common::association::murtys::murtys_algorithm_wrapper;
use crate::lmb::association::LmbAssociationResult;
use nalgebra::{DMatrix, DVector};

/// Compute marginals using Loopy Belief Propagation
///
/// # Arguments
/// * `association_result` - Association matrices and posterior parameters
/// * `epsilon` - Convergence tolerance
/// * `max_iterations` - Maximum LBP iterations
///
/// # Returns
/// Tuple of (r, W) where:
/// - r: Posterior existence probabilities (n x 1)
/// - W: Marginal association probabilities (n x (m+1))
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn lmb_lbp(
    association_result: &LmbAssociationResult,
    epsilon: f64,
    max_iterations: usize,
) -> (DVector<f64>, DMatrix<f64>) {
    let result = loopy_belief_propagation(&association_result.lbp, epsilon, max_iterations);
    (result.r, result.w)
}

/// Compute marginals using fixed-iteration Loopy Belief Propagation
///
/// # Arguments
/// * `association_result` - Association matrices and posterior parameters
/// * `max_iterations` - Number of LBP iterations
///
/// # Returns
/// Tuple of (r, W)
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn lmb_lbp_fixed(
    association_result: &LmbAssociationResult,
    max_iterations: usize,
) -> (DVector<f64>, DMatrix<f64>) {
    let result = fixed_loopy_belief_propagation(&association_result.lbp, max_iterations);
    (result.r, result.w)
}

/// Compute marginals using Gibbs sampling
///
/// # Arguments
/// * `rng` - Random number generator
/// * `association_result` - Association matrices and posterior parameters
/// * `num_samples` - Number of Gibbs samples
///
/// # Returns
/// Tuple of (r, W)
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn lmb_gibbs(
    rng: &mut impl crate::common::rng::Rng,
    association_result: &LmbAssociationResult,
    num_samples: usize,
) -> (DVector<f64>, DMatrix<f64>) {
    let result = lmb_gibbs_sampling(rng, &association_result.gibbs, num_samples);
    (result.r, result.w)
}

/// Compute marginals using Murty's algorithm
///
/// Determines posterior existence probabilities and marginal association
/// probabilities using Murty's algorithm.
///
/// # Arguments
/// * `association_result` - Association matrices and posterior parameters
/// * `num_assignments` - Number of best assignments to find
///
/// # Returns
/// Tuple of (r, W, V) where:
/// - r: Posterior existence probabilities (n x 1)
/// - W: Marginal association probabilities (n x (m+1))
/// - V: Association events (num_assignments x n), 1-indexed
///
/// # Implementation Notes
/// Matches MATLAB lmbMurtysAlgorithm.m lines 26-40:
/// 1. Run Murty's algorithm to get K-best assignments
/// 2. Compute marginal distributions from assignment events
/// 3. Normalize and compute existence probabilities
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn lmb_murtys(
    association_result: &LmbAssociationResult,
    num_assignments: usize,
) -> (DVector<f64>, DMatrix<f64>, DMatrix<usize>) {
    let n = association_result.cost.nrows();
    let m = association_result.cost.ncols();

    // Run Murty's algorithm
    let murtys_result = murtys_algorithm_wrapper(&association_result.cost, num_assignments);
    let v = murtys_result.assignments; // (k x n)

    // Determine marginal distributions
    // W = repmat(V, 1, 1, m+1) == reshape(0:m, 1, 1, m+1)
    // This creates indicator matrices for each measurement (including 0=miss)
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
    // This extracts likelihoods for the assigned measurements
    let mut j_matrix = DMatrix::zeros(k, n);
    for i in 0..k {
        for obj_idx in 0..n {
            let meas_idx = v[(i, obj_idx)]; // 0-indexed in V
            j_matrix[(i, obj_idx)] = association_result.gibbs.l[(obj_idx, meas_idx)];
        }
    }

    // L = permute(sum(prod(J, 2) .* W, 1), [2 1 3])
    // This computes weighted marginals
    let mut l_marg = Vec::with_capacity(m + 1);
    for meas_idx in 0..=m {
        let mut l_col = DVector::zeros(n);
        for obj_idx in 0..n {
            let mut sum = 0.0;
            for event_idx in 0..k {
                // prod(J, 2): product across objects for this event
                let mut prod = 1.0;
                for j in 0..n {
                    prod *= j_matrix[(event_idx, j)];
                }
                // .* W: multiply by indicator
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

    // Tau = (Sigma .* associationMatrices.R) ./ sum(Sigma, 2)
    let mut tau = DMatrix::zeros(n, m + 1);
    for obj_idx in 0..n {
        let row_sum: f64 = sigma.row(obj_idx).sum();
        if row_sum > 1e-15 {
            for meas_idx in 0..=m {
                tau[(obj_idx, meas_idx)] =
                    (sigma[(obj_idx, meas_idx)] * association_result.gibbs.r[(obj_idx, meas_idx)])
                        / row_sum;
            }
        }
    }

    // r = sum(Tau, 2)
    let mut r = DVector::zeros(n);
    for obj_idx in 0..n {
        r[obj_idx] = tau.row(obj_idx).sum();
    }

    // W = Tau ./ r
    let mut w_result = DMatrix::zeros(n, m + 1);
    for obj_idx in 0..n {
        if r[obj_idx] > 1e-15 {
            for meas_idx in 0..=m {
                w_result[(obj_idx, meas_idx)] = tau[(obj_idx, meas_idx)] / r[obj_idx];
            }
        }
    }

    (r, w_result, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};
    use crate::lmb::association::generate_lmb_association_matrices;

    #[test]
    fn test_lmb_lbp() {
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

        let (r, w) = lmb_lbp(&association_result, 1e-6, 1000);

        // Check dimensions
        assert_eq!(r.len(), objects.len());
        assert_eq!(w.nrows(), objects.len());
        assert_eq!(w.ncols(), 2); // miss + 1 measurement

        // Check normalization
        for i in 0..objects.len() {
            let row_sum: f64 = w.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lmb_gibbs() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let objects = model.birth_parameters.clone();
        let measurements = vec![DVector::from_vec(vec![0.0, 0.0])];

        let association_result =
            generate_lmb_association_matrices(&objects, &measurements, &model);

        let mut rng2 = crate::common::rng::SimpleRng::new(42);
        let (r, w) = lmb_gibbs(&mut rng2, &association_result, 100);

        // Check dimensions
        assert_eq!(r.len(), objects.len());
        assert_eq!(w.nrows(), objects.len());
        assert_eq!(w.ncols(), 2);

        // Check normalization
        for i in 0..objects.len() {
            let row_sum: f64 = w.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lmb_murtys() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Murty,
            ScenarioType::Fixed,
            None,
        );

        let objects = model.birth_parameters.clone();
        let measurements = vec![DVector::from_vec(vec![0.0, 0.0])];

        let association_result =
            generate_lmb_association_matrices(&objects, &measurements, &model);

        let (r, w, v) = lmb_murtys(&association_result, 10);

        // Check dimensions
        assert_eq!(r.len(), objects.len());
        assert_eq!(w.nrows(), objects.len());
        assert_eq!(w.ncols(), 2);
        assert!(v.nrows() <= 10); // May be less if fewer valid assignments
        assert_eq!(v.ncols(), objects.len());

        // Check normalization
        for i in 0..objects.len() {
            let row_sum: f64 = w.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }
}
