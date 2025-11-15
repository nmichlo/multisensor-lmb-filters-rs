//! Marginal evaluation tests - LBP vs Murty's validation
//!
//! Ports MATLAB evaluateSmallExamples.m to validate that LBP approximation
//! produces marginals close to the exact Murty's algorithm.
//!
//! This test compares:
//! - Existence probability marginals (r)
//! - Association probability marginals (W)
//!
//! Using both KL divergence and Hellinger distance metrics.

mod test_utils;

use test_utils::{
    average_hellinger_distance, average_kullback_leibler_divergence,
    calculate_number_of_association_events, generate_association_matrices,
    generate_simplified_model,
};

use nalgebra::DMatrix;
use prak::common::association::lbp::{loopy_belief_propagation, AssociationMatrices};
use prak::common::association::murtys::murtys_algorithm_wrapper;
use prak::common::rng::SimpleRng;

/// Convert test association matrices to LBP association matrices format
fn convert_to_lbp_matrices(
    test_matrices: &test_utils::TestAssociationMatrices,
) -> AssociationMatrices {
    AssociationMatrices {
        eta: test_matrices.eta.clone(),
        phi: test_matrices.phi.clone(),
        psi: test_matrices.psi.clone(),
    }
}

/// Compute exact marginals using Murty's algorithm
///
/// Implements MATLAB evaluateSmallExamples.m lines 39-54
///
/// # Arguments
/// * `test_matrices` - Test association matrices
/// * `n` - Number of objects
/// * `num_events` - Total number of association events to generate
///
/// # Returns
/// (r_murty, w_murty) - Existence probabilities and association marginals
fn compute_exact_marginals_murty(
    test_matrices: &test_utils::TestAssociationMatrices,
    n: usize,
    num_events: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    // Run Murty's algorithm exhaustively
    let assignments = murtys_algorithm_wrapper(&test_matrices.c, num_events);

    // Extract assignment matrices V (n × num_events)
    let mut v = vec![vec![0usize; n]; num_events];
    for event_idx in 0..assignments.assignments.nrows() {
        for i in 0..n {
            v[event_idx][i] = assignments.assignments[(event_idx, i)];
        }
    }

    // Compute linear indices: ell = n * V + (1:n)
    // MATLAB is 1-indexed, Rust is 0-indexed
    let m = test_matrices.l.ncols() - 1; // Number of measurements (exclude eta column)
    let mut ell_indices = vec![vec![0usize; n]; num_events];
    for event_idx in 0..num_events {
        for i in 0..n {
            // Linear index: ell = n * v[event_idx][i] + i
            ell_indices[event_idx][i] = n * v[event_idx][i] + i;
        }
    }

    // Determine marginals: J = reshape(associationMatrices.L(ell), numberOfEvents, n)
    let mut j = DMatrix::zeros(num_events, n);
    for event_idx in 0..num_events {
        for i in 0..n {
            let row = i;
            let col = v[event_idx][i]; // Column in L matrix (includes eta at index 0)
            j[(event_idx, i)] = test_matrices.l[(row, col)];
        }
    }

    // W = repmat(V, 1, 1, n+1) == reshape(0:n, 1, 1, n+1)
    // This creates a 3D array where W(:,:,k) indicates which assignments equal k-1
    // We'll compute this for each measurement index
    let mut w_3d = vec![DMatrix::zeros(num_events, n); m + 1];
    for k in 0..=m {
        for event_idx in 0..num_events {
            for i in 0..n {
                w_3d[k][(event_idx, i)] = if v[event_idx][i] == k { 1.0 } else { 0.0 };
            }
        }
    }

    // L = permute(sum(prod(J, 2) .* W, 1), [2 1 3])
    // First: prod(J, 2) - product across columns for each event
    let mut prod_j = vec![0.0; num_events];
    for event_idx in 0..num_events {
        let mut prod = 1.0;
        for i in 0..n {
            prod *= j[(event_idx, i)];
        }
        prod_j[event_idx] = prod;
    }

    // For each measurement k: sum(prod_j .* W[:,:,k], dim=0)
    let mut l_result = DMatrix::zeros(n, m + 1);
    for k in 0..=m {
        for i in 0..n {
            let mut sum = 0.0;
            for event_idx in 0..num_events {
                sum += prod_j[event_idx] * w_3d[k][(event_idx, i)];
            }
            l_result[(i, k)] = sum;
        }
    }

    // Sigma = reshape(L, n, n+1)
    let sigma = l_result;

    // Normalize: Tau = (Sigma .* R) ./ sum(Sigma, 2)
    let mut tau = DMatrix::zeros(n, m + 1);
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..=m {
            row_sum += sigma[(i, j)];
        }
        if row_sum > 1e-15 {
            for j in 0..=m {
                tau[(i, j)] = sigma[(i, j)] * test_matrices.r_mat[(i, j)] / row_sum;
            }
        }
    }

    // Existence probabilities: rMurty = sum(Tau, 2)
    let mut r_murty = DMatrix::zeros(n, 2);
    for i in 0..n {
        let r_exist: f64 = tau.row(i).iter().sum();
        r_murty[(i, 0)] = 1.0 - r_exist;
        r_murty[(i, 1)] = r_exist;
    }

    // Marginal association probabilities: WMurty = Tau ./ rMurty
    let mut w_murty = DMatrix::zeros(n, m + 1);
    for i in 0..n {
        let r_exist = r_murty[(i, 1)];
        if r_exist > 1e-15 {
            for j in 0..=m {
                w_murty[(i, j)] = tau[(i, j)] / r_exist;
            }
        } else {
            // If existence probability is 0, use uniform distribution
            for j in 0..=m {
                w_murty[(i, j)] = 1.0 / (m + 1) as f64;
            }
        }
    }

    (r_murty, w_murty)
}

/// Compute approximate marginals using LBP
///
/// # Arguments
/// * `lbp_matrices` - LBP association matrices
/// * `lbp_tolerance` - Convergence tolerance
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
/// (r_lbp, w_lbp) - Existence probabilities and association marginals
fn compute_approximate_marginals_lbp(
    lbp_matrices: &AssociationMatrices,
    lbp_tolerance: f64,
    max_iterations: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let result = loopy_belief_propagation(lbp_matrices, lbp_tolerance, max_iterations);

    let n = result.r.len();

    // Convert r to matrix format [1-r, r]
    let mut r_lbp = DMatrix::zeros(n, 2);
    for i in 0..n {
        r_lbp[(i, 0)] = 1.0 - result.r[i];
        r_lbp[(i, 1)] = result.r[i];
    }

    (r_lbp, result.w)
}

/// Run a single marginal evaluation trial
///
/// # Arguments
/// * `rng` - Random number generator
/// * `n` - Number of objects
///
/// # Returns
/// (r_kl, w_kl, r_h, w_h) - KL divergence and Hellinger distance errors
fn run_marginal_evaluation_trial(
    rng: &mut SimpleRng,
    n: usize,
) -> (f64, f64, f64, f64) {
    // Generate simplified model and association matrices
    let model = generate_simplified_model(rng, n, 0.95, 0.0);
    let test_matrices = generate_association_matrices(rng, &model);

    // Run LBP
    let lbp_matrices = convert_to_lbp_matrices(&test_matrices);
    let (r_lbp, w_lbp) = compute_approximate_marginals_lbp(
        &lbp_matrices,
        model.lbp_tolerance,
        model.max_lbp_iterations,
    );

    // Run Murty's algorithm (exhaustively)
    let m = test_matrices.psi.ncols();
    let num_events = calculate_number_of_association_events(n, m);

    // For large problems, limit the number of events
    let num_events = num_events.min(100000);

    let (r_murty, w_murty) = compute_exact_marginals_murty(&test_matrices, n, num_events);

    // Compute errors
    let r_kl = average_kullback_leibler_divergence(&r_murty, &r_lbp);
    let w_kl = average_kullback_leibler_divergence(&w_murty, &w_lbp);
    let r_h = average_hellinger_distance(&r_murty, &r_lbp);
    let w_h = average_hellinger_distance(&w_murty, &w_lbp);

    (r_kl, w_kl, r_h, w_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbp_vs_murtys_small_n1() {
        let mut rng = SimpleRng::new(42);
        let num_trials = 100;

        let mut r_kl_sum = 0.0;
        let mut w_kl_sum = 0.0;
        let mut r_h_sum = 0.0;
        let mut w_h_sum = 0.0;

        for _ in 0..num_trials {
            let (r_kl, w_kl, r_h, w_h) = run_marginal_evaluation_trial(&mut rng, 1);
            r_kl_sum += r_kl;
            w_kl_sum += w_kl;
            r_h_sum += r_h;
            w_h_sum += w_h;
        }

        let r_kl_avg = r_kl_sum / num_trials as f64;
        let w_kl_avg = w_kl_sum / num_trials as f64;
        let r_h_avg = r_h_sum / num_trials as f64;
        let w_h_avg = w_h_sum / num_trials as f64;

        println!("n=1: r_KL={:.6}, w_KL={:.6}, r_H={:.6}, w_H={:.6}",
                 r_kl_avg, w_kl_avg, r_h_avg, w_h_avg);

        // For n=1, LBP should be very accurate
        assert!(r_kl_avg < 0.1, "r_KL too large: {}", r_kl_avg);
        assert!(w_kl_avg < 0.1, "w_KL too large: {}", w_kl_avg);
        assert!(r_h_avg < 0.1, "r_H too large: {}", r_h_avg);
        assert!(w_h_avg < 0.1, "w_H too large: {}", w_h_avg);
    }

    #[test]
    fn test_lbp_vs_murtys_small_n2() {
        let mut rng = SimpleRng::new(42);
        let num_trials = 50;

        let mut r_kl_sum = 0.0;
        let mut w_kl_sum = 0.0;

        for _ in 0..num_trials {
            let (r_kl, w_kl, _, _) = run_marginal_evaluation_trial(&mut rng, 2);
            r_kl_sum += r_kl;
            w_kl_sum += w_kl;
        }

        let r_kl_avg = r_kl_sum / num_trials as f64;
        let w_kl_avg = w_kl_sum / num_trials as f64;

        println!("n=2: r_KL={:.6}, w_KL={:.6}", r_kl_avg, w_kl_avg);

        // For n=2, LBP should still be reasonably accurate
        assert!(r_kl_avg < 0.2, "r_KL too large: {}", r_kl_avg);
        assert!(w_kl_avg < 0.2, "w_KL too large: {}", w_kl_avg);
    }

    #[test]
    #[ignore] // This test is computationally expensive
    fn test_lbp_vs_murtys_small_n3() {
        let mut rng = SimpleRng::new(42);
        let num_trials = 20;

        let mut r_kl_sum = 0.0;
        let mut w_kl_sum = 0.0;

        for i in 0..num_trials {
            println!("Trial {}/{}", i + 1, num_trials);
            let (r_kl, w_kl, _, _) = run_marginal_evaluation_trial(&mut rng, 3);
            r_kl_sum += r_kl;
            w_kl_sum += w_kl;
        }

        let r_kl_avg = r_kl_sum / num_trials as f64;
        let w_kl_avg = w_kl_sum / num_trials as f64;

        println!("n=3: r_KL={:.6}, w_KL={:.6}", r_kl_avg, w_kl_avg);

        // For n=3, LBP approximation error increases but should still be reasonable
        assert!(r_kl_avg < 0.5, "r_KL too large: {}", r_kl_avg);
        assert!(w_kl_avg < 0.5, "w_KL too large: {}", w_kl_avg);
    }

    #[test]
    #[ignore] // This test is very computationally expensive
    fn test_lbp_vs_murtys_comprehensive() {
        // This test runs the full evaluation from MATLAB evaluateSmallExamples.m
        // but with fewer trials due to computational constraints
        let num_trials = 10;
        let max_objects = 5;

        for n in 1..=max_objects {
            let mut rng = SimpleRng::new(42 + n as u64);
            let mut r_kl_sum = 0.0;
            let mut w_kl_sum = 0.0;
            let mut r_h_sum = 0.0;
            let mut w_h_sum = 0.0;

            for trial in 0..num_trials {
                println!("n={}, trial={}/{}", n, trial + 1, num_trials);
                let (r_kl, w_kl, r_h, w_h) = run_marginal_evaluation_trial(&mut rng, n);
                r_kl_sum += r_kl;
                w_kl_sum += w_kl;
                r_h_sum += r_h;
                w_h_sum += w_h;
            }

            let r_kl_avg = r_kl_sum / num_trials as f64;
            let w_kl_avg = w_kl_sum / num_trials as f64;
            let r_h_avg = r_h_sum / num_trials as f64;
            let w_h_avg = w_h_sum / num_trials as f64;

            println!(
                "n={}: r_KL={:.6}, w_KL={:.6}, r_H={:.6}, w_H={:.6}",
                n, r_kl_avg, w_kl_avg, r_h_avg, w_h_avg
            );

            // Errors should be bounded (looser bounds for larger n)
            let max_kl = 0.1 + 0.1 * n as f64;
            let max_h = 0.1 + 0.1 * n as f64;
            assert!(r_kl_avg < max_kl, "r_KL too large for n={}: {}", n, r_kl_avg);
            assert!(w_kl_avg < max_kl, "w_KL too large for n={}: {}", n, w_kl_avg);
            assert!(r_h_avg < max_h, "r_H too large for n={}: {}", n, r_h_avg);
            assert!(w_h_avg < max_h, "w_H too large for n={}: {}", n, w_h_avg);
        }
    }
}
