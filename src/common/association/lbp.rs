//! Loopy Belief Propagation for data association
//!
//! Implements LBP algorithm for computing marginal association probabilities
//! and posterior existence probabilities. Matches MATLAB loopyBeliefPropagation.m
//! and fixedLoopyBeliefPropagation.m exactly.

use nalgebra::{DMatrix, DVector};

/// LBP result containing existence probabilities and association weights
#[derive(Debug, Clone)]
pub struct LbpResult {
    /// Posterior existence probabilities (n x 1)
    pub r: DVector<f64>,
    /// Marginal association probabilities (n x m)
    /// Each row contains an object's association probabilities
    pub w: DMatrix<f64>,
}

/// Association matrices for LBP
#[derive(Debug, Clone)]
pub struct AssociationMatrices {
    /// Psi matrix (n x m): likelihood ratios L / eta
    pub psi: DMatrix<f64>,
    /// Phi vector (n x 1): missed detection probabilities
    pub phi: DVector<f64>,
    /// Eta vector (n x 1): normalization factors
    pub eta: DVector<f64>,
}

// ============================================================================
// Shared helper functions (extracted to eliminate duplication)
// ============================================================================

/// Perform one iteration of LBP message passing
///
/// Updates sigma_mt in place by:
/// 1. Computing messages from objects to measurements (sigma_tm)
/// 2. Computing messages from measurements to objects (sigma_mt)
#[inline]
fn lbp_message_passing_iteration(matrices: &AssociationMatrices, sigma_mt: &mut DMatrix<f64>) {
    let n_objects = matrices.psi.nrows();
    let n_measurements = matrices.psi.ncols();

    // Pass messages from object to measurement clusters
    // B = Psi .* sigma_mt
    let b = matrices.psi.component_mul(sigma_mt);

    // sigma_tm = Psi ./ (-B + sum(B, 2) + 1)
    // Match MATLAB exactly: sum in column order (j=0,1,2,...) like MATLAB's sum(B, 2)
    let mut sigma_tm = DMatrix::zeros(n_objects, n_measurements);
    for i in 0..n_objects {
        // Explicit left-to-right sum to match MATLAB's sum(B, 2)
        let mut row_sum: f64 = 0.0;
        for j in 0..n_measurements {
            row_sum += b[(i, j)];
        }
        for j in 0..n_measurements {
            let denom = -b[(i, j)] + row_sum + 1.0;
            sigma_tm[(i, j)] = matrices.psi[(i, j)] / denom;
        }
    }

    // Pass messages from measurement to object clusters
    // sigma_mt = 1 ./ (-sigma_tm + sum(sigma_tm, 1) + 1)
    // Match MATLAB exactly: sum in row order (i=0,1,2,...) like MATLAB's sum(X, 1)
    let mut col_sums: Vec<f64> = vec![0.0; n_measurements];
    for j in 0..n_measurements {
        for i in 0..n_objects {
            col_sums[j] += sigma_tm[(i, j)];
        }
    }

    for i in 0..n_objects {
        for j in 0..n_measurements {
            let denom = -sigma_tm[(i, j)] + col_sums[j] + 1.0;
            sigma_mt[(i, j)] = 1.0 / denom;
        }
    }
}

/// Compute final LBP result from messages
///
/// IMPORTANT: Takes sigma_mt_old (from BEFORE the last message passing iteration)
/// to match MATLAB exactly. In MATLAB, B is computed at the START of each iteration
/// before sigma_mt is updated, and after the loop Gamma uses that B.
///
/// # Arguments
/// * `matrices` - Association matrices (Psi, phi, eta)
/// * `sigma_mt_old` - Messages from BEFORE the last iteration (not the final updated sigma_mt)
#[inline]
fn compute_lbp_result(matrices: &AssociationMatrices, sigma_mt_old: &DMatrix<f64>) -> LbpResult {
    let n_objects = matrices.psi.nrows();
    let n_measurements = matrices.psi.ncols();

    // B = Psi .* sigma_mt_old (uses sigma_mt from before the last update, matching MATLAB)
    let b = matrices.psi.component_mul(sigma_mt_old);

    // Gamma = [phi, B .* eta]
    let mut gamma = DMatrix::zeros(n_objects, n_measurements + 1);
    for i in 0..n_objects {
        gamma[(i, 0)] = matrices.phi[i];
        for j in 0..n_measurements {
            gamma[(i, j + 1)] = b[(i, j)] * matrices.eta[i];
        }
    }

    // q = sum(Gamma, 2)
    // Match MATLAB exactly: explicit left-to-right sum
    let q: Vec<f64> = (0..n_objects)
        .map(|i| {
            let mut sum = 0.0;
            for j in 0..(n_measurements + 1) {
                sum += gamma[(i, j)];
            }
            sum
        })
        .collect();

    // W = Gamma ./ q
    // Match MATLAB exactly - no division guards
    let mut w = DMatrix::zeros(n_objects, n_measurements + 1);
    for i in 0..n_objects {
        for j in 0..(n_measurements + 1) {
            w[(i, j)] = gamma[(i, j)] / q[i];
        }
    }

    // r = q ./ (eta + q - phi)
    // Match MATLAB exactly - no division guards
    let mut r = DVector::zeros(n_objects);
    for i in 0..n_objects {
        let denom = matrices.eta[i] + q[i] - matrices.phi[i];
        r[i] = q[i] / denom;
    }

    LbpResult { r, w }
}

// ============================================================================
// Public API
// ============================================================================

/// Loopy Belief Propagation with convergence check
///
/// Determines posterior existence probabilities and marginal association
/// probabilities using loopy belief propagation.
///
/// # Arguments
/// * `matrices` - Association matrices (Psi, phi, eta)
/// * `epsilon` - Convergence tolerance
/// * `max_iterations` - Maximum number of LBP iterations
///
/// # Returns
/// LbpResult with posterior existence probabilities and association weights
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn loopy_belief_propagation(
    matrices: &AssociationMatrices,
    epsilon: f64,
    max_iterations: usize,
) -> LbpResult {
    let n_objects = matrices.psi.nrows();
    let n_measurements = matrices.psi.ncols();

    // Initialize messages
    let mut sigma_mt = DMatrix::from_element(n_objects, n_measurements, 1.0);
    let mut sigma_mt_old = sigma_mt.clone();
    let mut not_converged = true;
    let mut counter = 0;

    // Message passing loop with convergence check
    while not_converged {
        // Cache previous iteration's messages (MATLAB uses this for B computation)
        sigma_mt_old = sigma_mt.clone();

        // Perform one iteration of message passing
        lbp_message_passing_iteration(matrices, &mut sigma_mt);

        // Check for convergence
        counter += 1;
        let delta = (&sigma_mt - &sigma_mt_old).abs();
        let max_delta = delta.iter().cloned().fold(0.0, f64::max);
        not_converged = max_delta > epsilon && counter < max_iterations;
    }

    // Compute final results using sigma_mt_old (matches MATLAB which uses B from start of last iteration)
    compute_lbp_result(matrices, &sigma_mt_old)
}

/// Fixed-iteration Loopy Belief Propagation
///
/// Same as loopy_belief_propagation but runs for a fixed number of iterations
/// without convergence checking. Used for computational complexity verification.
///
/// # Arguments
/// * `matrices` - Association matrices (Psi, phi, eta)
/// * `max_iterations` - Number of LBP iterations to run
///
/// # Returns
/// LbpResult with posterior existence probabilities and association weights
pub fn fixed_loopy_belief_propagation(
    matrices: &AssociationMatrices,
    max_iterations: usize,
) -> LbpResult {
    let n_objects = matrices.psi.nrows();
    let n_measurements = matrices.psi.ncols();

    // Initialize messages
    let mut sigma_mt = DMatrix::from_element(n_objects, n_measurements, 1.0);
    let mut sigma_mt_old = sigma_mt.clone();

    // Fixed number of iterations (no convergence check)
    for _ in 0..max_iterations {
        // Cache messages before update (MATLAB uses this for B computation)
        sigma_mt_old = sigma_mt.clone();
        lbp_message_passing_iteration(matrices, &mut sigma_mt);
    }

    // Compute final results using sigma_mt_old (matches MATLAB which uses B from start of last iteration)
    compute_lbp_result(matrices, &sigma_mt_old)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbp_simple() {
        // Simple 2 objects, 3 measurements case
        let psi = DMatrix::from_row_slice(2, 3, &[0.8, 0.2, 0.1, 0.1, 0.7, 0.3]);
        let phi = DVector::from_vec(vec![0.05, 0.05]);
        let eta = DVector::from_vec(vec![0.95, 0.95]);

        let matrices = AssociationMatrices { psi, phi, eta };

        let result = loopy_belief_propagation(&matrices, 1e-6, 1000);

        // Verify shapes
        assert_eq!(result.r.len(), 2);
        assert_eq!(result.w.nrows(), 2);
        assert_eq!(result.w.ncols(), 4); // 3 measurements + 1 miss

        // Verify probabilities sum to 1 for each object
        for i in 0..2 {
            let row_sum: f64 = result.w.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sum: {}", i, row_sum);
        }

        // Verify existence probabilities are in [0, 1]
        for i in 0..2 {
            assert!(result.r[i] >= 0.0 && result.r[i] <= 1.0);
        }
    }

    #[test]
    fn test_fixed_lbp() {
        let psi = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.6, 0.4]);
        let phi = DVector::from_vec(vec![0.1, 0.1]);
        let eta = DVector::from_vec(vec![0.9, 0.9]);

        let matrices = AssociationMatrices { psi, phi, eta };

        let result = fixed_loopy_belief_propagation(&matrices, 10);

        // Verify probabilities sum to 1
        for i in 0..2 {
            let row_sum: f64 = result.w.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }
}
