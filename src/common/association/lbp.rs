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
pub fn loopy_belief_propagation(
    matrices: &AssociationMatrices,
    epsilon: f64,
    max_iterations: usize,
) -> LbpResult {
    let n_objects = matrices.psi.nrows();
    let n_measurements = matrices.psi.ncols();

    // Initialize messages
    let mut sigma_mt = DMatrix::from_element(n_objects, n_measurements, 1.0);
    let mut not_converged = true;
    let mut counter = 0;

    // Message passing loop
    while not_converged {
        // Cache previous iteration's messages
        let sigma_mt_old = sigma_mt.clone();

        // Pass messages from object to measurement clusters
        let b = matrices.psi.component_mul(&sigma_mt);

        // sigma_tm = Psi ./ (-B + sum(B, 2) + 1)
        let mut sigma_tm = DMatrix::zeros(n_objects, n_measurements);
        for i in 0..n_objects {
            let row_sum: f64 = b.row(i).sum();
            for j in 0..n_measurements {
                let denom = -b[(i, j)] + row_sum + 1.0;
                sigma_tm[(i, j)] = if denom.abs() > 1e-15 {
                    matrices.psi[(i, j)] / denom
                } else {
                    0.0
                };
            }
        }

        // Pass messages from measurement to object clusters
        // sigma_mt = 1 ./ (-sigma_tm + sum(sigma_tm, 1) + 1)
        let col_sums: Vec<f64> = (0..n_measurements)
            .map(|j| sigma_tm.column(j).sum())
            .collect();

        for i in 0..n_objects {
            for j in 0..n_measurements {
                let denom = -sigma_tm[(i, j)] + col_sums[j] + 1.0;
                sigma_mt[(i, j)] = if denom.abs() > 1e-15 {
                    1.0 / denom
                } else {
                    0.0
                };
            }
        }

        // Check for convergence
        counter += 1;
        let delta = (&sigma_mt - &sigma_mt_old).abs();
        let max_delta = delta.iter().cloned().fold(0.0, f64::max);
        not_converged = max_delta > epsilon && counter < max_iterations;
    }

    // Compute final results
    let b = matrices.psi.component_mul(&sigma_mt);

    // Gamma = [phi, B .* eta]
    let mut gamma = DMatrix::zeros(n_objects, n_measurements + 1);
    for i in 0..n_objects {
        gamma[(i, 0)] = matrices.phi[i];
        for j in 0..n_measurements {
            gamma[(i, j + 1)] = b[(i, j)] * matrices.eta[i];
        }
    }

    // q = sum(Gamma, 2)
    let q: Vec<f64> = (0..n_objects).map(|i| gamma.row(i).sum()).collect();

    // W = Gamma ./ q
    let mut w = DMatrix::zeros(n_objects, n_measurements + 1);
    for i in 0..n_objects {
        if q[i].abs() > 1e-15 {
            for j in 0..(n_measurements + 1) {
                w[(i, j)] = gamma[(i, j)] / q[i];
            }
        }
    }

    // r = q ./ (eta + q - phi)
    let mut r = DVector::zeros(n_objects);
    for i in 0..n_objects {
        let denom = matrices.eta[i] + q[i] - matrices.phi[i];
        r[i] = if denom.abs() > 1e-15 {
            q[i] / denom
        } else {
            0.0
        };
    }

    LbpResult { r, w }
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

    // Fixed number of iterations
    for _ in 0..max_iterations {
        // Pass messages from object to measurement clusters
        let b = matrices.psi.component_mul(&sigma_mt);

        // sigma_tm = Psi ./ (-B + sum(B, 2) + 1)
        let mut sigma_tm = DMatrix::zeros(n_objects, n_measurements);
        for i in 0..n_objects {
            let row_sum: f64 = b.row(i).sum();
            for j in 0..n_measurements {
                let denom = -b[(i, j)] + row_sum + 1.0;
                sigma_tm[(i, j)] = if denom.abs() > 1e-15 {
                    matrices.psi[(i, j)] / denom
                } else {
                    0.0
                };
            }
        }

        // Pass messages from measurement to object clusters
        let col_sums: Vec<f64> = (0..n_measurements)
            .map(|j| sigma_tm.column(j).sum())
            .collect();

        for i in 0..n_objects {
            for j in 0..n_measurements {
                let denom = -sigma_tm[(i, j)] + col_sums[j] + 1.0;
                sigma_mt[(i, j)] = if denom.abs() > 1e-15 {
                    1.0 / denom
                } else {
                    0.0
                };
            }
        }
    }

    // Compute final results (same as converged version)
    let b = matrices.psi.component_mul(&sigma_mt);

    let mut gamma = DMatrix::zeros(n_objects, n_measurements + 1);
    for i in 0..n_objects {
        gamma[(i, 0)] = matrices.phi[i];
        for j in 0..n_measurements {
            gamma[(i, j + 1)] = b[(i, j)] * matrices.eta[i];
        }
    }

    let q: Vec<f64> = (0..n_objects).map(|i| gamma.row(i).sum()).collect();

    let mut w = DMatrix::zeros(n_objects, n_measurements + 1);
    for i in 0..n_objects {
        if q[i].abs() > 1e-15 {
            for j in 0..(n_measurements + 1) {
                w[(i, j)] = gamma[(i, j)] / q[i];
            }
        }
    }

    let mut r = DVector::zeros(n_objects);
    for i in 0..n_objects {
        let denom = matrices.eta[i] + q[i] - matrices.phi[i];
        r[i] = if denom.abs() > 1e-15 {
            q[i] / denom
        } else {
            0.0
        };
    }

    LbpResult { r, w }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbp_simple() {
        // Simple 2 objects, 3 measurements case
        let psi = DMatrix::from_row_slice(2, 3, &[
            0.8, 0.2, 0.1,
            0.1, 0.7, 0.3,
        ]);
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