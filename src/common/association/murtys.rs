//! Murty's algorithm for K-best assignments
//!
//! Implements Murty's algorithm for finding the K-best ranked optimal assignments.
//! Matches MATLAB murtysAlgorithm.m and murtysAlgorithmWrapper.m exactly.

use crate::common::types::DMatrix;
use ndarray::Array2;
use ndarray::s;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Result from Murty's algorithm
#[derive(Debug, Clone)]
pub struct MurtysResult {
    /// K-best assignments (k x n matrix)
    /// Each row is an assignment vector
    pub assignments: DMatrix<usize>,
    /// Costs of each assignment
    pub costs: Vec<f64>,
}

/// Priority queue entry for Murty's algorithm
#[derive(Debug, Clone)]
struct QueueEntry {
    cost: f64,
    assignment: Vec<usize>,
    problem: DMatrix<f64>,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.cost.partial_cmp(&self.cost) // Reverse for min-heap
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Murty's algorithm wrapper
///
/// Determines the m most likely data association events.
/// Handles padding with dummy variables and cost normalization.
///
/// # Arguments
/// * `p0` - Cost matrix (n x m)
/// * `m` - Number of best assignments to find
///
/// # Returns
/// MurtysResult with assignments and costs
pub fn murtys_algorithm_wrapper(p0: &DMatrix<f64>, m: usize) -> MurtysResult {
    if m == 0 {
        return MurtysResult {
            assignments: Array2::zeros((0, 0)),
            costs: Vec::new(),
        };
    }

    let n1 = p0.nrows();
    let n2 = p0.ncols();

    // Padding block for dummy variables (matching MATLAB: -log(ones(n1,n1)) = 0)
    let blk1 = Array2::from_elem((n1, n1), -(1.0_f64).ln());

    // Concatenate: P0 = [P0 blk1]
    let mut p0_padded = Array2::zeros((n1, n2 + n1));
    p0_padded.slice_mut(s![0..n1, 0..n2]).assign(p0);
    p0_padded.slice_mut(s![0..n1, n2..n2+n1]).assign(&blk1);

    // Make costs non-negative
    let x = p0_padded.iter().cloned().fold(f64::INFINITY, f64::min);
    for val in p0_padded.iter_mut() {
        *val -= x;
    }

    // Run Murty's algorithm
    let result = murtys_algorithm(&p0_padded, m);

    // Restore correct costs
    let mut costs = Vec::new();
    let mut assignments_mat = Array2::zeros((result.assignments.nrows(), n1));

    for i in 0..result.assignments.nrows() {
        let num_assigned = result.assignments.row(i).iter().filter(|&&v| v > 0).count();
        costs.push(result.costs[i] + x * num_assigned as f64);

        // Strip dummy variables and copy to output
        for j in 0..n1 {
            let val = result.assignments[(i, j)];
            // Dummy assignments (> n2) become 0 (clutter)
            assignments_mat[(i, j)] = if val > n2 { 0 } else { val };
        }
    }

    MurtysResult {
        assignments: assignments_mat,
        costs,
    }
}

/// Core Murty's algorithm
///
/// Finds m-best ranked optimal assignments.
///
/// # Arguments
/// * `p0` - Cost matrix (n x m)
/// * `m` - Number of assignments to find
///
/// # Returns
/// MurtysResult with assignments and costs
fn murtys_algorithm(p0: &DMatrix<f64>, m: usize) -> MurtysResult {
    let num_rows = p0.nrows();
    let num_cols = p0.ncols();

    // Find optimal solution
    let initial = super::hungarian::hungarian(p0);
    let mut s0 = vec![0; num_rows];
    for i in 0..num_rows {
        for j in 0..num_cols {
            if initial.matching[(i, j)] == 1.0 {
                s0[i] = j + 1; // 1-indexed
                break;
            }
        }
    }
    let c0 = initial.cost;

    if m == 1 {
        let mut assignments = Array2::zeros((1, num_rows));
        for j in 0..num_rows {
            assignments[(0, j)] = s0[j];
        }
        return MurtysResult {
            assignments,
            costs: vec![c0],
        };
    }

    // Priority queue
    let mut queue = BinaryHeap::new();
    queue.push(QueueEntry {
        cost: c0,
        assignment: s0.clone(),
        problem: p0.clone(),
    });

    let mut assignments_list = Vec::new();
    let mut costs_list = Vec::new();

    for _ in 0..m {
        if queue.is_empty() {
            break;
        }

        // Get lowest cost entry
        let entry = queue.pop().unwrap();
        assignments_list.push(entry.assignment.clone());
        costs_list.push(entry.cost);

        let mut p_now = entry.problem;
        let s_now = entry.assignment;

        // Generate children
        for a in 0..s_now.len() {
            let aj = s_now[a];

            if aj != 0 {
                // Remove assignment and solve
                let mut p_tmp = p_now.clone();
                if aj <= num_cols - num_rows {
                    p_tmp[(a, aj - 1)] = f64::INFINITY;
                } else {
                    for col in (num_cols - num_rows)..num_cols {
                        p_tmp[(a, col)] = f64::INFINITY;
                    }
                }

                let result_tmp = super::hungarian::hungarian(&p_tmp);
                let mut s_tmp = vec![0; num_rows];
                for i in 0..num_rows {
                    for j in 0..num_cols {
                        if result_tmp.matching[(i, j)] == 1.0 {
                            s_tmp[i] = j + 1;
                            break;
                        }
                    }
                }

                // Only add if all assigned
                if s_tmp.iter().all(|&v| v != 0) {
                    queue.push(QueueEntry {
                        cost: result_tmp.cost,
                        assignment: s_tmp,
                        problem: p_tmp.clone(),
                    });
                }

                // Enforce current assignment (modifies P_now in place for next iteration)
                // MATLAB: v_tmp = P_now(aw,aj); P_now(aw,:) = inf; P_now(:,aj) = inf; P_now(aw,aj) = v_tmp;
                let v_tmp = p_now[(a, aj - 1)];
                for col in 0..num_cols {
                    p_now[(a, col)] = f64::INFINITY;
                }
                for row in 0..num_rows {
                    p_now[(row, aj - 1)] = f64::INFINITY;
                }
                p_now[(a, aj - 1)] = v_tmp;
            }
        }
    }

    // Convert to matrix
    let mut assignments = Array2::zeros((assignments_list.len(), num_rows));
    for (i, assign) in assignments_list.iter().enumerate() {
        for (j, &val) in assign.iter().enumerate() {
            assignments[(i, j)] = val;
        }
    }

    MurtysResult {
        assignments,
        costs: costs_list,
    }
}

#[cfg(test)]
mod tests {
    use crate::common::types::MatrixExt;
    use super::*;

    #[test]
    fn test_murtys_simple() {
        let p0 = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0,
        ]);

        let result = murtys_algorithm_wrapper(&p0, 3);

        assert_eq!(result.assignments.nrows(), 3);
        assert_eq!(result.assignments.ncols(), 3);
        assert_eq!(result.costs.len(), 3);

        // Costs should be increasing
        assert!(result.costs[0] <= result.costs[1]);
        assert!(result.costs[1] <= result.costs[2]);
    }

    #[test]
    fn test_murtys_single() {
        let p0 = DMatrix::from_row_slice(2, 2, &[
            1.0, 3.0,
            3.0, 1.0,
        ]);

        let result = murtys_algorithm_wrapper(&p0, 1);

        assert_eq!(result.assignments.nrows(), 1);
        assert_eq!(result.costs.len(), 1);
    }
}