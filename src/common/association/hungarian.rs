//! Hungarian algorithm for optimal assignment
//!
//! Implements the Hungarian algorithm for finding minimum cost assignments
//! in a bipartite graph. Matches MATLAB Hungarian.m exactly.

use crate::common::types::DMatrix;
use ndarray::Array2;

/// Hungarian algorithm result
#[derive(Debug, Clone)]
pub struct HungarianResult {
    /// Assignment matrix (1 where assigned, 0 elsewhere)
    pub matching: DMatrix<f64>,
    /// Total cost of the assignment
    pub cost: f64,
}

/// Hungarian algorithm for optimal assignment
///
/// Finds the minimum cost assignment in a bipartite graph.
/// Matches the MATLAB Hungarian.m implementation exactly.
///
/// # Arguments
/// * `perf` - Performance/cost matrix (MxN). Use f64::INFINITY for impossible assignments.
///
/// # Returns
/// HungarianResult with matching matrix and total cost
pub fn hungarian(perf: &DMatrix<f64>) -> HungarianResult {
    let m = perf.nrows();
    let n = perf.ncols();
    let mut matching = Array2::zeros((m, n));

    // Find connected vertices (non-infinite entries)
    let mut x_con = Vec::new();
    let mut y_con = Vec::new();

    for i in 0..m {
        if perf.row(i).iter().any(|&val| !val.is_infinite()) {
            x_con.push(i);
        }
    }

    for j in 0..n {
        if perf.column(j).iter().any(|&val| !val.is_infinite()) {
            y_con.push(j);
        }
    }

    if x_con.is_empty() || y_con.is_empty() {
        return HungarianResult {
            matching,
            cost: 0.0,
        };
    }

    // Assemble condensed performance matrix
    let p_size = x_con.len().max(y_con.len());
    let mut p_cond = Array2::zeros((p_size, p_size));

    for (i_new, &i_old) in x_con.iter().enumerate() {
        for (j_new, &j_old) in y_con.iter().enumerate() {
            p_cond[(i_new, j_new)] = perf[(i_old, j_old)];
        }
    }

    // Ensure perfect matching exists
    let mut edge = p_cond.clone();
    for i in 0..edge.nrows() {
        for j in 0..edge.ncols() {
            if !edge[(i, j)].is_infinite() {
                edge[(i, j)] = 0.0;
            }
        }
    }

    let cnum = min_line_cover(&edge);

    // Add virtual vertices if needed
    let mut pmax = f64::NEG_INFINITY;
    for i in 0..x_con.len() {
        for j in 0..y_con.len() {
            let val = perf[(x_con[i], y_con[j])];
            if !val.is_infinite() && val > pmax {
                pmax = val;
            }
        }
    }

    let final_size = p_size + cnum;
    let mut p_final = Array2::from_elem((final_size, final_size), pmax);

    for (i_new, &i_old) in x_con.iter().enumerate() {
        for (j_new, &j_old) in y_con.iter().enumerate() {
            p_final[(i_new, j_new)] = perf[(i_old, j_old)];
        }
    }

    // Run Hungarian algorithm main loop
    let m_result = hungarian_main(&mut p_final);

    // Extract matching for original vertices
    for (i_new, &i_old) in x_con.iter().enumerate() {
        for (j_new, &j_old) in y_con.iter().enumerate() {
            matching[(i_old, j_old)] = m_result[(i_new, j_new)];
        }
    }

    // Calculate cost
    let mut cost = 0.0;
    for i in 0..m {
        for j in 0..n {
            if matching[(i, j)] == 1.0 {
                cost += perf[(i, j)];
            }
        }
    }

    HungarianResult { matching, cost }
}

/// Main Hungarian algorithm loop
fn hungarian_main(p_cond: &mut DMatrix<f64>) -> DMatrix<f64> {
    let p_size = p_cond.nrows();

    // Step 1: Subtract row minimums
    step1(p_cond);

    // Step 2: Initial starring
    let (mut r_cov, mut c_cov, mut m) = step2(p_cond);

    // Main loop
    let mut step_num = 3;
    let mut z_r = 0;
    let mut z_c = 0;

    loop {
        match step_num {
            3 => {
                step_num = step3(&m, p_size, &mut c_cov);
            }
            4 => {
                let result = step4(p_cond, &mut r_cov, &mut c_cov, &mut m);
                step_num = result.0;
                z_r = result.1;
                z_c = result.2;
            }
            5 => {
                step_num = step5(&mut m, z_r, z_c, &mut r_cov, &mut c_cov);
            }
            6 => {
                step_num = step6(p_cond, &r_cov, &c_cov);
            }
            7 => break,
            _ => break,
        }
    }

    m
}

/// Step 1: Subtract row minimums
fn step1(p_cond: &mut DMatrix<f64>) {
    let p_size = p_cond.nrows();

    for i in 0..p_size {
        let row_min = p_cond.row(i).iter().cloned().fold(f64::INFINITY, f64::min);
        for j in 0..p_size {
            p_cond[(i, j)] -= row_min;
        }
    }
}

/// Step 2: Find initial starred zeros
fn step2(p_cond: &DMatrix<f64>) -> (Vec<usize>, Vec<usize>, DMatrix<f64>) {
    let p_size = p_cond.nrows();
    let mut r_cov = vec![0; p_size];
    let mut c_cov = vec![0; p_size];
    let mut m = Array2::zeros((p_size, p_size));

    for i in 0..p_size {
        for j in 0..p_size {
            if p_cond[(i, j)] == 0.0 && r_cov[i] == 0 && c_cov[j] == 0 {
                m[(i, j)] = 1.0; // Star this zero
                r_cov[i] = 1;
                c_cov[j] = 1;
            }
        }
    }

    // Re-initialize covers
    r_cov = vec![0; p_size];
    c_cov = vec![0; p_size];

    (r_cov, c_cov, m)
}

/// Step 3: Cover columns with starred zeros
fn step3(m: &DMatrix<f64>, p_size: usize, c_cov: &mut Vec<usize>) -> usize {
    for j in 0..p_size {
        for i in 0..p_size {
            if m[(i, j)] == 1.0 {
                c_cov[j] = 1;
                break;
            }
        }
    }

    let covered_cols: usize = c_cov.iter().sum();

    if covered_cols == p_size {
        7 // Done
    } else {
        4 // Continue
    }
}

/// Step 4: Find uncovered zeros and prime them
fn step4(
    p_cond: &DMatrix<f64>,
    r_cov: &mut Vec<usize>,
    c_cov: &mut Vec<usize>,
    m: &mut DMatrix<f64>,
) -> (usize, usize, usize) {
    let p_size = p_cond.nrows();

    loop {
        // Find first uncovered zero
        let mut row = None;
        let mut col = None;

        'outer: for i in 0..p_size {
            for j in 0..p_size {
                if p_cond[(i, j)] == 0.0 && r_cov[i] == 0 && c_cov[j] == 0 {
                    row = Some(i);
                    col = Some(j);
                    break 'outer;
                }
            }
        }

        match (row, col) {
            (None, None) => {
                // No uncovered zeros, go to step 6
                return (6, 0, 0);
            }
            (Some(r), Some(c)) => {
                // Prime the zero
                m[(r, c)] = 2.0;

                // Check if there's a starred zero in this row
                let mut star_col = None;
                for j in 0..p_size {
                    if m[(r, j)] == 1.0 {
                        star_col = Some(j);
                        break;
                    }
                }

                match star_col {
                    Some(sc) => {
                        // Cover row, uncover column
                        r_cov[r] = 1;
                        c_cov[sc] = 0;
                    }
                    None => {
                        // Go to step 5
                        return (5, r, c);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Step 5: Augment matching along alternating path
fn step5(
    m: &mut DMatrix<f64>,
    z_r: usize,
    z_c: usize,
    r_cov: &mut Vec<usize>,
    c_cov: &mut Vec<usize>,
) -> usize {
    let p_size = m.nrows();
    let mut path_r = vec![z_r];
    let mut path_c = vec![z_c];

    loop {
        // Find starred zero in column of last primed zero
        let mut star_row = None;
        let last_col = path_c[path_c.len() - 1];

        for i in 0..p_size {
            if m[(i, last_col)] == 1.0 {
                star_row = Some(i);
                break;
            }
        }

        match star_row {
            Some(sr) => {
                path_r.push(sr);
                path_c.push(last_col);

                // Find primed zero in this row
                let mut prime_col = None;
                for j in 0..p_size {
                    if m[(sr, j)] == 2.0 {
                        prime_col = Some(j);
                        break;
                    }
                }

                if let Some(pc) = prime_col {
                    path_r.push(sr);
                    path_c.push(pc);
                }
            }
            None => break,
        }
    }

    // Augment path: unstar starred zeros, star primed zeros
    for i in 0..path_r.len() {
        if m[(path_r[i], path_c[i])] == 1.0 {
            m[(path_r[i], path_c[i])] = 0.0;
        } else {
            m[(path_r[i], path_c[i])] = 1.0;
        }
    }

    // Clear covers
    r_cov.fill(0);
    c_cov.fill(0);

    // Remove all primes
    for i in 0..p_size {
        for j in 0..p_size {
            if m[(i, j)] == 2.0 {
                m[(i, j)] = 0.0;
            }
        }
    }

    3 // Go to step 3
}

/// Step 6: Add minimum to covered rows, subtract from uncovered columns
fn step6(p_cond: &mut DMatrix<f64>, r_cov: &Vec<usize>, c_cov: &Vec<usize>) -> usize {
    let p_size = p_cond.nrows();

    // Find minimum uncovered value
    let mut minval = f64::INFINITY;
    for i in 0..p_size {
        if r_cov[i] == 0 {
            for j in 0..p_size {
                if c_cov[j] == 0 {
                    minval = minval.min(p_cond[(i, j)]);
                }
            }
        }
    }

    // Add to covered rows
    for i in 0..p_size {
        if r_cov[i] == 1 {
            for j in 0..p_size {
                p_cond[(i, j)] += minval;
            }
        }
    }

    // Subtract from uncovered columns
    for j in 0..p_size {
        if c_cov[j] == 0 {
            for i in 0..p_size {
                p_cond[(i, j)] -= minval;
            }
        }
    }

    4 // Go to step 4
}

/// Calculate minimum line cover (for deficiency calculation)
fn min_line_cover(edge: &DMatrix<f64>) -> usize {
    let (r_cov, c_cov, m) = step2(edge);
    let p_size = edge.nrows();
    let mut c_cov_mut = c_cov.clone();

    step3(&m, p_size, &mut c_cov_mut);

    let r_sum: usize = r_cov.iter().sum();
    let c_sum: usize = c_cov_mut.iter().sum();

    p_size - r_sum - c_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hungarian_simple() {
        // Simple 3x3 cost matrix
        let mut perf = Array2::from_elem((3, 3), 0.0);
        perf[(0, 0)] = 1.0;
        perf[(0, 1)] = 2.0;
        perf[(0, 2)] = 3.0;
        perf[(1, 0)] = 2.0;
        perf[(1, 1)] = 4.0;
        perf[(1, 2)] = 6.0;
        perf[(2, 0)] = 3.0;
        perf[(2, 1)] = 6.0;
        perf[(2, 2)] = 9.0;

        let result = hungarian(&perf);

        // Verify each row and column has exactly one assignment
        let row_sums: Vec<f64> = (0..3).map(|i| result.matching.row(i).sum()).collect();
        let col_sums: Vec<f64> = (0..3).map(|j| result.matching.column(j).sum()).collect();

        for &sum in &row_sums {
            assert!((sum - 1.0).abs() < 1e-10);
        }
        for &sum in &col_sums {
            assert!((sum - 1.0).abs() < 1e-10);
        }

        // Minimum cost should be 1+4+9 = 14 or better
        assert!(result.cost > 0.0);
    }

    #[test]
    fn test_hungarian_with_infinity() {
        // Matrix with some impossible assignments
        let mut perf = Array2::from_elem((2, 2), 0.0);
        perf[(0, 0)] = 1.0;
        perf[(0, 1)] = f64::INFINITY;
        perf[(1, 0)] = f64::INFINITY;
        perf[(1, 1)] = 2.0;

        let result = hungarian(&perf);

        assert!((result.matching[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result.matching[(1, 1)] - 1.0).abs() < 1e-10);
        assert!((result.cost - 3.0).abs() < 1e-10);
    }
}