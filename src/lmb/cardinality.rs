//! LMB cardinality estimation
//!
//! Implements MAP cardinality estimation for LMB filters.
//! Matches MATLAB lmbMapCardinalityEstimate.m and esf.m exactly.

/// Elementary Symmetric Function (ESF)
///
/// Calculates elementary symmetric function using Mahler's recursive formula.
/// This is Vo and Vo's code ported to Rust.
///
/// # Arguments
/// * `z` - Input array
///
/// # Returns
/// Vector of ESF values [e_0, e_1, ..., e_n] where:
/// - e_0 = 1
/// - e_k = sum of all products of k elements from z
///
/// # Implementation Notes
/// Matches MATLAB esf.m exactly using Mahler's recursive formula:
/// - F(n,k) = F(n-1,k) + z(n)*F(n-1,k-1)
/// - Uses two-row buffer to save memory
pub fn elementary_symmetric_function(z: &[f64]) -> Vec<f64> {
    if z.is_empty() {
        return vec![1.0];
    }

    let n_z = z.len();
    let mut f = vec![vec![0.0; n_z]; 2];

    let mut i_n = 0;
    let mut i_nminus = 1;

    for n in 0..n_z {
        // F(i_n,0) = F(i_nminus,0) + Z(n)
        f[i_n][0] = f[i_nminus][0] + z[n];

        for k in 1..=n {
            if k == n {
                // F(i_n,k) = Z(n)*F(i_nminus,k-1)
                f[i_n][k] = z[n] * f[i_nminus][k - 1];
            } else {
                // F(i_n,k) = F(i_nminus,k) + Z(n)*F(i_nminus,k-1)
                f[i_n][k] = f[i_nminus][k] + z[n] * f[i_nminus][k - 1];
            }
        }

        // Swap indices
        std::mem::swap(&mut i_n, &mut i_nminus);
    }

    // Build result: [1; F(i_nminus,:)']
    let mut result = vec![1.0];
    result.extend_from_slice(&f[i_nminus][..n_z]);
    result
}

/// LMB MAP cardinality estimate
///
/// Determines approximate MAP estimate for LMB filter using Mahler's algorithm.
///
/// # Arguments
/// * `r` - Posterior existence probabilities for each object
///
/// # Returns
/// Tuple of (n_map, map_indices) where:
/// - n_map: MAP estimate for cardinality
/// - map_indices: Indices of the n_map objects with highest existence probabilities
///
/// # Implementation Notes
/// Matches MATLAB lmbMapCardinalityEstimate.m exactly:
/// 1. Compute LMB cardinality distribution: rho = prod(1-r) * esf(r/(1-r))
/// 2. Find maximum of rho
/// 3. Cap to number of objects
/// 4. Return indices of n_map largest existence probabilities
pub fn lmb_map_cardinality_estimate(r: &[f64]) -> (usize, Vec<usize>) {
    if r.is_empty() {
        return (0, Vec::new());
    }

    // Adjust existence probabilities to avoid unit values
    // r = r - 1e-6 (does not work with unit existence probabilities)
    let r_adjusted: Vec<f64> = r.iter().map(|&ri| ri - 1e-6).collect();

    // Compute rho = prod(1 - r)*esf(r/(1-r))
    let mut prod_1_minus_r = 1.0;
    let mut r_ratio = Vec::with_capacity(r_adjusted.len());

    for &ri in &r_adjusted {
        prod_1_minus_r *= 1.0 - ri;
        r_ratio.push(ri / (1.0 - ri));
    }

    let esf_values = elementary_symmetric_function(&r_ratio);

    // rho = prod(1-r) * esf(r/(1-r))
    let rho: Vec<f64> = esf_values.iter().map(|&e| prod_1_minus_r * e).collect();

    // Find maximum cardinality index
    let max_cardinality_index = rho
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // MAP estimate cannot be larger than the number of objects
    let n_map = std::cmp::min(max_cardinality_index, r.len());

    // Sort r in descending order and get indices
    let mut indexed_r: Vec<(usize, f64)> = r.iter().enumerate().map(|(i, &val)| (i, val)).collect();
    indexed_r.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    // Choose the nMap largest indices of r
    let map_indices: Vec<usize> = indexed_r.iter().take(n_map).map(|(i, _)| *i).collect();

    (n_map, map_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_esf_empty() {
        let result = elementary_symmetric_function(&[]);
        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn test_esf_simple() {
        // ESF of [1, 2] should be:
        // e_0 = 1
        // e_1 = 1 + 2 = 3
        // e_2 = 1*2 = 2
        let result = elementary_symmetric_function(&[1.0, 2.0]);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_esf_three_elements() {
        // ESF of [1, 2, 3] should be:
        // e_0 = 1
        // e_1 = 1 + 2 + 3 = 6
        // e_2 = 1*2 + 1*3 + 2*3 = 11
        // e_3 = 1*2*3 = 6
        let result = elementary_symmetric_function(&[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 6.0).abs() < 1e-10);
        assert!((result[2] - 11.0).abs() < 1e-10);
        assert!((result[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_cardinality_empty() {
        let (n_map, indices) = lmb_map_cardinality_estimate(&[]);
        assert_eq!(n_map, 0);
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_map_cardinality_simple() {
        // High existence probabilities should lead to MAP estimate ≈ number of objects
        let r = vec![0.9, 0.85, 0.8, 0.3, 0.2];
        let (n_map, indices) = lmb_map_cardinality_estimate(&r);

        // Should select high-probability objects
        assert!(n_map >= 3);
        assert_eq!(indices.len(), n_map);

        // Indices should be sorted by existence probability (descending)
        for i in 1..indices.len() {
            assert!(r[indices[i - 1]] >= r[indices[i]]);
        }
    }

    #[test]
    fn test_map_cardinality_low_prob() {
        // Low existence probabilities should lead to low MAP estimate
        let r = vec![0.1, 0.05, 0.02];
        let (n_map, _indices) = lmb_map_cardinality_estimate(&r);

        // MAP estimate should be low (0 or 1)
        assert!(n_map <= 1);
    }
}
