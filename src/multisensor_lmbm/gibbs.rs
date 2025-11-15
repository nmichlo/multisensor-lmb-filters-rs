//! Multi-sensor LMBM Gibbs sampling
//!
//! Implements Gibbs sampling for multi-sensor LMBM data association.
//! Matches MATLAB multisensorLmbmGibbsSampling.m and generateMultisensorAssociationEvent.m exactly.

use nalgebra::DMatrix;
use std::collections::HashSet;

/// Generate association events using multi-sensor Gibbs sampler
///
/// Generates a set of posterior hypotheses for a given prior hypothesis
/// using Gibbs sampling on the high-dimensional likelihood matrix.
///
/// # Arguments
/// * `l` - Flattened likelihood matrix (see association.rs for structure)
/// * `dimensions` - Dimensions [m1+1, m2+1, ..., ms+1, n]
/// * `number_of_samples` - Number of Gibbs samples to generate
///
/// # Returns
/// Array of distinct association events, where each row is an association event
///
/// # Implementation Notes
/// Matches MATLAB multisensorLmbmGibbsSampling.m exactly:
/// 1. Initialize V (n x S) and W (max(m) x S) association matrices
/// 2. For each sample: call generateMultisensorAssociationEvent
/// 3. Store and keep only unique association events
pub fn multisensor_lmbm_gibbs_sampling(
    rng: &mut impl crate::common::rng::Rng,
    l: &[f64],
    dimensions: &[usize],
    number_of_samples: usize,
) -> DMatrix<usize> {
    let number_of_sensors = dimensions.len() - 1;
    let number_of_objects = dimensions[number_of_sensors];

    // m = dimensions[0..S] - 1
    let m: Vec<usize> = dimensions[0..number_of_sensors]
        .iter()
        .map(|&d| d - 1)
        .collect();
    let max_m = *m.iter().max().unwrap_or(&0);

    // Initialize association matrices
    let mut v = DMatrix::zeros(number_of_objects, number_of_sensors);
    let mut w = DMatrix::zeros(max_m, number_of_sensors);

    // Store samples as HashSet to keep only unique
    let mut unique_samples = HashSet::new();

    // Gibbs sampling
    for _ in 0..number_of_samples {
        // Generate new Gibbs sample
        generate_multisensor_association_event(rng, l, dimensions, &m, &mut v, &mut w);

        // Store sample (flatten V row-wise)
        let mut sample = Vec::with_capacity(number_of_objects * number_of_sensors);
        for i in 0..number_of_objects {
            for s in 0..number_of_sensors {
                sample.push(v[(i, s)]);
            }
        }
        unique_samples.insert(sample);
    }

    // Convert HashSet to DMatrix
    let num_unique = unique_samples.len();
    let mut a = DMatrix::zeros(num_unique, number_of_objects * number_of_sensors);
    for (row_idx, sample) in unique_samples.iter().enumerate() {
        for (col_idx, &val) in sample.iter().enumerate() {
            a[(row_idx, col_idx)] = val;
        }
    }

    a
}

/// Generate a single multi-sensor association event using Gibbs sampling
///
/// Updates association vectors V and W by sampling object-measurement associations
/// for each sensor sequentially.
///
/// # Arguments
/// * `l` - Flattened likelihood matrix
/// * `dimensions` - Dimensions [m1+1, m2+1, ..., ms+1, n]
/// * `m` - Number of measurements per sensor [m1, m2, ..., ms]
/// * `v` - Object-to-measurement association matrix (n x S), modified in-place
/// * `w` - Measurement-to-object association matrix (max(m) x S), modified in-place
///
/// # Implementation Notes
/// Matches MATLAB generateMultisensorAssociationEvent.m exactly:
/// 1. For each sensor s:
///    - For each object i:
///      - For each measurement j:
///        - If measurement j is unassociated or associated to object i:
///          - Compute sample probability: P = 1 / (exp(L_miss - L_detect) + 1)
///          - Sample: if rand() < P, associate object i to measurement j
fn generate_multisensor_association_event(
    rng: &mut impl crate::common::rng::Rng,
    l: &[f64],
    dimensions: &[usize],
    m: &[usize],
    v: &mut DMatrix<usize>,
    w: &mut DMatrix<usize>,
) {
    let number_of_sensors = m.len();
    let number_of_objects = dimensions[number_of_sensors];

    // For each sensor
    for s in 0..number_of_sensors {
        // For each object
        for i in 0..number_of_objects {
            let k = if v[(i, s)] == 0 { 0 } else { v[(i, s)] };

            // Build association vector u = [v1+1, v2+1, ..., vs+1, i+1] (MATLAB 1-indexed)
            let mut u = vec![0; number_of_sensors + 1];
            for s_idx in 0..number_of_sensors {
                u[s_idx] = v[(i, s_idx)] + 1; // Convert to 1-indexed
            }
            u[number_of_sensors] = i + 1; // Object index (1-indexed)

            // For each measurement of sensor s
            for j in k..m[s] {
                // Check if measurement j is unassociated or associated to object i
                if w[(j, s)] == 0 || w[(j, s)] == i + 1 {
                    // Compute sample probability
                    // Detection: object i generates measurement j
                    u[s] = j + 2; // j is 0-indexed, MATLAB uses j+1, so +2 for detection
                    let q_idx = determine_linear_index(&u, dimensions);

                    // Miss: object i does not generate measurement j
                    u[s] = 1; // Miss is index 1 in MATLAB (0 in Rust + 1)
                    let r_idx = determine_linear_index(&u, dimensions);

                    // P = 1 / (exp(L_miss - L_detect) + 1)
                    let p = 1.0 / ((l[r_idx] - l[q_idx]).exp() + 1.0);

                    // Sample
                    if rng.rand() < p {
                        // Object i generated measurement j
                        v[(i, s)] = j + 1; // Store as 1-indexed
                        w[(j, s)] = i + 1;
                        break;
                    } else {
                        // Object i did not generate measurement j
                        v[(i, s)] = 0;
                        if w[(j, s)] == i + 1 {
                            w[(j, s)] = 0;
                        }
                    }
                }
            }
        }
    }
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

    #[test]
    fn test_multisensor_lmbm_gibbs_sampling() {
        // Simple test with 2 sensors, 2 objects
        // Dimensions: [2+1, 2+1, 2] = [3, 3, 2]
        let dimensions = vec![3, 3, 2];
        let total_size = 3 * 3 * 2;

        // Create simple likelihood matrix (all ones for simplicity)
        let l = vec![0.0; total_size];

        let samples = multisensor_lmbm_gibbs_sampling(&l, &dimensions, 10);

        // Should have at least 1 sample, at most 10
        assert!(samples.nrows() >= 1 && samples.nrows() <= 10);

        // Each sample should have 2 sensors * 2 objects = 4 elements
        assert_eq!(samples.ncols(), 4);
    }
}
