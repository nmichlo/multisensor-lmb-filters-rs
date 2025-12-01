//! Multi-sensor LMBM Gibbs sampling
//!
//! Implements Gibbs sampling for multi-sensor LMBM data association.
//! Matches MATLAB multisensorLmbmGibbsSampling.m and generateMultisensorAssociationEvent.m exactly.

use super::determine_linear_index;
use super::lazy::LazyLikelihood;
use nalgebra::DMatrix;
use std::collections::HashSet;

// Access pattern tracing for lazy likelihood investigation
#[cfg(feature = "gibbs-trace")]
use std::cell::RefCell;

#[cfg(feature = "gibbs-trace")]
thread_local! {
    static ACCESSED_INDICES: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
    static TOTAL_ENTRIES: RefCell<usize> = RefCell::new(0);
}

/// Reset access pattern tracking (call before sampling)
#[cfg(feature = "gibbs-trace")]
pub fn reset_access_trace(total_entries: usize) {
    ACCESSED_INDICES.with(|indices| indices.borrow_mut().clear());
    TOTAL_ENTRIES.with(|total| *total.borrow_mut() = total_entries);
}

/// Get access pattern statistics: (unique_accesses, total_entries, access_ratio)
#[cfg(feature = "gibbs-trace")]
pub fn get_access_stats() -> (usize, usize, f64) {
    let unique = ACCESSED_INDICES.with(|indices| indices.borrow().len());
    let total = TOTAL_ENTRIES.with(|total| *total.borrow());
    let ratio = if total > 0 { unique as f64 / total as f64 } else { 0.0 };
    (unique, total, ratio)
}

/// Print access pattern report
#[cfg(feature = "gibbs-trace")]
pub fn print_access_report() {
    let (unique, total, ratio) = get_access_stats();
    eprintln!("[gibbs-trace] Accessed {}/{} entries ({:.2}%)", unique, total, ratio * 100.0);
    eprintln!("[gibbs-trace] Potential savings: {:.2}% of likelihood computations", (1.0 - ratio) * 100.0);
}

#[cfg(feature = "gibbs-trace")]
#[inline]
fn record_access(idx: usize) {
    ACCESSED_INDICES.with(|indices| indices.borrow_mut().insert(idx));
}

/// Generate association events using multi-sensor Gibbs sampler
///
/// Generates a set of posterior hypotheses for a given prior hypothesis
/// using Gibbs sampling on the high-dimensional likelihood matrix.
///
/// # Arguments
/// * `lazy` - Lazy likelihood computer (computes values on-demand)
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
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn multisensor_lmbm_gibbs_sampling(
    rng: &mut impl crate::common::rng::Rng,
    lazy: &LazyLikelihood,
    number_of_samples: usize,
) -> DMatrix<usize> {
    let dimensions = lazy.dimensions();
    let number_of_sensors = lazy.number_of_sensors();
    let number_of_objects = lazy.number_of_objects();

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
    for _sample_idx in 0..number_of_samples {
        // Generate new Gibbs sample
        generate_multisensor_association_event(rng, lazy, dimensions, &m, &mut v, &mut w);

        // Store sample (flatten V column-major to match MATLAB's reshape behavior)
        // MATLAB: reshape(V, 1, n * numberOfSensors) flattens column-major
        let mut sample = Vec::with_capacity(number_of_objects * number_of_sensors);
        for s in 0..number_of_sensors {
            for i in 0..number_of_objects {
                sample.push(v[(i, s)]);
            }
        }
        unique_samples.insert(sample);
    }

    // Convert HashSet to sorted Vec to match MATLAB's unique(A, 'rows') behavior
    let mut unique_vec: Vec<Vec<usize>> = unique_samples.into_iter().collect();
    unique_vec.sort();

    let num_unique = unique_vec.len();

    let mut a = DMatrix::zeros(num_unique, number_of_objects * number_of_sensors);
    for (row_idx, sample) in unique_vec.iter().enumerate() {
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
/// * `lazy` - Lazy likelihood computer
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
    lazy: &LazyLikelihood,
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
            // k = V(i, s) + (V(i, s) == 0) in MATLAB
            // If V=0 (miss): k=1 (start from measurement 1)
            // If V=j (meas j): k=j (start from measurement j)
            // In Rust (0-indexed): If v=0: k=0, If v=j (1-indexed): k=j-1 (convert to 0-indexed)
            let k = if v[(i, s)] == 0 { 0 } else { v[(i, s)] - 1 };

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

                    // Record accesses for lazy likelihood analysis
                    #[cfg(feature = "gibbs-trace")]
                    {
                        record_access(q_idx);
                        record_access(r_idx);
                    }

                    // P = 1 / (exp(L_miss - L_detect) + 1)
                    // Use lazy likelihood - only computes values that are actually accessed
                    let l_miss = lazy.get_l(r_idx);
                    let l_detect = lazy.get_l(q_idx);
                    let p = 1.0 / ((l_miss - l_detect).exp() + 1.0);

                    // Sample
                    let rand_val = rng.rand();

                    if rand_val < p {
                        // Object i generated measurement j
                        v[(i, s)] = j + 1; // Store as 1-indexed
                        w[(j, s)] = i + 1;
                        break;
                    } else {
                        // Object i did not generate measurement j
                        v[(i, s)] = 0;
                        w[(j, s)] = 0; // Unconditionally clear to match MATLAB line 36
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, Hypothesis, ScenarioType};
    use nalgebra::DVector;

    #[test]
    fn test_multisensor_lmbm_gibbs_sampling() {
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
            birth_location: vec![0, 0],
            birth_time: vec![0, 0],
            w: 1.0,
            r: vec![0.8, 0.7],
            mu: vec![
                DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
                DVector::from_vec(vec![10.0, 10.0, 0.0, 0.0]),
            ],
            sigma: vec![
                nalgebra::DMatrix::identity(4, 4) * 10.0,
                nalgebra::DMatrix::identity(4, 4) * 10.0,
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

        let mut rng = SimpleRng::new(42);
        let samples = multisensor_lmbm_gibbs_sampling(&mut rng, &lazy, 10);

        // Should have at least 1 sample, at most 10
        assert!(samples.nrows() >= 1 && samples.nrows() <= 10);

        // Each sample should have 2 sensors * 2 objects = 4 elements
        assert_eq!(samples.ncols(), 4);
    }
}
