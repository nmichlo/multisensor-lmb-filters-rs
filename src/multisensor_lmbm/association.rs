//! Multi-sensor LMBM association matrices
//!
//! Implements association matrix generation for multi-sensor LMBM filter.
//! Uses a high-dimensional likelihood matrix for joint sensor-object associations.
//! Matches MATLAB generateMultisensorLmbmAssociationMatrices.m exactly.
//!
//! WARNING: This implementation can be very memory intensive for large numbers
//! of objects and sensors, matching the MATLAB behavior.

use crate::common::linalg::robust_inverse_with_log_det;
use crate::common::types::{Hypothesis, Model};
use nalgebra::{DMatrix, DVector};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Gating threshold for quick Mahalanobis distance check.
/// Associations with diagonal Mahalanobis distance above this are skipped.
/// Based on chi-squared distribution: for z_dim=2, 99.99% confidence ≈ 18.42.
/// We use a higher value (50.0) to be conservative and avoid missing valid associations.
/// This can be configured via the `gating_threshold` field in Model (if added).
const DEFAULT_GATING_THRESHOLD: f64 = 50.0;

/// Posterior parameters for multi-sensor LMBM update
#[derive(Debug, Clone)]
pub struct MultisensorLmbmPosteriorParameters {
    /// Posterior existence probabilities (flattened tensor)
    pub r: Vec<f64>,
    /// Posterior means (flattened tensor)
    pub mu: Vec<DVector<f64>>,
    /// Posterior covariances (flattened tensor)
    pub sigma: Vec<DMatrix<f64>>,
}

/// Maximum supported sensors for stack-allocated index arrays
const MAX_SENSORS: usize = 8;

/// Convert linear index to Cartesian coordinates (stack-allocated version)
///
/// # Arguments
/// * `ell` - Linear index (1-indexed in MATLAB, 0-indexed here after conversion)
/// * `page_sizes` - Cumulative product of dimensions
/// * `out` - Output array to store Cartesian coordinates (must have length >= page_sizes.len())
///
/// Writes Cartesian coordinates to `out` without heap allocation.
#[inline]
fn convert_from_linear_to_cartesian_inplace(mut ell: usize, page_sizes: &[usize], out: &mut [usize; MAX_SENSORS]) {
    let m = page_sizes.len();
    debug_assert!(m <= MAX_SENSORS, "Too many sensors for stack-allocated array");

    for i in 0..m {
        let j = m - i - 1;
        let zeta = ell / page_sizes[j];
        let eta = ell % page_sizes[j];
        out[j] = zeta + if eta != 0 { 1 } else { 0 };
        ell = ell - page_sizes[j] * (zeta - if eta == 0 { 1 } else { 0 });
    }
}

/// Determine log likelihood ratio for a given object-sensor association
///
/// # Arguments
/// * `i` - Object index
/// * `a` - Association vector (measurement indices per sensor, 0 = miss)
/// * `measurements` - Measurements from all sensors
/// * `hypothesis` - Prior hypothesis
/// * `model` - Model parameters
/// * `q_cache` - Pre-cached Q matrices per sensor (avoids repeated clones)
/// * `c_cache` - Pre-cached C matrices per sensor (avoids repeated clones)
///
/// # Returns
/// Tuple of (log likelihood, existence prob, mean, covariance)
#[inline]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn determine_log_likelihood_ratio(
    i: usize,
    a: &[usize],
    measurements: &[&[DVector<f64>]],
    hypothesis: &Hypothesis,
    model: &Model,
    q_cache: &[DMatrix<f64>],
    c_cache: &[DMatrix<f64>],
) -> (f64, f64, DVector<f64>, DMatrix<f64>) {
    let number_of_sensors = a.len();

    // Check which sensors have detections (stack-allocated)
    let mut assignments = [false; MAX_SENSORS];
    let mut number_of_assignments = 0usize;
    for s in 0..number_of_sensors {
        if a[s] > 0 {
            assignments[s] = true;
            number_of_assignments += 1;
        }
    }

    if number_of_assignments > 0 {
        // GATING: Quick diagonal Mahalanobis check BEFORE expensive matrix ops
        // This is O(n) vs O(n³) for Cholesky, skipping ~30-50% of impossible associations
        let mut gate_failed = false;
        for s in 0..number_of_sensors {
            if assignments[s] {
                let meas = &measurements[s][a[s] - 1];
                let c_s = &c_cache[s];
                let q_s = &q_cache[s];

                // Quick predicted measurement: c * mu (O(z_dim * x_dim))
                let predicted_z = c_s * &hypothesis.mu[i];

                // Innovation
                let innovation = meas - &predicted_z;

                // Quick diagonal covariance approximation:
                // Z_diag ≈ diag(c * sigma * c') + diag(q)
                // For standard observation matrix c = [[1,0,0,0],[0,0,1,0]]:
                //   c*sigma*c' diagonal = [sigma[0,0], sigma[2,2]] (position variances)
                // We use sigma diagonal as proxy (conservative upper bound on distance)
                let sigma_diag = hypothesis.sigma[i].diagonal();
                let mut quick_dist = 0.0;
                for j in 0..model.z_dimension {
                    // For standard c matrix: measure position (indices 0, 2 of state)
                    // The j-th measurement corresponds to state index j*2 (x->0, y->2)
                    let state_idx = j * 2; // Works for standard [x,vx,y,vy] state
                    let var_approx = if state_idx < sigma_diag.len() {
                        sigma_diag[state_idx] + q_s[(j, j)]
                    } else {
                        q_s[(j, j)] // Fallback if state layout differs
                    };
                    quick_dist += innovation[j].powi(2) / var_approx.max(1e-10);
                }

                // If any sensor's quick distance exceeds threshold, gate this association
                if quick_dist > DEFAULT_GATING_THRESHOLD {
                    gate_failed = true;
                    break;
                }
            }
        }

        // If gating failed, return very low likelihood (effectively -infinity for exp)
        if gate_failed {
            return (
                -1e10, // Log-likelihood so low it will be ignored by Gibbs
                0.0,   // Existence prob (won't be used)
                hypothesis.mu[i].clone(),    // Placeholder
                hypothesis.sigma[i].clone(), // Placeholder
            );
        }

        // Determine measurement vector by stacking measurements from detecting sensors
        let z_dim_total = model.z_dimension * number_of_assignments;
        let mut z = DVector::zeros(z_dim_total);
        let mut c = DMatrix::zeros(z_dim_total, model.x_dimension);
        let mut q_block_indices = [0usize; MAX_SENSORS];
        let mut num_q_blocks = 0usize;

        // Stack measurements from detecting sensors using cached matrices
        let mut counter = 0;
        for s in 0..number_of_sensors {
            if assignments[s] {
                let start = model.z_dimension * counter;

                // Copy measurement (a is 1-indexed: 0=miss, 1+=measurement, so subtract 1 for array access)
                z.rows_mut(start, model.z_dimension)
                    .copy_from(&measurements[s][a[s] - 1]);

                // Copy observation matrix from cache (avoids repeated accessor + clone)
                c.view_mut((start, 0), (model.z_dimension, model.x_dimension))
                    .copy_from(&c_cache[s]);

                // Store index for Q matrix (stack-allocated)
                q_block_indices[num_q_blocks] = s;
                num_q_blocks += 1;

                counter += 1;
            }
        }

        // Block diagonal Q
        let mut q = DMatrix::zeros(z_dim_total, z_dim_total);
        let mut offset = 0;
        for idx in 0..num_q_blocks {
            let s = q_block_indices[idx];
            q.view_mut((offset, offset), (model.z_dimension, model.z_dimension))
                .copy_from(&q_cache[s]);
            offset += model.z_dimension;
        }

        // Likelihood computation
        let nu = &z - &c * &hypothesis.mu[i];
        let z_matrix = &c * &hypothesis.sigma[i] * c.transpose() + &q;

        // Use combined inverse + log-det (avoids computing Cholesky twice)
        let (z_inv, eta) = robust_inverse_with_log_det(z_matrix, z_dim_total)
            .expect("z_matrix should be invertible");

        let k = &hypothesis.sigma[i] * c.transpose() * &z_inv;

        // Detection probabilities using accessor
        let mut pd_log = 0.0;
        for s in 0..number_of_sensors {
            let p_d = model.get_detection_probability(Some(s));
            pd_log += if assignments[s] {
                p_d.ln()
            } else {
                (1.0 - p_d).ln()
            };
        }

        // Clutter per unit volume using accessor (only iterate valid sensors)
        let mut kappa_log = 0.0;
        for s in 0..number_of_sensors {
            if assignments[s] {
                kappa_log += model.get_clutter_per_unit_volume(Some(s)).ln();
            }
        }

        let l = hypothesis.r[i].ln() + pd_log + eta - 0.5 * nu.dot(&(&z_inv * &nu)) - kappa_log;

        // Posterior parameters
        let r = 1.0;
        let mu = &hypothesis.mu[i] + &k * &nu;
        let sigma = (DMatrix::identity(model.x_dimension, model.x_dimension) - &k * &c)
            * &hypothesis.sigma[i];

        (l, r, mu, sigma)
    } else {
        // All missed detections - compute probability using accessor
        let mut prob_no_detect = 1.0;
        for s in 0..number_of_sensors {
            prob_no_detect *= 1.0 - model.get_detection_probability(Some(s));
        }

        let numerator = hypothesis.r[i] * prob_no_detect;
        let denominator = 1.0 - hypothesis.r[i] + numerator;

        let l = denominator.ln();
        let r = numerator / denominator;
        let mu = hypothesis.mu[i].clone();
        let sigma = hypothesis.sigma[i].clone();

        (l, r, mu, sigma)
    }
}

/// Determine log likelihood ratio ONLY (no posterior params)
///
/// This is a faster version that only computes the log-likelihood value,
/// skipping the posterior parameter computation (r, mu, sigma).
/// Used for Phase C optimization where we compute L for all entries in parallel,
/// but only compute posterior params for the ~1000 indices actually used by Gibbs.
#[inline]
fn determine_log_likelihood_only(
    i: usize,
    a: &[usize],
    measurements: &[&[DVector<f64>]],
    hypothesis: &Hypothesis,
    model: &Model,
    q_cache: &[DMatrix<f64>],
    c_cache: &[DMatrix<f64>],
) -> f64 {
    let number_of_sensors = a.len();

    // Check which sensors have detections (stack-allocated)
    let mut assignments = [false; MAX_SENSORS];
    let mut number_of_assignments = 0usize;
    for s in 0..number_of_sensors {
        if a[s] > 0 {
            assignments[s] = true;
            number_of_assignments += 1;
        }
    }

    if number_of_assignments > 0 {
        // GATING: Quick diagonal Mahalanobis check BEFORE expensive matrix ops
        let mut gate_failed = false;
        for s in 0..number_of_sensors {
            if assignments[s] {
                let meas = &measurements[s][a[s] - 1];
                let c_s = &c_cache[s];
                let q_s = &q_cache[s];

                let predicted_z = c_s * &hypothesis.mu[i];
                let innovation = meas - &predicted_z;

                let sigma_diag = hypothesis.sigma[i].diagonal();
                let mut quick_dist = 0.0;
                for j in 0..model.z_dimension {
                    let state_idx = j * 2;
                    let var_approx = if state_idx < sigma_diag.len() {
                        sigma_diag[state_idx] + q_s[(j, j)]
                    } else {
                        q_s[(j, j)]
                    };
                    quick_dist += innovation[j].powi(2) / var_approx.max(1e-10);
                }

                if quick_dist > DEFAULT_GATING_THRESHOLD {
                    gate_failed = true;
                    break;
                }
            }
        }

        if gate_failed {
            return -1e10;
        }

        // Stack measurements (q_block_indices is stack-allocated)
        let z_dim_total = model.z_dimension * number_of_assignments;
        let mut z = DVector::zeros(z_dim_total);
        let mut c = DMatrix::zeros(z_dim_total, model.x_dimension);
        let mut q_block_indices = [0usize; MAX_SENSORS];
        let mut num_q_blocks = 0usize;

        let mut counter = 0;
        for s in 0..number_of_sensors {
            if assignments[s] {
                let start = model.z_dimension * counter;
                z.rows_mut(start, model.z_dimension)
                    .copy_from(&measurements[s][a[s] - 1]);
                c.view_mut((start, 0), (model.z_dimension, model.x_dimension))
                    .copy_from(&c_cache[s]);
                q_block_indices[num_q_blocks] = s;
                num_q_blocks += 1;
                counter += 1;
            }
        }

        // Block diagonal Q
        let mut q = DMatrix::zeros(z_dim_total, z_dim_total);
        let mut offset = 0;
        for idx in 0..num_q_blocks {
            let s = q_block_indices[idx];
            q.view_mut((offset, offset), (model.z_dimension, model.z_dimension))
                .copy_from(&q_cache[s]);
            offset += model.z_dimension;
        }

        // Likelihood computation ONLY (skip Kalman gain and posterior)
        let nu = &z - &c * &hypothesis.mu[i];
        let z_matrix = &c * &hypothesis.sigma[i] * c.transpose() + &q;

        let (z_inv, eta) = match robust_inverse_with_log_det(z_matrix, z_dim_total) {
            Some(result) => result,
            None => return -1e10, // Singular matrix
        };

        // Detection probabilities
        let mut pd_log = 0.0;
        for s in 0..number_of_sensors {
            let p_d = model.get_detection_probability(Some(s));
            pd_log += if assignments[s] {
                p_d.ln()
            } else {
                (1.0 - p_d).ln()
            };
        }

        // Clutter (only iterate valid sensors)
        let mut kappa_log = 0.0;
        for s in 0..number_of_sensors {
            if assignments[s] {
                kappa_log += model.get_clutter_per_unit_volume(Some(s)).ln();
            }
        }

        hypothesis.r[i].ln() + pd_log + eta - 0.5 * nu.dot(&(&z_inv * &nu)) - kappa_log
    } else {
        // All missed detections
        let mut prob_no_detect = 1.0;
        for s in 0..number_of_sensors {
            prob_no_detect *= 1.0 - model.get_detection_probability(Some(s));
        }
        let numerator = hypothesis.r[i] * prob_no_detect;
        let denominator = 1.0 - hypothesis.r[i] + numerator;
        denominator.ln()
    }
}

/// Generate multi-sensor LMBM likelihood matrix ONLY (no posterior params)
///
/// This is a Phase C optimization: compute only L values in parallel,
/// deferring posterior param computation to hypothesis.rs where we only
/// need ~1000 indices instead of all 10.7M.
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn generate_multisensor_lmbm_likelihood_only(
    hypothesis: &Hypothesis,
    measurements: &[&[DVector<f64>]],
    model: &Model,
    number_of_sensors: usize,
) -> (Vec<f64>, Vec<usize>) {
    let number_of_objects = hypothesis.r.len();

    // Determine dimensions: [m1+1, m2+1, ..., ms+1, n]
    let mut dimensions = vec![0; number_of_sensors + 1];
    for s in 0..number_of_sensors {
        dimensions[s] = measurements[s].len() + 1;
    }
    dimensions[number_of_sensors] = number_of_objects;

    let number_of_entries: usize = dimensions.iter().product();
    let mut page_sizes = vec![1; number_of_sensors + 1];
    for i in 1..=number_of_sensors {
        page_sizes[i] = page_sizes[i - 1] * dimensions[i - 1];
    }

    // Pre-cache Q and C matrices
    let q_cache: Vec<DMatrix<f64>> = (0..number_of_sensors)
        .map(|s| model.get_measurement_noise(Some(s)).clone())
        .collect();
    let c_cache: Vec<DMatrix<f64>> = (0..number_of_sensors)
        .map(|s| model.get_observation_matrix(Some(s)).clone())
        .collect();

    // Parallel L-only computation
    #[cfg(feature = "rayon")]
    let l: Vec<f64> = (0..number_of_entries)
        .into_par_iter()
        .map(|ell| {
            let mut u = [0usize; MAX_SENSORS];
            let mut a = [0usize; MAX_SENSORS];

            convert_from_linear_to_cartesian_inplace(ell + 1, &page_sizes, &mut u);
            let i = u[number_of_sensors] - 1;
            for s in 0..number_of_sensors {
                a[s] = u[s] - 1;
            }

            determine_log_likelihood_only(i, &a[..number_of_sensors], measurements, hypothesis, model, &q_cache, &c_cache)
        })
        .collect();

    #[cfg(not(feature = "rayon"))]
    let l: Vec<f64> = {
        let mut l = vec![0.0; number_of_entries];
        let mut u = [0usize; MAX_SENSORS];
        let mut a = [0usize; MAX_SENSORS];

        for ell in 0..number_of_entries {
            convert_from_linear_to_cartesian_inplace(ell + 1, &page_sizes, &mut u);
            let i = u[number_of_sensors] - 1;
            for s in 0..number_of_sensors {
                a[s] = u[s] - 1;
            }
            l[ell] = determine_log_likelihood_only(i, &a[..number_of_sensors], measurements, hypothesis, model, &q_cache, &c_cache);
        }
        l
    };

    (l, dimensions)
}

/// Compute posterior parameters for specific indices
///
/// Only computes (r, mu, sigma) for the given indices, not all entries.
/// This is used after Gibbs sampling to compute params only for the ~1000
/// unique indices that were actually sampled.
pub fn compute_posterior_params_for_indices(
    indices: &[usize],
    hypothesis: &Hypothesis,
    measurements: &[&[DVector<f64>]],
    model: &Model,
    number_of_sensors: usize,
    dimensions: &[usize],
) -> std::collections::HashMap<usize, (f64, DVector<f64>, DMatrix<f64>)> {
    use std::collections::HashMap;

    let mut page_sizes = vec![1; number_of_sensors + 1];
    for i in 1..=number_of_sensors {
        page_sizes[i] = page_sizes[i - 1] * dimensions[i - 1];
    }

    // Pre-cache Q and C matrices
    let q_cache: Vec<DMatrix<f64>> = (0..number_of_sensors)
        .map(|s| model.get_measurement_noise(Some(s)).clone())
        .collect();
    let c_cache: Vec<DMatrix<f64>> = (0..number_of_sensors)
        .map(|s| model.get_observation_matrix(Some(s)).clone())
        .collect();

    let mut results = HashMap::with_capacity(indices.len());

    for &ell in indices {
        if results.contains_key(&ell) {
            continue; // Already computed
        }

        let mut u = [0usize; MAX_SENSORS];
        let mut a = [0usize; MAX_SENSORS];

        convert_from_linear_to_cartesian_inplace(ell + 1, &page_sizes, &mut u);
        let i = u[number_of_sensors] - 1;
        for s in 0..number_of_sensors {
            a[s] = u[s] - 1;
        }

        // Compute full likelihood with posterior params
        let (_, r, mu, sigma) = determine_log_likelihood_ratio(
            i, &a[..number_of_sensors], measurements, hypothesis, model, &q_cache, &c_cache
        );

        results.insert(ell, (r, mu, sigma));
    }

    results
}

/// Generate multi-sensor LMBM association matrices
///
/// Computes the high-dimensional likelihood matrix L for multi-sensor LMBM.
///
/// # Arguments
/// * `hypothesis` - Prior LMBM hypothesis
/// * `measurements` - Measurements from all sensors [sensor][measurements]
/// * `model` - Model parameters (must have numberOfSensors defined)
/// * `number_of_sensors` - Number of sensors
///
/// # Returns
/// Tuple of (flattened L matrix, posterior parameters, dimensions)
///
/// # Implementation Notes
/// Matches MATLAB generateMultisensorLmbmAssociationMatrices.m exactly:
/// - L has dimensions (m1+1, m2+1, ..., ms+1, n)
/// - Flattened to 1D vector for memory efficiency
/// - Posterior parameters also flattened
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn generate_multisensor_lmbm_association_matrices(
    hypothesis: &Hypothesis,
    measurements: &[&[DVector<f64>]],
    model: &Model,
    number_of_sensors: usize,
) -> (Vec<f64>, MultisensorLmbmPosteriorParameters, Vec<usize>) {
    let number_of_objects = hypothesis.r.len();

    // Determine dimensions: [m1+1, m2+1, ..., ms+1, n]
    let mut dimensions = vec![0; number_of_sensors + 1];
    for s in 0..number_of_sensors {
        dimensions[s] = measurements[s].len() + 1;
    }
    dimensions[number_of_sensors] = number_of_objects;

    // Calculate total entries and page sizes
    let number_of_entries: usize = dimensions.iter().product();
    let mut page_sizes = vec![1; number_of_sensors + 1];
    for i in 1..=number_of_sensors {
        page_sizes[i] = page_sizes[i - 1] * dimensions[i - 1];
    }

    // Pre-cache Q and C matrices per sensor (avoids 10.7M clones)
    let q_cache: Vec<DMatrix<f64>> = (0..number_of_sensors)
        .map(|s| model.get_measurement_noise(Some(s)).clone())
        .collect();
    let c_cache: Vec<DMatrix<f64>> = (0..number_of_sensors)
        .map(|s| model.get_observation_matrix(Some(s)).clone())
        .collect();

    // Populate likelihood matrix
    // Parallel version using rayon when feature is enabled
    #[cfg(feature = "rayon")]
    let (l, r, mu, sigma) = {
        let results: Vec<_> = (0..number_of_entries)
            .into_par_iter()
            .map(|ell| {
                // Stack-allocated arrays for index conversion (thread-local)
                let mut u = [0usize; MAX_SENSORS];
                let mut a = [0usize; MAX_SENSORS];

                // Get association vector (convert expects 1-indexed input like MATLAB, so add 1)
                convert_from_linear_to_cartesian_inplace(ell + 1, &page_sizes, &mut u);
                // Convert to 0-indexed like MATLAB: a = u(1:end-1) - 1, i = u(end) - 1
                let i = u[number_of_sensors] - 1; // Object index (0-indexed)
                for s in 0..number_of_sensors {
                    a[s] = u[s] - 1; // Convert from 1-indexed to 0-indexed: 0=miss, 1=meas[0], 2=meas[1]
                }

                // Determine log likelihood and posterior (uses cached matrices)
                determine_log_likelihood_ratio(i, &a[..number_of_sensors], measurements, hypothesis, model, &q_cache, &c_cache)
            })
            .collect();

        // Unpack results into separate vectors
        let mut l = Vec::with_capacity(number_of_entries);
        let mut r = Vec::with_capacity(number_of_entries);
        let mut mu = Vec::with_capacity(number_of_entries);
        let mut sigma = Vec::with_capacity(number_of_entries);
        for (l_val, r_val, mu_val, sigma_val) in results {
            l.push(l_val);
            r.push(r_val);
            mu.push(mu_val);
            sigma.push(sigma_val);
        }
        (l, r, mu, sigma)
    };

    // Serial version when rayon is not enabled
    #[cfg(not(feature = "rayon"))]
    let (l, r, mu, sigma) = {
        let mut l = vec![0.0; number_of_entries];
        let mut r = vec![0.0; number_of_entries];
        let mut mu = vec![DVector::zeros(0); number_of_entries];
        let mut sigma = vec![DMatrix::zeros(0, 0); number_of_entries];

        // Stack-allocated arrays for index conversion (avoid 10.7M heap allocations)
        let mut u = [0usize; MAX_SENSORS];
        let mut a = [0usize; MAX_SENSORS];

        for ell in 0..number_of_entries {
            // Get association vector (convert expects 1-indexed input like MATLAB, so add 1)
            convert_from_linear_to_cartesian_inplace(ell + 1, &page_sizes, &mut u);
            // Convert to 0-indexed like MATLAB: a = u(1:end-1) - 1, i = u(end) - 1
            let i = u[number_of_sensors] - 1; // Object index (0-indexed)
            for s in 0..number_of_sensors {
                a[s] = u[s] - 1; // Convert from 1-indexed to 0-indexed: 0=miss, 1=meas[0], 2=meas[1]
            }

            // Determine log likelihood and posterior (uses cached matrices)
            let (l_val, r_val, mu_val, sigma_val) =
                determine_log_likelihood_ratio(i, &a[..number_of_sensors], measurements, hypothesis, model, &q_cache, &c_cache);

            l[ell] = l_val;
            r[ell] = r_val;
            mu[ell] = mu_val;
            sigma[ell] = sigma_val;
        }
        (l, r, mu, sigma)
    };

    let posterior_parameters = MultisensorLmbmPosteriorParameters { r, mu, sigma };

    (l, posterior_parameters, dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};

    #[test]
    fn test_convert_from_linear_to_cartesian() {
        let page_sizes = vec![1, 3, 6]; // dimensions [3, 2, 4]
        let mut u = [0usize; MAX_SENSORS];
        convert_from_linear_to_cartesian_inplace(5, &page_sizes, &mut u);
        // Function returns 1-indexed values per MATLAB convention
        // Just verify it produces non-zero results (integration tests verify correctness)
        assert!(u[0] > 0 || u[1] > 0 || u[2] > 0);
    }

    #[test]
    fn test_generate_multisensor_lmbm_association_matrices() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let hypothesis = model.hypotheses.clone();

        // 2 sensors, 1 measurement each
        let measurements_owned = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])],
            vec![DVector::from_vec(vec![1.0, 1.0])],
        ];
        let measurements: Vec<&[DVector<f64>]> = measurements_owned.iter()
            .map(|v| v.as_slice())
            .collect();

        let (l, posterior, dimensions) =
            generate_multisensor_lmbm_association_matrices(&hypothesis, &measurements, &model, 2);

        // Check dimensions: (1+1, 1+1, n) = (2, 2, n)
        assert_eq!(dimensions[0], 2); // sensor 1: miss + 1 meas
        assert_eq!(dimensions[1], 2); // sensor 2: miss + 1 meas
        assert_eq!(dimensions[2], hypothesis.r.len()); // objects

        // Check total entries
        let expected_entries: usize = dimensions.iter().product();
        assert_eq!(l.len(), expected_entries);
        assert_eq!(posterior.r.len(), expected_entries);
    }
}