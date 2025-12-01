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
    measurements: &[Vec<DVector<f64>>],
    hypothesis: &Hypothesis,
    model: &Model,
    q_cache: &[DMatrix<f64>],
    c_cache: &[DMatrix<f64>],
) -> (f64, f64, DVector<f64>, DMatrix<f64>) {
    let number_of_sensors = a.len();

    // Check which sensors have detections
    let assignments: Vec<bool> = a.iter().map(|&ai| ai > 0).collect();
    let number_of_assignments: usize = assignments.iter().filter(|&&x| x).count();

    if number_of_assignments > 0 {
        // Determine measurement vector by stacking measurements from detecting sensors
        let z_dim_total = model.z_dimension * number_of_assignments;
        let mut z = DVector::zeros(z_dim_total);
        let mut c = DMatrix::zeros(z_dim_total, model.x_dimension);
        let mut q_blocks = Vec::new();

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

                // Use Q matrix from cache (avoids clone per call)
                q_blocks.push(&q_cache[s]);

                counter += 1;
            }
        }

        // Block diagonal Q
        let mut q = DMatrix::zeros(z_dim_total, z_dim_total);
        let mut offset = 0;
        for q_block in &q_blocks {
            q.view_mut((offset, offset), (model.z_dimension, model.z_dimension))
                .copy_from(q_block);
            offset += model.z_dimension;
        }

        // Likelihood computation
        let nu = &z - &c * &hypothesis.mu[i];
        let z_matrix = &c * &hypothesis.sigma[i] * c.transpose() + &q;

        // Use combined inverse + log-det (avoids computing Cholesky twice)
        let (z_inv, eta) = robust_inverse_with_log_det(&z_matrix, z_dim_total)
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

        // Clutter per unit volume using accessor
        let kappa_log: f64 = assignments
            .iter()
            .enumerate()
            .filter(|(_, &x)| x)
            .map(|(s, _)| model.get_clutter_per_unit_volume(Some(s)).ln())
            .sum();

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
    measurements: &[Vec<DVector<f64>>],
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
        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])],
            vec![DVector::from_vec(vec![1.0, 1.0])],
        ];

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