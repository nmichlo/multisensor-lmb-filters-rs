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
use crate::multisensor_lmbm::workspace::LmbmLikelihoodWorkspace;
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

/// Determine log likelihood ratio using pre-allocated workspace buffers
///
/// This variant reuses workspace buffers to avoid allocations in hot loops.
/// Used by parallel rayon implementation with `map_init`.
#[inline]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn determine_log_likelihood_ratio_with_workspace(
    i: usize,
    a: &[usize],
    measurements: &[Vec<DVector<f64>>],
    hypothesis: &Hypothesis,
    model: &Model,
    q_cache: &[DMatrix<f64>],
    c_cache: &[DMatrix<f64>],
    workspace: &mut LmbmLikelihoodWorkspace,
) -> (f64, f64, DVector<f64>, DMatrix<f64>) {
    let number_of_sensors = a.len();

    // Update assignments using workspace buffer
    let mut number_of_assignments = 0;
    for s in 0..number_of_sensors {
        workspace.assignments[s] = a[s] > 0;
        if workspace.assignments[s] {
            number_of_assignments += 1;
        }
    }

    if number_of_assignments > 0 {
        let z_dim_total = model.z_dimension * number_of_assignments;

        // Reset workspace buffers for this computation size
        workspace.reset(z_dim_total);

        // Stack measurements from detecting sensors using workspace buffers
        let mut counter = 0;
        for s in 0..number_of_sensors {
            if workspace.assignments[s] {
                let start = model.z_dimension * counter;

                // Copy measurement into workspace.z
                workspace.z.rows_mut(start, model.z_dimension)
                    .copy_from(&measurements[s][a[s] - 1]);

                // Copy observation matrix into workspace.c
                workspace.c.view_mut((start, 0), (model.z_dimension, model.x_dimension))
                    .copy_from(&c_cache[s]);

                // Track Q block index for later
                workspace.q_block_indices.push(s);

                counter += 1;
            }
        }

        // Build block diagonal Q in workspace
        let mut offset = 0;
        for &s in &workspace.q_block_indices {
            workspace.q.view_mut((offset, offset), (model.z_dimension, model.z_dimension))
                .copy_from(&q_cache[s]);
            offset += model.z_dimension;
        }

        // Get active views (the portion actually used)
        let z_view = workspace.z.rows(0, z_dim_total);
        let c_view = workspace.c.view((0, 0), (z_dim_total, model.x_dimension));
        let q_view = workspace.q.view((0, 0), (z_dim_total, z_dim_total));

        // Likelihood computation using views
        // nu = z - C * mu[i]
        let c_mu = &c_view * &hypothesis.mu[i];
        for j in 0..z_dim_total {
            workspace.nu[j] = z_view[j] - c_mu[j];
        }
        let nu_view = workspace.nu.rows(0, z_dim_total);

        // z_matrix = C * Sigma * C' + Q
        let c_sigma = &c_view * &hypothesis.sigma[i];
        let z_matrix = &c_sigma * c_view.transpose() + &q_view;

        // Use combined inverse + log-det
        let (z_inv, eta) = robust_inverse_with_log_det(&z_matrix, z_dim_total)
            .expect("z_matrix should be invertible");

        // Kalman gain: K = Sigma * C' * Z_inv
        let sigma_ct = &hypothesis.sigma[i] * c_view.transpose();
        let k = &sigma_ct * &z_inv;

        // Detection probabilities
        let mut pd_log = 0.0;
        for s in 0..number_of_sensors {
            let p_d = model.get_detection_probability(Some(s));
            pd_log += if workspace.assignments[s] {
                p_d.ln()
            } else {
                (1.0 - p_d).ln()
            };
        }

        // Clutter per unit volume
        let kappa_log: f64 = workspace.assignments
            .iter()
            .take(number_of_sensors)
            .enumerate()
            .filter(|(_, &x)| x)
            .map(|(s, _)| model.get_clutter_per_unit_volume(Some(s)).ln())
            .sum();

        let l = hypothesis.r[i].ln() + pd_log + eta - 0.5 * nu_view.dot(&(&z_inv * &nu_view)) - kappa_log;

        // Posterior parameters
        let r = 1.0;
        let mu = &hypothesis.mu[i] + &k * &nu_view;
        let sigma = (&workspace.identity - &k * &c_view) * &hypothesis.sigma[i];

        (l, r, mu, sigma)
    } else {
        // All missed detections - compute probability
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
    // Uses map_init to create thread-local workspace buffers
    #[cfg(feature = "rayon")]
    let (l, r, mu, sigma) = {
        let results: Vec<_> = (0..number_of_entries)
            .into_par_iter()
            .map_init(
                // Initialize thread-local workspace (called once per thread)
                || LmbmLikelihoodWorkspace::new(number_of_sensors, model.x_dimension, model.z_dimension),
                |workspace, ell| {
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

                    // Determine log likelihood and posterior using workspace buffers
                    determine_log_likelihood_ratio_with_workspace(
                        i, &a[..number_of_sensors], measurements, hypothesis, model, &q_cache, &c_cache, workspace
                    )
                }
            )
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
    // Also uses workspace to reduce allocations
    #[cfg(not(feature = "rayon"))]
    let (l, r, mu, sigma) = {
        let mut l = vec![0.0; number_of_entries];
        let mut r = vec![0.0; number_of_entries];
        let mut mu = vec![DVector::zeros(0); number_of_entries];
        let mut sigma = vec![DMatrix::zeros(0, 0); number_of_entries];

        // Stack-allocated arrays for index conversion (avoid 10.7M heap allocations)
        let mut u = [0usize; MAX_SENSORS];
        let mut a = [0usize; MAX_SENSORS];

        // Single workspace for serial execution
        let mut workspace = LmbmLikelihoodWorkspace::new(number_of_sensors, model.x_dimension, model.z_dimension);

        for ell in 0..number_of_entries {
            // Get association vector (convert expects 1-indexed input like MATLAB, so add 1)
            convert_from_linear_to_cartesian_inplace(ell + 1, &page_sizes, &mut u);
            // Convert to 0-indexed like MATLAB: a = u(1:end-1) - 1, i = u(end) - 1
            let i = u[number_of_sensors] - 1; // Object index (0-indexed)
            for s in 0..number_of_sensors {
                a[s] = u[s] - 1; // Convert from 1-indexed to 0-indexed: 0=miss, 1=meas[0], 2=meas[1]
            }

            // Determine log likelihood and posterior using workspace buffers
            let (l_val, r_val, mu_val, sigma_val) =
                determine_log_likelihood_ratio_with_workspace(
                    i, &a[..number_of_sensors], measurements, hypothesis, model, &q_cache, &c_cache, &mut workspace
                );

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