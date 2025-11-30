//! Multi-sensor LMBM association matrices
//!
//! Implements association matrix generation for multi-sensor LMBM filter.
//! Uses a high-dimensional likelihood matrix for joint sensor-object associations.
//! Matches MATLAB generateMultisensorLmbmAssociationMatrices.m exactly.
//!
//! WARNING: This implementation can be very memory intensive for large numbers
//! of objects and sensors, matching the MATLAB behavior.

use crate::common::types::{DMatrix, DVector, Hypothesis, Model};
use std::f64::consts::PI;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Determinant, Inverse, Solve, SVD, UPLO};

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

/// Convert linear index to Cartesian coordinates
///
/// # Arguments
/// * `ell` - Linear index (1-indexed in MATLAB, 0-indexed here after conversion)
/// * `page_sizes` - Cumulative product of dimensions
///
/// # Returns
/// Cartesian coordinates (0-indexed)
fn convert_from_linear_to_cartesian(mut ell: usize, page_sizes: &[usize]) -> Vec<usize> {
    let m = page_sizes.len();
    let mut u = vec![0; m];

    for i in 0..m {
        let j = m - i - 1;
        let zeta = ell / page_sizes[j];
        let eta = ell % page_sizes[j];
        u[j] = zeta + if eta != 0 { 1 } else { 0 };
        ell = ell - page_sizes[j] * (zeta - if eta == 0 { 1 } else { 0 });
    }

    u
}

/// Determine log likelihood ratio for a given object-sensor association
///
/// # Arguments
/// * `i` - Object index
/// * `a` - Association vector (measurement indices per sensor, 0 = miss)
/// * `measurements` - Measurements from all sensors
/// * `hypothesis` - Prior hypothesis
/// * `model` - Model parameters
///
/// # Returns
/// Tuple of (log likelihood, existence prob, mean, covariance)
fn determine_log_likelihood_ratio(
    i: usize,
    a: &[usize],
    measurements: &[Vec<DVector<f64>>],
    hypothesis: &Hypothesis,
    model: &Model,
) -> (f64, f64, DVector<f64>, DMatrix<f64>) {
    let number_of_sensors = a.len();

    // Check which sensors have detections
    let assignments: Vec<bool> = a.iter().map(|&ai| ai > 0).collect();
    let number_of_assignments: usize = assignments.iter().filter(|&&x| x).count();

    if number_of_assignments > 0 {
        // Determine measurement vector by stacking measurements from detecting sensors
        let z_dim_total = model.z_dimension * number_of_assignments;
        let mut z = Array1::zeros(z_dim_total);
        let mut c = Array2::zeros((z_dim_total, model.x_dimension));
        let mut q_blocks = Vec::new();

        // Get per-sensor observation matrices and noise covariances
        let c_multisensor = model.c_multisensor.as_ref()
            .expect("Multisensor model must have per-sensor observation matrices");
        let q_multisensor = model.q_multisensor.as_ref()
            .expect("Multisensor model must have per-sensor measurement noise covariances");

        let mut counter = 0;
        for s in 0..number_of_sensors {
            if assignments[s] {
                let start = model.z_dimension * counter;
                let end = start + model.z_dimension;

                // Copy measurement (a is 1-indexed: 0=miss, 1+=measurement, so subtract 1 for array access)
                z.slice_mut(ndarray::s![start..end]).assign(&measurements[s][a[s] - 1]);

                // Copy observation matrix for sensor s
                c.slice_mut(ndarray::s![start..end, ..]).assign(&c_multisensor[s]);

                // Collect Q matrix for sensor s
                q_blocks.push(q_multisensor[s].clone());

                counter += 1;
            }
        }

        // Block diagonal Q
        let mut q = Array2::zeros((z_dim_total, z_dim_total));
        let mut offset = 0;
        for q_block in &q_blocks {
            let end_offset = offset + model.z_dimension;
            q.slice_mut(ndarray::s![offset..end_offset, offset..end_offset]).assign(q_block);
            offset += model.z_dimension;
        }

        // Likelihood computation
        let nu = &z - c.dot(&hypothesis.mu[i]);
        let z_matrix = c.dot(&hypothesis.sigma[i]).dot(&c.t()) + &q;

        let z_inv = z_matrix.clone().inv().expect("Cannot invert innovation covariance z_matrix");

        let k = hypothesis.sigma[i].dot(&c.t()).dot(&z_inv);
        // eta = -0.5 * log(det(2*pi*Z))
        // For n×n matrix: det(c*A) = c^n * det(A)
        // So det(2*pi*Z) = (2*pi)^n * det(Z) where n = z_dim_total
        let z_matrix_det = z_matrix.det().expect("Innovation covariance z_matrix should be positive definite");
        let eta = -0.5 * ((2.0 * PI).powi(z_dim_total as i32) * z_matrix_det).ln();

        // Detection probabilities (use per-sensor values for multisensor)
        let detection_probs = model.detection_probability_multisensor.as_ref()
            .expect("Multisensor model must have per-sensor detection probabilities");

        let mut pd_log = 0.0;
        for s in 0..number_of_sensors {
            pd_log += if assignments[s] {
                detection_probs[s].ln()
            } else {
                (1.0 - detection_probs[s]).ln()
            };
        }

        // Clutter per unit volume (use per-sensor values for multisensor)
        let clutter_per_unit_volumes = model.clutter_per_unit_volume_multisensor.as_ref()
            .expect("Multisensor model must have per-sensor clutter per unit volume");

        let kappa_log: f64 = assignments
            .iter()
            .enumerate()
            .filter(|(_, &x)| x)
            .map(|(s, _)| clutter_per_unit_volumes[s].ln())
            .sum();

        let quadratic = nu.dot(&z_inv.dot(&nu));
        let l = hypothesis.r[i].ln() + pd_log + eta - 0.5 * quadratic - kappa_log;

        // Posterior parameters
        let r = 1.0;
        let mu = &hypothesis.mu[i] + k.dot(&nu);
        let sigma = (Array2::eye(model.x_dimension) - k.dot(&c))
            .dot(&hypothesis.sigma[i]);

        (l, r, mu, sigma)
    } else {
        // All missed detections (use per-sensor detection probabilities)
        let detection_probs = model.detection_probability_multisensor.as_ref()
            .expect("Multisensor model must have per-sensor detection probabilities");

        let mut prob_no_detect = 1.0;
        for s in 0..number_of_sensors {
            prob_no_detect *= 1.0 - detection_probs[s];
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
pub fn generate_multisensor_lmbm_association_matrices(
    hypothesis: &Hypothesis,
    measurements: &[Vec<DVector<f64>>],
    _model: &Model,
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

    // Preallocate flattened arrays
    let mut l = vec![0.0; number_of_entries];
    let mut r = vec![0.0; number_of_entries];
    let mut mu = vec![Array1::zeros(0); number_of_entries];
    let mut sigma = vec![Array2::zeros((0, 0)); number_of_entries];

    // Populate likelihood matrix
    for ell in 0..number_of_entries {
        // Get association vector (convert expects 1-indexed input like MATLAB, so add 1)
        let u = convert_from_linear_to_cartesian(ell + 1, &page_sizes);
        // Convert to 0-indexed like MATLAB: a = u(1:end-1) - 1, i = u(end) - 1
        let i = u[number_of_sensors] - 1; // Object index (0-indexed)
        let a: Vec<usize> = u[0..number_of_sensors].iter()
            .map(|&x| x - 1) // Convert from 1-indexed to 0-indexed: 0=miss, 1=meas[0], 2=meas[1]
            .collect();

        // Determine log likelihood and posterior
        let (l_val, r_val, mu_val, sigma_val) =
            determine_log_likelihood_ratio(i, &a, measurements, hypothesis, _model);

        l[ell] = l_val;
        r[ell] = r_val;
        mu[ell] = mu_val;
        sigma[ell] = sigma_val;
    }

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
        let u = convert_from_linear_to_cartesian(5, &page_sizes);
        // This should give a specific Cartesian coordinate
        assert_eq!(u.len(), 3);
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