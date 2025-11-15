//! Multi-sensor LMB association matrices
//!
//! Implements association matrix generation for multi-sensor LMB filters.
//! Matches MATLAB generateLmbSensorAssociationMatrices.m exactly.
//!
//! NOTE: Full multi-sensor support requires extending Model to include:
//! - `number_of_sensors: usize`
//! - `c: Vec<DMatrix<f64>>` (one per sensor, currently single matrix)
//! - `q: Vec<DMatrix<f64>>` (one per sensor, currently single matrix)
//! - `detection_probability: Vec<f64>` (one per sensor, currently single f64)
//! - `clutter_per_unit_volume: Vec<f64>` (one per sensor, currently single f64)

use crate::common::types::{Model, Object};
use nalgebra::{DMatrix, DVector};

/// Association matrices for multi-sensor LMB data association
#[derive(Debug, Clone)]
pub struct LmbAssociationMatrices {
    /// Existence probabilities (n x 1)
    pub r: Vec<f64>,
    /// LBP psi matrix (n x m)
    pub psi: DMatrix<f64>,
    /// LBP phi vector (n x 1)
    pub phi: DVector<f64>,
    /// LBP eta vector (n x 1)
    pub eta: DVector<f64>,
    /// Gibbs probability matrix (n x m)
    pub p: DMatrix<f64>,
    /// Gibbs log-likelihood matrix (n x (m+1))
    pub l: DMatrix<f64>,
    /// Gibbs existence ratio matrix (n x (m+1))
    pub gibbs_r: DMatrix<f64>,
    /// Cost matrix for Murty's algorithm (n x m)
    pub cost: DMatrix<f64>,
}

/// Posterior parameters for multi-sensor LMB update
#[derive(Debug, Clone)]
pub struct LmbPosteriorParameters {
    /// Posterior weights (n x (m+1) x num_gm_components) - normalized, not log-space
    /// For each object, rows are [miss, meas1, ..., measm]
    pub w: Vec<Vec<Vec<f64>>>,
    /// Posterior means (n x (m+1) x num_gm_components)
    pub mu: Vec<Vec<Vec<DVector<f64>>>>,
    /// Posterior covariances (n x (m+1) x num_gm_components)
    pub sigma: Vec<Vec<Vec<DMatrix<f64>>>>,
}

/// Generate LMB sensor association matrices for a specific sensor
///
/// Computes association matrices for a single sensor in a multi-sensor system.
/// Handles Gaussian mixture components in the prior distribution.
///
/// # Arguments
/// * `objects` - Prior LMB Bernoulli components (may have GM components)
/// * `measurements` - Measurements from this sensor
/// * `model` - Model parameters
/// * `sensor_idx` - Sensor index (0-based, for future multi-sensor Model support)
///
/// # Returns
/// Tuple of (association matrices, posterior parameters)
///
/// # Implementation Notes
/// Matches MATLAB generateLmbSensorAssociationMatrices.m exactly:
/// 1. For each object and GM component, compute likelihood ratios
/// 2. Aggregate likelihoods across GM components
/// 3. Build association matrices for LBP, Gibbs, and Murty's algorithms
/// 4. Compute posterior parameters for each measurement
///
/// # Multi-sensor Note
/// Currently uses single-sensor Model parameters (c, q, detection_probability, etc.)
/// For true multi-sensor support, Model needs per-sensor parameters indexed by sensor_idx
pub fn generate_lmb_sensor_association_matrices(
    objects: &[Object],
    measurements: &[DVector<f64>],
    model: &Model,
    _sensor_idx: usize, // Reserved for future multi-sensor Model support
) -> (LmbAssociationMatrices, LmbPosteriorParameters) {
    let number_of_objects = objects.len();
    let number_of_measurements = measurements.len();

    // Auxiliary matrices
    let mut l_matrix = DMatrix::zeros(number_of_objects, number_of_measurements);
    let mut phi = vec![0.0; number_of_objects];
    let mut eta = vec![0.0; number_of_objects];

    // Posterior parameters (weights are log-space, one row per measurement+miss)
    let mut posterior_w = Vec::with_capacity(number_of_objects);
    let mut posterior_mu = Vec::with_capacity(number_of_objects);
    let mut posterior_sigma = Vec::with_capacity(number_of_objects);

    // For each object
    for i in 0..number_of_objects {
        let num_gm = objects[i].number_of_gm_components;

        // Preallocate posterior components (miss + measurements)
        let mut w_obj = vec![vec![0.0; num_gm]; number_of_measurements + 1];
        let mut mu_obj = vec![vec![DVector::zeros(0); num_gm]; number_of_measurements + 1];
        let mut sigma_obj = vec![vec![DMatrix::zeros(0, 0); num_gm]; number_of_measurements + 1];

        // Initialize with miss detection
        let log_miss_weight = (objects[i].r * (1.0 - model.detection_probability)).ln();
        for j in 0..num_gm {
            w_obj[0][j] = log_miss_weight;
            mu_obj[0][j] = objects[i].mu[j].clone();
            sigma_obj[0][j] = objects[i].sigma[j].clone();
        }

        // Populate auxiliary parameters
        phi[i] = (1.0 - model.detection_probability) * objects[i].r;
        eta[i] = 1.0 - model.detection_probability * objects[i].r;

        // For each GM component
        for j in 0..num_gm {
            // Predicted measurement and innovation covariance
            let mu_z = &model.c * &objects[i].mu[j];
            let z_cov = &model.c * &objects[i].sigma[j] * model.c.transpose() + &model.q;

            // Log Gaussian normalizing constant
            let log_gauss_norm = if let Some(chol) = z_cov.clone().cholesky() {
                let log_det = 2.0 * chol.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
                -(0.5 * model.z_dimension as f64) * (2.0 * std::f64::consts::PI).ln() - 0.5 * log_det
            } else {
                -(0.5 * model.z_dimension as f64) * (2.0 * std::f64::consts::PI).ln()
                    - 0.5 * z_cov.determinant().ln()
            };

            let log_likelihood_ratio_terms = objects[i].r.ln()
                + model.detection_probability.ln()
                + objects[i].w[j].ln()
                - model.clutter_per_unit_volume.ln();

            // Compute Z inverse and Kalman gain
            let z_inv = if let Some(chol) = z_cov.clone().cholesky() {
                let identity = DMatrix::identity(z_cov.nrows(), z_cov.ncols());
                chol.solve(&identity)
            } else {
                z_cov.clone().try_inverse().unwrap_or_else(|| {
                    let svd = z_cov.svd(true, true);
                    svd.pseudo_inverse(1e-10).unwrap()
                })
            };

            let k = &objects[i].sigma[j] * model.c.transpose() * &z_inv;
            let sigma_updated = (DMatrix::identity(model.x_dimension, model.x_dimension)
                - &k * &model.c)
                * &objects[i].sigma[j];

            // For each measurement
            for k_idx in 0..number_of_measurements {
                let nu = &measurements[k_idx] - &mu_z;
                let quadratic_form = nu.transpose() * &z_inv * &nu;
                let gauss_log_likelihood = log_gauss_norm - 0.5 * quadratic_form[(0, 0)];

                // Accumulate to L matrix (linear space)
                l_matrix[(i, k_idx)] +=
                    (log_likelihood_ratio_terms + gauss_log_likelihood).exp();

                // Posterior weights (log space)
                w_obj[k_idx + 1][j] = objects[i].w[j].ln()
                    + gauss_log_likelihood
                    + model.detection_probability.ln()
                    - model.clutter_per_unit_volume.ln();

                // Posterior mean and covariance
                mu_obj[k_idx + 1][j] = &objects[i].mu[j] + &k * &nu;
                sigma_obj[k_idx + 1][j] = sigma_updated.clone();
            }
        }

        // Normalize weights (per measurement row)
        for k_idx in 0..=number_of_measurements {
            let max_w = w_obj[k_idx]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let offset_w: Vec<f64> = w_obj[k_idx].iter().map(|&w| w - max_w).collect();
            let sum_exp_w: f64 = offset_w.iter().map(|&w| w.exp()).sum();
            w_obj[k_idx] = offset_w.iter().map(|&w| (w.exp()) / sum_exp_w).collect();
        }

        posterior_w.push(w_obj);
        posterior_mu.push(mu_obj);
        posterior_sigma.push(sigma_obj);
    }

    // Build association matrices
    let r = objects.iter().map(|obj| obj.r).collect();

    // LBP matrices
    let mut psi = DMatrix::zeros(number_of_objects, number_of_measurements);
    for i in 0..number_of_objects {
        for j in 0..number_of_measurements {
            psi[(i, j)] = l_matrix[(i, j)] / eta[i];
        }
    }

    let phi_vec = DVector::from_vec(phi.clone());
    let eta_vec = DVector::from_vec(eta.clone());

    // Gibbs matrices
    let mut p_gibbs = DMatrix::zeros(number_of_objects, number_of_measurements);
    for i in 0..number_of_objects {
        for j in 0..number_of_measurements {
            p_gibbs[(i, j)] = l_matrix[(i, j)] / (l_matrix[(i, j)] + eta[i]);
        }
    }

    let mut l_gibbs = DMatrix::zeros(number_of_objects, number_of_measurements + 1);
    for i in 0..number_of_objects {
        l_gibbs[(i, 0)] = eta[i];
        for j in 0..number_of_measurements {
            l_gibbs[(i, j + 1)] = l_matrix[(i, j)];
        }
    }

    let mut r_gibbs = DMatrix::zeros(number_of_objects, number_of_measurements + 1);
    for i in 0..number_of_objects {
        r_gibbs[(i, 0)] = phi[i] / eta[i];
        for j in 1..=number_of_measurements {
            r_gibbs[(i, j)] = 1.0;
        }
    }

    // Murty's cost matrix
    let cost = l_matrix.map(|x| -(x.ln()));

    let association_matrices = LmbAssociationMatrices {
        r,
        psi,
        phi: phi_vec,
        eta: eta_vec,
        p: p_gibbs,
        l: l_gibbs,
        gibbs_r: r_gibbs,
        cost,
    };

    let posterior_parameters = LmbPosteriorParameters {
        w: posterior_w,
        mu: posterior_mu,
        sigma: posterior_sigma,
    };

    (association_matrices, posterior_parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};

    #[test]
    fn test_generate_lmb_sensor_association_matrices() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let objects = model.birth_parameters.clone();
        let measurements = vec![DVector::from_vec(vec![0.0, 0.0])];

        let (assoc, posterior) = generate_lmb_sensor_association_matrices(&objects, &measurements, &model, 0);

        // Check dimensions
        assert_eq!(assoc.psi.nrows(), objects.len());
        assert_eq!(assoc.psi.ncols(), measurements.len());
        assert_eq!(posterior.w.len(), objects.len());
    }
}