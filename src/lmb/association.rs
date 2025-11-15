//! LMB association matrix generation
//!
//! Computes association matrices required for data association algorithms.
//! Matches MATLAB generateLmbAssociationMatrices.m exactly.

use crate::common::association::gibbs::GibbsAssociationMatrices;
use crate::common::association::lbp::AssociationMatrices;
use crate::common::types::{Model, Object};
use nalgebra::{DMatrix, DVector};

/// Posterior spatial distribution parameters for one object
#[derive(Debug, Clone)]
pub struct PosteriorParameters {
    /// Log-weights for posterior GM components (m+1 x num_components)
    /// Row 0: missed detection
    /// Rows 1..m: detection by measurement j
    pub w: DMatrix<f64>,
    /// Posterior means (m+1 x num_components)
    pub mu: Vec<Vec<DVector<f64>>>,
    /// Posterior covariances (m+1 x num_components)
    pub sigma: Vec<Vec<DMatrix<f64>>>,
}

/// Complete association matrices and posterior parameters
#[derive(Debug, Clone)]
pub struct LmbAssociationResult {
    /// LBP association matrices
    pub lbp: AssociationMatrices,
    /// Gibbs association matrices
    pub gibbs: GibbsAssociationMatrices,
    /// Cost matrix for Murty's algorithm (n x m)
    pub cost: DMatrix<f64>,
    /// Posterior parameters for each object
    pub posterior_parameters: Vec<PosteriorParameters>,
}

/// Generate LMB association matrices
///
/// Computes the association matrices required by LBP, Gibbs sampler, and Murty's
/// algorithm. Also determines measurement-updated components for posterior spatial
/// distribution.
///
/// # Arguments
/// * `objects` - Prior LMB Bernoulli components
/// * `measurements` - Measurements at current time-step
/// * `model` - Model parameters
///
/// # Returns
/// LmbAssociationResult with all association matrices and posterior parameters
///
/// # Implementation Notes
/// Matches MATLAB generateLmbAssociationMatrices.m exactly:
/// 1. For each object, compute marginal likelihood ratio L(i,j) for each measurement
/// 2. Compute auxiliary parameters phi(i) and eta(i)
/// 3. Compute posterior GM components for each measurement association
/// 4. Build association matrices for LBP, Gibbs, and Murty's
pub fn generate_lmb_association_matrices(
    objects: &[Object],
    measurements: &[DVector<f64>],
    model: &Model,
) -> LmbAssociationResult {
    let number_of_objects = objects.len();
    let number_of_measurements = measurements.len();

    // Auxiliary matrices
    let mut l_matrix = DMatrix::zeros(number_of_objects, number_of_measurements);
    let mut phi = DVector::zeros(number_of_objects);
    let mut eta = DVector::zeros(number_of_objects);

    // Posterior parameters for each object
    let mut posterior_parameters = Vec::with_capacity(number_of_objects);

    // Populate arrays and compute posterior components
    for i in 0..number_of_objects {
        let obj = &objects[i];

        // Predeclare object's posterior components (m+1 rows, num_components cols)
        let num_comp = obj.number_of_gm_components;
        let mut w_log = DMatrix::zeros(number_of_measurements + 1, num_comp);
        let mut mu_posterior = vec![vec![DVector::zeros(model.x_dimension); num_comp]; number_of_measurements + 1];
        let mut sigma_posterior = vec![vec![DMatrix::zeros(model.x_dimension, model.x_dimension); num_comp]; number_of_measurements + 1];

        // Row 0: missed detection event
        for j in 0..num_comp {
            w_log[(0, j)] = (obj.w[j] * (1.0 - model.detection_probability)).ln();
            mu_posterior[0][j] = obj.mu[j].clone();
            sigma_posterior[0][j] = obj.sigma[j].clone();
        }

        // Auxiliary LBP parameters
        phi[i] = (1.0 - model.detection_probability) * obj.r;
        eta[i] = 1.0 - model.detection_probability * obj.r;

        // Determine marginal likelihood ratio for each measurement
        for j in 0..num_comp {
            // Predicted measurement and innovation covariance
            let mu_z = &model.c * &obj.mu[j];
            let z_cov = &model.c * &obj.sigma[j] * model.c.transpose() + &model.q;

            // Log normalizing constant
            let log_gaussian_norm = -(0.5 * model.z_dimension as f64) * (2.0 * std::f64::consts::PI).ln()
                - 0.5 * z_cov.determinant().ln();

            // Likelihood ratio terms
            let log_likelihood_ratio_terms = obj.r.ln()
                + model.detection_probability.ln()
                + obj.w[j].ln()
                - model.clutter_per_unit_volume.ln();

            // Kalman gain and updated covariance
            let z_inv = match z_cov.clone().cholesky() {
                Some(chol) => chol.inverse(),
                None => {
                    // Singular covariance, skip this component
                    continue;
                }
            };

            let k = &obj.sigma[j] * model.c.transpose() * &z_inv;
            let sigma_updated =
                (DMatrix::identity(model.x_dimension, model.x_dimension) - &k * &model.c) * &obj.sigma[j];

            // Process each measurement
            for (meas_idx, z) in measurements.iter().enumerate() {
                // Innovation
                let nu = z - &mu_z;

                // Gaussian log-likelihood
                let gaussian_log_likelihood = log_gaussian_norm - 0.5 * nu.dot(&(&z_inv * &nu));

                // Update L matrix
                l_matrix[(i, meas_idx)] += (log_likelihood_ratio_terms + gaussian_log_likelihood).exp();

                // Posterior component parameters
                w_log[(meas_idx + 1, j)] = obj.w[j].ln()
                    + gaussian_log_likelihood
                    + model.detection_probability.ln()
                    - model.clutter_per_unit_volume.ln();

                mu_posterior[meas_idx + 1][j] = &obj.mu[j] + &k * &nu;
                sigma_posterior[meas_idx + 1][j] = sigma_updated.clone();
            }
        }

        // Normalize weights using log-sum-exp
        for row in 0..=number_of_measurements {
            let max_w = (0..num_comp).map(|j| w_log[(row, j)]).fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = (0..num_comp).map(|j| (w_log[(row, j)] - max_w).exp()).sum();

            for j in 0..num_comp {
                w_log[(row, j)] = ((w_log[(row, j)] - max_w).exp() / sum_exp).ln();
            }
        }

        // Convert log-weights to linear
        let mut w_normalized = DMatrix::zeros(number_of_measurements + 1, num_comp);
        for row in 0..=number_of_measurements {
            for j in 0..num_comp {
                w_normalized[(row, j)] = w_log[(row, j)].exp();
            }
        }

        posterior_parameters.push(PosteriorParameters {
            w: w_normalized,
            mu: mu_posterior,
            sigma: sigma_posterior,
        });
    }

    // Build association matrices

    // LBP matrices: Psi = L ./ eta (broadcast division)
    let mut psi = DMatrix::zeros(number_of_objects, number_of_measurements);
    for i in 0..number_of_objects {
        for j in 0..number_of_measurements {
            psi[(i, j)] = l_matrix[(i, j)] / eta[i];
        }
    }

    // Clone phi and eta for use in Gibbs matrices
    let phi_gibbs = phi.clone();
    let eta_gibbs = eta.clone();

    let lbp = AssociationMatrices { psi, phi, eta };

    // Gibbs matrices: P = L ./ (L + eta) (broadcast division)
    let mut p = DMatrix::zeros(number_of_objects, number_of_measurements);
    for i in 0..number_of_objects {
        for j in 0..number_of_measurements {
            p[(i, j)] = l_matrix[(i, j)] / (l_matrix[(i, j)] + eta_gibbs[i]);
        }
    }

    // L = [eta, L]
    let mut l_gibbs = DMatrix::zeros(number_of_objects, number_of_measurements + 1);
    l_gibbs.column_mut(0).copy_from(&eta_gibbs);
    l_gibbs.view_mut((0, 1), (number_of_objects, number_of_measurements))
        .copy_from(&l_matrix);

    // R = [phi./eta, ones(n,m)]
    let mut r_gibbs = DMatrix::zeros(number_of_objects, number_of_measurements + 1);
    for i in 0..number_of_objects {
        r_gibbs[(i, 0)] = phi_gibbs[i] / eta_gibbs[i];
        for j in 1..=number_of_measurements {
            r_gibbs[(i, j)] = 1.0;
        }
    }

    let gibbs = GibbsAssociationMatrices {
        p,
        l: l_gibbs,
        r: r_gibbs,
        c: l_matrix.map(|val| -val.ln()),
    };

    // Murty's cost matrix: C = -log(L)
    let cost = l_matrix.map(|val| if val > 1e-300 { -val.ln() } else { 1e10 });

    LmbAssociationResult {
        lbp,
        gibbs,
        cost,
        posterior_parameters,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};

    #[test]
    fn test_generate_association_matrices_no_measurements() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let objects = model.birth_parameters.clone();
        let measurements = vec![];

        let result = generate_lmb_association_matrices(&objects, &measurements, &model);

        assert_eq!(result.lbp.psi.nrows(), objects.len());
        assert_eq!(result.lbp.psi.ncols(), 0);
        assert_eq!(result.posterior_parameters.len(), objects.len());
    }

    #[test]
    fn test_generate_association_matrices_with_measurements() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let objects = model.birth_parameters.clone();
        let measurements = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![10.0, 10.0]),
        ];

        let result = generate_lmb_association_matrices(&objects, &measurements, &model);

        // Check dimensions
        assert_eq!(result.lbp.psi.nrows(), objects.len());
        assert_eq!(result.lbp.psi.ncols(), 2);
        assert_eq!(result.lbp.phi.len(), objects.len());
        assert_eq!(result.lbp.eta.len(), objects.len());

        // Check Gibbs matrices
        assert_eq!(result.gibbs.p.nrows(), objects.len());
        assert_eq!(result.gibbs.p.ncols(), 2);
        assert_eq!(result.gibbs.l.ncols(), 3); // [eta, L1, L2]

        // Check posterior parameters
        assert_eq!(result.posterior_parameters.len(), objects.len());
        for params in &result.posterior_parameters {
            assert_eq!(params.w.nrows(), 3); // [miss, meas1, meas2]
            assert_eq!(params.mu.len(), 3);
            assert_eq!(params.sigma.len(), 3);
        }
    }

    #[test]
    fn test_posterior_weights_normalized() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let objects = model.birth_parameters.clone();
        let measurements = vec![DVector::from_vec(vec![0.0, 0.0])];

        let result = generate_lmb_association_matrices(&objects, &measurements, &model);

        // Check that weights are normalized for each row
        for params in &result.posterior_parameters {
            for row in 0..params.w.nrows() {
                let row_sum: f64 = params.w.row(row).sum();
                assert!((row_sum - 1.0).abs() < 1e-10, "Row sum: {}", row_sum);
            }
        }
    }
}
