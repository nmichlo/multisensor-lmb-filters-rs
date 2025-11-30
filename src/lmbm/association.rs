//! LMBM association matrices generation
//!
//! Implements association matrix generation for LMBM filter.
//! Matches MATLAB generateLmbmAssociationMatrices.m and lmbmGibbsSampling.m exactly.

use crate::common::association::gibbs::{generate_gibbs_sample, initialize_gibbs_association_vectors};
use crate::common::linalg::{
    compute_innovation_params, compute_kalman_gain, compute_kalman_updated_mean,
    compute_measurement_log_likelihood, log_gaussian_normalizing_constant,
};
use crate::common::types::{Hypothesis, Model};
use nalgebra::{DMatrix, DVector};

/// Posterior parameters for LMBM update
#[derive(Debug, Clone)]
pub struct LmbmPosteriorParameters {
    /// Posterior existence probabilities for miss detection (n x 1)
    pub r: Vec<f64>,
    /// Posterior means (n x (m+1)) - index 0 is miss, 1..m+1 are measurements
    pub mu: Vec<Vec<DVector<f64>>>,
    /// Posterior covariances (n x 1) - same for all measurements
    pub sigma: Vec<DMatrix<f64>>,
}

/// Association matrices for LMBM data association
#[derive(Debug, Clone)]
pub struct LmbmAssociationMatrices {
    /// Probability matrix for Gibbs sampler (n x m)
    pub p: DMatrix<f64>,
    /// Log likelihood matrix for Gibbs (n x (m+1)) - column 0 is log(eta), rest is R
    pub l: DMatrix<f64>,
    /// Cost matrix for Murty's algorithm (n x m)
    pub c: DMatrix<f64>,
}

/// Result containing both association matrices and posterior parameters
#[derive(Debug, Clone)]
pub struct LmbmAssociationResult {
    pub association_matrices: LmbmAssociationMatrices,
    pub posterior_parameters: LmbmPosteriorParameters,
}

/// Generate LMBM association matrices
///
/// Computes the association matrices required by data association algorithms
/// and the measurement-updated components for posterior spatial distributions.
///
/// # Arguments
/// * `hypothesis` - Prior LMBM hypothesis containing Bernoulli components
/// * `measurements` - Measurements for current time-step
/// * `model` - Model parameters
///
/// # Returns
/// LmbmAssociationResult containing association matrices and posterior parameters
///
/// # Implementation Notes
/// Matches MATLAB generateLmbmAssociationMatrices.m exactly:
/// 1. Compute log likelihood matrix R
/// 2. Compute auxiliary variables phi and eta
/// 3. Compute posterior parameters for miss and each measurement
/// 4. Build association matrices P, L, C
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn generate_lmbm_association_matrices(
    hypothesis: &Hypothesis,
    measurements: &[DVector<f64>],
    model: &Model,
) -> LmbmAssociationResult {
    let number_of_objects = hypothesis.r.len();
    let number_of_measurements = measurements.len();

    // Log likelihood matrix
    let mut r_matrix = DMatrix::zeros(number_of_objects, number_of_measurements);

    // Auxiliary variables
    let mut phi = vec![0.0; number_of_objects];
    let mut eta = vec![0.0; number_of_objects];
    for i in 0..number_of_objects {
        phi[i] = (1.0 - model.detection_probability) * hypothesis.r[i];
        eta[i] = 1.0 - model.detection_probability * hypothesis.r[i];
    }

    // Updated components for posterior spatial distributions
    let mut posterior_r = vec![0.0; number_of_objects];
    for i in 0..number_of_objects {
        posterior_r[i] = phi[i] / eta[i];
    }

    // Posterior mu: n x (m+1), where column 0 is miss, columns 1..m+1 are measurements
    let mut posterior_mu = vec![vec![DVector::zeros(0); number_of_measurements + 1]; number_of_objects];

    // Posterior Sigma: same as prior (updated only once per object)
    let mut posterior_sigma = hypothesis.sigma.clone();

    // Populate association matrices and compute posterior components
    for i in 0..number_of_objects {
        // Missed detection event
        posterior_mu[i][0] = hypothesis.mu[i].clone();

        // Determine posterior parameters for each object's spatial distribution
        let (mu_z, z_cov) =
            compute_innovation_params(&hypothesis.mu[i], &hypothesis.sigma[i], &model.c, &model.q);

        // Compute log Gaussian normalizing constant
        let log_gaussian_norm = log_gaussian_normalizing_constant(&z_cov, model.z_dimension);

        let log_likelihood_ratio_terms = hypothesis.r[i].ln()
            + model.detection_probability.ln()
            - model.clutter_per_unit_volume.ln();

        // Compute Z inverse and Kalman gain
        let (k, sigma_updated, z_inv) =
            match compute_kalman_gain(&hypothesis.sigma[i], &model.c, &z_cov, model.x_dimension) {
                Some(result) => result,
                None => {
                    // Singular covariance, skip this object (shouldn't happen in normal operation)
                    continue;
                }
            };
        posterior_sigma[i] = sigma_updated;

        // Determine total marginal likelihood and posterior components
        for j in 0..number_of_measurements {
            // Determine marginal likelihood ratio
            let gaussian_log_likelihood =
                compute_measurement_log_likelihood(&measurements[j], &mu_z, &z_inv, log_gaussian_norm);
            r_matrix[(i, j)] = log_likelihood_ratio_terms + gaussian_log_likelihood;

            // Determine posterior mean for each measurement
            posterior_mu[i][j + 1] =
                compute_kalman_updated_mean(&hypothesis.mu[i], &k, &measurements[j], &mu_z);
        }
    }

    // Determine Gibbs sampler parameters
    let r_linear = r_matrix.map(|x| x.exp());

    // Gibbs sampler association matrices
    // P = RLinear ./ (RLinear + eta)
    let mut p_matrix = DMatrix::zeros(number_of_objects, number_of_measurements);
    for i in 0..number_of_objects {
        for j in 0..number_of_measurements {
            p_matrix[(i, j)] = r_linear[(i, j)] / (r_linear[(i, j)] + eta[i]);
        }
    }

    // L = [log(eta) R]
    let mut l_matrix = DMatrix::zeros(number_of_objects, number_of_measurements + 1);
    for i in 0..number_of_objects {
        l_matrix[(i, 0)] = eta[i].ln();
        for j in 0..number_of_measurements {
            l_matrix[(i, j + 1)] = r_matrix[(i, j)];
        }
    }

    // Murty's algorithm cost matrix
    // C = -R
    let c_matrix = r_matrix.map(|x| -x);

    LmbmAssociationResult {
        association_matrices: LmbmAssociationMatrices {
            p: p_matrix,
            l: l_matrix,
            c: c_matrix,
        },
        posterior_parameters: LmbmPosteriorParameters {
            r: posterior_r,
            mu: posterior_mu,
            sigma: posterior_sigma,
        },
    }
}

/// Generate association events using Gibbs sampling for LMBM
///
/// Generates a set of posterior hypotheses using Gibbs sampling.
///
/// # Arguments
/// * `p` - (n x m) matrix of sampling probabilities
/// * `c` - (n x m) cost matrix for initialization
/// * `number_of_samples` - Number of Gibbs samples to generate
///
/// # Returns
/// Matrix of distinct association events (rows are events)
///
/// # Implementation Notes
/// Matches MATLAB lmbmGibbsSampling.m exactly:
/// 1. Initialize association vectors using Hungarian algorithm
/// 2. Generate Gibbs samples
/// 3. Return only unique samples
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn lmbm_gibbs_sampling(
    rng: &mut impl crate::common::rng::Rng,
    p: &DMatrix<f64>,
    c: &DMatrix<f64>,
    number_of_samples: usize,
) -> DMatrix<usize> {
    let n = p.nrows();

    // Initialize Gibbs association vectors
    let (mut v, mut w) = initialize_gibbs_association_vectors(c);

    // Association vectors storage
    let mut v_samples = DMatrix::zeros(number_of_samples, n);

    // Gibbs sampling
    for i in 0..number_of_samples {
        // Generate a new Gibbs sample
        let (v_new, w_new) = generate_gibbs_sample(rng, p, v.clone(), w.clone());
        v = v_new;
        w = w_new;

        // Store Gibbs sample
        for (j, &val) in v.iter().enumerate() {
            v_samples[(i, j)] = val;
        }
    }

    // Keep only distinct samples
    let mut unique_samples = Vec::new();
    for i in 0..number_of_samples {
        let row: Vec<usize> = v_samples.row(i).iter().copied().collect();
        if !unique_samples.contains(&row) {
            unique_samples.push(row);
        }
    }

    // Sort to match MATLAB's unique(V, 'rows') behavior
    unique_samples.sort();

    // Convert to matrix
    let num_unique = unique_samples.len();
    let mut result = DMatrix::zeros(num_unique, n);
    for (i, row) in unique_samples.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[(i, j)] = val;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};

    #[test]
    fn test_generate_lmbm_association_matrices() {
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
        let measurements = vec![DVector::from_vec(vec![0.0, 0.0])];

        let result = generate_lmbm_association_matrices(&hypothesis, &measurements, &model);

        let n = hypothesis.r.len();
        let m = measurements.len();

        // Check association matrix dimensions
        assert_eq!(result.association_matrices.p.nrows(), n);
        assert_eq!(result.association_matrices.p.ncols(), m);
        assert_eq!(result.association_matrices.l.nrows(), n);
        assert_eq!(result.association_matrices.l.ncols(), m + 1);
        assert_eq!(result.association_matrices.c.nrows(), n);
        assert_eq!(result.association_matrices.c.ncols(), m);

        // Check posterior parameter dimensions
        assert_eq!(result.posterior_parameters.r.len(), n);
        assert_eq!(result.posterior_parameters.mu.len(), n);
        assert_eq!(result.posterior_parameters.sigma.len(), n);

        for i in 0..n {
            assert_eq!(result.posterior_parameters.mu[i].len(), m + 1);
        }
    }

    #[test]
    fn test_lmbm_gibbs_sampling() {
        let mut rng = crate::common::rng::SimpleRng::new(42);
        let p = DMatrix::from_row_slice(3, 2, &[0.5, 0.5, 0.6, 0.4, 0.3, 0.7]);
        let c = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        let samples = lmbm_gibbs_sampling(&mut rng, &p, &c, 50);

        // Check dimensions
        assert!(samples.nrows() > 0);
        assert_eq!(samples.ncols(), 3);

        // Check all values are valid (0 to m)
        for i in 0..samples.nrows() {
            for j in 0..samples.ncols() {
                assert!(samples[(i, j)] <= 2);
            }
        }
    }

    #[test]
    fn test_lmbm_association_posterior_probabilities() {
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
        let measurements = vec![DVector::from_vec(vec![0.0, 0.0])];

        let result = generate_lmbm_association_matrices(&hypothesis, &measurements, &model);

        // Check P matrix entries are probabilities
        for i in 0..result.association_matrices.p.nrows() {
            for j in 0..result.association_matrices.p.ncols() {
                let p_val = result.association_matrices.p[(i, j)];
                assert!(p_val >= 0.0 && p_val <= 1.0);
            }
        }

        // Check posterior existence probabilities are valid
        for &r in &result.posterior_parameters.r {
            assert!(r >= 0.0 && r <= 1.0);
        }
    }
}