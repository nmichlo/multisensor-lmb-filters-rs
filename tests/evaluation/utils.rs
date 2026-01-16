//! Test utilities for marginal evaluations and validation
//!
//! Implements helper functions for comparing LBP/Gibbs approximations
//! against exact Murty's algorithm marginals.

use multisensor_lmb_filters_rs::common::rng::Rng;
use nalgebra::{DMatrix, DVector};

/// Calculate the total number of association events for n objects and m measurements
///
/// Implements the combinatorial formula from MATLAB evaluateSmallExamples.m lines 101-106:
/// ```matlab
/// function numberOfEvents = calculateNumberOfAssociationEvents(n, m)
/// numberOfEvents = 0;
/// for k = 0:min(n, m)
///     numberOfEvents = numberOfEvents + factorial(k) * nchoosek(n, k) * nchoosek(m, k);
/// end
/// end
/// ```
///
/// # Arguments
/// * `n` - Number of objects
/// * `m` - Number of measurements
///
/// # Returns
/// Total number of possible association events
pub fn calculate_number_of_association_events(n: usize, m: usize) -> usize {
    let mut total = 0;
    let max_k = n.min(m);

    for k in 0..=max_k {
        let factorial_k = factorial(k);
        let n_choose_k = binomial_coefficient(n, k);
        let m_choose_k = binomial_coefficient(m, k);
        total += factorial_k * n_choose_k * m_choose_k;
    }

    total
}

/// Compute factorial of n
fn factorial(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        (2..=n).product()
    }
}

/// Compute binomial coefficient "n choose k"
fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        0
    } else if k == 0 || k == n {
        1
    } else {
        let k = k.min(n - k); // Optimize using symmetry
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}

/// Compute average Kullback-Leibler divergence
///
/// Implements MATLAB evaluateSmallExamples.m lines 108-112:
/// ```matlab
/// function kl = averageKullbackLeiblerDivergence(p, q)
/// logPQ = log(p ./ q);
/// logPQ(isinf(logPQ) | isnan(logPQ)) = 0;
/// kl = mean(sum(p .* logPQ, 2));
/// end
/// ```
///
/// # Arguments
/// * `p` - True probability distribution (n × m matrix)
/// * `q` - Approximate probability distribution (n × m matrix)
///
/// # Returns
/// Average KL divergence across rows
pub fn average_kullback_leibler_divergence(p: &DMatrix<f64>, q: &DMatrix<f64>) -> f64 {
    assert_eq!(
        p.nrows(),
        q.nrows(),
        "p and q must have same number of rows"
    );
    assert_eq!(
        p.ncols(),
        q.ncols(),
        "p and q must have same number of columns"
    );

    let mut kl_sum = 0.0;

    for i in 0..p.nrows() {
        let mut row_kl = 0.0;
        for j in 0..p.ncols() {
            let p_val = p[(i, j)];
            let q_val = q[(i, j)];

            if p_val > 0.0 && q_val > 0.0 {
                let log_pq = (p_val / q_val).ln();
                if log_pq.is_finite() {
                    row_kl += p_val * log_pq;
                }
            }
        }
        kl_sum += row_kl;
    }

    kl_sum / (p.nrows() as f64)
}

/// Compute average Hellinger distance
///
/// Implements MATLAB evaluateSmallExamples.m lines 114-117:
/// ```matlab
/// function h = averageHellingerDistance(p, q)
/// hDist = sqrt(1 - sum( sqrt(p .* q), 2));
/// h = mean(real(hDist));
/// end
/// ```
///
/// # Arguments
/// * `p` - True probability distribution (n × m matrix)
/// * `q` - Approximate probability distribution (n × m matrix)
///
/// # Returns
/// Average Hellinger distance across rows
pub fn average_hellinger_distance(p: &DMatrix<f64>, q: &DMatrix<f64>) -> f64 {
    assert_eq!(
        p.nrows(),
        q.nrows(),
        "p and q must have same number of rows"
    );
    assert_eq!(
        p.ncols(),
        q.ncols(),
        "p and q must have same number of columns"
    );

    let mut h_sum = 0.0;

    for i in 0..p.nrows() {
        let mut bc = 0.0; // Bhattacharyya coefficient
        for j in 0..p.ncols() {
            bc += (p[(i, j)] * q[(i, j)]).sqrt();
        }
        let h_dist = (1.0 - bc).max(0.0).sqrt();
        h_sum += h_dist;
    }

    h_sum / (p.nrows() as f64)
}

/// Generate a simplified model for testing
///
/// Implements MATLAB generateSimplifiedModel.m
/// Creates a random model with specified number of objects and parameters
///
/// # Arguments
/// * `rng` - Random number generator
/// * `num_objects` - Number of objects in the scenario
/// * `detection_prob` - Probability of detecting each object
/// * `clutter_rate` - Expected number of clutter measurements
///
/// # Returns
/// Model structure for testing
pub fn generate_simplified_model(
    rng: &mut impl Rng,
    num_objects: usize,
    detection_prob: f64,
    clutter_rate: f64,
) -> SimplifiedModel {
    let x_dim = 2;
    let z_dim = 2;

    // Generate random existence probabilities
    let mut r = Vec::with_capacity(num_objects);
    for _ in 0..num_objects {
        r.push(rng.rand());
    }

    // State space limits
    let state_space_limits = vec![(0.0, 2.0), (0.0, 2.0)];

    // Generate random means
    let mut mu = Vec::with_capacity(num_objects);
    for _ in 0..num_objects {
        let mut mean = DVector::zeros(x_dim);
        for d in 0..x_dim {
            mean[d] = state_space_limits[d].0 + 2.0 * state_space_limits[d].1 * rng.rand();
        }
        mu.push(mean);
    }

    // Generate random covariance matrices
    let mut sigma = Vec::with_capacity(num_objects);
    for _ in 0..num_objects {
        // Generate random Q matrix
        let mut q_mat = DMatrix::zeros(x_dim, x_dim);
        for i in 0..x_dim {
            for j in 0..x_dim {
                q_mat[(i, j)] = rng.randn();
            }
        }

        // Generate random diagonal S matrix
        let mut s_diag = vec![0.0; x_dim];
        for i in 0..x_dim {
            s_diag[i] = (25.0 + rng.randn()).abs(); // 5^2 + randn
        }
        let s_mat = DMatrix::from_diagonal(&DVector::from_vec(s_diag));

        // Sigma = Q' * S * Q
        let cov = q_mat.transpose() * s_mat * q_mat;
        sigma.push(cov);
    }

    // Measurement model
    let mut c_mat = DMatrix::zeros(z_dim, x_dim);
    for i in 0..z_dim.min(x_dim) {
        c_mat[(i, i)] = 1.0;
    }

    let q_noise = DMatrix::from_diagonal(&DVector::from_element(z_dim, 4.0)); // 2^2 = 4

    // Clutter parameters
    let observation_space_limits = vec![(0.0, 2.0), (0.0, 2.0)];
    let obs_volume: f64 = observation_space_limits
        .iter()
        .map(|(low, high)| high - low)
        .product();

    let expected_clutter = if clutter_rate == 0.0 {
        clutter_rate
    } else {
        clutter_rate
    };
    let clutter_density = if clutter_rate == 0.0 {
        1.0 / obs_volume
    } else {
        expected_clutter / obs_volume
    };

    SimplifiedModel {
        x_dim,
        z_dim,
        num_objects,
        r,
        mu,
        sigma,
        c: c_mat,
        q: q_noise,
        detection_prob,
        clutter_density,
        expected_clutter,
        observation_space_limits,
        lbp_tolerance: 1e-18,
        max_lbp_iterations: 10000,
    }
}

/// Simplified model structure for testing
#[derive(Debug, Clone)]
pub struct SimplifiedModel {
    pub x_dim: usize,
    pub z_dim: usize,
    pub num_objects: usize,
    pub r: Vec<f64>,              // Existence probabilities
    pub mu: Vec<DVector<f64>>,    // Means
    pub sigma: Vec<DMatrix<f64>>, // Covariances
    pub c: DMatrix<f64>,          // Measurement matrix
    pub q: DMatrix<f64>,          // Measurement noise
    pub detection_prob: f64,
    pub clutter_density: f64,
    pub expected_clutter: f64,
    pub observation_space_limits: Vec<(f64, f64)>,
    pub lbp_tolerance: f64,
    pub max_lbp_iterations: usize,
}

/// Association matrices for testing
#[derive(Debug, Clone)]
pub struct TestAssociationMatrices {
    /// LBP matrices
    pub eta: DVector<f64>,
    pub phi: DVector<f64>,
    pub psi: DMatrix<f64>,

    /// Gibbs matrices
    pub p: DMatrix<f64>,
    pub l: DMatrix<f64>,
    pub r_mat: DMatrix<f64>,

    /// Murty's cost matrix
    pub c: DMatrix<f64>,
}

/// Generate association matrices from a simplified model
///
/// Implements MATLAB generateAssociationMatrices.m
/// Simulates measurements and computes association matrices
///
/// # Arguments
/// * `rng` - Random number generator
/// * `model` - Simplified model
///
/// # Returns
/// Association matrices for data association algorithms
pub fn generate_association_matrices(
    rng: &mut impl Rng,
    model: &SimplifiedModel,
) -> TestAssociationMatrices {
    // Generate measurements (all objects detected in this test)
    let mut detected = vec![false; model.num_objects];
    for i in 0..model.num_objects {
        detected[i] = rng.rand() < 1.0; // All objects detected for testing
    }

    let num_detected = detected.iter().filter(|&&d| d).count();

    // Generate clutter
    let num_clutter = if model.expected_clutter > 0.0 {
        rng.poissrnd(model.expected_clutter)
    } else {
        0
    };

    let num_measurements = num_detected + num_clutter;

    // Generate measurements
    let mut measurements = Vec::with_capacity(num_measurements);

    // Object-generated measurements
    let q_chol = model.q.clone().cholesky().unwrap().l();
    for i in 0..model.num_objects {
        if detected[i] {
            let mut noise = DVector::zeros(model.z_dim);
            for d in 0..model.z_dim {
                noise[d] = rng.randn();
            }
            let z = &model.c * &model.mu[i] + &q_chol * noise;
            measurements.push(z);
        }
    }

    // Clutter measurements
    for _ in 0..num_clutter {
        let mut z = DVector::zeros(model.z_dim);
        for d in 0..model.z_dim {
            z[d] = model.observation_space_limits[d].0
                + 2.0 * model.observation_space_limits[d].1 * rng.rand();
        }
        measurements.push(z);
    }

    // Shuffle measurements (simplified - just use them as is for determinism)

    // Generate association matrices
    let n = model.num_objects;
    let m = num_measurements;

    let mut l_mat = DMatrix::zeros(n, m);

    for i in 0..n {
        // Marginal distribution
        let mu_pred = &model.c * &model.mu[i];
        let s = &model.c * &model.sigma[i] * model.c.transpose() + &model.q;

        // Compute K = S^{-1} using Cholesky decomposition
        let k = match s.clone().cholesky() {
            Some(chol) => chol.inverse(),
            None => {
                // If singular, skip this object
                continue;
            }
        };

        let det_k = k.determinant();
        if det_k <= 0.0 {
            continue;
        }

        let normalizing_constant = (det_k / (2.0 * std::f64::consts::PI)).sqrt();

        // Compute likelihoods
        for j in 0..m {
            let nu = &measurements[j] - &mu_pred;
            let mahalanobis = nu.dot(&(&k * &nu));
            l_mat[(i, j)] = model.r[i]
                * model.detection_prob
                * normalizing_constant
                * (-0.5 * mahalanobis).exp()
                / model.clutter_density;
        }
    }

    // LBP association matrices
    let eta = DVector::from_iterator(n, model.r.iter().map(|&r| 1.0 - model.detection_prob * r));
    let phi = DVector::from_iterator(n, model.r.iter().map(|&r| (1.0 - model.detection_prob) * r));

    let mut psi = DMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            psi[(i, j)] = l_mat[(i, j)] / eta[i];
        }
    }

    // Gibbs matrices
    let mut p = DMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            p[(i, j)] = l_mat[(i, j)] / (eta[i] + l_mat[(i, j)]);
        }
    }

    // L matrix includes eta as first column
    let mut l_full = DMatrix::zeros(n, m + 1);
    for i in 0..n {
        l_full[(i, 0)] = eta[i];
        for j in 0..m {
            l_full[(i, j + 1)] = l_mat[(i, j)];
        }
    }

    // R matrix
    let mut r_mat = DMatrix::zeros(n, m + 1);
    for i in 0..n {
        r_mat[(i, 0)] = phi[i] / eta[i];
        for j in 0..m {
            r_mat[(i, j + 1)] = 1.0;
        }
    }

    // Murty's cost matrix (negative log-likelihood)
    let mut c_mat = DMatrix::from_element(n, m, 0.0);
    for i in 0..n {
        for j in 0..m {
            if l_mat[(i, j)] > 0.0 {
                c_mat[(i, j)] = -l_mat[(i, j)].ln();
            } else {
                c_mat[(i, j)] = 1e10; // Large cost for impossible assignments
            }
        }
    }

    TestAssociationMatrices {
        eta,
        phi,
        psi,
        p,
        l: l_full,
        r_mat,
        c: c_mat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use multisensor_lmb_filters_rs::common::rng::SimpleRng;

    #[test]
    fn test_calculate_number_of_association_events() {
        // Test cases from MATLAB
        assert_eq!(calculate_number_of_association_events(1, 1), 2);
        assert_eq!(calculate_number_of_association_events(2, 2), 7);
        assert_eq!(calculate_number_of_association_events(3, 3), 34);
        assert_eq!(calculate_number_of_association_events(0, 0), 1);
        assert_eq!(calculate_number_of_association_events(1, 0), 1);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 0), 1);
        assert_eq!(binomial_coefficient(5, 5), 1);
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(10, 3), 120);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = DMatrix::from_row_slice(2, 3, &[0.3, 0.4, 0.3, 0.5, 0.3, 0.2]);
        let kl = average_kullback_leibler_divergence(&p, &p);
        assert!(
            kl.abs() < 1e-10,
            "KL divergence of identical distributions should be 0"
        );
    }

    #[test]
    fn test_hellinger_distance_identical() {
        let p = DMatrix::from_row_slice(2, 3, &[0.3, 0.4, 0.3, 0.5, 0.3, 0.2]);
        let h = average_hellinger_distance(&p, &p);
        assert!(
            h.abs() < 1e-10,
            "Hellinger distance of identical distributions should be 0"
        );
    }

    #[test]
    fn test_generate_simplified_model() {
        let mut rng = SimpleRng::new(42);
        let model = generate_simplified_model(&mut rng, 3, 0.95, 5.0);

        assert_eq!(model.num_objects, 3);
        assert_eq!(model.r.len(), 3);
        assert_eq!(model.mu.len(), 3);
        assert_eq!(model.sigma.len(), 3);
        assert_eq!(model.detection_prob, 0.95);
    }
}
