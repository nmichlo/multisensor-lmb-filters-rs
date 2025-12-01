//! Lazy likelihood computation for multi-sensor LMBM
//!
//! Computes likelihood values on-demand with memoization instead of
//! precomputing all entries. Access pattern analysis shows only 5-17%
//! of entries are accessed during Gibbs sampling, making lazy computation
//! significantly more efficient.

use crate::common::linalg::robust_inverse_with_log_det;
use crate::common::types::{Hypothesis, Model};
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;
use std::collections::HashMap;

/// Maximum supported sensors for stack-allocated index arrays
const MAX_SENSORS: usize = 8;

/// Cached computation result for a single entry
#[derive(Debug, Clone)]
struct CachedEntry {
    l: f64,
    r: f64,
    mu: DVector<f64>,
    sigma: DMatrix<f64>,
}

/// Lazy likelihood computation with memoization
///
/// Instead of precomputing all likelihood entries upfront (which can be 10M+ entries),
/// this struct computes values on-demand and caches results for reuse.
///
/// # Example
/// ```ignore
/// let lazy = LazyLikelihood::new(&hypothesis, &measurements, &model, number_of_sensors);
/// let l_value = lazy.get_l(index);  // Computed on first access
/// let l_again = lazy.get_l(index);  // Retrieved from cache
/// ```
pub struct LazyLikelihood<'a> {
    /// Prior hypothesis
    hypothesis: &'a Hypothesis,
    /// Measurements from all sensors [sensor][measurements]
    measurements: &'a [Vec<DVector<f64>>],
    /// Model parameters
    model: &'a Model,
    /// Pre-cached measurement noise matrices per sensor
    q_cache: Vec<DMatrix<f64>>,
    /// Pre-cached observation matrices per sensor
    c_cache: Vec<DMatrix<f64>>,
    /// Dimensions: [m1+1, m2+1, ..., ms+1, n]
    dimensions: Vec<usize>,
    /// Page sizes for index conversion (cumulative products)
    page_sizes: Vec<usize>,
    /// Number of sensors
    number_of_sensors: usize,
    /// Total number of entries in the likelihood tensor
    number_of_entries: usize,
    /// Memoization cache (interior mutability for lazy computation)
    cache: RefCell<HashMap<usize, CachedEntry>>,
}

impl<'a> LazyLikelihood<'a> {
    /// Create a new lazy likelihood computer
    ///
    /// # Arguments
    /// * `hypothesis` - Prior LMBM hypothesis
    /// * `measurements` - Measurements from all sensors [sensor][measurements]
    /// * `model` - Model parameters
    /// * `number_of_sensors` - Number of sensors
    pub fn new(
        hypothesis: &'a Hypothesis,
        measurements: &'a [Vec<DVector<f64>>],
        model: &'a Model,
        number_of_sensors: usize,
    ) -> Self {
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

        // Pre-cache Q and C matrices per sensor
        let q_cache: Vec<DMatrix<f64>> = (0..number_of_sensors)
            .map(|s| model.get_measurement_noise(Some(s)).clone())
            .collect();
        let c_cache: Vec<DMatrix<f64>> = (0..number_of_sensors)
            .map(|s| model.get_observation_matrix(Some(s)).clone())
            .collect();

        Self {
            hypothesis,
            measurements,
            model,
            q_cache,
            c_cache,
            dimensions,
            page_sizes,
            number_of_sensors,
            number_of_entries,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// Get the dimensions of the likelihood tensor
    #[inline]
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Get the total number of entries
    #[inline]
    pub fn number_of_entries(&self) -> usize {
        self.number_of_entries
    }

    /// Get the number of objects
    #[inline]
    pub fn number_of_objects(&self) -> usize {
        self.hypothesis.r.len()
    }

    /// Get the number of sensors
    #[inline]
    pub fn number_of_sensors(&self) -> usize {
        self.number_of_sensors
    }

    /// Get log likelihood value at index (computes on-demand)
    #[inline]
    pub fn get_l(&self, index: usize) -> f64 {
        self.ensure_computed(index);
        self.cache.borrow()[&index].l
    }

    /// Get posterior existence probability at index (computes on-demand)
    #[inline]
    pub fn get_r(&self, index: usize) -> f64 {
        self.ensure_computed(index);
        self.cache.borrow()[&index].r
    }

    /// Get posterior mean at index (computes on-demand)
    #[inline]
    pub fn get_mu(&self, index: usize) -> DVector<f64> {
        self.ensure_computed(index);
        self.cache.borrow()[&index].mu.clone()
    }

    /// Get posterior covariance at index (computes on-demand)
    #[inline]
    pub fn get_sigma(&self, index: usize) -> DMatrix<f64> {
        self.ensure_computed(index);
        self.cache.borrow()[&index].sigma.clone()
    }

    /// Get all cached values at once (avoids multiple cache lookups)
    #[inline]
    pub fn get_all(&self, index: usize) -> (f64, f64, DVector<f64>, DMatrix<f64>) {
        self.ensure_computed(index);
        let cache = self.cache.borrow();
        let entry = &cache[&index];
        (entry.l, entry.r, entry.mu.clone(), entry.sigma.clone())
    }

    /// Check if an index has been computed
    #[inline]
    pub fn is_computed(&self, index: usize) -> bool {
        self.cache.borrow().contains_key(&index)
    }

    /// Get the number of entries that have been computed
    #[inline]
    pub fn computed_count(&self) -> usize {
        self.cache.borrow().len()
    }

    /// Convert linear index to Cartesian coordinates
    ///
    /// Returns (object_index, association_array) where association_array[s] is
    /// the measurement index for sensor s (0 = miss, 1+ = measurement index)
    #[inline]
    fn index_to_association(&self, ell: usize) -> (usize, [usize; MAX_SENSORS]) {
        let mut u = [0usize; MAX_SENSORS];

        // Convert from 0-indexed to 1-indexed for MATLAB-style computation
        let mut remaining = ell + 1;
        let m = self.page_sizes.len();

        for i in 0..m {
            let j = m - i - 1;
            let zeta = remaining / self.page_sizes[j];
            let eta = remaining % self.page_sizes[j];
            u[j] = zeta + if eta != 0 { 1 } else { 0 };
            remaining = remaining - self.page_sizes[j] * (zeta - if eta == 0 { 1 } else { 0 });
        }

        // Extract object index and convert to 0-indexed associations
        let object_index = u[self.number_of_sensors] - 1;
        let mut a = [0usize; MAX_SENSORS];
        for s in 0..self.number_of_sensors {
            a[s] = u[s] - 1; // 0 = miss, 1+ = measurement index (1-indexed)
        }

        (object_index, a)
    }

    /// Ensure the entry at index is computed and cached
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn ensure_computed(&self, index: usize) {
        // Fast path: already computed
        if self.cache.borrow().contains_key(&index) {
            return;
        }

        // Compute the entry
        let (object_index, associations) = self.index_to_association(index);
        let (l, r, mu, sigma) = self.compute_likelihood(object_index, &associations);

        // Cache the result
        self.cache.borrow_mut().insert(
            index,
            CachedEntry { l, r, mu, sigma },
        );
    }

    /// Compute likelihood for a specific object and association
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn compute_likelihood(
        &self,
        i: usize,
        a: &[usize; MAX_SENSORS],
    ) -> (f64, f64, DVector<f64>, DMatrix<f64>) {
        // Count assignments
        let mut assignments = [false; MAX_SENSORS];
        let mut number_of_assignments = 0;
        for s in 0..self.number_of_sensors {
            assignments[s] = a[s] > 0;
            if assignments[s] {
                number_of_assignments += 1;
            }
        }

        if number_of_assignments > 0 {
            let z_dim_total = self.model.z_dimension * number_of_assignments;

            // Build stacked measurement vector
            let mut z = DVector::zeros(z_dim_total);
            let mut c = DMatrix::zeros(z_dim_total, self.model.x_dimension);
            let mut q = DMatrix::zeros(z_dim_total, z_dim_total);

            let mut counter = 0;
            for s in 0..self.number_of_sensors {
                if assignments[s] {
                    let start = self.model.z_dimension * counter;

                    // Copy measurement
                    z.rows_mut(start, self.model.z_dimension)
                        .copy_from(&self.measurements[s][a[s] - 1]);

                    // Copy observation matrix
                    c.view_mut((start, 0), (self.model.z_dimension, self.model.x_dimension))
                        .copy_from(&self.c_cache[s]);

                    // Copy measurement noise (block diagonal)
                    q.view_mut((start, start), (self.model.z_dimension, self.model.z_dimension))
                        .copy_from(&self.q_cache[s]);

                    counter += 1;
                }
            }

            // nu = z - C * mu[i]
            let nu = &z - &c * &self.hypothesis.mu[i];

            // z_matrix = C * Sigma * C' + Q
            let c_sigma = &c * &self.hypothesis.sigma[i];
            let z_matrix = &c_sigma * c.transpose() + &q;

            // Use combined inverse + log-det
            let (z_inv, eta) = robust_inverse_with_log_det(&z_matrix, z_dim_total)
                .expect("z_matrix should be invertible");

            // Kalman gain: K = Sigma * C' * Z_inv
            let sigma_ct = &self.hypothesis.sigma[i] * c.transpose();
            let k = &sigma_ct * &z_inv;

            // Detection probabilities
            let mut pd_log = 0.0;
            for s in 0..self.number_of_sensors {
                let p_d = self.model.get_detection_probability(Some(s));
                pd_log += if assignments[s] {
                    p_d.ln()
                } else {
                    (1.0 - p_d).ln()
                };
            }

            // Clutter per unit volume
            let kappa_log: f64 = assignments
                .iter()
                .take(self.number_of_sensors)
                .enumerate()
                .filter(|(_, &x)| x)
                .map(|(s, _)| self.model.get_clutter_per_unit_volume(Some(s)).ln())
                .sum();

            let l = self.hypothesis.r[i].ln() + pd_log + eta
                - 0.5 * nu.dot(&(&z_inv * &nu))
                - kappa_log;

            // Posterior parameters
            let r = 1.0;
            let mu = &self.hypothesis.mu[i] + &k * &nu;
            let identity = DMatrix::identity(self.model.x_dimension, self.model.x_dimension);
            let sigma = (&identity - &k * &c) * &self.hypothesis.sigma[i];

            (l, r, mu, sigma)
        } else {
            // All missed detections - compute probability
            let mut prob_no_detect = 1.0;
            for s in 0..self.number_of_sensors {
                prob_no_detect *= 1.0 - self.model.get_detection_probability(Some(s));
            }

            let numerator = self.hypothesis.r[i] * prob_no_detect;
            let denominator = 1.0 - self.hypothesis.r[i] + numerator;

            let l = denominator.ln();
            let r = numerator / denominator;
            let mu = self.hypothesis.mu[i].clone();
            let sigma = self.hypothesis.sigma[i].clone();

            (l, r, mu, sigma)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};

    #[test]
    fn test_lazy_likelihood_basic() {
        use crate::common::types::Hypothesis;

        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        // Create a hypothesis with 2 objects (the model might have 0)
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
                DMatrix::identity(4, 4) * 10.0,
                DMatrix::identity(4, 4) * 10.0,
            ],
        };

        // 2 sensors, 1 measurement each
        let measurements = vec![
            vec![DVector::from_vec(vec![0.0, 0.0])],
            vec![DVector::from_vec(vec![1.0, 1.0])],
        ];

        let lazy = LazyLikelihood::new(&hypothesis, &measurements, &model, 2);

        // Check dimensions: (1+1, 1+1, n) = (2, 2, 2)
        assert_eq!(lazy.dimensions()[0], 2); // sensor 1: miss + 1 meas
        assert_eq!(lazy.dimensions()[1], 2); // sensor 2: miss + 1 meas
        assert_eq!(lazy.dimensions()[2], 2); // 2 objects

        // Total entries: 2 * 2 * 2 = 8
        assert_eq!(lazy.number_of_entries(), 8);

        // Nothing computed yet
        assert_eq!(lazy.computed_count(), 0);

        // Access a few entries
        let _l0 = lazy.get_l(0);
        assert_eq!(lazy.computed_count(), 1);

        let _l1 = lazy.get_l(1);
        assert_eq!(lazy.computed_count(), 2);

        // Accessing same index shouldn't increase count
        let _l0_again = lazy.get_l(0);
        assert_eq!(lazy.computed_count(), 2);
    }

    #[test]
    fn test_lazy_vs_eager_equivalence() {
        use crate::common::types::Hypothesis;
        use crate::multisensor_lmbm::association::generate_multisensor_lmbm_association_matrices;

        let mut rng = crate::common::rng::SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        // Create a hypothesis with 3 objects
        let hypothesis = Hypothesis {
            birth_location: vec![0, 0, 0],
            birth_time: vec![0, 0, 0],
            w: 1.0,
            r: vec![0.8, 0.7, 0.6],
            mu: vec![
                DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
                DVector::from_vec(vec![10.0, 10.0, 0.0, 0.0]),
                DVector::from_vec(vec![20.0, 20.0, 0.0, 0.0]),
            ],
            sigma: vec![
                DMatrix::identity(4, 4) * 10.0,
                DMatrix::identity(4, 4) * 10.0,
                DMatrix::identity(4, 4) * 10.0,
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

        // Compute eagerly
        let (l_eager, posterior_eager, dims_eager) =
            generate_multisensor_lmbm_association_matrices(&hypothesis, &measurements, &model, 2);

        // Compute lazily
        let lazy = LazyLikelihood::new(&hypothesis, &measurements, &model, 2);

        // Verify dimensions match
        assert_eq!(lazy.dimensions(), &dims_eager);

        // Verify all entries match
        for idx in 0..lazy.number_of_entries() {
            let l_lazy = lazy.get_l(idx);
            let r_lazy = lazy.get_r(idx);

            assert!(
                (l_eager[idx] - l_lazy).abs() < 1e-10,
                "L mismatch at {}: eager={}, lazy={}",
                idx,
                l_eager[idx],
                l_lazy
            );
            assert!(
                (posterior_eager.r[idx] - r_lazy).abs() < 1e-10,
                "R mismatch at {}: eager={}, lazy={}",
                idx,
                posterior_eager.r[idx],
                r_lazy
            );
        }

        // Verify we actually tested something
        // 3 objects * 3 sensor1 options * 3 sensor2 options = 27 entries
        assert_eq!(lazy.number_of_entries(), 27);
        assert_eq!(lazy.computed_count(), 27);
    }
}
