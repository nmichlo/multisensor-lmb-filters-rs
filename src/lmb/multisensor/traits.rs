//! Multi-sensor associator traits and implementations.
//!
//! This module provides the [`MultisensorAssociator`] trait for multi-sensor
//! joint data association. Unlike the single-sensor [`Associator`] trait which
//! operates on a 2D cost matrix, multi-sensor association operates on a
//! Cartesian product space of all sensor measurements.
//!
//! # Why a Separate Trait?
//!
//! Multi-sensor association has fundamentally different inputs:
//! - Single-sensor: `n × m` cost matrix
//! - Multi-sensor: Flattened tensor of shape `[m_1+1, m_2+1, ..., m_S+1, n]`
//!
//! The output is also different: multi-sensor returns per-object assignments
//! for each sensor, not just measurement indices.

use nalgebra::DMatrix;
use rand::distributions::Distribution;

use super::super::config::AssociationConfig;
use super::super::errors::AssociationError;
use super::super::simple_rng::Uniform01;

/// Result of multi-sensor data association.
///
/// Contains sampled association events where each sample specifies which
/// measurement (if any) each object is associated with for each sensor.
#[derive(Debug, Clone)]
pub struct MultisensorAssociationResult {
    /// Sampled association events.
    ///
    /// Each inner `Vec<usize>` is a flattened association matrix of shape
    /// `[n_objects, n_sensors]` stored column-major:
    /// - `sample[s * n_objects + i]` = measurement assigned to object i by sensor s
    /// - Value 0 means miss (no detection)
    /// - Value 1..=m_s means measurement index (1-indexed)
    pub samples: Vec<Vec<usize>>,

    /// Number of Gibbs iterations performed.
    pub iterations: usize,
}

impl MultisensorAssociationResult {
    /// Create a new result with samples.
    pub fn new(samples: Vec<Vec<usize>>, iterations: usize) -> Self {
        Self {
            samples,
            iterations,
        }
    }

    /// Create an empty result.
    pub fn empty() -> Self {
        Self {
            samples: Vec::new(),
            iterations: 0,
        }
    }

    /// Number of unique samples.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }
}

/// Multi-sensor data association trait.
///
/// This trait defines the interface for joint association across multiple sensors.
/// Unlike single-sensor [`Associator`] which computes marginal probabilities,
/// multi-sensor association typically produces discrete samples from the
/// joint association posterior.
///
/// # Implementation
///
/// The primary implementation is [`MultisensorGibbsAssociator`] which uses
/// Gibbs sampling to generate association samples. The samples explore the
/// Cartesian product space of all possible sensor-object assignments.
pub trait MultisensorAssociator: Send + Sync + Default {
    /// Perform multi-sensor data association.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `log_likelihoods` - Flattened log-likelihood tensor
    /// * `dimensions` - Tensor dimensions `[m_1+1, ..., m_S+1, n]`
    /// * `config` - Association algorithm configuration
    ///
    /// # Returns
    /// Association samples or error.
    fn associate<R: rand::Rng>(
        &self,
        rng: &mut R,
        log_likelihoods: &[f64],
        dimensions: &[usize],
        config: &AssociationConfig,
    ) -> Result<MultisensorAssociationResult, AssociationError>;

    /// Algorithm name for logging and debugging.
    fn name(&self) -> &'static str;
}

/// Multi-sensor Gibbs sampling associator.
///
/// Uses Gibbs sampling to generate association samples from the joint posterior
/// over all sensor-object assignments. The algorithm iteratively samples each
/// object's association for each sensor, conditioned on all other assignments.
///
/// This is the standard method for multi-sensor LMBM filters where discrete
/// samples are needed to generate posterior hypotheses.
#[derive(Debug, Clone, Default)]
pub struct MultisensorGibbsAssociator;

impl MultisensorGibbsAssociator {
    /// Create a new Gibbs associator.
    pub fn new() -> Self {
        Self
    }

    /// Convert linear index to Cartesian coordinates (MATLAB-style, 1-indexed).
    #[allow(dead_code)] // Used in tests
    fn linear_to_cartesian(ell: usize, page_sizes: &[usize]) -> Vec<usize> {
        let m = page_sizes.len();
        let mut u = vec![0; m];
        let mut remaining = ell;

        for i in 0..m {
            let j = m - i - 1;
            let zeta = remaining / page_sizes[j];
            let eta = remaining % page_sizes[j];
            u[j] = zeta + if eta != 0 { 1 } else { 0 };
            remaining -= page_sizes[j] * (zeta - if eta == 0 { 1 } else { 0 });
        }

        u
    }

    /// Convert Cartesian coordinates to linear index (MATLAB-style, 1-indexed).
    fn cartesian_to_linear(u: &[usize], dimensions: &[usize]) -> usize {
        let mut ell = u[0];
        let mut pi = 1;

        for i in 1..u.len() {
            pi *= dimensions[i - 1];
            ell += pi * (u[i] - 1);
        }

        ell - 1 // Convert to 0-indexed
    }

    /// Generate a single association event using Gibbs sampling.
    fn generate_association_event<R: rand::Rng>(
        rng: &mut R,
        log_likelihoods: &[f64],
        dimensions: &[usize],
        m: &[usize],
        v: &mut DMatrix<usize>,
        w: &mut DMatrix<usize>,
    ) {
        let num_sensors = m.len();
        let num_objects = dimensions[num_sensors];

        let debug = false; // Set to true for detailed debugging

        for s in 0..num_sensors {
            for i in 0..num_objects {
                // Starting measurement index (CRITICAL: must match MATLAB's k calculation)
                // MATLAB: k = V(i,s) + (V(i,s) == 0) where V stores measurement numbers 0,1,2,...
                // If V=0 (miss): k = 0 + 1 = 1 (start from measurement 1)
                // If V=1 (meas 1): k = 1 + 0 = 1 (start from measurement 1, reconsider current)
                // If V=2 (meas 2): k = 2 + 0 = 2 (start from measurement 2, reconsider current)
                // Rust v also stores measurement numbers 0,1,2,...
                // To match MATLAB's j loop (1-indexed inclusive), Rust needs j loop (0-indexed exclusive)
                // MATLAB j=1 corresponds to Rust j=0 (both are first measurement)
                // When v=0: MATLAB k=1 means start at j=1, Rust needs j=0
                // When v=1: MATLAB k=1 means start at j=1, Rust needs j=0
                // When v=2: MATLAB k=2 means start at j=2, Rust needs j=1
                let k = if v[(i, s)] == 0 { 0 } else { v[(i, s)] - 1 };

                if debug {
                    eprintln!(
                        "[GEN] sensor={}, object={}, v[(i,s)]={}, k={}, m[s]={}",
                        s,
                        i,
                        v[(i, s)],
                        k,
                        m[s]
                    );
                }

                // Build association vector u (1-indexed for internal indexing)
                let mut u: Vec<usize> = (0..num_sensors).map(|s_idx| v[(i, s_idx)] + 1).collect();
                u.push(i + 1); // Object index

                let mut found_association = false;

                for j in k..m[s] {
                    // Check if measurement j is available
                    if w[(j, s)] == 0 || w[(j, s)] == i + 1 {
                        // Detection case: j is 0-indexed, MATLAB uses (j+1)+1 for 1-indexed measurement
                        u[s] = j + 2;
                        let q_idx = Self::cartesian_to_linear(&u, dimensions);

                        // Miss case
                        u[s] = 1;
                        let r_idx = Self::cartesian_to_linear(&u, dimensions);

                        // Sampling probability
                        let p =
                            1.0 / ((log_likelihoods[r_idx] - log_likelihoods[q_idx]).exp() + 1.0);

                        let sample = Uniform01.sample(rng);

                        if debug {
                            eprintln!("  [LOOP] j={}, w[(j,s)]={}, L[r]={:.3}, L[q]={:.3}, p={:.3}, sample={:.3}",
                                j, w[(j, s)], log_likelihoods[r_idx], log_likelihoods[q_idx], p, sample);
                        }

                        if sample < p {
                            // Associate object i with measurement j
                            v[(i, s)] = j + 1;
                            w[(j, s)] = i + 1;
                            found_association = true;
                            if debug {
                                eprintln!("    -> ASSOCIATED v[(i,s)]={}", j + 1);
                            }
                            break;
                        } else {
                            // No association
                            v[(i, s)] = 0;
                            w[(j, s)] = 0;
                            if debug {
                                eprintln!("    -> MISS");
                            }
                        }
                    } else if debug {
                        eprintln!("  [SKIP] j={}, w[(j,s)]={} (not available)", j, w[(j, s)]);
                    }
                }

                if debug && !found_association {
                    eprintln!("  [END] No association made, v[(i,s)]={}", v[(i, s)]);
                }
            }
        }
    }
}

impl MultisensorAssociator for MultisensorGibbsAssociator {
    fn associate<R: rand::Rng>(
        &self,
        rng: &mut R,
        log_likelihoods: &[f64],
        dimensions: &[usize],
        config: &AssociationConfig,
    ) -> Result<MultisensorAssociationResult, AssociationError> {
        let num_sensors = dimensions.len() - 1;
        let num_objects = dimensions[num_sensors];

        if num_objects == 0 {
            return Ok(MultisensorAssociationResult::empty());
        }

        // m[s] = number of measurements from sensor s
        let m: Vec<usize> = dimensions[0..num_sensors].iter().map(|&d| d - 1).collect();
        let max_m = m.iter().copied().max().unwrap_or(0);

        // V[i, s] = measurement index assigned to object i by sensor s (0 = miss)
        let mut v = DMatrix::zeros(num_objects, num_sensors);
        // W[j, s] = object index assigned to measurement j by sensor s (0 = none)
        let mut w = DMatrix::zeros(max_m, num_sensors);

        let mut unique_samples = std::collections::HashSet::new();

        for _ in 0..config.gibbs_samples {
            // Generate one Gibbs sample
            Self::generate_association_event(rng, log_likelihoods, dimensions, &m, &mut v, &mut w);

            // Flatten V column-major to match MATLAB reshape(V, 1, n*S)
            // MATLAB reshape reads column-by-column: V(1,1), V(2,1), ..., V(n,1), V(1,2), V(2,2), ...
            let mut sample = Vec::with_capacity(num_objects * num_sensors);
            for s in 0..num_sensors {
                for i in 0..num_objects {
                    sample.push(v[(i, s)]);
                }
            }

            unique_samples.insert(sample);
        }

        let result: Vec<Vec<usize>> = unique_samples.into_iter().collect();
        eprintln!(
            "[DEBUG] MultisensorGibbs: {} Gibbs iterations -> {} unique samples",
            config.gibbs_samples,
            result.len()
        );

        Ok(MultisensorAssociationResult::new(
            result,
            config.gibbs_samples,
        ))
    }

    fn name(&self) -> &'static str {
        "MultisensorGibbs"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gibbs_associator_creation() {
        let associator = MultisensorGibbsAssociator::new();
        assert_eq!(associator.name(), "MultisensorGibbs");
    }

    #[test]
    fn test_empty_association() {
        let associator = MultisensorGibbsAssociator::new();
        let mut rng = rand::thread_rng();
        let config = AssociationConfig::default();

        // Empty case: 0 objects
        let dimensions = vec![3, 4, 0]; // 2 sensors, 0 objects
        let log_likelihoods = vec![];

        let result = associator
            .associate(&mut rng, &log_likelihoods, &dimensions, &config)
            .unwrap();
        assert_eq!(result.num_samples(), 0);
    }

    #[test]
    fn test_cartesian_linear_conversion() {
        let dimensions = vec![3, 4, 2]; // 2 measurements sensor 1, 3 measurements sensor 2, 2 objects

        // Test round-trip
        for ell in 1..=24 {
            // page_sizes for conversion
            let mut page_sizes = vec![1; dimensions.len()];
            for i in 1..dimensions.len() {
                page_sizes[i] = page_sizes[i - 1] * dimensions[i - 1];
            }

            let u = MultisensorGibbsAssociator::linear_to_cartesian(ell, &page_sizes);
            let ell_back = MultisensorGibbsAssociator::cartesian_to_linear(&u, &dimensions);

            // ell is 1-indexed going in, 0-indexed coming out
            assert_eq!(ell_back, ell - 1, "Round-trip failed for ell={}", ell);
        }
    }
}
