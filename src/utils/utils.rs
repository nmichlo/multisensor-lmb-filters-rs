//! Gaussian Mixture utilities for LMB filters
//!
//! Implements pruning and capping algorithms for Gaussian mixture components.
//! Matches MATLAB computePosteriorLmbSpatialDistributions.m and
//! lmbmNormalisationAndGating.m exactly.

/// Result from pruning a Gaussian mixture
#[derive(Debug, Clone)]
pub struct PrunedGaussianMixture {
    /// Pruned and normalized weights
    pub weights: Vec<f64>,
    /// Indices of kept components (in sorted order)
    pub indices: Vec<usize>,
    /// Number of components after pruning
    pub num_components: usize,
}

/// Prune Gaussian mixture components
///
/// Implements the "crude mixture reduction algorithm" from MATLAB.
/// Sorts components by weight, discards insignificant ones, and caps
/// to maximum number of components.
///
/// # Arguments
/// * `weights` - Component weights (unnormalized is OK)
/// * `weight_threshold` - Minimum weight to keep (e.g., 1e-6)
/// * `max_components` - Maximum number of components to keep (e.g., 5)
///
/// # Returns
/// PrunedGaussianMixture with normalized weights and sorted indices
///
/// # Implementation Notes
/// Matches MATLAB computePosteriorLmbSpatialDistributions.m lines 32-45:
/// - Sort weights in descending order
/// - Discard weights below threshold
/// - Renormalize
/// - Cap to maximum number of components
/// - Return sorted indices for reordering mu/Sigma
pub fn prune_gaussian_mixture(
    weights: &[f64],
    weight_threshold: f64,
    max_components: usize,
) -> PrunedGaussianMixture {
    let n = weights.len();

    // Normalize weights first
    let sum: f64 = weights.iter().sum();
    let normalized: Vec<f64> = if sum > 1e-15 {
        weights.iter().map(|w| w / sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    };

    // Create (weight, original_index) pairs and sort descending
    // Use stable sort (lower index wins for equal weights) to match MATLAB's stable sort behavior
    let mut indexed_weights: Vec<(f64, usize)> = normalized
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, i))
        .collect();
    // Use epsilon comparison to handle floating point precision differences
    // Weights within 1e-12 relative tolerance are considered equal
    indexed_weights.sort_by(|a, b| {
        let diff = (b.0 - a.0).abs();
        let max_val = b.0.abs().max(a.0.abs());
        let relative_diff = if max_val > 1e-15 {
            diff / max_val
        } else {
            diff
        };

        if relative_diff < 1e-12 {
            // Weights are effectively equal - use stable sort (lower index first)
            a.1.cmp(&b.1)
        } else {
            // Weights are different - sort descending by weight
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    // Filter out components below threshold
    let significant: Vec<(f64, usize)> = indexed_weights
        .into_iter()
        .filter(|(w, _)| *w > weight_threshold)
        .collect();

    // If all components were discarded, keep the largest one
    let significant = if significant.is_empty() {
        vec![(normalized[0], 0)]
    } else {
        significant
    };

    // Cap to maximum number of components
    let final_components: Vec<(f64, usize)> =
        significant.into_iter().take(max_components).collect();

    // Extract weights and indices
    let final_weights: Vec<f64> = final_components.iter().map(|(w, _)| *w).collect();
    let indices: Vec<usize> = final_components.iter().map(|(_, i)| *i).collect();
    let num_components = indices.len();

    // Renormalize final weights
    let final_sum: f64 = final_weights.iter().sum();
    let normalized_weights: Vec<f64> = if final_sum > 1e-15 {
        final_weights.iter().map(|w| w / final_sum).collect()
    } else {
        vec![1.0 / final_weights.len() as f64; final_weights.len()]
    };

    PrunedGaussianMixture {
        weights: normalized_weights,
        indices,
        num_components,
    }
}

/// Update existence probability for missed detection
///
/// Computes the Bayesian update when no measurement is associated to an object.
/// Formula: r' = r*(1-p_d) / (1 - r*p_d)
///
/// # Arguments
/// * `r` - Prior existence probability
/// * `detection_probability` - Detection probability p_d
///
/// # Returns
/// Updated existence probability
///
/// # Implementation Notes
/// Matches MATLAB runLmbFilter.m lines 54-56:
/// r' = r * (1 - p_D) / (1 - r * p_D)
#[inline]
pub fn update_existence_missed_detection(r: f64, detection_probability: f64) -> f64 {
    (r * (1.0 - detection_probability)) / (1.0 - r * detection_probability)
}

/// Prune objects by existence probability
///
/// Filters objects with existence probability below threshold.
/// Matches MATLAB lmbmNormalisationAndGating.m lines 41-42.
///
/// # Arguments
/// * `existence_probs` - Vector of existence probabilities
/// * `existence_threshold` - Minimum existence probability (e.g., 1e-2)
///
/// # Returns
/// Vector of booleans indicating which objects to keep
pub fn gate_objects_by_existence(existence_probs: &[f64], existence_threshold: f64) -> Vec<bool> {
    existence_probs
        .iter()
        .map(|&r| r > existence_threshold)
        .collect()
}

/// Normalize and prune hypothesis weights
///
/// Implements hypothesis management from MATLAB lmbmNormalisationAndGating.m:
/// - Normalizes log-weights
/// - Filters hypotheses below threshold
/// - Caps to maximum number of hypotheses
///
/// # Arguments
/// * `log_weights` - Log-space hypothesis weights
/// * `weight_threshold` - Minimum hypothesis weight (e.g., 1e-3)
/// * `max_hypotheses` - Maximum number of hypotheses (e.g., 25)
///
/// # Returns
/// Tuple of (normalized_weights, kept_indices)
pub fn prune_hypotheses(
    log_weights: &[f64],
    weight_threshold: f64,
    max_hypotheses: usize,
) -> (Vec<f64>, Vec<usize>) {
    // Normalize log-weights using log-sum-exp
    let max_log_w = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_shifted: Vec<f64> = log_weights.iter().map(|w| (w - max_log_w).exp()).collect();
    let sum_exp: f64 = exp_shifted.iter().sum();
    let weights: Vec<f64> = exp_shifted.iter().map(|w| w / sum_exp).collect();

    // Filter likely hypotheses
    let likely: Vec<(f64, usize)> = weights
        .iter()
        .enumerate()
        .filter_map(|(i, &w)| {
            if w > weight_threshold {
                Some((w, i))
            } else {
                None
            }
        })
        .collect();

    // If no hypotheses survive, keep the best one
    let likely = if likely.is_empty() {
        let (max_idx, &max_weight) = weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        vec![(max_weight, max_idx)]
    } else {
        likely
    };

    // Sort by weight descending
    let mut sorted = likely.clone();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Cap to maximum number of hypotheses
    let final_hyps: Vec<(f64, usize)> = sorted.into_iter().take(max_hypotheses).collect();

    // Extract and renormalize
    let final_weights: Vec<f64> = final_hyps.iter().map(|(w, _)| *w).collect();
    let indices: Vec<usize> = final_hyps.iter().map(|(_, i)| *i).collect();

    let sum: f64 = final_weights.iter().sum();
    let normalized: Vec<f64> = if sum > 1e-15 {
        final_weights.iter().map(|w| w / sum).collect()
    } else {
        vec![1.0 / final_weights.len() as f64; final_weights.len()]
    };

    (normalized, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_existence_missed_detection() {
        // r' = r*(1-p_d) / (1 - r*p_d)
        // For r=0.8, p_d=0.9: 0.8*0.1 / (1 - 0.8*0.9) = 0.08/0.28 ≈ 0.2857
        let r = update_existence_missed_detection(0.8, 0.9);
        let expected = (0.8 * 0.1) / (1.0 - 0.8 * 0.9);
        assert!((r - expected).abs() < 1e-10);

        // For r=0.5, p_d=0.5: 0.5*0.5 / (1 - 0.5*0.5) = 0.25/0.75 ≈ 0.3333
        let r = update_existence_missed_detection(0.5, 0.5);
        let expected = (0.5 * 0.5) / (1.0 - 0.5 * 0.5);
        assert!((r - expected).abs() < 1e-10);
    }

    #[test]
    fn test_prune_gaussian_mixture_simple() {
        let weights = vec![0.5, 0.3, 0.15, 0.04, 0.01];
        let result = prune_gaussian_mixture(&weights, 0.05, 10);

        // Should keep first 3 components (0.5, 0.3, 0.15)
        assert_eq!(result.num_components, 3);
        assert_eq!(result.indices, vec![0, 1, 2]);

        // Check normalization
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prune_gaussian_mixture_capping() {
        let weights = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let result = prune_gaussian_mixture(&weights, 0.01, 3);

        // Should cap to 3 components
        assert_eq!(result.num_components, 3);

        // Check normalization
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prune_gaussian_mixture_all_small() {
        let weights = vec![0.001, 0.0005, 0.0003];
        let result = prune_gaussian_mixture(&weights, 0.5, 10);

        // Should keep at least one (the largest)
        assert_eq!(result.num_components, 1);
        assert_eq!(result.indices, vec![0]);
        assert!((result.weights[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gate_objects_by_existence() {
        let existence_probs = vec![0.8, 0.05, 0.02, 0.5, 0.01];
        let keep = gate_objects_by_existence(&existence_probs, 0.03);

        assert_eq!(keep, vec![true, true, false, true, false]);
    }

    #[test]
    fn test_prune_hypotheses_simple() {
        let log_weights = vec![-1.0, -2.0, -0.5, -5.0];
        let (weights, _indices) = prune_hypotheses(&log_weights, 0.1, 10);

        // Should keep hypotheses with reasonable weights
        assert!(weights.len() <= 4);
        assert!(weights.len() > 0);

        // Check normalization
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Should be sorted by weight descending
        for i in 1..weights.len() {
            assert!(weights[i - 1] >= weights[i]);
        }
    }

    #[test]
    fn test_prune_hypotheses_capping() {
        let log_weights: Vec<f64> = (0..30).map(|i| -0.1 * i as f64).collect();
        let (weights, indices) = prune_hypotheses(&log_weights, 1e-6, 10);

        // Should cap to 10 hypotheses
        assert_eq!(weights.len(), 10);
        assert_eq!(indices.len(), 10);

        // Check normalization
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prune_hypotheses_all_low() {
        let log_weights = vec![-100.0, -101.0, -102.0];
        let (weights, indices) = prune_hypotheses(&log_weights, 0.9, 10);

        // Should keep at least one (the best)
        assert_eq!(weights.len(), 1);
        assert_eq!(indices.len(), 1);
        assert!((weights[0] - 1.0).abs() < 1e-10);
    }
}
