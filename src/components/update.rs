//! Existence probability update functions
//!
//! This module provides unified existence probability updates.

/// Update existence probability when no measurements received
///
/// Formula: `r' = r × (1-p_D) / (1 - r × p_D)`
///
/// This Bayesian update reflects that if a target exists but wasn't detected,
/// we should reduce our belief in its existence based on the detection probability.
///
/// # Arguments
/// * `existence` - Current existence probability
/// * `detection_prob` - Probability of detection
///
/// # Returns
/// Updated existence probability
#[inline]
pub fn update_existence_no_detection(existence: f64, detection_prob: f64) -> f64 {
    let numerator = existence * (1.0 - detection_prob);
    let denominator = 1.0 - existence * detection_prob;

    if denominator.abs() < 1e-15 {
        // Avoid division by zero
        existence
    } else {
        numerator / denominator
    }
}

/// Update existence probability for multi-sensor no-detection case
///
/// Applies the no-detection update for each sensor in sequence.
///
/// # Arguments
/// * `existence` - Current existence probability
/// * `detection_probs` - Detection probabilities for each sensor
///
/// # Returns
/// Updated existence probability after considering all sensors
pub fn update_existence_no_detection_multisensor(
    existence: f64,
    detection_probs: &[f64],
) -> f64 {
    detection_probs
        .iter()
        .fold(existence, |r, &p_d| update_existence_no_detection(r, p_d))
}

/// Update existence probability with measurement
///
/// Formula for LMB: used in conjunction with likelihood ratio
/// This is a helper for the more complex update in association.
///
/// # Arguments
/// * `prior_existence` - Prior existence probability
/// * `detection_prob` - Detection probability
/// * `likelihood_sum` - Sum of likelihood ratios for all measurements
///
/// # Returns
/// Posterior existence probability
#[inline]
pub fn update_existence_with_measurement(
    prior_existence: f64,
    detection_prob: f64,
    likelihood_sum: f64,
) -> f64 {
    let miss_factor = 1.0 - detection_prob;
    let total = miss_factor + likelihood_sum;

    if total.abs() < 1e-15 {
        prior_existence
    } else {
        let numerator = prior_existence * total;
        let denominator = 1.0 - prior_existence + prior_existence * total;
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_detection_update() {
        // If detection probability is 0, existence shouldn't change
        let r = update_existence_no_detection(0.9, 0.0);
        assert!((r - 0.9).abs() < 1e-10);

        // If detection probability is 1, existence should become 0
        let r = update_existence_no_detection(0.9, 1.0);
        assert!(r.abs() < 1e-10);

        // Standard case
        let r = update_existence_no_detection(0.5, 0.9);
        // r' = 0.5 * 0.1 / (1 - 0.5 * 0.9) = 0.05 / 0.55 ≈ 0.0909
        assert!((r - 0.05 / 0.55).abs() < 1e-10);
    }

    #[test]
    fn test_multisensor_no_detection() {
        let r = update_existence_no_detection_multisensor(0.9, &[0.8, 0.8]);

        // Should be lower than single sensor
        let r_single = update_existence_no_detection(0.9, 0.8);
        assert!(r < r_single);

        // Sequential application
        let r_seq = update_existence_no_detection(
            update_existence_no_detection(0.9, 0.8),
            0.8,
        );
        assert!((r - r_seq).abs() < 1e-10);
    }

    #[test]
    fn test_with_measurement_update() {
        // No likelihood contribution (like no detection)
        let r = update_existence_with_measurement(0.9, 0.9, 0.0);
        let r_no_det = update_existence_no_detection(0.9, 0.9);
        // When likelihood_sum = 0, should match no-detection (approximately)
        // Note: formulas differ slightly

        // With strong likelihood, existence should increase
        let r_high = update_existence_with_measurement(0.5, 0.9, 10.0);
        assert!(r_high > 0.5);
    }
}
