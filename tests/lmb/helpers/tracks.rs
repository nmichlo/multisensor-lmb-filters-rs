//! Track and Hypothesis comparison helpers
//!
//! This module provides reusable functions for comparing Track and
//! LmbmHypothesis structures against MATLAB fixtures.

use multisensor_lmb_filters_rs::lmb::{LmbmHypothesis, Track};

use super::assertions::{assert_dvector_close, assert_scalar_close};

/// Compare single Track against expected values
///
/// # Arguments
/// * `actual` - Rust Track to validate
/// * `expected_r` - Expected existence probability
/// * `expected_mu` - Expected component means (num_components × state_dim)
/// * `expected_sigma` - Expected covariances (num_components × dim × dim)
/// * `expected_w` - Expected component weights
/// * `tolerance` - Numerical tolerance (typically 1e-10)
/// * `track_idx` - Index for error messages
pub fn assert_track_close<T>(actual: &Track, expected: &T, tolerance: f64, track_idx: usize)
where
    T: TrackDataAccess,
{
    // Compare existence
    assert_scalar_close(
        actual.existence,
        expected.r(),
        tolerance,
        &format!("Track {} existence", track_idx),
    );

    // Compare number of components
    assert_eq!(
        actual.components.len(),
        expected.mu().len(),
        "Track {} component count mismatch",
        track_idx
    );

    // Compare each component
    for (comp_idx, comp) in actual.components.iter().enumerate() {
        // Component weight
        assert_scalar_close(
            comp.weight,
            expected.w()[comp_idx],
            tolerance,
            &format!("Track {} component {} weight", track_idx, comp_idx),
        );

        // Component mean
        let expected_mean_vec = nalgebra::DVector::from_vec(expected.mu()[comp_idx].clone());
        assert_dvector_close(
            &comp.mean,
            &expected_mean_vec,
            tolerance,
            &format!("Track {} component {} mean", track_idx, comp_idx),
        );

        // Component covariance
        let dim = comp.covariance.nrows();
        for row in 0..dim {
            for col in 0..dim {
                assert_scalar_close(
                    comp.covariance[(row, col)],
                    expected.sigma()[comp_idx][row][col],
                    tolerance,
                    &format!(
                        "Track {} component {} Sigma[{},{}]",
                        track_idx, comp_idx, row, col
                    ),
                );
            }
        }
    }

    // Compare label (if available)
    if let Some(label) = expected.label() {
        assert_eq!(
            actual.label.birth_time, label[1],
            "Track {} birthTime mismatch",
            track_idx
        );
        assert_eq!(
            actual.label.birth_location, label[0],
            "Track {} birthLocation mismatch",
            track_idx
        );
    }
}

/// Compare vector of Tracks
pub fn assert_tracks_close<T>(actual: &[Track], expected: &[T], tolerance: f64)
where
    T: TrackDataAccess,
{
    assert_eq!(
        actual.len(),
        expected.len(),
        "Track count mismatch (actual={}, expected={})",
        actual.len(),
        expected.len()
    );

    for (i, (actual_track, expected_track)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_track_close(actual_track, expected_track, tolerance, i);
    }
}

/// Compare single LmbmHypothesis against expected values
///
/// # Arguments
/// * `actual` - Rust LmbmHypothesis to validate
/// * `expected_w` - Expected log-weight
/// * `expected_r` - Expected existence probabilities (one per track)
/// * `expected_mu` - Expected means (one per track, each is state_dim)
/// * `expected_sigma` - Expected covariances (one per track, each is dim × dim)
/// * `expected_birth_time` - Expected birth times
/// * `expected_birth_location` - Expected birth locations
/// * `tolerance` - Numerical tolerance (typically 1e-10)
/// * `hyp_idx` - Index for error messages
pub fn assert_hypothesis_close<T>(
    actual: &LmbmHypothesis,
    expected: &T,
    tolerance: f64,
    hyp_idx: usize,
) where
    T: HypothesisDataAccess,
{
    // Compare log-weight (fixture w field stores log weights in LMBM context)
    assert_scalar_close(
        actual.log_weight,
        expected.w(),
        tolerance,
        &format!("Hypothesis {} log_weight", hyp_idx),
    );

    // Compare number of tracks
    assert_eq!(
        actual.tracks.len(),
        expected.r().len(),
        "Hypothesis {} track count mismatch",
        hyp_idx
    );

    // Compare existence probabilities (r)
    for (j, (track, &expected_r)) in actual.tracks.iter().zip(expected.r()).enumerate() {
        assert_scalar_close(
            track.existence,
            expected_r,
            tolerance,
            &format!("Hypothesis {} track {} existence", hyp_idx, j),
        );
    }

    // Compare means (mu)
    for (j, (track, expected_mu)) in actual.tracks.iter().zip(expected.mu()).enumerate() {
        assert_eq!(
            track.components.len(),
            1,
            "LMBM track should have exactly 1 component"
        );
        let actual_mu = &track.components[0].mean;
        assert_eq!(
            actual_mu.len(),
            expected_mu.len(),
            "Hypothesis {} track {} state dimension mismatch",
            hyp_idx,
            j
        );
        for (k, (&actual_val, &expected_val)) in
            actual_mu.iter().zip(expected_mu.iter()).enumerate()
        {
            assert_scalar_close(
                actual_val,
                expected_val,
                tolerance,
                &format!("Hypothesis {} track {} mu[{}]", hyp_idx, j, k),
            );
        }
    }

    // Compare covariances (Sigma)
    for (j, (track, expected_sigma)) in actual.tracks.iter().zip(expected.sigma()).enumerate() {
        let actual_sigma = &track.components[0].covariance;
        let dim = actual_sigma.nrows();
        assert_eq!(
            dim,
            expected_sigma.len(),
            "Hypothesis {} track {} covariance dimension mismatch",
            hyp_idx,
            j
        );
        for row in 0..dim {
            for col in 0..dim {
                assert_scalar_close(
                    actual_sigma[(row, col)],
                    expected_sigma[row][col],
                    tolerance,
                    &format!("Hypothesis {} track {} Sigma[{},{}]", hyp_idx, j, row, col),
                );
            }
        }
    }

    // Compare birthTime
    for (j, (track, &expected_bt)) in actual.tracks.iter().zip(expected.birth_time()).enumerate() {
        assert_eq!(
            track.label.birth_time, expected_bt,
            "Hypothesis {} track {} birthTime mismatch",
            hyp_idx, j
        );
    }

    // Compare birthLocation
    for (j, (track, &expected_bl)) in actual
        .tracks
        .iter()
        .zip(expected.birth_location())
        .enumerate()
    {
        assert_eq!(
            track.label.birth_location, expected_bl,
            "Hypothesis {} track {} birthLocation mismatch",
            hyp_idx, j
        );
    }
}

/// Compare vector of hypotheses
pub fn assert_hypotheses_close<T>(actual: &[LmbmHypothesis], expected: &[T], tolerance: f64)
where
    T: HypothesisDataAccess,
{
    assert_eq!(
        actual.len(),
        expected.len(),
        "Hypothesis count mismatch (actual={}, expected={})",
        actual.len(),
        expected.len()
    );

    for (i, (actual_hyp, expected_hyp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_hypothesis_close(actual_hyp, expected_hyp, tolerance, i);
    }
}

/// Trait to abstract over different Track data types
///
/// This allows the comparison function to work with different fixture types.
pub trait TrackDataAccess {
    fn r(&self) -> f64;
    fn mu(&self) -> &[Vec<f64>];
    fn sigma(&self) -> &[Vec<Vec<f64>>];
    fn w(&self) -> &[f64];
    fn label(&self) -> Option<&[usize]> {
        None
    }
    /// Birth time for variant fixtures that use separate fields
    fn birth_time(&self) -> usize {
        0
    }
    /// Birth location for variant fixtures that use separate fields
    fn birth_location(&self) -> usize {
        0
    }
}

/// Compare tracks using birth_time/birth_location fields (variant fixture format)
pub fn assert_variant_tracks_close<T>(actual: &[Track], expected: &[T], tolerance: f64)
where
    T: TrackDataAccess,
{
    assert_eq!(
        actual.len(),
        expected.len(),
        "Track count mismatch (actual={}, expected={})",
        actual.len(),
        expected.len()
    );

    for (i, (actual_track, expected_track)) in actual.iter().zip(expected.iter()).enumerate() {
        // Compare existence
        assert_scalar_close(
            actual_track.existence,
            expected_track.r(),
            tolerance,
            &format!("Track {} existence", i),
        );

        // Compare label using birth_time/birth_location methods
        assert_eq!(
            actual_track.label.birth_time,
            expected_track.birth_time(),
            "Track {} birthTime mismatch",
            i
        );
        assert_eq!(
            actual_track.label.birth_location,
            expected_track.birth_location(),
            "Track {} birthLocation mismatch",
            i
        );

        // Compare number of components
        assert_eq!(
            actual_track.components.len(),
            expected_track.mu().len(),
            "Track {} component count mismatch",
            i
        );

        // Compare each component
        for (comp_idx, comp) in actual_track.components.iter().enumerate() {
            // Component weight
            assert_scalar_close(
                comp.weight,
                expected_track.w()[comp_idx],
                tolerance,
                &format!("Track {} component {} weight", i, comp_idx),
            );

            // Component mean
            let expected_mean_vec =
                nalgebra::DVector::from_vec(expected_track.mu()[comp_idx].clone());
            assert_dvector_close(
                &comp.mean,
                &expected_mean_vec,
                tolerance,
                &format!("Track {} component {} mean", i, comp_idx),
            );

            // Component covariance
            let dim = comp.covariance.nrows();
            for row in 0..dim {
                for col in 0..dim {
                    assert_scalar_close(
                        comp.covariance[(row, col)],
                        expected_track.sigma()[comp_idx][row][col],
                        tolerance,
                        &format!("Track {} component {} Sigma[{},{}]", i, comp_idx, row, col),
                    );
                }
            }
        }
    }
}

/// Trait to abstract over different Hypothesis data types
///
/// This allows the comparison function to work with different fixture types.
pub trait HypothesisDataAccess {
    fn w(&self) -> f64;
    fn r(&self) -> &[f64];
    fn mu(&self) -> &[Vec<f64>];
    fn sigma(&self) -> &[Vec<Vec<f64>>];
    fn birth_time(&self) -> &[usize];
    fn birth_location(&self) -> &[usize];
}
