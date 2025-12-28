//! Association result comparison helpers
//!
//! This module provides reusable functions for comparing complex association
//! results against MATLAB fixtures.

use nalgebra::DVector;

use multisensor_lmb_filters_rs::association::PosteriorGrid;
use multisensor_lmb_filters_rs::lmb::AssociationResult;

use super::assertions::{assert_scalar_close, assert_vec_close};

/// Compare AssociationResult against MATLAB r and W
///
/// # Arguments
/// * `actual` - Rust AssociationResult from Gibbs/Murty/LBP
/// * `expected_r` - MATLAB posterior existence probabilities (n_tracks)
/// * `expected_w` - MATLAB marginal weights (n_tracks × n_measurements+1)
///                  Column 0 is miss probability, columns 1+ are measurements
/// * `tolerance` - Numerical tolerance (typically 1e-10)
/// * `test_name` - Name for error messages
///
/// # MATLAB vs Rust Structure
/// - MATLAB: W is (n_tracks, n_measurements+1) where col 0 is miss
/// - Rust: miss_weights (vector) + marginal_weights (matrix) separate
pub fn assert_association_result_close(
    actual: &AssociationResult,
    expected_r: &[f64],
    expected_w: &[Vec<f64>],
    tolerance: f64,
    test_name: &str,
) {
    // Compare r (posterior existence probabilities)
    // Convert DVector to slice for comparison
    assert_vec_close(
        actual.posterior_existence.as_slice(),
        expected_r,
        tolerance,
        &format!("{} r (posterior_existence)", test_name),
    );

    // Compare W (marginal association weights)
    // MATLAB W has shape (n_tracks, n_measurements+1) where col 0 is miss probability
    // Rust separates: miss_weights (col 0) and marginal_weights (cols 1+)
    let n_tracks = actual.marginal_weights.nrows();
    let n_meas = actual.marginal_weights.ncols();

    assert_eq!(
        n_tracks,
        expected_w.len(),
        "{} W: row count mismatch (Rust={}, MATLAB={})",
        test_name,
        n_tracks,
        expected_w.len()
    );

    let expected_num_cols = expected_w[0].len();
    assert_eq!(
        n_meas + 1,
        expected_num_cols,
        "{} W: column count mismatch (Rust={} + 1 miss col, MATLAB={})",
        test_name,
        n_meas,
        expected_num_cols
    );

    for i in 0..n_tracks {
        // Compare miss probability (column 0 in MATLAB)
        // Access DVector element
        assert_scalar_close(
            actual.miss_weights.as_slice()[i],
            expected_w[i][0],
            tolerance,
            &format!("{} W[{},0] (miss)", test_name, i),
        );

        // Compare measurement associations (columns 1+ in MATLAB)
        for j in 0..n_meas {
            let rust_w = actual.marginal_weights[(i, j)];
            let expected_w_val = expected_w[i][j + 1]; // MATLAB col j+1 is Rust measurement j
            assert_scalar_close(
                rust_w,
                expected_w_val,
                tolerance,
                &format!("{} W[{},{}]", test_name, i, j + 1),
            );
        }
    }
}

/// Compare PosteriorGrid against MATLAB posteriorParameters
///
/// # Arguments
/// * `actual_posteriors` - Rust PosteriorGrid from AssociationBuilder
/// * `expected_params` - MATLAB posteriorParameters (one entry per track)
/// * `num_measurements` - Number of measurements (needed for indexing)
/// * `tolerance` - Numerical tolerance (typically 1e-10)
///
/// # MATLAB Serialization Quirks
/// - Single-component tracks: w is 1D array [18 values] → deserialized as [[val1, ..., val18]]
/// - Multi-component tracks: w is 2D array [[c1,c2,...], [c1,c2,...], ...]
/// - mu/Sigma use column-major indexing: flat_idx = comp_idx * (num_meas + 1) + (meas_idx + 1)
pub fn assert_posterior_parameters_close<T>(
    actual_posteriors: &PosteriorGrid,
    expected_params: &[T],
    num_measurements: usize,
    tolerance: f64,
) where
    T: PosteriorParamsAccess,
{
    let num_tracks = expected_params.len();

    for track_idx in 0..num_tracks {
        let expected = &expected_params[track_idx];
        let num_components = actual_posteriors.num_components(track_idx);

        // Determine number of components from MATLAB fixture
        // MATLAB serializes single-component tracks as 1D arrays [18 values]
        // which deserialize_posterior_w converts to [[val1, ..., val18]]
        // Multi-component tracks are 2D: [[c1,c2,...], [c1,c2,...], ...]
        let matlab_num_components = if expected.w().len() == num_measurements + 1 {
            // Multi-component: (num_meas+1) rows × num_comp columns
            expected.w()[0].len()
        } else {
            // Single-component: 1D array deserialized as 1 row × (num_meas+1) columns
            1
        };

        assert_eq!(
            num_components, matlab_num_components,
            "Track {} component count mismatch: Rust={}, MATLAB={}",
            track_idx, num_components, matlab_num_components
        );

        // Compare component weights, means, and covariances for each measurement
        for meas_idx in 0..num_measurements {
            for comp_idx in 0..num_components {
                // Component weights (likelihood-normalized)
                let rust_w = actual_posteriors
                    .get_component_weight(track_idx, meas_idx, comp_idx)
                    .expect("component weight should exist");

                // Index into MATLAB w based on structure
                let expected_w_val = if matlab_num_components > 1 {
                    // Multi-component: w[meas+1][comp]
                    expected.w()[meas_idx + 1][comp_idx]
                } else {
                    // Single-component: w[0][meas+1]
                    expected.w()[0][meas_idx + 1]
                };

                assert_scalar_close(
                    rust_w,
                    expected_w_val,
                    tolerance,
                    &format!(
                        "Track {} meas {} comp {} weight",
                        track_idx, meas_idx, comp_idx
                    ),
                );

                // Posterior means
                let rust_mean = actual_posteriors
                    .get_mean_for_component(track_idx, meas_idx, comp_idx)
                    .expect("posterior mean should exist");

                // MATLAB mu is a 2D cell array (rows=meas+1, cols=comp) flattened in COLUMN-MAJOR order
                // flat_idx = comp_idx * (num_meas + 1) + (meas_idx + 1)
                let flat_idx = comp_idx * (num_measurements + 1) + (meas_idx + 1);
                let expected_mean = &expected.mu()[flat_idx];

                for state_idx in 0..rust_mean.len() {
                    assert_scalar_close(
                        rust_mean[state_idx],
                        expected_mean[state_idx],
                        tolerance,
                        &format!(
                            "Track {} meas {} comp {} mu[{}]",
                            track_idx, meas_idx, comp_idx, state_idx
                        ),
                    );
                }

                // Posterior covariances
                let rust_cov = actual_posteriors
                    .get_covariance_for_component(track_idx, meas_idx, comp_idx)
                    .expect("posterior covariance should exist");

                // Use same column-major flat_idx as mu
                let sigma_flat_idx = comp_idx * (num_measurements + 1) + (meas_idx + 1);
                let expected_cov = &expected.sigma()[sigma_flat_idx];

                for row in 0..rust_cov.nrows() {
                    for col in 0..rust_cov.ncols() {
                        assert_scalar_close(
                            rust_cov[(row, col)],
                            expected_cov[row][col],
                            tolerance,
                            &format!(
                                "Track {} meas {} comp {} Sigma[{},{}]",
                                track_idx, meas_idx, comp_idx, row, col
                            ),
                        );
                    }
                }
            }
        }
    }
}

/// Trait to abstract over different PosteriorParams types
///
/// This allows the comparison function to work with different test fixture types
/// that all have w, mu, and sigma fields.
pub trait PosteriorParamsAccess {
    fn w(&self) -> &[Vec<f64>];
    fn mu(&self) -> &[Vec<f64>];
    fn sigma(&self) -> &[Vec<Vec<f64>>];
}
