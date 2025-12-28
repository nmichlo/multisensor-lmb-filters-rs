//! Fixture loading and test setup utilities
//!
//! This module centralizes fixture loading and conversion to eliminate
//! duplicate setup code across test files.

use nalgebra::{DMatrix, DVector};
use smallvec::SmallVec;
use std::fs;

use multisensor_lmb_filters_rs::lmb::{GaussianComponent, SensorModel, Track, TrackLabel};

// Re-export fixture types (these will need to be made public in the test modules)
// For now, we'll define the conversion functions that can be used with the existing types

/// Preloaded fixture and common test setup for association tests
pub struct AssociationTestSetup {
    pub sensor: SensorModel,
    pub tracks: Vec<Track>,
    pub measurements: Vec<DVector<f64>>,
}

/// Convert MATLAB model data to SensorModel
///
/// # Arguments
/// * `c` - Measurement matrix (row-major nested Vec from MATLAB)
/// * `q` - Measurement noise covariance (row-major nested Vec)
/// * `p_d` - Probability of detection
/// * `clutter_per_unit_volume` - Clutter intensity
pub fn model_to_sensor(
    c: &[Vec<f64>],
    q: &[Vec<f64>],
    p_d: f64,
    clutter_per_unit_volume: f64,
) -> SensorModel {
    let z_dim = c.len();
    let x_dim = c[0].len();

    let c_matrix = DMatrix::from_row_slice(
        z_dim,
        x_dim,
        &c.iter()
            .flat_map(|row| row.iter())
            .copied()
            .collect::<Vec<_>>(),
    );

    let q_matrix = DMatrix::from_row_slice(
        z_dim,
        z_dim,
        &q.iter()
            .flat_map(|row| row.iter())
            .copied()
            .collect::<Vec<_>>(),
    );

    // MATLAB uses clutter_per_unit_volume directly
    // observation_space_volume = 40000 (default: 200x200 area)
    let observation_space_volume = 40000.0;
    let clutter_rate = clutter_per_unit_volume * observation_space_volume;

    SensorModel::new(
        c_matrix,
        q_matrix,
        p_d,
        clutter_rate,
        observation_space_volume,
    )
}

/// Convert MATLAB object data to Track
///
/// # Arguments
/// * `r` - Existence probability
/// * `label` - Label as [birthLocation, birthTime]
/// * `mu` - Component means (row-major nested Vec, num_components × state_dim)
/// * `sigma` - Component covariances (3D nested Vec, num_components × dim × dim)
/// * `w` - Component weights
pub fn object_data_to_track(
    r: f64,
    label: &[usize],
    mu: &[Vec<f64>],
    sigma: &[Vec<Vec<f64>>],
    w: &[f64],
) -> Track {
    let track_label = TrackLabel {
        birth_time: if label.len() >= 2 { label[1] } else { 0 },
        birth_location: if !label.is_empty() { label[0] } else { 0 },
    };

    let components: SmallVec<[GaussianComponent; 4]> = mu
        .iter()
        .zip(sigma.iter())
        .zip(w.iter())
        .map(|((mean_vec, cov_nested), &weight)| {
            let state_dim = mean_vec.len();
            let mean = DVector::from_vec(mean_vec.clone());

            // Convert nested Vec to DMatrix (row-major from MATLAB)
            let cov = DMatrix::from_row_slice(
                state_dim,
                state_dim,
                &cov_nested
                    .iter()
                    .flat_map(|row| row.iter())
                    .copied()
                    .collect::<Vec<_>>(),
            );

            GaussianComponent {
                weight,
                mean,
                covariance: cov,
            }
        })
        .collect();

    Track {
        existence: r,
        label: track_label,
        components,
        trajectory: None,
    }
}

/// Convert MATLAB measurements to Vec<DVector<f64>>
pub fn measurements_to_dvectors(measurements: &[Vec<f64>]) -> Vec<DVector<f64>> {
    measurements
        .iter()
        .map(|m| DVector::from_vec(m.clone()))
        .collect()
}

/// Load LMB fixture from standard JSON path
pub fn load_lmb_fixture_path() -> &'static str {
    "tests/data/step_by_step/lmb_step_by_step_seed42.json"
}

/// Load LMBM fixture from standard JSON path
pub fn load_lmbm_fixture_path() -> &'static str {
    "tests/data/step_by_step/lmbm_step_by_step_seed42.json"
}

/// Generic fixture loading helper
pub fn load_fixture_from_path<T: serde::de::DeserializeOwned>(path: &str) -> T {
    let fixture_data = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", path, e));
    serde_json::from_str(&fixture_data).unwrap_or_else(|e| panic!("Failed to parse fixture: {}", e))
}
