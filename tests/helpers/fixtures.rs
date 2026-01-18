//! Fixture loading and deserialization utilities
//!
//! This module centralizes fixture loading and conversion to eliminate
//! duplicate setup code across test files.

use nalgebra::{DMatrix, DVector};
use smallvec::SmallVec;
use std::fs;

use multisensor_lmb_filters_rs::lmb::{GaussianComponent, SensorModel, Track, TrackLabel};

//=============================================================================
// Deserialization Helpers
//=============================================================================

/// Deserialize MATLAB weight field that can be scalar or array
///
/// MATLAB serializes single-component weights as scalars, but multi-component
/// weights as arrays. This handles both cases.
pub fn deserialize_w<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Deserialize};

    struct WVisitor;

    impl<'de> de::Visitor<'de> for WVisitor {
        type Value = Vec<f64>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a number or array of numbers")
        }

        fn visit_f64<E>(self, value: f64) -> Result<Vec<f64>, E>
        where
            E: de::Error,
        {
            Ok(vec![value])
        }

        fn visit_i64<E>(self, value: i64) -> Result<Vec<f64>, E>
        where
            E: de::Error,
        {
            Ok(vec![value as f64])
        }

        fn visit_u64<E>(self, value: u64) -> Result<Vec<f64>, E>
        where
            E: de::Error,
        {
            Ok(vec![value as f64])
        }

        fn visit_seq<A>(self, seq: A) -> Result<Vec<f64>, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(WVisitor)
}

/// Deserialize MATLAB P_s field that can be scalar or single-element array
///
/// MATLAB sometimes serializes scalar values as 1-element arrays.
pub fn deserialize_p_s<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct PSVisitor;

    impl<'de> de::Visitor<'de> for PSVisitor {
        type Value = f64;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a float or array of floats")
        }

        fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(value)
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            seq.next_element()?
                .ok_or_else(|| de::Error::custom("empty array for P_s"))
        }
    }

    deserializer.deserialize_any(PSVisitor)
}

/// Deserialize MATLAB matrix that may contain null values (representing infinity)
///
/// MATLAB JSON export uses null for infinity values. This converts them to f64::INFINITY.
pub fn deserialize_matrix<'de, D>(deserializer: D) -> Result<Vec<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let matrix: Vec<Vec<Option<f64>>> = Deserialize::deserialize(deserializer)?;
    Ok(matrix
        .iter()
        .map(|row| row.iter().map(|&v| v.unwrap_or(f64::INFINITY)).collect())
        .collect())
}

/// Deserialize MATLAB posterior weights that can be 1D or 2D
///
/// MATLAB serializes single-component posterior weights as 1D arrays,
/// but multi-component as 2D arrays. This handles both cases.
pub fn deserialize_posterior_w<'de, D>(deserializer: D) -> Result<Vec<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct WVisitor;

    impl<'de> de::Visitor<'de> for WVisitor {
        type Value = Vec<Vec<f64>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a 1D or 2D array of numbers")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Vec<Vec<f64>>, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut result = Vec::new();

            if let Some(first) = seq.next_element::<serde_json::Value>()? {
                if first.is_array() {
                    let first_row: Vec<f64> = serde_json::from_value(first).map_err(|e| {
                        de::Error::custom(format!("Failed to parse first row: {}", e))
                    })?;
                    result.push(first_row);

                    while let Some(row) = seq.next_element::<Vec<f64>>()? {
                        result.push(row);
                    }
                } else {
                    let first_val: f64 = serde_json::from_value(first).map_err(|e| {
                        de::Error::custom(format!("Failed to parse first value: {}", e))
                    })?;
                    let mut row = vec![first_val];

                    while let Some(val) = seq.next_element::<f64>()? {
                        row.push(val);
                    }
                    result.push(row);
                }
            }

            Ok(result)
        }
    }

    deserializer.deserialize_seq(WVisitor)
}

/// Deserialize MATLAB V matrix (integer assignment matrix)
pub fn deserialize_v_matrix<'de, D>(deserializer: D) -> Result<Vec<Vec<i32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    Deserialize::deserialize(deserializer)
}

/// Deserialize MATLAB i32 matrix
pub fn deserialize_matrix_i32<'de, D>(deserializer: D) -> Result<Vec<Vec<i32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    Deserialize::deserialize(deserializer)
}

//=============================================================================
// Fixture Loading Helpers
//=============================================================================

/// Preloaded fixture and utils test setup for association tests
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
    "tests/fixtures/step_ss_lmb_seed42.json"
}

/// Load LMBM fixture from standard JSON path
pub fn load_lmbm_fixture_path() -> &'static str {
    "tests/fixtures/step_ss_lmbm_seed42.json"
}

/// Load multisensor LMB fixture from standard JSON path
pub fn load_multisensor_lmb_fixture_path() -> &'static str {
    "tests/fixtures/step_ms_lmb_seed42.json"
}

/// Load multisensor LMBM fixture from standard JSON path
pub fn load_multisensor_lmbm_fixture_path() -> &'static str {
    "tests/fixtures/step_ms_lmbm_seed42.json"
}

/// Generic fixture loading helper
pub fn load_fixture_from_path<T: serde::de::DeserializeOwned>(path: &str) -> T {
    let fixture_data = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", path, e));
    serde_json::from_str(&fixture_data).unwrap_or_else(|e| panic!("Failed to parse fixture: {}", e))
}
