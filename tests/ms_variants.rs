//! MATLAB Equivalence Tests for ALL Multisensor LMB Variants
//!
//! These tests verify that ALL multisensor LMB filter variants (AA, GA, PU, IC)
//! produce IDENTICAL numerical results to the MATLAB reference implementation.
//!
//! Tests for each variant:
//! - AA-LMB (Arithmetic Average)
//! - GA-LMB (Geometric Average)
//! - PU-LMB (Parallel Update)
//! - IC-LMB (Iterated Corrector)

mod helpers;

use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use smallvec::SmallVec;
use std::fs;

use multisensor_lmb_filters_rs::association::AssociationBuilder;
use multisensor_lmb_filters_rs::lmb::{
    AssociationConfig, Associator, AssociatorLbp, DataAssociationMethod, GaussianComponent,
    SensorModel, Track, TrackLabel, Updater, UpdaterMarginal,
};

use helpers::fixtures::{deserialize_matrix, deserialize_posterior_w, deserialize_w};

const TOLERANCE: f64 = 1e-10;

/// Deserialize step5_fusion which can be either null, empty array [], or actual data
fn deserialize_optional_fusion<'de, D>(
    deserializer: D,
) -> Result<Option<VariantFusionStep>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct FusionVisitor;

    impl<'de> Visitor<'de> for FusionVisitor {
        type Value = Option<VariantFusionStep>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, empty array, or fusion step data")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            // Empty array means no fusion
            if seq.next_element::<serde::de::IgnoredAny>()?.is_none() {
                Ok(None)
            } else {
                Err(de::Error::custom("expected empty array for no fusion"))
            }
        }

        fn visit_map<M>(self, map: M) -> Result<Self::Value, M::Error>
        where
            M: de::MapAccess<'de>,
        {
            // Actual fusion data
            let fusion =
                VariantFusionStep::deserialize(de::value::MapAccessDeserializer::new(map))?;
            Ok(Some(fusion))
        }
    }

    deserializer.deserialize_any(FusionVisitor)
}

//=============================================================================
// Fixture Data Structures (new format with birthTime/birthLocation)
//=============================================================================

#[derive(Debug, Deserialize)]
struct VariantFixture {
    seed: u64,
    timestep: usize,
    #[serde(rename = "filterType")]
    filter_type: String,
    #[serde(rename = "numberOfSensors")]
    number_of_sensors: usize,
    model: VariantModelData,
    measurements: Vec<Vec<Vec<f64>>>,
    step1_prediction: VariantPredictionStep,
    #[serde(rename = "sensorUpdates")]
    sensor_updates: Vec<VariantSensorUpdateData>,
    #[serde(
        rename = "step5_fusion",
        deserialize_with = "deserialize_optional_fusion"
    )]
    step5_fusion: Option<VariantFusionStep>,
    #[serde(rename = "stepFinal_cardinality")]
    step_final_cardinality: VariantCardinalityStep,
}

#[derive(Debug, Deserialize)]
struct VariantModelData {
    #[serde(rename = "A")]
    a: Vec<Vec<f64>>,
    #[serde(rename = "R")]
    r: Vec<Vec<f64>>,
    #[serde(rename = "C")]
    c: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "Q")]
    q: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "P_s")]
    p_s: f64,
    #[serde(rename = "P_d")]
    p_d: Vec<f64>,
    clutter_per_unit_volume: Vec<f64>,
    #[serde(rename = "numberOfSensors")]
    number_of_sensors: usize,
}

#[derive(Debug, Deserialize)]
struct VariantObjectData {
    r: f64,
    #[serde(rename = "birthTime")]
    birth_time: usize,
    #[serde(rename = "birthLocation")]
    birth_location: usize,
    #[serde(rename = "numberOfGmComponents")]
    number_of_gm_components: usize,
    mu: Vec<Vec<f64>>,
    #[serde(rename = "Sigma")]
    sigma: Vec<Vec<Vec<f64>>>,
    #[serde(deserialize_with = "deserialize_w")]
    w: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct VariantPredictionStep {
    input: VariantPredictionInput,
    output: VariantPredictionOutput,
}

#[derive(Debug, Deserialize)]
struct VariantPredictionInput {
    prior_objects: Vec<VariantObjectData>,
}

#[derive(Debug, Deserialize)]
struct VariantPredictionOutput {
    predicted_objects: Vec<VariantObjectData>,
}

#[derive(Debug, Deserialize)]
struct VariantSensorUpdateData {
    #[serde(rename = "sensorIndex")]
    sensor_index: usize,
    input: VariantSensorInput,
    association: Option<VariantAssociationData>,
    #[serde(rename = "dataAssociation")]
    data_association: Option<VariantDataAssociationOutput>,
    output: VariantSensorOutput,
}

#[derive(Debug, Deserialize)]
struct VariantSensorInput {
    objects: Vec<VariantObjectData>,
    measurements: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct VariantAssociationData {
    #[serde(rename = "C", deserialize_with = "deserialize_matrix")]
    c: Vec<Vec<f64>>,
    #[serde(rename = "L", deserialize_with = "deserialize_matrix")]
    l: Vec<Vec<f64>>,
    #[serde(rename = "R", deserialize_with = "deserialize_matrix")]
    r: Vec<Vec<f64>>,
    #[serde(rename = "P", deserialize_with = "deserialize_matrix")]
    p: Vec<Vec<f64>>,
    eta: Vec<f64>,
    #[serde(rename = "posteriorParameters")]
    posterior_parameters: Vec<VariantPosteriorParams>,
}

#[derive(Debug, Deserialize)]
struct VariantPosteriorParams {
    mu: Vec<Vec<f64>>,
    #[serde(rename = "Sigma")]
    sigma: Vec<Vec<Vec<f64>>>,
    #[serde(deserialize_with = "deserialize_posterior_w")]
    w: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct VariantDataAssociationOutput {
    r: Vec<f64>,
    #[serde(rename = "W", deserialize_with = "deserialize_matrix")]
    w: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct VariantSensorOutput {
    updated_objects: Vec<VariantObjectData>,
}

#[derive(Debug, Deserialize)]
struct VariantFusionStep {
    fusion_type: String,
    input: VariantFusionInput,
    output: VariantFusionOutput,
}

#[derive(Debug, Deserialize)]
struct VariantFusionInput {
    per_sensor_objects: Vec<Vec<VariantObjectData>>,
    predicted_objects: Vec<VariantObjectData>,
}

#[derive(Debug, Deserialize)]
struct VariantFusionOutput {
    fused_objects: Vec<VariantObjectData>,
}

#[derive(Debug, Deserialize)]
struct VariantCardinalityStep {
    input: VariantCardinalityInput,
    output: VariantCardinalityOutput,
}

#[derive(Debug, Deserialize)]
struct VariantCardinalityInput {
    existence_probs: Vec<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MapIndices {
    Single(usize),
    Multiple(Vec<usize>),
}

impl MapIndices {
    fn as_vec(&self) -> Vec<usize> {
        match self {
            MapIndices::Single(v) => vec![*v],
            MapIndices::Multiple(v) => v.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct VariantCardinalityOutput {
    n_estimated: usize,
    map_indices: MapIndices,
}

//=============================================================================
// Trait Implementations for Helper Functions
//=============================================================================

impl helpers::tracks::TrackDataAccess for VariantObjectData {
    fn r(&self) -> f64 {
        self.r
    }
    fn mu(&self) -> &[Vec<f64>] {
        &self.mu
    }
    fn sigma(&self) -> &[Vec<Vec<f64>>] {
        &self.sigma
    }
    fn w(&self) -> &[f64] {
        &self.w
    }
    fn label(&self) -> Option<&[usize]> {
        None
    }
    fn birth_time(&self) -> usize {
        self.birth_time
    }
    fn birth_location(&self) -> usize {
        self.birth_location
    }
}

//=============================================================================
// Conversion Helpers
//=============================================================================

fn variant_objects_to_tracks(objects: &[VariantObjectData]) -> Vec<Track> {
    objects
        .iter()
        .map(|obj| {
            let label = TrackLabel {
                birth_time: obj.birth_time,
                birth_location: obj.birth_location,
            };

            let n_components = obj.w.len();
            let mut components = SmallVec::with_capacity(n_components);

            for i in 0..n_components {
                let mean = DVector::from_vec(obj.mu[i].clone());
                let n = obj.sigma[i].len();
                let cov = DMatrix::from_row_slice(
                    n,
                    n,
                    &obj.sigma[i]
                        .iter()
                        .flat_map(|row| row.iter())
                        .copied()
                        .collect::<Vec<_>>(),
                );

                components.push(GaussianComponent {
                    weight: obj.w[i],
                    mean,
                    covariance: cov,
                });
            }

            Track {
                label,
                existence: obj.r,
                components,
                trajectory: None,
            }
        })
        .collect()
}

fn variant_model_to_sensor(model: &VariantModelData, sensor_idx: usize) -> SensorModel {
    let c_sensor = &model.c[sensor_idx];
    let q_sensor = &model.q[sensor_idx];

    let z_dim = c_sensor.len();
    let x_dim = c_sensor[0].len();
    let c = DMatrix::from_row_slice(
        z_dim,
        x_dim,
        &c_sensor
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .collect::<Vec<_>>(),
    );
    let q = DMatrix::from_row_slice(
        z_dim,
        z_dim,
        &q_sensor
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .collect::<Vec<_>>(),
    );

    let observation_space_volume = 40000.0;
    let clutter_rate = model.clutter_per_unit_volume[sensor_idx] * observation_space_volume;

    SensorModel::new(
        c,
        q,
        model.p_d[sensor_idx],
        clutter_rate,
        observation_space_volume,
    )
}

fn measurements_to_dvectors(measurements: &[Vec<f64>]) -> Vec<DVector<f64>> {
    measurements
        .iter()
        .map(|m| DVector::from_vec(m.clone()))
        .collect()
}

//=============================================================================
// Test Loading Functions
//=============================================================================

fn load_variant_fixture(variant: &str) -> VariantFixture {
    let fixture_path = format!("tests/fixtures/step_ms_{}_lmb_seed42.json", variant);
    let fixture_data = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", fixture_path, e));
    serde_json::from_str(&fixture_data)
        .unwrap_or_else(|e| panic!("Failed to parse fixture {}: {}", fixture_path, e))
}

//=============================================================================
// Common Test Functions
//=============================================================================

fn test_prediction_equivalence(variant: &str) {
    let fixture = load_variant_fixture(variant);
    println!(
        "Testing {}-LMB prediction step against MATLAB...",
        variant.to_uppercase()
    );

    let prior = &fixture.step1_prediction.input.prior_objects;
    let expected = &fixture.step1_prediction.output.predicted_objects;

    let p_s = fixture.model.p_s;
    for (i, (prior_obj, expected_obj)) in prior.iter().zip(expected.iter()).enumerate() {
        let computed_r = prior_obj.r * p_s;
        let diff = (computed_r - expected_obj.r).abs();
        assert!(
            diff <= TOLERANCE,
            "{}-LMB Track {} existence: computed {} vs MATLAB {} (diff: {:.2e})",
            variant.to_uppercase(),
            i,
            computed_r,
            expected_obj.r,
            diff
        );
    }

    println!(
        "  ✓ {}-LMB prediction matches MATLAB",
        variant.to_uppercase()
    );
}

fn test_association_matrices_equivalence(variant: &str) {
    let fixture = load_variant_fixture(variant);

    for sensor_update in &fixture.sensor_updates {
        let sensor_idx = sensor_update.sensor_index - 1;

        let association = match &sensor_update.association {
            Some(a) => a,
            None => continue,
        };

        println!(
            "Testing {}-LMB sensor {} association matrices against MATLAB...",
            variant.to_uppercase(),
            sensor_idx
        );

        let sensor = variant_model_to_sensor(&fixture.model, sensor_idx);
        let tracks = variant_objects_to_tracks(&sensor_update.input.objects);
        let measurements = measurements_to_dvectors(&sensor_update.input.measurements);

        let mut builder = AssociationBuilder::new(&tracks, &sensor);
        let matrices = builder.build(&measurements);

        // Compare P matrix
        for (i, expected_row) in association.p.iter().enumerate() {
            for (j, &expected_val) in expected_row.iter().enumerate() {
                let rust_val = matrices.sampling_prob[(i, j)];
                let diff = (rust_val - expected_val).abs();
                assert!(
                    diff <= TOLERANCE,
                    "{}-LMB sensor {} P[{},{}]: {} vs MATLAB {} (diff: {:.2e})",
                    variant.to_uppercase(),
                    sensor_idx,
                    i,
                    j,
                    rust_val,
                    expected_val,
                    diff
                );
            }
        }

        // Compare eta
        for (i, &expected_val) in association.eta.iter().enumerate() {
            let rust_val = matrices.eta[i];
            let diff = (rust_val - expected_val).abs();
            assert!(
                diff <= TOLERANCE,
                "{}-LMB sensor {} eta[{}]: {} vs MATLAB {} (diff: {:.2e})",
                variant.to_uppercase(),
                sensor_idx,
                i,
                rust_val,
                expected_val,
                diff
            );
        }

        println!(
            "  ✓ {}-LMB sensor {} association matrices match MATLAB",
            variant.to_uppercase(),
            sensor_idx
        );
    }
}

fn test_data_association_equivalence(variant: &str) {
    let fixture = load_variant_fixture(variant);

    for sensor_update in &fixture.sensor_updates {
        let sensor_idx = sensor_update.sensor_index - 1;

        let expected_da = match &sensor_update.data_association {
            Some(da) => da,
            None => continue,
        };

        println!(
            "Testing {}-LMB sensor {} data association against MATLAB...",
            variant.to_uppercase(),
            sensor_idx
        );

        let sensor = variant_model_to_sensor(&fixture.model, sensor_idx);
        let tracks = variant_objects_to_tracks(&sensor_update.input.objects);
        let measurements = measurements_to_dvectors(&sensor_update.input.measurements);

        let mut builder = AssociationBuilder::new(&tracks, &sensor);
        let matrices = builder.build(&measurements);

        let config = AssociationConfig {
            method: DataAssociationMethod::Lbp,
            lbp_max_iterations: 1000,
            lbp_tolerance: 1e-6,
            ..Default::default()
        };

        let associator = AssociatorLbp;
        let mut rng = rand::thread_rng();
        let result = associator.associate(&matrices, &config, &mut rng).unwrap();

        helpers::association::assert_association_result_close(
            &result,
            &expected_da.r,
            &expected_da.w,
            TOLERANCE,
            &format!(
                "{}-LMB sensor {} data association",
                variant.to_uppercase(),
                sensor_idx
            ),
        );

        println!(
            "  ✓ {}-LMB sensor {} data association matches MATLAB",
            variant.to_uppercase(),
            sensor_idx
        );
    }
}

fn test_update_output_equivalence(variant: &str) {
    use multisensor_lmb_filters_rs::utils::common_ops::update_existence_from_marginals;

    let fixture = load_variant_fixture(variant);

    for sensor_update in &fixture.sensor_updates {
        let sensor_idx = sensor_update.sensor_index - 1;

        if sensor_update.association.is_none() {
            continue;
        }

        println!(
            "Testing {}-LMB sensor {} update output against MATLAB...",
            variant.to_uppercase(),
            sensor_idx
        );

        let sensor = variant_model_to_sensor(&fixture.model, sensor_idx);
        let mut tracks = variant_objects_to_tracks(&sensor_update.input.objects);
        let measurements = measurements_to_dvectors(&sensor_update.input.measurements);

        let mut builder = AssociationBuilder::new(&tracks, &sensor);
        let matrices = builder.build(&measurements);

        let config = AssociationConfig {
            method: DataAssociationMethod::Lbp,
            lbp_max_iterations: 1000,
            lbp_tolerance: 1e-6,
            ..Default::default()
        };

        let associator = AssociatorLbp;
        let mut rng = rand::thread_rng();
        let result = associator.associate(&matrices, &config, &mut rng).unwrap();

        // Use MATLAB's parameters: gm_weight=1e-6, max_components=20
        let updater = UpdaterMarginal::with_thresholds(1e-6, 20, f64::INFINITY);
        updater.update(&mut tracks, &result, &matrices.posteriors);
        update_existence_from_marginals(&mut tracks, &result);

        helpers::tracks::assert_variant_tracks_close(
            &tracks,
            &sensor_update.output.updated_objects,
            TOLERANCE,
        );

        println!(
            "  ✓ {}-LMB sensor {} update output matches MATLAB",
            variant.to_uppercase(),
            sensor_idx
        );
    }
}

fn test_fusion_equivalence(variant: &str) {
    use multisensor_lmb_filters_rs::lmb::{
        Merger, MergerAverageArithmetic, MergerAverageGeometric, MergerParallelUpdate,
    };

    let fixture = load_variant_fixture(variant);

    // IC-LMB has no fusion step
    if fixture.step5_fusion.is_none() {
        println!(
            "  ⏭ {}-LMB has no fusion step (sequential processing)",
            variant.to_uppercase()
        );
        return;
    }

    let fusion = fixture.step5_fusion.as_ref().unwrap();

    println!(
        "Testing {}-LMB fusion against MATLAB...",
        variant.to_uppercase()
    );

    // Convert per-sensor tracks from fixture
    let per_sensor_tracks: Vec<Vec<Track>> = fusion
        .input
        .per_sensor_objects
        .iter()
        .map(|objs| variant_objects_to_tracks(objs))
        .collect();

    // Get sensor weights from model (uniform for this fixture)
    let num_sensors = fixture.model.p_d.len();
    let weights: Vec<f64> = vec![1.0 / num_sensors as f64; num_sensors];

    // Apply the appropriate merger
    let fused_tracks: Vec<Track> = match variant {
        "aa" => {
            let merger = MergerAverageArithmetic::with_weights(weights.clone(), 20);
            merger.merge(&per_sensor_tracks, Some(&weights))
        }
        "ga" => {
            let merger = MergerAverageGeometric::with_weights(weights.clone());
            merger.merge(&per_sensor_tracks, Some(&weights))
        }
        "pu" => {
            // PU merger needs predicted tracks (prior) for decorrelation
            let predicted_tracks = variant_objects_to_tracks(&fusion.input.predicted_objects);
            let mut merger = MergerParallelUpdate::new(Vec::new());
            merger.set_prior(predicted_tracks);
            merger.merge(&per_sensor_tracks, None)
        }
        _ => panic!("Unknown variant: {}", variant),
    };

    // Compare against expected
    let expected = &fusion.output.fused_objects;

    assert_eq!(
        fused_tracks.len(),
        expected.len(),
        "{}-LMB fusion: track count mismatch (got {}, expected {})",
        variant.to_uppercase(),
        fused_tracks.len(),
        expected.len()
    );

    for (i, (actual, exp)) in fused_tracks.iter().zip(expected.iter()).enumerate() {
        helpers::assertions::assert_scalar_close(
            actual.existence,
            exp.r,
            TOLERANCE,
            &format!("{}-LMB fused track {} existence", variant.to_uppercase(), i),
        );

        assert_eq!(
            actual.label.birth_time,
            exp.birth_time,
            "{}-LMB fused track {} birth_time mismatch",
            variant.to_uppercase(),
            i
        );

        assert_eq!(
            actual.label.birth_location,
            exp.birth_location,
            "{}-LMB fused track {} birth_location mismatch",
            variant.to_uppercase(),
            i
        );

        assert_eq!(
            actual.components.len(),
            exp.mu.len(),
            "{}-LMB fused track {} component count mismatch",
            variant.to_uppercase(),
            i
        );

        for (j, comp) in actual.components.iter().enumerate() {
            helpers::assertions::assert_scalar_close(
                comp.weight,
                exp.w[j],
                TOLERANCE,
                &format!(
                    "{}-LMB fused track {} component {} weight",
                    variant.to_uppercase(),
                    i,
                    j
                ),
            );

            for (k, (&actual_val, &expected_val)) in
                comp.mean.iter().zip(exp.mu[j].iter()).enumerate()
            {
                helpers::assertions::assert_scalar_close(
                    actual_val,
                    expected_val,
                    TOLERANCE,
                    &format!(
                        "{}-LMB fused track {} component {} mean[{}]",
                        variant.to_uppercase(),
                        i,
                        j,
                        k
                    ),
                );
            }

            for row in 0..comp.covariance.nrows() {
                for col in 0..comp.covariance.ncols() {
                    helpers::assertions::assert_scalar_close(
                        comp.covariance[(row, col)],
                        exp.sigma[j][row][col],
                        TOLERANCE,
                        &format!(
                            "{}-LMB fused track {} component {} Sigma[{},{}]",
                            variant.to_uppercase(),
                            i,
                            j,
                            row,
                            col
                        ),
                    );
                }
            }
        }
    }

    println!(
        "  ✓ {}-LMB fusion matches MATLAB ({} tracks)",
        variant.to_uppercase(),
        fused_tracks.len()
    );
}

fn test_cardinality_equivalence(variant: &str) {
    use multisensor_lmb_filters_rs::cardinality::lmb_map_cardinality_estimate;

    let fixture = load_variant_fixture(variant);

    println!(
        "Testing {}-LMB cardinality against MATLAB...",
        variant.to_uppercase()
    );

    let existence_probs = &fixture.step_final_cardinality.input.existence_probs;
    let expected = &fixture.step_final_cardinality.output;

    let (n_estimated, map_indices) = lmb_map_cardinality_estimate(existence_probs);

    assert_eq!(
        n_estimated,
        expected.n_estimated,
        "{}-LMB n_estimated: expected {}, got {}",
        variant.to_uppercase(),
        expected.n_estimated,
        n_estimated
    );

    let expected_indices = expected.map_indices.as_vec();
    assert_eq!(
        map_indices.len(),
        expected_indices.len(),
        "{}-LMB map_indices length mismatch",
        variant.to_uppercase()
    );

    for (i, (&actual, &expected_val)) in map_indices.iter().zip(expected_indices.iter()).enumerate()
    {
        assert_eq!(
            actual + 1,
            expected_val,
            "{}-LMB map_indices[{}]: expected {}, got {}",
            variant.to_uppercase(),
            i,
            expected_val,
            actual + 1
        );
    }

    println!(
        "  ✓ {}-LMB cardinality matches MATLAB",
        variant.to_uppercase()
    );
}

//=============================================================================
// AA-LMB Tests
//=============================================================================

#[test]
fn test_aa_lmb_prediction_equivalence() {
    test_prediction_equivalence("aa");
}

#[test]
fn test_aa_lmb_association_matrices_equivalence() {
    test_association_matrices_equivalence("aa");
}

#[test]
fn test_aa_lmb_data_association_equivalence() {
    test_data_association_equivalence("aa");
}

#[test]
fn test_aa_lmb_update_output_equivalence() {
    test_update_output_equivalence("aa");
}

#[test]
fn test_aa_lmb_cardinality_equivalence() {
    test_cardinality_equivalence("aa");
}

#[test]
fn test_aa_lmb_fusion_equivalence() {
    test_fusion_equivalence("aa");
}

//=============================================================================
// GA-LMB Tests
//=============================================================================

#[test]
fn test_ga_lmb_prediction_equivalence() {
    test_prediction_equivalence("ga");
}

#[test]
fn test_ga_lmb_association_matrices_equivalence() {
    test_association_matrices_equivalence("ga");
}

#[test]
fn test_ga_lmb_data_association_equivalence() {
    test_data_association_equivalence("ga");
}

#[test]
fn test_ga_lmb_update_output_equivalence() {
    test_update_output_equivalence("ga");
}

#[test]
fn test_ga_lmb_cardinality_equivalence() {
    test_cardinality_equivalence("ga");
}

#[test]
fn test_ga_lmb_fusion_equivalence() {
    test_fusion_equivalence("ga");
}

//=============================================================================
// PU-LMB Tests
//=============================================================================

#[test]
fn test_pu_lmb_prediction_equivalence() {
    test_prediction_equivalence("pu");
}

#[test]
fn test_pu_lmb_association_matrices_equivalence() {
    test_association_matrices_equivalence("pu");
}

#[test]
fn test_pu_lmb_data_association_equivalence() {
    test_data_association_equivalence("pu");
}

#[test]
fn test_pu_lmb_update_output_equivalence() {
    test_update_output_equivalence("pu");
}

#[test]
fn test_pu_lmb_cardinality_equivalence() {
    test_cardinality_equivalence("pu");
}

#[test]
fn test_pu_lmb_fusion_equivalence() {
    test_fusion_equivalence("pu");
}

//=============================================================================
// IC-LMB Tests
//=============================================================================

#[test]
fn test_ic_lmb_prediction_equivalence() {
    test_prediction_equivalence("ic");
}

#[test]
fn test_ic_lmb_association_matrices_equivalence() {
    test_association_matrices_equivalence("ic");
}

#[test]
fn test_ic_lmb_data_association_equivalence() {
    test_data_association_equivalence("ic");
}

#[test]
fn test_ic_lmb_update_output_equivalence() {
    test_update_output_equivalence("ic");
}

#[test]
fn test_ic_lmb_cardinality_equivalence() {
    test_cardinality_equivalence("ic");
}

//=============================================================================
// Summary Test
//=============================================================================

#[test]
fn test_multisensor_lmb_variants_matlab_equivalence_summary() {
    println!("\n========================================");
    println!("Multisensor LMB Variants MATLAB Equivalence");
    println!("========================================");
    println!("Testing AA-LMB, GA-LMB, PU-LMB, IC-LMB variants");
    println!("Tolerance: {:.0e}", TOLERANCE);
    println!("----------------------------------------\n");
}
