// Save this to tests/test_gibbs_actual.rs and run with: cargo test --test test_gibbs_actual -- --nocapture
use nalgebra::{DMatrix, DVector};
use prak::common::rng::SimpleRng;
use prak::common::types::Hypothesis;
use prak::multisensor_lmbm::gibbs::multisensor_lmbm_gibbs_sampling;
use prak::multisensor_lmbm::lazy::LazyLikelihood;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    model: FixtureModel,
    measurements: Vec<Vec<Vec<f64>>>,
    step1_prediction: Step1Prediction,
    step3_gibbs: Step3Gibbs,
}

#[derive(Debug, Deserialize)]
struct FixtureModel {
    #[serde(rename = "C")]
    c: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "Q")]
    q: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "P_d")]
    p_d: Vec<f64>,
    clutter_per_unit_volume: Vec<f64>,
    #[serde(rename = "numberOfSensors")]
    number_of_sensors: usize,
}

#[derive(Debug, Deserialize)]
struct Step1Prediction {
    output: PredictionOutput,
}

#[derive(Debug, Deserialize)]
struct PredictionOutput {
    predicted_hypothesis: FixtureHypothesis,
}

#[derive(Debug, Deserialize)]
struct FixtureHypothesis {
    w: f64,
    r: Vec<f64>,
    mu: Vec<Vec<f64>>,
    #[serde(rename = "Sigma")]
    sigma: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "birthTime")]
    birth_time: Vec<usize>,
    #[serde(rename = "birthLocation")]
    birth_location: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct Step3Gibbs {
    input: GibbsInput,
    output: GibbsOutput,
}

#[derive(Debug, Deserialize)]
struct GibbsInput {
    #[serde(rename = "numberOfSamples")]
    number_of_samples: usize,
    rng_seed: u64,
}

#[derive(Debug, Deserialize)]
struct GibbsOutput {
    #[serde(rename = "A")]
    a: Vec<Vec<usize>>,
}

fn convert_hypothesis(fh: &FixtureHypothesis) -> Hypothesis {
    Hypothesis {
        w: fh.w,
        r: fh.r.clone(),
        mu: fh.mu.iter().map(|v| DVector::from_vec(v.clone())).collect(),
        sigma: fh.sigma.iter().map(|m| {
            let n = m.len();
            let mut mat = DMatrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    mat[(i, j)] = m[i][j];
                }
            }
            mat
        }).collect(),
        birth_time: fh.birth_time.clone(),
        birth_location: fh.birth_location.clone(),
    }
}

fn convert_measurements(measurements: &[Vec<Vec<f64>>]) -> Vec<Vec<DVector<f64>>> {
    measurements
        .iter()
        .map(|sensor| sensor.iter().map(|m| DVector::from_vec(m.clone())).collect())
        .collect()
}

fn build_model_for_test(fixture: &Fixture) -> prak::common::types::Model {
    use prak::common::types::{DataAssociationMethod, ScenarioType};

    // Build a minimal model with the fixture's parameters
    let mut rng = SimpleRng::new(42);
    let mut model = prak::common::model::generate_model(
        &mut rng,
        10.0,
        0.9,
        DataAssociationMethod::Gibbs,
        ScenarioType::Fixed,
        None,
    );

    // Override with fixture values
    model.number_of_sensors = Some(fixture.model.number_of_sensors);
    model.detection_probability_multisensor = Some(fixture.model.p_d.clone());
    model.clutter_per_unit_volume_multisensor = Some(fixture.model.clutter_per_unit_volume.clone());

    // Set observation and noise matrices for each sensor
    model.c_multisensor = Some(fixture.model.c.iter().map(|c| {
        let rows = c.len();
        let cols = c[0].len();
        let mut mat = DMatrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                mat[(i, j)] = c[i][j];
            }
        }
        mat
    }).collect());

    model.q_multisensor = Some(fixture.model.q.iter().map(|q| {
        let n = q.len();
        let mut mat = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                mat[(i, j)] = q[i][j];
            }
        }
        mat
    }).collect());

    model
}

#[test]
fn test_gibbs_directly_from_fixture() {
    let fixture_path = "tests/data/step_by_step/multisensor_lmbm_step_by_step_seed42.json";
    let contents = std::fs::read_to_string(fixture_path).expect("Failed to read fixture");
    let fixture: Fixture = serde_json::from_str(&contents).expect("Failed to parse JSON");

    let input = &fixture.step3_gibbs.input;

    // Convert fixture data to Rust types
    let hypothesis = convert_hypothesis(&fixture.step1_prediction.output.predicted_hypothesis);
    let measurements = convert_measurements(&fixture.measurements);
    let model = build_model_for_test(&fixture);

    // Create lazy likelihood
    let lazy = LazyLikelihood::new(&hypothesis, &measurements, &model, fixture.model.number_of_sensors);

    let mut rng = SimpleRng::new(input.rng_seed);
    let samples = multisensor_lmbm_gibbs_sampling(&mut rng, &lazy, input.number_of_samples);

    println!("\nUsing ACTUAL library code with LazyLikelihood:");
    println!("Number of unique samples: {}", samples.nrows());
    println!("Entries computed: {} out of {} total ({:.1}%)",
        lazy.computed_count(),
        lazy.number_of_entries(),
        100.0 * lazy.computed_count() as f64 / lazy.number_of_entries() as f64
    );

    // Print first 5 samples
    for i in 0..5.min(samples.nrows()) {
        print!("Sample {}: [", i + 1);
        for j in 0..samples.ncols() {
            print!("{}", samples[(i, j)]);
            if j < samples.ncols() - 1 {
                print!(", ");
            }
        }
        println!("]");
    }

    // Compare with expected output
    let expected = &fixture.step3_gibbs.output.a;
    assert_eq!(samples.nrows(), expected.len(), "Should produce {} unique samples", expected.len());
}
