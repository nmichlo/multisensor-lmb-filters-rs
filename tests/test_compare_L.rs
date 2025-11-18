use prak::common::types::{Hypothesis, Model};
use prak::multisensor_lmbm::association::generate_multisensor_lmbm_association_matrices;
use prak::common::rng::SimpleRng;
use nalgebra::{DMatrix, DVector};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    number_of_sensors: usize,
    step2_association: Step2,
}

#[derive(Debug, Deserialize)]
struct Step2 {
    input: AssocInput,
    output: AssocOutput,
}

#[derive(Debug, Deserialize)]
struct AssocInput {
    predicted_hypothesis: HypothesisData,
    measurements: Vec<Vec<Vec<f64>>>,
}

#[derive(Debug, Deserialize)]
struct AssocOutput {
    #[serde(rename = "L")]
    l: Vec<Vec<Vec<f64>>>,
}

#[derive(Debug, Deserialize)]
struct HypothesisData {
    w: f64,
    r: Vec<f64>,
    mu: Vec<Vec<f64>>,
    sigma: Vec<Vec<Vec<f64>>>,
}

#[test]
fn test_compare_L_matrices() {
    let fixture_path = "tests/data/step_by_step/multisensor_lmbm_step_by_step_seed42.json";
    let contents = std::fs::read_to_string(fixture_path).expect("Failed to read fixture");
    let fixture: Fixture = serde_json::from_str(&contents).expect("Failed to parse JSON");

    // Convert measurements
    let measurements: Vec<Vec<DVector<f64>>> = fixture.step2_association.input.measurements.iter()
        .map(|sensor_meas| sensor_meas.iter()
            .map(|m| DVector::from_vec(m.clone()))
            .collect())
        .collect();

    // Convert hypothesis
    let mu: Vec<DVector<f64>> = fixture.step2_association.input.predicted_hypothesis.mu.iter()
        .map(|v| DVector::from_vec(v.clone()))
        .collect();
    let sigma: Vec<DMatrix<f64>> = fixture.step2_association.input.predicted_hypothesis.sigma.iter()
        .map(|s| DMatrix::from_row_slice(s.len(), s[0].len(), &s.iter().flatten().cloned().collect::<Vec<_>>()))
        .collect();

    let hypothesis = Hypothesis {
        w: fixture.step2_association.input.predicted_hypothesis.w,
        r: fixture.step2_association.input.predicted_hypothesis.r.clone(),
        mu,
        sigma,
    };

    // Create a minimal model (values don't matter for association matrices)
    let model = Model {
        x_dimension: 4,
        z_dimension: 2,
        a: DMatrix::identity(4, 4),
        r_process: DMatrix::identity(4, 4),
        c: DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        q: DMatrix::identity(2, 2),
        survival_probability: 0.99,
        detection_probability: 0.9,
        clutter_intensity: 1e-6,
        birth_intensity: Vec::new(),
        number_of_birth_locations: 0,
        hypotheses: Vec::new(),
        trajectory: Vec::new(),
        number_of_samples: 1000,
        gating_threshold: 1e-3,
        extraction_threshold: 0.5,
    };

    // Generate L using library function
    let (l_generated, _params, dimensions) = generate_multisensor_lmbm_association_matrices(
        &hypothesis,
        &measurements,
        &model,
        fixture.number_of_sensors
    );

    println!("\nGenerated dimensions: {:?}", dimensions);
    println!("Generated L length: {}", l_generated.len());

    // Get expected L from fixture
    let expected_l = &fixture.step2_association.output.l;
    println!("Expected L shape: [{}, {}, {}]", expected_l.len(), expected_l[0].len(), expected_l[0][0].len());

    let expected_total = expected_l.len() * expected_l[0].len() * expected_l[0][0].len();
    println!("Expected L length: {}", expected_total);

    // Compare first few values
    println!("\nFirst 10 generated L values:");
    for i in 0..10.min(l_generated.len()) {
        println!("  L[{}] = {:.16}", i, l_generated[i]);
    }

    println!("\nFirst 10 expected L values (flattened):");
    let mut idx = 0;
    'outer: for k in 0..expected_l[0][0].len() {
        for j in 0..expected_l[0].len() {
            for i in 0..expected_l.len() {
                if idx >= 10 {
                    break 'outer;
                }
                println!("  L[{}] = {:.16}", idx, expected_l[i][j][k]);
                idx += 1;
            }
        }
    }

    assert_eq!(l_generated.len(), expected_total, "L length mismatch");
}
