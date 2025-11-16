//! Accuracy trial tests - Validate exact numerical equivalence with MATLAB
//!
//! This test suite validates that the Rust implementation produces identical results
//! to the MATLAB implementation for accuracy trials. It uses deterministic fixtures
//! generated from MATLAB with SimpleRng to enable exact numerical comparison.
//!
//! Based on MATLAB singleSensorAccuracyTrial.m and multiSensorAccuracyTrial.m.

use prak::common::ground_truth::generate_ground_truth;
use prak::common::metrics::ospa;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::filter::run_lmb_filter;
use prak::lmbm::filter::run_lmbm_filter;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

/// Fixture data for a single filter variant's results for one trial
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FilterVariantMetrics {
    name: String,
    #[serde(rename = "eOspa")]
    e_ospa: Vec<f64>,
    #[serde(rename = "hOspa")]
    h_ospa: Vec<f64>,
    cardinality: Vec<usize>,
}

/// Complete fixture data for a single trial (all filter variants)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrialFixture {
    seed: u64,
    filter_variants: Vec<FilterVariantMetrics>,
}

/// Load a fixture file from the tests/data directory
fn load_fixture(seed: u64) -> TrialFixture {
    // Get path to fixture file (in tests/data directory)
    let fixture_path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "data",
        &format!("single_trial_{}.json", seed),
    ]
    .iter()
    .collect();

    // Read and parse JSON
    let json_str = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read fixture file {:?}: {}", fixture_path, e));

    serde_json::from_str(&json_str)
        .unwrap_or_else(|e| panic!("Failed to parse fixture file {:?}: {}", fixture_path, e))
}

/// Run a single-sensor trial with given seed and data association method
/// Returns (e_ospa, h_ospa, cardinality) vectors for all timesteps
fn run_single_sensor_trial(
    seed: u64,
    method: DataAssociationMethod,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    // Generate model with fixed seed 0 (same for all trials)
    let mut model_rng = SimpleRng::new(0);
    let mut model = generate_model(
        &mut model_rng,
        2.0, // clutterRate - matches MATLAB generateModel(2, 0.95, 'LBP')
        0.95, // detectionProbability
        method,
        ScenarioType::Fixed,
        None,
    );

    // Update model's data association method
    model.data_association_method = method;

    // Generate ground truth with trial-specific seed
    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth_output = generate_ground_truth(&mut trial_rng, &model, None);

    // Run filter with seed+1000 (matches MATLAB fixture generation)
    let mut filter_rng = SimpleRng::new(seed + 1000);
    let state_estimates = match method {
        DataAssociationMethod::LBP
        | DataAssociationMethod::Gibbs
        | DataAssociationMethod::Murty
        | DataAssociationMethod::LBPFixed => {
            run_lmb_filter(&mut filter_rng, &model, &ground_truth_output.measurements)
        }
    };

    // Compute OSPA metrics for each timestep
    let simulation_length = ground_truth_output.measurements.len();
    let mut e_ospa = Vec::with_capacity(simulation_length);
    let mut h_ospa = Vec::with_capacity(simulation_length);
    let mut cardinality = Vec::with_capacity(simulation_length);

    for t in 0..simulation_length {
        let metrics = ospa(
            &ground_truth_output.ground_truth_rfs.x[t],
            &ground_truth_output.ground_truth_rfs.mu[t],
            &ground_truth_output.ground_truth_rfs.sigma[t],
            &state_estimates.mu[t],
            &state_estimates.sigma[t],
            &model.ospa_parameters,
        );

        e_ospa.push(metrics.e_ospa.total);
        h_ospa.push(metrics.h_ospa.total);
        cardinality.push(state_estimates.mu[t].len());
    }

    (e_ospa, h_ospa, cardinality)
}

/// Run a single-sensor LMBM trial (only first 10 timesteps for quick validation)
fn run_single_sensor_lmbm_trial(
    seed: u64,
    method: DataAssociationMethod,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    // Generate model with fixed seed 0
    let mut model_rng = SimpleRng::new(0);
    let mut model = generate_model(
        &mut model_rng,
        2.0,
        0.95,
        method,
        ScenarioType::Fixed,
        None,
    );
    model.data_association_method = method;

    // Generate ground truth (full 100 timesteps)
    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth_output = generate_ground_truth(&mut trial_rng, &model, None);

    // Run LMBM filter with only first 10 measurements (quick validation)
    const LMBM_SIMULATION_LENGTH: usize = 10;
    let mut filter_rng = SimpleRng::new(seed + 1000);
    let measurements_short: Vec<_> = ground_truth_output.measurements.iter()
        .take(LMBM_SIMULATION_LENGTH)
        .cloned()
        .collect();
    let state_estimates = run_lmbm_filter(&mut filter_rng, &model, &measurements_short);

    // Compute OSPA metrics for first 10 timesteps only
    let mut e_ospa = Vec::with_capacity(LMBM_SIMULATION_LENGTH);
    let mut h_ospa = Vec::with_capacity(LMBM_SIMULATION_LENGTH);
    let mut cardinality = Vec::with_capacity(LMBM_SIMULATION_LENGTH);

    for t in 0..LMBM_SIMULATION_LENGTH {
        let metrics = ospa(
            &ground_truth_output.ground_truth_rfs.x[t],
            &ground_truth_output.ground_truth_rfs.mu[t],
            &ground_truth_output.ground_truth_rfs.sigma[t],
            &state_estimates.mu[t],
            &state_estimates.sigma[t],
            &model.ospa_parameters,
        );

        e_ospa.push(metrics.e_ospa.total);
        h_ospa.push(metrics.h_ospa.total);
        cardinality.push(state_estimates.mu[t].len());
    }

    (e_ospa, h_ospa, cardinality)
}

/// Compare two f64 vectors element-wise with tolerance
fn assert_vec_close(actual: &[f64], expected: &[f64], tolerance: f64, metric_name: &str, filter_name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{} length mismatch for {}: {} vs {}",
        metric_name,
        filter_name,
        actual.len(),
        expected.len()
    );

    for (t, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tolerance,
            "{} mismatch at timestep {} for {}: Rust={}, MATLAB={}, diff={}",
            metric_name,
            t,
            filter_name,
            a,
            e,
            diff
        );
    }
}

/// Compare two usize vectors element-wise (exact match)
fn assert_vec_exact(actual: &[usize], expected: &[usize], metric_name: &str, filter_name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{} length mismatch for {}: {} vs {}",
        metric_name,
        filter_name,
        actual.len(),
        expected.len()
    );

    for (t, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a,
            e,
            "{} mismatch at timestep {} for {}: Rust={}, MATLAB={}",
            metric_name,
            t,
            filter_name,
            a,
            e
        );
    }
}

/// Test helper: validate a single fixture
fn validate_fixture(seed: u64) {
    println!("\n=== Testing seed {} ===", seed);
    let fixture = load_fixture(seed);
    assert_eq!(fixture.seed, seed, "Fixture seed mismatch");

    const TOLERANCE: f64 = 1e-10;

    for variant in &fixture.filter_variants {
        println!("  Testing {} ... (expected t=0: E-OSPA={:.6}, H-OSPA={:.6}, Card={})",
                 variant.name, variant.e_ospa[0], variant.h_ospa[0], variant.cardinality[0]);

        // Determine method and filter type from name
        let (e_ospa, h_ospa, cardinality) = if variant.name.starts_with("LMB-") {
            let method = match variant.name.as_str() {
                "LMB-LBP" => DataAssociationMethod::LBP,
                "LMB-Gibbs" => DataAssociationMethod::Gibbs,
                "LMB-Murty" => DataAssociationMethod::Murty,
                _ => panic!("Unknown LMB variant: {}", variant.name),
            };
            run_single_sensor_trial(seed, method)
        } else if variant.name.starts_with("LMBM-") {
            let method = match variant.name.as_str() {
                "LMBM-Gibbs" => DataAssociationMethod::Gibbs,
                "LMBM-Murty" => DataAssociationMethod::Murty,
                _ => panic!("Unknown LMBM variant: {}", variant.name),
            };
            run_single_sensor_lmbm_trial(seed, method)
        } else {
            panic!("Unknown filter variant: {}", variant.name);
        };

        // Show actual Rust results for comparison
        println!("    Rust t=0: E-OSPA={:.6}, H-OSPA={:.6}, Card={}", e_ospa[0], h_ospa[0], cardinality[0]);

        // Compare results
        assert_vec_close(&e_ospa, &variant.e_ospa, TOLERANCE, "E-OSPA", &variant.name);
        assert_vec_close(&h_ospa, &variant.h_ospa, TOLERANCE, "H-OSPA", &variant.name);
        assert_vec_exact(&cardinality, &variant.cardinality, "Cardinality", &variant.name);

        println!("    ✓ E-OSPA, H-OSPA, Cardinality match (tolerance < {})", TOLERANCE);
    }

    println!("=== Seed {} PASSED ===", seed);
}

//
// Test cases for seed 42 (representative fixture validation)
//

#[test]
fn test_accuracy_seed_42() {
    validate_fixture(42);
}

//
// Determinism verification test
//

#[test]
fn test_single_sensor_determinism() {
    // Verify that running the same seed twice produces identical results
    let seed = 42;
    let method = DataAssociationMethod::LBP;

    let (e1, h1, c1) = run_single_sensor_trial(seed, method);
    let (e2, h2, c2) = run_single_sensor_trial(seed, method);

    assert_vec_close(&e1, &e2, 0.0, "E-OSPA (determinism)", "LMB-LBP");
    assert_vec_close(&h1, &h2, 0.0, "H-OSPA (determinism)", "LMB-LBP");
    assert_vec_exact(&c1, &c2, "Cardinality (determinism)", "LMB-LBP");

    println!("✓ Determinism verified: same seed produces identical results");
}
