//! Multisensor clutter sensitivity trial tests - Validate exact numerical equivalence with MATLAB
//!
//! This test suite validates that the Rust multisensor implementations produce identical results
//! to the MATLAB implementation for clutter sensitivity trials. It uses deterministic fixtures
//! generated from MATLAB with SimpleRng to enable exact numerical comparison.
//!
//! Based on MATLAB multiSensorClutterTrial.m.

use prak::common::ground_truth::generate_multisensor_ground_truth;
use prak::common::metrics::ospa;
use prak::common::model::generate_multisensor_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ParallelUpdateMode, ScenarioType};
use prak::multisensor_lmb::iterated_corrector::run_ic_lmb_filter;
use prak::multisensor_lmb::parallel_update::run_parallel_update_lmb_filter;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

/// Fixture data for a single filter variant's results across all clutter rates
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FilterVariantMetrics {
    name: String,
    #[serde(rename = "eOspa")]
    e_ospa: Vec<f64>, // Mean E-OSPA for each clutter rate
    #[serde(rename = "hOspa")]
    h_ospa: Vec<f64>, // Mean H-OSPA for each clutter rate
}

/// Complete fixture data for multisensor clutter sensitivity trial
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClutterTrialFixture {
    seed: u64,
    clutter_rates: Vec<f64>,
    detection_probabilities: Vec<f64>, // Per sensor
    number_of_sensors: usize,
    simulation_length: usize,
    filter_variants: Vec<FilterVariantMetrics>,
}

/// Load a fixture file from the tests/data directory
fn load_fixture(filename: &str) -> ClutterTrialFixture {
    // Get path to fixture file (in tests/data directory)
    let fixture_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "data", filename]
        .iter()
        .collect();

    // Read and parse JSON
    let json_str = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read fixture file {:?}: {}", fixture_path, e));

    serde_json::from_str(&json_str)
        .unwrap_or_else(|e| panic!("Failed to parse fixture file {:?}: {}", fixture_path, e))
}

/// Run a multisensor clutter trial with given parameters and compute mean OSPA
/// Returns (mean_e_ospa, mean_h_ospa)
fn run_multisensor_clutter_trial(
    seed: u64,
    clutter_rate: f64,
    detection_probabilities: &[f64],
    number_of_sensors: usize,
    q_values: &[f64],
    simulation_length: usize,
    update_mode: Option<ParallelUpdateMode>,
) -> (f64, f64) {
    // Generate model with fixed seed 0 (same for all trials)
    // Clutter rate is replicated across all sensors
    let clutter_rates = vec![clutter_rate; number_of_sensors];

    let mut model_rng = SimpleRng::new(0);
    let mut model = generate_multisensor_model(
        &mut model_rng,
        number_of_sensors,
        clutter_rates,
        detection_probabilities.to_vec(),
        q_values.to_vec(),
        ParallelUpdateMode::PU, // default mode
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    // Update model's parallel update mode if specified
    if let Some(mode) = update_mode {
        model.lmb_parallel_update_mode = Some(mode);
    }

    // Generate ground truth with trial-specific seed
    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth_output = generate_multisensor_ground_truth(&mut trial_rng, &model, None);

    // Truncate measurements if needed (all tests use full 100 timesteps)
    let measurements: Vec<Vec<_>> = ground_truth_output
        .measurements
        .iter()
        .map(|sensor_meas| {
            sensor_meas
                .iter()
                .take(simulation_length)
                .cloned()
                .collect()
        })
        .collect();

    // Run appropriate filter with RNG
    let mut filter_rng = SimpleRng::new(seed + 1000);
    let state_estimates = if update_mode.is_none() {
        // IC-LMB filter
        run_ic_lmb_filter(
            &mut filter_rng,
            &model,
            &measurements,
            number_of_sensors,
        )
    } else {
        // PU/GA/AA-LMB filters
        run_parallel_update_lmb_filter(
            &mut filter_rng,
            &model,
            &measurements,
            number_of_sensors,
            update_mode.unwrap(),
        )
    };

    // Compute OSPA metrics for each timestep
    let mut e_ospa_values = Vec::new();
    let mut h_ospa_values = Vec::new();

    for t in 0..simulation_length {
        let metrics = ospa(
            &ground_truth_output.ground_truth_rfs.x[t],
            &ground_truth_output.ground_truth_rfs.mu[t],
            &ground_truth_output.ground_truth_rfs.sigma[t],
            &state_estimates.mu[t],
            &state_estimates.sigma[t],
            &model.ospa_parameters,
        );
        e_ospa_values.push(metrics.e_ospa.total);
        h_ospa_values.push(metrics.h_ospa.total);
    }

    // Return mean values
    let mean_e_ospa = e_ospa_values.iter().sum::<f64>() / e_ospa_values.len() as f64;
    let mean_h_ospa = h_ospa_values.iter().sum::<f64>() / h_ospa_values.len() as f64;

    (mean_e_ospa, mean_h_ospa)
}

/// Validate a single fixture by running all filter variants and clutter rates
fn validate_fixture(filename: &str) {
    let fixture = load_fixture(filename);

    println!("=== Testing multisensor clutter sensitivity (seed {}) ===", fixture.seed);
    println!("  Clutter rates: {:?}", fixture.clutter_rates);
    println!("  Detection probabilities (per sensor): {:?}", fixture.detection_probabilities);
    println!("  Number of sensors: {}", fixture.number_of_sensors);
    println!("  Simulation length: {}", fixture.simulation_length);

    // Q values for multisensor model (from MATLAB multiSensorClutterTrial.m)
    let q_values = vec![4.0, 3.0, 2.0];

    // Tolerance for numerical equivalence (relaxed due to averaging over 100 timesteps)
    // Minor floating-point differences accumulate in GA-LMB merging
    const TOLERANCE: f64 = 1e-6;

    for variant in &fixture.filter_variants {
        println!("  Testing {} ...", variant.name);

        // Determine update mode from name
        let update_mode = match variant.name.as_str() {
            "IC-LMB" => None, // IC uses different function
            "PU-LMB" => Some(ParallelUpdateMode::PU),
            "GA-LMB" => Some(ParallelUpdateMode::GA),
            "AA-LMB" => Some(ParallelUpdateMode::AA),
            _ => panic!("Unknown multisensor filter variant: {}", variant.name),
        };

        // Test each clutter rate
        for (idx, &clutter_rate) in fixture.clutter_rates.iter().enumerate() {
            let (rust_e_ospa, rust_h_ospa) = run_multisensor_clutter_trial(
                fixture.seed,
                clutter_rate,
                &fixture.detection_probabilities,
                fixture.number_of_sensors,
                &q_values,
                fixture.simulation_length,
                update_mode,
            );

            let matlab_e_ospa = variant.e_ospa[idx];
            let matlab_h_ospa = variant.h_ospa[idx];

            let e_diff = (rust_e_ospa - matlab_e_ospa).abs();
            let h_diff = (rust_h_ospa - matlab_h_ospa).abs();

            println!(
                "    Clutter={}: E-OSPA: Rust={:.6}, MATLAB={:.6}, diff={:.2e}",
                clutter_rate, rust_e_ospa, matlab_e_ospa, e_diff
            );
            println!(
                "               H-OSPA: Rust={:.6}, MATLAB={:.6}, diff={:.2e}",
                rust_h_ospa, matlab_h_ospa, h_diff
            );

            assert!(
                e_diff <= TOLERANCE,
                "E-OSPA mismatch for {} at clutter rate {}: Rust={}, MATLAB={}, diff={}",
                variant.name,
                clutter_rate,
                rust_e_ospa,
                matlab_e_ospa,
                e_diff
            );

            assert!(
                h_diff <= TOLERANCE,
                "H-OSPA mismatch for {} at clutter rate {}: Rust={}, MATLAB={}, diff={}",
                variant.name,
                clutter_rate,
                rust_h_ospa,
                matlab_h_ospa,
                h_diff
            );
        }

        println!("    ✓ All clutter rates match (tolerance < {})", TOLERANCE);
    }

    println!("=== Multisensor clutter sensitivity seed {} PASSED ===", fixture.seed);
}

//
// Test cases for seed 42 (quick validation with 2 clutter rates)
//

#[test]
fn test_multisensor_clutter_sensitivity_quick() {
    validate_fixture("multisensor_clutter_trial_42_quick.json");
}

//
// Determinism verification test
//

#[test]
fn test_multisensor_clutter_determinism() {
    // Verify that running the same seed twice produces identical results
    let seed = 42;
    let clutter_rate = 10.0;
    let detection_probabilities = vec![0.67, 0.70, 0.73];
    let number_of_sensors = 3;
    let q_values = vec![4.0, 3.0, 2.0];
    let simulation_length = 100;
    let update_mode = Some(ParallelUpdateMode::PU);

    let (e1, h1) = run_multisensor_clutter_trial(
        seed,
        clutter_rate,
        &detection_probabilities,
        number_of_sensors,
        &q_values,
        simulation_length,
        update_mode,
    );

    let (e2, h2) = run_multisensor_clutter_trial(
        seed,
        clutter_rate,
        &detection_probabilities,
        number_of_sensors,
        &q_values,
        simulation_length,
        update_mode,
    );

    assert_eq!(e1, e2, "E-OSPA determinism check failed");
    assert_eq!(h1, h2, "H-OSPA determinism check failed");

    println!("✓ Determinism verified: same seed produces identical results");
}
