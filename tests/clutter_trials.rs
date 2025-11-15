//! Clutter sensitivity trial tests - Validate exact numerical equivalence with MATLAB
//!
//! This test suite validates that the Rust implementation produces identical results
//! to the MATLAB implementation for clutter sensitivity trials. It uses deterministic
//! fixtures generated from MATLAB with SimpleRng to enable exact numerical comparison.
//!
//! Based on MATLAB singleSensorClutterTrial.m.

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

/// Complete fixture data for clutter sensitivity trial
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClutterTrialFixture {
    seed: u64,
    clutter_rates: Vec<f64>,
    detection_probability: f64,
    simulation_length_gibbs: usize,
    simulation_length_other: usize,
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

/// Run a single-sensor trial with given parameters and compute mean OSPA
/// Returns (mean_e_ospa, mean_h_ospa)
fn run_clutter_trial(
    seed: u64,
    method: DataAssociationMethod,
    clutter_rate: f64,
    detection_probability: f64,
    simulation_length: usize,
) -> (f64, f64) {
    // Generate model with fixed seed 0 (same for all trials)
    let mut model_rng = SimpleRng::new(0);
    let mut model = generate_model(
        &mut model_rng,
        clutter_rate,
        detection_probability,
        method,
        ScenarioType::Fixed,
        None,
    );
    model.data_association_method = method;

    // Generate ground truth with trial-specific seed (full 100 timesteps)
    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth_output = generate_ground_truth(&mut trial_rng, &model, None);

    // Truncate measurements if needed
    let measurements: Vec<_> = ground_truth_output
        .measurements
        .iter()
        .take(simulation_length)
        .cloned()
        .collect();

    // Run filter with seed+1000 (matches MATLAB fixture generation)
    let mut filter_rng = SimpleRng::new(seed + 1000);
    let state_estimates = match method {
        DataAssociationMethod::LBP
        | DataAssociationMethod::Gibbs
        | DataAssociationMethod::Murty
        | DataAssociationMethod::LBPFixed => run_lmb_filter(&mut filter_rng, &model, &measurements),
    };

    // Compute OSPA metrics for each timestep
    let mut e_ospa_values = Vec::with_capacity(simulation_length);
    let mut h_ospa_values = Vec::with_capacity(simulation_length);

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

/// Run a single-sensor LMBM trial with given parameters and compute mean OSPA
fn run_clutter_trial_lmbm(
    seed: u64,
    method: DataAssociationMethod,
    clutter_rate: f64,
    detection_probability: f64,
    simulation_length: usize,
) -> (f64, f64) {
    // Generate model with fixed seed 0
    let mut model_rng = SimpleRng::new(0);
    let mut model = generate_model(
        &mut model_rng,
        clutter_rate,
        detection_probability,
        method,
        ScenarioType::Fixed,
        None,
    );
    model.data_association_method = method;

    // Generate ground truth (full 100 timesteps)
    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth_output = generate_ground_truth(&mut trial_rng, &model, None);

    // Truncate measurements
    let measurements: Vec<_> = ground_truth_output
        .measurements
        .iter()
        .take(simulation_length)
        .cloned()
        .collect();

    // Run LMBM filter
    let mut filter_rng = SimpleRng::new(seed + 1000);
    let state_estimates = run_lmbm_filter(&mut filter_rng, &model, &measurements);

    // Compute OSPA metrics
    let mut e_ospa_values = Vec::with_capacity(simulation_length);
    let mut h_ospa_values = Vec::with_capacity(simulation_length);

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

/// Test helper: validate a clutter sensitivity fixture
fn validate_clutter_fixture(filename: &str) {
    println!("\n=== Testing {} ===", filename);
    let fixture = load_fixture(filename);

    println!("  Seed: {}", fixture.seed);
    println!("  Clutter rates: {:?}", fixture.clutter_rates);
    println!("  Detection probability: {}", fixture.detection_probability);
    println!(
        "  Simulation lengths: Gibbs={}, Other={}",
        fixture.simulation_length_gibbs, fixture.simulation_length_other
    );

    const TOLERANCE: f64 = 1e-9; // Relaxed for averaging across timesteps

    for variant in &fixture.filter_variants {
        println!("\n  Testing {} ...", variant.name);

        // Determine method and filter type from name
        let is_lmbm = variant.name.starts_with("LMBM-");
        let is_gibbs = variant.name.ends_with("Gibbs");

        let method = if variant.name.contains("LBP") {
            DataAssociationMethod::LBP
        } else if variant.name.contains("Gibbs") {
            DataAssociationMethod::Gibbs
        } else if variant.name.contains("Murty") {
            DataAssociationMethod::Murty
        } else {
            panic!("Unknown method in variant: {}", variant.name);
        };

        let simulation_length = if is_gibbs {
            fixture.simulation_length_gibbs
        } else {
            fixture.simulation_length_other
        };

        // Run trial for each clutter rate
        let mut rust_e_ospa = Vec::with_capacity(fixture.clutter_rates.len());
        let mut rust_h_ospa = Vec::with_capacity(fixture.clutter_rates.len());

        for (i, &clutter_rate) in fixture.clutter_rates.iter().enumerate() {
            let (mean_e, mean_h) = if is_lmbm {
                run_clutter_trial_lmbm(
                    fixture.seed,
                    method,
                    clutter_rate,
                    fixture.detection_probability,
                    simulation_length,
                )
            } else {
                run_clutter_trial(
                    fixture.seed,
                    method,
                    clutter_rate,
                    fixture.detection_probability,
                    simulation_length,
                )
            };

            rust_e_ospa.push(mean_e);
            rust_h_ospa.push(mean_h);

            println!(
                "    Clutter rate {}: MATLAB E-OSPA={:.6}, Rust E-OSPA={:.6}, diff={:.2e}",
                clutter_rate as usize,
                variant.e_ospa[i],
                mean_e,
                (variant.e_ospa[i] - mean_e).abs()
            );
            println!(
                "                    MATLAB H-OSPA={:.6}, Rust H-OSPA={:.6}, diff={:.2e}",
                variant.h_ospa[i],
                mean_h,
                (variant.h_ospa[i] - mean_h).abs()
            );
        }

        // Compare results
        for (i, (rust_e, matlab_e)) in rust_e_ospa.iter().zip(variant.e_ospa.iter()).enumerate() {
            let diff = (rust_e - matlab_e).abs();
            assert!(
                diff < TOLERANCE,
                "E-OSPA mismatch at clutter rate {} for {}: Rust={}, MATLAB={}, diff={}",
                fixture.clutter_rates[i],
                variant.name,
                rust_e,
                matlab_e,
                diff
            );
        }

        for (i, (rust_h, matlab_h)) in rust_h_ospa.iter().zip(variant.h_ospa.iter()).enumerate() {
            let diff = (rust_h - matlab_h).abs();
            assert!(
                diff < TOLERANCE,
                "H-OSPA mismatch at clutter rate {} for {}: Rust={}, MATLAB={}, diff={}",
                fixture.clutter_rates[i],
                variant.name,
                rust_h,
                matlab_h,
                diff
            );
        }

        println!("    ✓ All clutter rates match (tolerance < {})", TOLERANCE);
    }

    println!("\n=== {} PASSED ===", filename);
}

//
// Test cases
//

#[test]
fn test_clutter_sensitivity_quick() {
    validate_clutter_fixture("single_trial_42_quick.json");
}

//
// Determinism verification test
//

#[test]
fn test_clutter_trial_determinism() {
    // Verify that running the same seed twice produces identical results
    let seed = 42;
    let method = DataAssociationMethod::LBP;
    let clutter_rate = 10.0;
    let detection_prob = 0.95;
    let sim_length = 100;

    let (e1, h1) = run_clutter_trial(seed, method, clutter_rate, detection_prob, sim_length);
    let (e2, h2) = run_clutter_trial(seed, method, clutter_rate, detection_prob, sim_length);

    assert_eq!(e1, e2, "E-OSPA determinism check failed");
    assert_eq!(h1, h2, "H-OSPA determinism check failed");

    println!("✓ Determinism verified: same seed produces identical results");
}
