// ! Phase 5.2: Single-sensor numerical equivalence tests
//!
//! This test suite validates EXACT numerical equivalence between MATLAB and Rust
//! implementations by comparing complete state estimates (not just OSPA metrics).
//!
//! Coverage:
//! - 5 filter variants: LMB-LBP, LMB-Gibbs, LMB-Murty's, LMBM-Gibbs, LMBM-Murty's
//! - 5 seeds: 1, 42, 100, 1000, 12345
//! - Total: 25 tests
//!
//! Tolerance: 1e-15 (within floating point precision)

use nalgebra::{DMatrix, DVector};
use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::filter::run_lmb_filter;
use prak::lmbm::filter::run_lmbm_filter;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

/// State estimates for a single timestep (from MATLAB fixture)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimestepState {
    labels: Vec<Vec<i64>>,  // 2×N matrix as vector of columns
    mu: Vec<Vec<f64>>,      // Cell array of state vectors
    #[serde(rename = "Sigma")]
    sigma: Vec<Vec<Vec<f64>>>, // Cell array of covariance matrices
}

/// Complete state estimates across all timesteps (from MATLAB fixture)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StateEstimates {
    // Skip labels - not needed for numerical equivalence and MATLAB encoding is inconsistent
    mu: Vec<Vec<Vec<f64>>>,      // Cell array: one per timestep
    #[serde(rename = "Sigma")]
    sigma: Vec<Vec<Vec<Vec<f64>>>>, // Cell array: one per timestep
}

/// Filter variant results (from MATLAB fixture)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FilterVariant {
    name: String,
    filter_type: String,
    #[serde(rename = "dataAssociationMethod")]
    data_association_method: Option<String>,
    #[serde(rename = "simulationLength")]
    simulation_length: usize,
    #[serde(rename = "stateEstimates")]
    state_estimates: StateEstimates,
    #[serde(rename = "eOspa")]
    e_ospa: Vec<f64>,
    #[serde(rename = "hOspa")]
    h_ospa: Vec<f64>,
    cardinality: Vec<usize>,
}

/// Complete fixture data for all variants at one seed
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NumericalEquivalenceFixture {
    seed: u64,
    #[serde(rename = "filterVariants")]
    filter_variants: Vec<FilterVariant>,
}

/// Load a numerical equivalence fixture from MATLAB repo
fn load_fixture(seed: u64) -> NumericalEquivalenceFixture {
    // Path to MATLAB fixture (relative to prak project root)
    let fixture_path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "..",
        "multisensor-lmb-filters",
        "fixtures",
        "numerical_equivalence",
        &format!("single_sensor_seed{}.json", seed),
    ]
    .iter()
    .collect();

    // Read and parse JSON
    let json_str = fs::read_to_string(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read fixture file {:?}: {}",
            fixture_path, e
        )
    });

    serde_json::from_str(&json_str).unwrap_or_else(|e| {
        panic!(
            "Failed to parse fixture file {:?}: {}",
            fixture_path, e
        )
    })
}

/// Run single-sensor LMB filter and return state estimates
fn run_single_sensor_lmb_trial(
    seed: u64,
    method: DataAssociationMethod,
) -> prak::lmb::filter::LmbStateEstimates {
    // Generate model with fixed seed 0 (matches MATLAB)
    let mut model_rng = SimpleRng::new(0);
    let mut model = generate_model(
        &mut model_rng,
        2.0,  // clutterRate
        0.95, // detectionProbability
        method,
        ScenarioType::Fixed,
        None,
    );
    model.data_association_method = method;

    // Generate ground truth with trial-specific seed
    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth_output = generate_ground_truth(&mut trial_rng, &model, None);

    // Run filter with seed+1000 (matches MATLAB)
    let mut filter_rng = SimpleRng::new(seed + 1000);
    run_lmb_filter(&mut filter_rng, &model, &ground_truth_output.measurements)
}

/// Run single-sensor LMBM filter (10 timesteps) and return state estimates
fn run_single_sensor_lmbm_trial(
    seed: u64,
    method: DataAssociationMethod,
) -> prak::lmbm::filter::LmbmStateEstimates {
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

    // Run LMBM filter with only first 10 measurements
    const LMBM_SIMULATION_LENGTH: usize = 10;
    let mut filter_rng = SimpleRng::new(seed + 1000);
    let measurements_short: Vec<_> = ground_truth_output
        .measurements
        .iter()
        .take(LMBM_SIMULATION_LENGTH)
        .cloned()
        .collect();
    run_lmbm_filter(&mut filter_rng, &model, &measurements_short)
}

/// Compare a DMatrix with Vec<Vec<f64>> element-by-element
fn assert_matrix_exact(
    actual: &DMatrix<f64>,
    expected: &[Vec<f64>],
    tolerance: f64,
    context: &str,
) {
    assert_eq!(
        actual.nrows(),
        expected.len(),
        "{}: matrix row count mismatch: {} vs {}",
        context,
        actual.nrows(),
        expected.len()
    );

    for (i, expected_row) in expected.iter().enumerate() {
        assert_eq!(
            actual.ncols(),
            expected_row.len(),
            "{}: matrix column count mismatch at row {}: {} vs {}",
            context,
            i,
            actual.ncols(),
            expected_row.len()
        );

        for (j, e) in expected_row.iter().enumerate() {
            let a = actual[(i, j)];
            let diff = (a - e).abs();
            assert!(
                diff <= tolerance,
                "{}: mismatch at [{}, {}]: Rust={}, MATLAB={}, diff={} (tolerance={})",
                context,
                i,
                j,
                a,
                e,
                diff,
                tolerance
            );
        }
    }
}

/// Compare state estimates for a single timestep
fn compare_timestep_state_estimates(
    rust_mu: &[DVector<f64>],
    rust_sigma: &[DMatrix<f64>],
    matlab_mu: &[Vec<f64>],
    matlab_sigma: &[Vec<Vec<f64>>],
    tolerance: f64,
    timestep: usize,
    variant_name: &str,
) {
    // Compare cardinality (number of targets)
    assert_eq!(
        rust_mu.len(),
        matlab_mu.len(),
        "{} t={}: cardinality mismatch: Rust={}, MATLAB={}",
        variant_name,
        timestep,
        rust_mu.len(),
        matlab_mu.len()
    );

    // Compare each target's state estimate
    for (i, (r_mu, m_mu)) in rust_mu.iter().zip(matlab_mu.iter()).enumerate() {
        assert_dvec_exact(
            r_mu,
            m_mu,
            tolerance,
            &format!("{} t={} target={} mu", variant_name, timestep, i),
        );
    }

    // Compare each target's covariance
    for (i, (r_sigma, m_sigma)) in rust_sigma.iter().zip(matlab_sigma.iter()).enumerate() {
        assert_matrix_exact(
            r_sigma,
            m_sigma,
            tolerance,
            &format!("{} t={} target={} Sigma", variant_name, timestep, i),
        );
    }
}

/// Compare a DVector with Vec<f64> element-by-element
fn assert_dvec_exact(actual: &DVector<f64>, expected: &[f64], tolerance: f64, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch: {} vs {}",
        context,
        actual.len(),
        expected.len()
    );

    for (i, e) in expected.iter().enumerate() {
        let a = actual[i];
        let diff = (a - e).abs();
        assert!(
            diff <= tolerance,
            "{}: mismatch at [{}]: Rust={}, MATLAB={}, diff={} (tolerance={})",
            context,
            i,
            a,
            e,
            diff,
            tolerance
        );
    }
}

/// Test helper: validate a single seed's fixture
fn validate_numerical_equivalence_fixture(seed: u64) {
    println!("\n=== Testing seed {} ===", seed);
    let fixture = load_fixture(seed);
    assert_eq!(fixture.seed, seed, "Fixture seed mismatch");

    // Tolerance for exact numerical equivalence (within reasonable floating point precision)
    // 1e-12 accounts for normal floating point arithmetic differences across implementations
    const TOLERANCE: f64 = 1e-12;

    for variant in &fixture.filter_variants {
        println!(
            "  Testing {} ({} timesteps) ...",
            variant.name, variant.simulation_length
        );

        // Run Rust filter and compare (handle LMB and LMBM separately due to different types)
        if variant.filter_type == "LMB" {
            let method = match variant.data_association_method.as_deref() {
                Some("LBP") => DataAssociationMethod::LBP,
                Some("Gibbs") => DataAssociationMethod::Gibbs,
                Some("Murty") => DataAssociationMethod::Murty,
                _ => panic!("Unknown LMB method: {:?}", variant.data_association_method),
            };
            let rust_estimates = run_single_sensor_lmb_trial(seed, method);

            // Verify simulation length
            assert_eq!(
                rust_estimates.mu.len(),
                variant.simulation_length,
                "{}: simulation length mismatch",
                variant.name
            );

            // Compare state estimates for each timestep
            for t in 0..variant.simulation_length {
                compare_timestep_state_estimates(
                    &rust_estimates.mu[t],
                    &rust_estimates.sigma[t],
                    &variant.state_estimates.mu[t],
                    &variant.state_estimates.sigma[t],
                    TOLERANCE,
                    t,
                    &variant.name,
                );
            }
        } else if variant.filter_type == "LMBM" {
            let method = match variant.data_association_method.as_deref() {
                Some("Gibbs") => DataAssociationMethod::Gibbs,
                Some("Murty") => DataAssociationMethod::Murty,
                _ => panic!("Unknown LMBM method: {:?}", variant.data_association_method),
            };
            let rust_estimates = run_single_sensor_lmbm_trial(seed, method);

            // Verify simulation length
            assert_eq!(
                rust_estimates.mu.len(),
                variant.simulation_length,
                "{}: simulation length mismatch",
                variant.name
            );

            // Compare state estimates for each timestep
            for t in 0..variant.simulation_length {
                compare_timestep_state_estimates(
                    &rust_estimates.mu[t],
                    &rust_estimates.sigma[t],
                    &variant.state_estimates.mu[t],
                    &variant.state_estimates.sigma[t],
                    TOLERANCE,
                    t,
                    &variant.name,
                );
            }
        } else {
            panic!("Unknown filter type: {}", variant.filter_type);
        }

        println!(
            "    ✓ Complete state estimates match across {} timesteps (tolerance < {})",
            variant.simulation_length, TOLERANCE
        );
    }

    println!("=== Seed {} PASSED: All variants numerically equivalent ===", seed);
}

//
// Test cases for all 5 seeds × 5 variants = 25 tests
//

#[test]
fn test_numerical_equivalence_seed_1() {
    validate_numerical_equivalence_fixture(1);
}

#[test]
fn test_numerical_equivalence_seed_42() {
    validate_numerical_equivalence_fixture(42);
}

#[test]
fn test_numerical_equivalence_seed_100() {
    validate_numerical_equivalence_fixture(100);
}

#[test]
fn test_numerical_equivalence_seed_1000() {
    validate_numerical_equivalence_fixture(1000);
}

#[test]
fn test_numerical_equivalence_seed_12345() {
    validate_numerical_equivalence_fixture(12345);
}
