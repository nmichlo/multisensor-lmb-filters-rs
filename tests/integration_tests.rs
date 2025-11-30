//! Integration tests for LMB and LMBM filters
//!
//! Tests that all filter variants run successfully with deterministic RNG.
//! These tests verify end-to-end functionality and serve as regression tests.

use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::common::rng::SimpleRng;
use prak::lmb::filter::run_lmb_filter;
use prak::lmbm::filter::run_lmbm_filter;

/// Test LMB filter with LBP data association
#[test]
fn test_lmb_filter_lbp() {
    let mut rng = SimpleRng::new(42);

    let model = generate_model(
        &mut rng,
        10.0,  // clutter rate
        0.95,  // detection probability
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);

    let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth.measurements);

    // Basic sanity checks
    assert!(!state_estimates.labels.is_empty(), "Should have time steps");
    assert_eq!(state_estimates.labels.len(), ground_truth.measurements.len());
    assert_eq!(state_estimates.mu.len(), ground_truth.measurements.len());
    assert_eq!(state_estimates.sigma.len(), ground_truth.measurements.len());

    // Note: The filter may track 0 objects if none pass the existence threshold
    // This is valid behavior, so we just check the filter runs without crashing
    // Detailed numerical verification happens in Phase 5
}

/// Test LMB filter with Gibbs sampling data association
#[test]
fn test_lmb_filter_gibbs() {
    let mut rng = SimpleRng::new(42);

    let model = generate_model(
        &mut rng,
        10.0,
        0.95,
        DataAssociationMethod::Gibbs,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);
    let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth.measurements);

    assert!(!state_estimates.labels.is_empty());
    // objects may be empty if no long trajectories were discarded yet
}

/// Test LMB filter with Murty's algorithm data association
#[test]
fn test_lmb_filter_murtys() {
    let mut rng = SimpleRng::new(42);

    let model = generate_model(
        &mut rng,
        10.0,
        0.95,
        DataAssociationMethod::Murty,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);
    let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth.measurements);

    assert!(!state_estimates.labels.is_empty());
    // objects may be empty if no long trajectories were discarded yet
}

/// Test LMB filter with fixed LBP iterations
#[test]
fn test_lmb_filter_lbp_fixed() {
    let mut rng = SimpleRng::new(42);

    let model = generate_model(
        &mut rng,
        10.0,
        0.95,
        DataAssociationMethod::LBPFixed,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);
    let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth.measurements);

    assert!(!state_estimates.labels.is_empty());
    // objects may be empty if no long trajectories were discarded yet
}

/// Test LMBM filter with Gibbs sampling
///
/// NOTE: This test takes ~1.5s in release mode.
#[test]
fn test_lmbm_filter_gibbs() {
    let mut rng = SimpleRng::new(42);

    // Use reduced parameters for faster testing
    let model = generate_model(
        &mut rng,
        5.0,  // Reduced clutter
        0.95,
        DataAssociationMethod::Gibbs,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);
    let state_estimates = run_lmbm_filter(&mut rng, &model, &ground_truth.measurements);

    assert!(!state_estimates.labels.is_empty());
    assert_eq!(state_estimates.mu.len(), ground_truth.measurements.len());
    assert_eq!(state_estimates.sigma.len(), ground_truth.measurements.len());
}

/// Test LMBM filter with Murty's algorithm
///
/// NOTE: This test takes ~1.5s in release mode.
#[test]
fn test_lmbm_filter_murtys() {
    let mut rng = SimpleRng::new(42);

    let model = generate_model(
        &mut rng,
        5.0,  // Reduced clutter
        0.95,
        DataAssociationMethod::Murty,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);
    let state_estimates = run_lmbm_filter(&mut rng, &model, &ground_truth.measurements);

    assert!(!state_estimates.labels.is_empty());
    assert_eq!(state_estimates.mu.len(), ground_truth.measurements.len());
}

/// Test determinism: same seed produces same results
#[test]
fn test_lmb_determinism() {
    let mut rng1 = SimpleRng::new(12345);
    let mut rng2 = SimpleRng::new(12345);

    let model1 = generate_model(
        &mut rng1,
        5.0,
        0.90,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let model2 = generate_model(
        &mut rng2,
        5.0,
        0.90,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let gt1 = generate_ground_truth(&mut rng1, &model1, None);
    let gt2 = generate_ground_truth(&mut rng2, &model2, None);

    let est1 = run_lmb_filter(&mut rng1, &model1, &gt1.measurements);
    let est2 = run_lmb_filter(&mut rng2, &model2, &gt2.measurements);

    // Same seed should produce same cardinality estimates
    assert_eq!(est1.labels.len(), est2.labels.len());
    for (labels1, labels2) in est1.labels.iter().zip(est2.labels.iter()) {
        assert_eq!(labels1.ncols(), labels2.ncols(), "Cardinality should match with same seed");
    }
}

/// Test with different clutter rates
#[test]
fn test_lmb_varying_clutter() {
    for clutter_rate in [0.0, 5.0, 10.0, 20.0] {
        let mut rng = SimpleRng::new(42);

        let model = generate_model(
            &mut rng,
            clutter_rate,
            0.95,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let ground_truth = generate_ground_truth(&mut rng, &model, None);
        let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth.measurements);

        assert!(!state_estimates.labels.is_empty(), "Filter should work with clutter rate {}", clutter_rate);
    }
}

/// Test with different detection probabilities
#[test]
fn test_lmb_varying_detection_probability() {
    for det_prob in [0.5, 0.7, 0.9, 0.99] {
        let mut rng = SimpleRng::new(42);

        let model = generate_model(
            &mut rng,
            10.0,
            det_prob,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let ground_truth = generate_ground_truth(&mut rng, &model, None);
        let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth.measurements);

        assert!(!state_estimates.labels.is_empty(), "Filter should work with det prob {}", det_prob);
    }
}

/// Test with random scenario
#[test]
fn test_lmb_random_scenario() {
    let mut rng = SimpleRng::new(42);

    let model = generate_model(
        &mut rng,
        10.0,
        0.95,
        DataAssociationMethod::LBP,
        ScenarioType::Random,
        Some(5),  // 5 birth locations
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, Some(8)); // 8 objects
    let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth.measurements);

    assert!(!state_estimates.labels.is_empty());
    // objects may be empty if no long trajectories were discarded yet
}
