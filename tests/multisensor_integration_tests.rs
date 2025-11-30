//! Integration tests for multi-sensor LMB and LMBM filters
//!
//! Tests that all multi-sensor filter variants run successfully with deterministic RNG.

use prak::common::ground_truth::generate_multisensor_ground_truth;
use prak::common::model::generate_multisensor_model;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::common::rng::SimpleRng;
use prak::multisensor_lmb::iterated_corrector::run_ic_lmb_filter;
use prak::multisensor_lmb::parallel_update::{run_parallel_update_lmb_filter, ParallelUpdateMode};
use prak::multisensor_lmbm::filter::run_multisensor_lmbm_filter;

/// Test Iterated Corrector LMB filter
#[test]
fn test_ic_lmb_filter() {
    let mut rng = SimpleRng::new(42);
    let num_sensors = 3;

    let model = generate_multisensor_model(
        &mut rng,
        num_sensors,
        vec![5.0; num_sensors],  // clutter rates
        vec![0.67, 0.70, 0.73],  // detection probabilities
        vec![4.0, 3.0, 2.0],     // q values
        ParallelUpdateMode::PU,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_multisensor_ground_truth(&mut rng, &model, None);

    let state_estimates = run_ic_lmb_filter(
        &mut rng,
        &model,
        &ground_truth.measurements,
        num_sensors,
    );

    // Basic sanity checks
    assert!(!state_estimates.labels.is_empty(), "Should have time steps");
    assert_eq!(state_estimates.labels.len(), ground_truth.measurements[0].len());
    assert!(!state_estimates.objects.is_empty(), "Should have tracked objects");

    for labels in &state_estimates.labels {
        assert!(labels.len() <= 25, "Cardinality should be reasonable");
    }
}

/// Test Parallel Update LMB filter (PU mode)
#[test]
fn test_parallel_update_lmb_pu() {
    let mut rng = SimpleRng::new(42);
    let num_sensors = 3;

    let model = generate_multisensor_model(
        &mut rng,
        num_sensors,
        vec![5.0; num_sensors],
        vec![0.67, 0.70, 0.73],
        vec![4.0, 3.0, 2.0],
        ParallelUpdateMode::PU,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_multisensor_ground_truth(&mut rng, &model, None);

    let state_estimates = run_parallel_update_lmb_filter(
        &mut rng,
        &model,
        &ground_truth.measurements,
        num_sensors,
        ParallelUpdateMode::PU,
    );

    assert!(!state_estimates.labels.is_empty());
    assert_eq!(state_estimates.labels.len(), ground_truth.measurements[0].len());
}

/// Test Parallel Update LMB filter (GA mode - Geometric Average)
#[test]
fn test_parallel_update_lmb_ga() {
    let mut rng = SimpleRng::new(42);
    let num_sensors = 3;

    let model = generate_multisensor_model(
        &mut rng,
        num_sensors,
        vec![5.0; num_sensors],
        vec![0.67, 0.70, 0.73],
        vec![4.0, 3.0, 2.0],
        ParallelUpdateMode::GA,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_multisensor_ground_truth(&mut rng, &model, None);

    let state_estimates = run_parallel_update_lmb_filter(
        &mut rng,
        &model,
        &ground_truth.measurements,
        num_sensors,
        ParallelUpdateMode::GA,
    );

    assert!(!state_estimates.labels.is_empty());
    assert!(!state_estimates.objects.is_empty());
}

/// Test Parallel Update LMB filter (AA mode - Arithmetic Average)
#[test]
fn test_parallel_update_lmb_aa() {
    let mut rng = SimpleRng::new(42);
    let num_sensors = 3;

    let model = generate_multisensor_model(
        &mut rng,
        num_sensors,
        vec![5.0; num_sensors],
        vec![0.67, 0.70, 0.73],
        vec![4.0, 3.0, 2.0],
        ParallelUpdateMode::AA,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_multisensor_ground_truth(&mut rng, &model, None);

    let state_estimates = run_parallel_update_lmb_filter(
        &mut rng,
        &model,
        &ground_truth.measurements,
        num_sensors,
        ParallelUpdateMode::AA,
    );

    assert!(!state_estimates.labels.is_empty());
    assert!(!state_estimates.objects.is_empty());
}

/// Test Multi-sensor LMBM filter
///
/// NOTE: This test takes ~13s in release mode.
#[test]
fn test_multisensor_lmbm_filter() {
    let mut rng = SimpleRng::new(42);
    let num_sensors = 3;

    let model = generate_multisensor_model(
        &mut rng,
        num_sensors,
        vec![5.0; num_sensors],
        vec![0.67, 0.70, 0.73],
        vec![4.0, 3.0, 2.0],
        ParallelUpdateMode::PU,
        DataAssociationMethod::Gibbs,  // LMBM uses Gibbs
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_multisensor_ground_truth(&mut rng, &model, None);

    let state_estimates = run_multisensor_lmbm_filter(
        &mut rng,
        &model,
        &ground_truth.measurements,
        num_sensors,
    );

    assert!(!state_estimates.labels.is_empty());
    assert_eq!(state_estimates.labels.len(), ground_truth.measurements[0].len());
}

/// Test multi-sensor determinism
#[test]
fn test_multisensor_determinism() {
    let num_sensors = 2;

    let mut rng1 = SimpleRng::new(99999);
    let mut rng2 = SimpleRng::new(99999);

    let model1 = generate_multisensor_model(
        &mut rng1,
        num_sensors,
        vec![3.0; num_sensors],
        vec![0.8; num_sensors],
        vec![3.0; num_sensors],
        ParallelUpdateMode::PU,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let model2 = generate_multisensor_model(
        &mut rng2,
        num_sensors,
        vec![3.0; num_sensors],
        vec![0.8; num_sensors],
        vec![3.0; num_sensors],
        ParallelUpdateMode::PU,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let gt1 = generate_multisensor_ground_truth(&mut rng1, &model1, None);
    let gt2 = generate_multisensor_ground_truth(&mut rng2, &model2, None);

    let est1 = run_parallel_update_lmb_filter(
        &mut rng1,
        &model1,
        &gt1.measurements,
        num_sensors,
        ParallelUpdateMode::PU,
    );

    let est2 = run_parallel_update_lmb_filter(
        &mut rng2,
        &model2,
        &gt2.measurements,
        num_sensors,
        ParallelUpdateMode::PU,
    );

    // Same seed should produce same cardinality estimates
    assert_eq!(est1.labels.len(), est2.labels.len());
    for (labels1, labels2) in est1.labels.iter().zip(est2.labels.iter()) {
        assert_eq!(
            labels1.len(),
            labels2.len(),
            "Cardinality should match with same seed"
        );
    }
}

/// Test with varying number of sensors
#[test]
fn test_varying_number_of_sensors() {
    for num_sensors in [2, 3, 4] {
        let mut rng = SimpleRng::new(42);

        let model = generate_multisensor_model(
            &mut rng,
            num_sensors,
            vec![5.0; num_sensors],
            vec![0.9; num_sensors],
            vec![3.0; num_sensors],
            ParallelUpdateMode::PU,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let ground_truth = generate_multisensor_ground_truth(&mut rng, &model, None);

        let state_estimates = run_parallel_update_lmb_filter(
            &mut rng,
            &model,
            &ground_truth.measurements,
            num_sensors,
            ParallelUpdateMode::PU,
        );

        assert!(
            !state_estimates.labels.is_empty(),
            "Filter should work with {} sensors",
            num_sensors
        );
    }
}

/// Test IC-LMB with Gibbs sampling
#[test]
fn test_ic_lmb_with_gibbs() {
    let mut rng = SimpleRng::new(42);
    let num_sensors = 2;

    let model = generate_multisensor_model(
        &mut rng,
        num_sensors,
        vec![3.0; num_sensors],
        vec![0.85; num_sensors],
        vec![3.0; num_sensors],
        ParallelUpdateMode::PU,
        DataAssociationMethod::Gibbs,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_multisensor_ground_truth(&mut rng, &model, None);

    let state_estimates = run_ic_lmb_filter(
        &mut rng,
        &model,
        &ground_truth.measurements,
        num_sensors,
    );

    assert!(!state_estimates.labels.is_empty());
    assert!(!state_estimates.objects.is_empty());
}
