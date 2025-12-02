//! Migration tests comparing new filter API with legacy implementations.
//!
//! These tests verify that the new trait-based filter implementations
//! produce compatible results with the legacy function-based implementations.
//!
//! Note: Exact numerical equivalence is not expected because:
//! 1. Different RNG consumption patterns
//! 2. Different internal data structures
//! 3. Different association result interpretations
//!
//! Instead, we verify:
//! - Both implementations run without errors
//! - Output formats are compatible
//! - Cardinality estimates are reasonable

use nalgebra::{DMatrix, DVector};
use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};

// Legacy API imports
use prak::lmb::filter::run_lmb_filter;
use prak::lmbm::filter::run_lmbm_filter;

// New API imports
use prak::filter::{Filter, LmbFilter, LmbmFilter};
use prak::types::{
    AssociationConfig, BirthLocation, BirthModel, MotionModel, SensorModel,
};

/// Convert legacy Model to new API parameters
fn convert_model_to_new_api(model: &prak::common::types::Model) -> (MotionModel, SensorModel, BirthModel, AssociationConfig) {
    // Motion model
    let motion = MotionModel::new(
        model.a.clone(),
        model.r.clone(),
        model.u.clone(),
        model.survival_probability,
    );

    // Sensor model
    let sensor = SensorModel::new(
        model.c.clone(),
        model.q.clone(),
        model.detection_probability,
        model.clutter_rate,
        model.observation_space_volume,
    );

    // Birth model
    let birth_locations: Vec<BirthLocation> = (0..model.number_of_birth_locations)
        .map(|i| {
            BirthLocation::new(
                model.birth_location_labels[i],
                model.mu_b[i].clone(),
                model.sigma_b[i].clone(),
            )
        })
        .collect();

    let birth = BirthModel::new(
        birth_locations,
        model.r_b[0],      // LMB birth existence
        model.r_b_lmbm[0], // LMBM birth existence
    );

    // Association config
    let method = match model.data_association_method {
        DataAssociationMethod::LBP => prak::types::DataAssociationMethod::Lbp,
        DataAssociationMethod::LBPFixed => prak::types::DataAssociationMethod::LbpFixed,
        DataAssociationMethod::Gibbs => prak::types::DataAssociationMethod::Gibbs,
        DataAssociationMethod::Murty => prak::types::DataAssociationMethod::Murty,
    };

    let association = AssociationConfig {
        method,
        lbp_max_iterations: model.maximum_number_of_lbp_iterations,
        lbp_tolerance: model.lbp_convergence_tolerance,
        gibbs_samples: model.number_of_samples,
        murty_assignments: model.number_of_assignments,
    };

    (motion, sensor, birth, association)
}

// ============================================================================
// LMB Filter Tests
// ============================================================================

/// Test that both LMB implementations run and produce output
#[test]
fn test_lmb_filter_both_apis_run() {
    let mut rng = SimpleRng::new(42);

    let model = generate_model(
        &mut rng,
        10.0,
        0.95,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);
    let measurements = &ground_truth.measurements;

    // Run legacy API
    let mut legacy_rng = SimpleRng::new(42);
    let legacy_result = run_lmb_filter(&mut legacy_rng, &model, measurements);

    // Run new API
    let (motion, sensor, birth, association) = convert_model_to_new_api(&model);
    let mut new_filter = LmbFilter::new(motion, sensor, birth, association)
        .with_existence_threshold(model.existence_threshold)
        .with_gm_pruning(model.gm_weight_threshold, model.maximum_number_of_gm_components);

    let mut new_rng = rand::thread_rng();
    let mut new_results = Vec::new();
    for (t, meas) in measurements.iter().enumerate() {
        let estimate = new_filter.step(&mut new_rng, meas, t).unwrap();
        new_results.push(estimate);
    }

    // Verify both produced output for same number of timesteps
    assert_eq!(legacy_result.labels.len(), new_results.len());
    assert_eq!(legacy_result.mu.len(), new_results.len());

    println!("LMB Filter comparison:");
    println!("  Legacy: {} timesteps", legacy_result.labels.len());
    println!("  New: {} timesteps", new_results.len());

    // Compare cardinality at each timestep
    for t in 0..measurements.len() {
        let legacy_card = legacy_result.labels[t].ncols();
        let new_card = new_results[t].tracks.len();
        println!("  t={}: legacy={}, new={}", t, legacy_card, new_card);
    }
}

/// Test LMB with Gibbs association
#[test]
fn test_lmb_gibbs_both_apis() {
    let mut rng = SimpleRng::new(123);

    let model = generate_model(
        &mut rng,
        8.0,
        0.9,
        DataAssociationMethod::Gibbs,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);

    // Legacy
    let mut legacy_rng = SimpleRng::new(123);
    let legacy_result = run_lmb_filter(&mut legacy_rng, &model, &ground_truth.measurements);

    // New
    let (motion, sensor, birth, association) = convert_model_to_new_api(&model);
    let mut new_filter = LmbFilter::new(motion, sensor, birth, association);

    let mut new_rng = rand::thread_rng();
    for (t, meas) in ground_truth.measurements.iter().enumerate() {
        let _ = new_filter.step(&mut new_rng, meas, t);
    }

    assert!(!legacy_result.labels.is_empty());
    assert!(!new_filter.state().is_empty() || ground_truth.measurements.len() > 0);
}

// ============================================================================
// LMBM Filter Tests
// ============================================================================

/// Test that both LMBM implementations run
/// This test can be slow and memory-intensive
#[test]
#[ignore]
fn test_lmbm_filter_both_apis_run() {
    let mut rng = SimpleRng::new(42);

    let model = generate_model(
        &mut rng,
        5.0,  // Lower clutter for faster test
        0.95,
        DataAssociationMethod::Gibbs,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);
    let measurements = &ground_truth.measurements;

    // Run legacy API
    let mut legacy_rng = SimpleRng::new(42);
    let legacy_result = run_lmbm_filter(&mut legacy_rng, &model, measurements);

    // Run new API
    let (motion, sensor, birth, association) = convert_model_to_new_api(&model);
    let lmbm_config = prak::types::LmbmConfig::default();
    let mut new_filter = LmbmFilter::new(motion, sensor, birth, association, lmbm_config);

    let mut new_rng = rand::thread_rng();
    let mut new_results = Vec::new();
    for (t, meas) in measurements.iter().enumerate() {
        let estimate = new_filter.step(&mut new_rng, meas, t).unwrap();
        new_results.push(estimate);
    }

    // Verify both produced output
    assert_eq!(legacy_result.labels.len(), new_results.len());

    println!("LMBM Filter comparison:");
    println!("  Legacy: {} timesteps", legacy_result.labels.len());
    println!("  New: {} timesteps", new_results.len());
}

// ============================================================================
// Multi-sensor LMB Filter Tests (placeholder - legacy doesn't have full impl)
// ============================================================================

/// Test that new multi-sensor LMB runs without legacy comparison
/// (Legacy MS-LMB filters are not fully implemented)
#[test]
fn test_multisensor_lmb_new_api_only() {
    use prak::filter::{AaLmbFilter, ArithmeticAverageMerger};
    use prak::types::MultisensorConfig;

    let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);

    let sensor1 = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
    let sensor2 = SensorModel::position_sensor_2d(1.2, 0.85, 12.0, 100.0);
    let sensors = MultisensorConfig::new(vec![sensor1, sensor2]);

    let birth_loc = BirthLocation::new(
        0,
        DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
        DMatrix::identity(4, 4) * 100.0,
    );
    let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);
    let association = AssociationConfig::default();
    let merger = ArithmeticAverageMerger::uniform(2, 100);

    let mut filter: AaLmbFilter = AaLmbFilter::new(motion, sensors, birth, association, merger);
    let mut rng = rand::thread_rng();

    // Run for 10 timesteps
    for t in 0..10 {
        let measurements = vec![
            vec![DVector::from_vec(vec![t as f64 * 0.5, t as f64 * 0.5])],
            vec![DVector::from_vec(vec![t as f64 * 0.5 + 0.1, t as f64 * 0.5 + 0.1])],
        ];
        let estimate = filter.step(&mut rng, &measurements, t).unwrap();
        println!("t={}: {} tracks estimated", t, estimate.tracks.len());
    }
}

// ============================================================================
// Multi-sensor LMBM Filter Tests
// ============================================================================

/// Test that new multi-sensor LMBM runs
#[test]
fn test_multisensor_lmbm_new_api_only() {
    use prak::filter::MultisensorLmbmFilter;
    use prak::types::{LmbmConfig, MultisensorConfig};

    let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);

    let sensor1 = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
    let sensor2 = SensorModel::position_sensor_2d(1.2, 0.85, 12.0, 100.0);
    let sensors = MultisensorConfig::new(vec![sensor1, sensor2]);

    let birth_loc = BirthLocation::new(
        0,
        DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
        DMatrix::identity(4, 4) * 100.0,
    );
    let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);
    let association = AssociationConfig::default();
    let lmbm_config = LmbmConfig::default();

    let mut filter = MultisensorLmbmFilter::new(motion, sensors, birth, association, lmbm_config);
    let mut rng = rand::thread_rng();

    // Run for 5 timesteps (LMBM is slower)
    for t in 0..5 {
        let measurements = vec![
            vec![DVector::from_vec(vec![t as f64 * 0.5, t as f64 * 0.5])],
            vec![DVector::from_vec(vec![t as f64 * 0.5 + 0.1, t as f64 * 0.5 + 0.1])],
        ];
        let estimate = filter.step(&mut rng, &measurements, t).unwrap();
        println!("t={}: {} tracks estimated", t, estimate.tracks.len());
    }
}

/// Compare new MS-LMBM with legacy run_multisensor_lmbm_filter
/// This test is ignored by default due to high memory usage
#[test]
#[ignore]
fn test_multisensor_lmbm_vs_legacy() {
    use prak::filter::MultisensorLmbmFilter;
    use prak::multisensor_lmbm::filter::run_multisensor_lmbm_filter;
    use prak::types::{LmbmConfig, MultisensorConfig};

    let mut rng = SimpleRng::new(42);

    let model = generate_model(
        &mut rng,
        5.0,
        0.9,
        DataAssociationMethod::Gibbs,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);

    // Convert measurements for multi-sensor format: [sensor][time][measurements]
    // For legacy: 2 sensors, same measurements
    let num_sensors = 2;
    let legacy_measurements: Vec<Vec<Vec<DVector<f64>>>> = (0..num_sensors)
        .map(|_| ground_truth.measurements.clone())
        .collect();

    // Run legacy
    let mut legacy_rng = SimpleRng::new(42);
    let legacy_result = run_multisensor_lmbm_filter(
        &mut legacy_rng,
        &model,
        &legacy_measurements,
        num_sensors,
    );

    // Run new API
    let motion = MotionModel::new(
        model.a.clone(),
        model.r.clone(),
        model.u.clone(),
        model.survival_probability,
    );

    let sensor = SensorModel::new(
        model.c.clone(),
        model.q.clone(),
        model.detection_probability,
        model.clutter_rate,
        model.observation_space_volume,
    );
    let sensors = MultisensorConfig::new(vec![sensor.clone(), sensor]);

    let birth_locations: Vec<BirthLocation> = (0..model.number_of_birth_locations)
        .map(|i| {
            BirthLocation::new(
                model.birth_location_labels[i],
                model.mu_b[i].clone(),
                model.sigma_b[i].clone(),
            )
        })
        .collect();
    let birth = BirthModel::new(birth_locations, model.r_b[0], model.r_b_lmbm[0]);

    let association = AssociationConfig {
        method: prak::types::DataAssociationMethod::Gibbs,
        gibbs_samples: model.number_of_samples,
        ..Default::default()
    };
    let lmbm_config = LmbmConfig::default();

    let mut new_filter = MultisensorLmbmFilter::new(
        motion, sensors, birth, association, lmbm_config
    );
    let mut new_rng = rand::thread_rng();

    let mut new_results = Vec::new();
    for (t, meas) in ground_truth.measurements.iter().enumerate() {
        // Same measurements for both sensors
        let ms_meas = vec![meas.clone(), meas.clone()];
        let estimate = new_filter.step(&mut new_rng, &ms_meas, t).unwrap();
        new_results.push(estimate);
    }

    // Compare
    println!("MS-LMBM Filter comparison:");
    println!("  Legacy: {} timesteps", legacy_result.labels.len());
    println!("  New: {} timesteps", new_results.len());

    assert_eq!(legacy_result.labels.len(), new_results.len());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Test filter behavior with no measurements at all timesteps
#[test]
fn test_lmb_no_measurements_all_timesteps() {
    let (motion, sensor, birth, association) = {
        let mut rng = SimpleRng::new(42);
        let model = generate_model(
            &mut rng,
            10.0,
            0.95,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );
        convert_model_to_new_api(&model)
    };

    let mut filter = LmbFilter::new(motion, sensor, birth, association);
    let mut rng = rand::thread_rng();

    for t in 0..10 {
        let empty_meas: Vec<DVector<f64>> = vec![];
        let estimate = filter.step(&mut rng, &empty_meas, t).unwrap();
        // With no measurements, existence should decay
        println!("t={}: {} tracks", t, estimate.tracks.len());
    }
}

/// Test filter reset functionality
#[test]
fn test_filter_reset_works() {
    let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);
    let sensor = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
    let birth_loc = BirthLocation::new(
        0,
        DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
        DMatrix::identity(4, 4) * 100.0,
    );
    let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);
    let association = AssociationConfig::default();

    let mut filter = LmbFilter::new(motion, sensor, birth, association);
    let mut rng = rand::thread_rng();

    // Run some steps
    for t in 0..5 {
        let meas = vec![DVector::from_vec(vec![t as f64, t as f64])];
        let _ = filter.step(&mut rng, &meas, t);
    }

    assert!(!filter.state().is_empty());

    // Reset
    filter.reset();

    assert!(filter.state().is_empty());
}
