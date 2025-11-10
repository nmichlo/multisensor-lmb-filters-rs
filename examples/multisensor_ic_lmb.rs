//! Multi-sensor Iterated Corrector (IC) LMB filter example
//!
//! This example demonstrates the IC-LMB filter for multi-sensor tracking.
//! The IC-LMB filter uses sequential sensor updates (iterated correction).
//!
//! This matches the MATLAB runMultisensorFilters.m script with filterType = 'IC'.

use prak::common::ground_truth::generate_multisensor_ground_truth;
use prak::common::model::generate_multisensor_model;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::multisensor_lmb::iterated_corrector::run_ic_lmb_filter;
use prak::multisensor_lmb::parallel_update::ParallelUpdateMode;

fn main() {
    println!("=== Multi-Sensor IC-LMB Filter Example ===\n");

    // Generate multisensor model
    println!("Generating multisensor model...");
    let number_of_sensors = 3;
    let clutter_rates = vec![5.0, 5.0, 5.0];
    let detection_probabilities = vec![0.67, 0.70, 0.73];
    let q_values = vec![4.0, 3.0, 2.0]; // Measurement noise std devs per sensor
    let lmb_parallel_update_mode = ParallelUpdateMode::PU;
    let data_association_method = DataAssociationMethod::LBPFixed;
    let scenario_type = ScenarioType::Fixed;

    let model = generate_multisensor_model(
        number_of_sensors,
        clutter_rates.clone(),
        detection_probabilities.clone(),
        q_values.clone(),
        lmb_parallel_update_mode,
        data_association_method,
        scenario_type,
        None,
    );

    println!("  Number of sensors: {}", number_of_sensors);
    for (s, ((clutter, pd), q)) in clutter_rates.iter().zip(&detection_probabilities).zip(&q_values).enumerate() {
        println!("  Sensor {}: clutter={}, Pd={}, q={}", s, clutter, pd, q);
    }
    println!("  Data association: {:?}", data_association_method);
    println!("  Scenario: {:?}", scenario_type);

    // Generate multisensor ground truth and measurements
    println!("\nGenerating multisensor ground truth and measurements...");
    let ground_truth_output = generate_multisensor_ground_truth(&model, None);
    println!("  Simulation length: {} time steps", ground_truth_output.measurements[0].len());
    println!("  Number of objects: {}", ground_truth_output.ground_truth.len());

    // Print measurement statistics per sensor
    for (s, sensor_meas) in ground_truth_output.measurements.iter().enumerate() {
        let total_meas: usize = sensor_meas.iter().map(|m| m.len()).sum();
        let avg_meas = total_meas as f64 / sensor_meas.len() as f64;
        println!("  Sensor {}: avg {:.1} measurements/timestep", s, avg_meas);
    }

    // Run IC-LMB filter
    println!("\nRunning IC-LMB filter...");
    let state_estimates = run_ic_lmb_filter(&model, &ground_truth_output.measurements, number_of_sensors);
    println!("  Filter completed successfully");

    // Print final statistics
    println!("\n=== Results ===");
    println!("Total trajectories tracked: {}", state_estimates.objects.len());
    println!("Final time step cardinality: {}", state_estimates.mu[state_estimates.mu.len() - 1].len());

    // Print per-trajectory info
    println!("\nTrajectory details:");
    for (i, obj) in state_estimates.objects.iter().take(10).enumerate() {
        println!("  Trajectory {}: birth_time={}, birth_location={}, trajectory_length={}",
                 i, obj.birth_time, obj.birth_location, obj.trajectory_length);
    }
    if state_estimates.objects.len() > 10 {
        println!("  ... and {} more trajectories", state_estimates.objects.len() - 10);
    }

    println!("\nExample completed successfully!");
}
