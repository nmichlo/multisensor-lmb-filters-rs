//! Multi-sensor LMBM filter example
//!
//! This example demonstrates the Multi-sensor LMBM (Labelled Multi-Bernoulli Mixture)
//! filter for multi-sensor multi-object tracking. The multisensor LMBM filter maintains
//! multiple hypotheses across sensors for improved tracking accuracy.
//!
//! This matches the MATLAB runMultisensorFilters.m script with filterType = 'LMBM'.
//!
//! Note: This example uses simplified multisensor setup. For production use,
//! implement proper multisensor model and ground truth generation.

use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::multisensor_lmbm::filter::run_multisensor_lmbm_filter;
use nalgebra::DVector;

fn main() {
    println!("=== Multi-Sensor LMBM Filter Example ===\n");

    // Generate model
    println!("Generating model...");
    let clutter_rate = 5.0; // Base clutter rate
    let detection_probability = 0.67; // Base detection probability
    let data_association_method = DataAssociationMethod::Gibbs; // LMBM typically uses Gibbs
    let scenario_type = ScenarioType::Fixed;
    let number_of_sensors = 3;

    let model = generate_model(
        clutter_rate,
        detection_probability,
        data_association_method,
        scenario_type,
        None,
    );
    println!("  Number of sensors: {}", number_of_sensors);
    println!("  Base clutter rate: {}", clutter_rate);
    println!("  Base detection probability: {}", detection_probability);
    println!("  Data association: {:?}", data_association_method);
    println!("  Scenario: {:?}", scenario_type);
    println!("  Max hypotheses: {}", model.maximum_number_of_posterior_hypotheses);

    // Generate ground truth and measurements
    println!("\nGenerating ground truth and measurements...");
    let ground_truth_output = generate_ground_truth(&model, None);
    println!("  Simulation length: {} time steps", ground_truth_output.measurements.len());
    println!("  Number of objects: {}", ground_truth_output.ground_truth.len());

    // Create multisensor measurements by replicating with different noise
    // Note: This is a simplified approach. In production, implement proper
    // multisensor ground truth generation with sensor-specific parameters.
    println!("\nCreating multi-sensor measurements...");
    let mut multisensor_measurements: Vec<Vec<Vec<DVector<f64>>>> = Vec::new();

    for s in 0..number_of_sensors {
        let sensor_measurements: Vec<Vec<DVector<f64>>> = ground_truth_output.measurements
            .iter()
            .map(|time_meas| {
                // For each sensor, use a subset of measurements with different characteristics
                // This simulates different sensor viewing angles and detection probabilities
                time_meas
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| (idx + s) % number_of_sensors == 0) // Different subset per sensor
                    .map(|(_, m)| m.clone())
                    .collect()
            })
            .collect();
        multisensor_measurements.push(sensor_measurements);
    }

    // Print measurement statistics per sensor
    for (s, sensor_meas) in multisensor_measurements.iter().enumerate() {
        let total_meas: usize = sensor_meas.iter().map(|m| m.len()).sum();
        let avg_meas = total_meas as f64 / sensor_meas.len() as f64;
        println!("  Sensor {}: avg {:.1} measurements/timestep", s, avg_meas);
    }

    // Run Multi-sensor LMBM filter
    println!("\nRunning Multi-sensor LMBM filter...");
    let state_estimates = run_multisensor_lmbm_filter(&model, &multisensor_measurements, number_of_sensors);
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
    println!("\nNote: This example uses simplified multisensor measurement generation.");
    println!("For production use, implement proper multisensor model and ground truth generation");
    println!("based on generateMultisensorModel.m and generateMultisensorGroundTruth.m");
}
