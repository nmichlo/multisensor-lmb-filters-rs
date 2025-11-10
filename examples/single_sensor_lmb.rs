//! Single-sensor LMB filter example
//!
//! This example demonstrates the basic usage of the LMB filter for multi-object tracking
//! with a single sensor. It generates a simulation scenario, runs the filter, and
//! computes OSPA metrics to evaluate performance.
//!
//! This matches the MATLAB runFilters.m script with useLmbFilter = true.

use prak::common::ground_truth::generate_ground_truth;
use prak::common::metrics::ospa;
use prak::common::model::generate_model;
use prak::common::types::{DataAssociationMethod, OspaParameters, ScenarioType};
use prak::lmb::filter::run_lmb_filter;

fn main() {
    println!("=== Single-Sensor LMB Filter Example ===\n");

    // Generate model
    println!("Generating model...");
    let clutter_rate = 10.0;
    let detection_probability = 0.95;
    let data_association_method = DataAssociationMethod::LBPFixed;
    let scenario_type = ScenarioType::Fixed;

    let model = generate_model(
        clutter_rate,
        detection_probability,
        data_association_method,
        scenario_type,
        None,
    );
    println!("  Clutter rate: {}", clutter_rate);
    println!("  Detection probability: {}", detection_probability);
    println!("  Data association: {:?}", data_association_method);
    println!("  Scenario: {:?}", scenario_type);

    // Generate ground truth and measurements
    println!("\nGenerating ground truth and measurements...");
    let ground_truth_output = generate_ground_truth(&model, None);
    println!("  Simulation length: {} time steps", ground_truth_output.measurements.len());
    println!("  Number of objects: {}", ground_truth_output.ground_truth.len());

    // Print measurement statistics
    let total_measurements: usize = ground_truth_output.measurements.iter()
        .map(|m| m.len())
        .sum();
    let avg_measurements = total_measurements as f64 / ground_truth_output.measurements.len() as f64;
    println!("  Average measurements per time step: {:.1}", avg_measurements);

    // Run LMB filter
    println!("\nRunning LMB filter...");
    let state_estimates = run_lmb_filter(&model, &ground_truth_output.measurements);
    println!("  Filter completed successfully");

    // Compute OSPA metrics
    println!("\nComputing OSPA metrics...");
    let ospa_params = OspaParameters {
        e_c: 100.0, // Euclidean cut-off
        e_p: 1.0,   // Euclidean order
        h_c: 1.0,   // Hellinger cut-off
        h_p: 1.0,   // Hellinger order
    };

    let mut ospa_values = Vec::new();
    for t in 0..ground_truth_output.measurements.len() {
        // Get ground truth data at time t (from RFS representation)
        let x_gt = &ground_truth_output.ground_truth_rfs.x[t];
        let mu_gt = &ground_truth_output.ground_truth_rfs.mu[t];
        let sigma_gt = &ground_truth_output.ground_truth_rfs.sigma[t];

        // Get filter estimates at time t
        let nu = &state_estimates.mu[t];
        let t_sigma = &state_estimates.sigma[t];

        // Compute OSPA
        let ospa_metrics = ospa(x_gt, mu_gt, sigma_gt, nu, t_sigma, &ospa_params);
        ospa_values.push(ospa_metrics);
    }

    let avg_e_ospa: f64 = ospa_values.iter().map(|m| m.e_ospa.total).sum::<f64>() / ospa_values.len() as f64;
    let max_e_ospa = ospa_values.iter().map(|m| m.e_ospa.total).fold(f64::NEG_INFINITY, f64::max);

    println!("  Average Euclidean OSPA: {:.2}", avg_e_ospa);
    println!("  Maximum Euclidean OSPA: {:.2}", max_e_ospa);

    // Print final statistics
    println!("\n=== Results ===");
    println!("Total objects tracked: {}", state_estimates.objects.len());
    println!("Final time step cardinality: {}", state_estimates.mu[state_estimates.mu.len() - 1].len());

    // Print per-object info
    println!("\nObject details:");
    for (i, obj) in state_estimates.objects.iter().enumerate() {
        println!("  Object {}: birth_time={}, birth_location={}, trajectory_length={}",
                 i, obj.birth_time, obj.birth_location, obj.trajectory_length);
    }

    println!("\nExample completed successfully!");
}
