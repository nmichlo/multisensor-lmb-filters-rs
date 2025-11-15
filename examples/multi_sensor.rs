//! Multi-sensor LMB/LMBM filter example
//!
//! Demonstrates running multi-sensor filters on a simulated tracking scenario.
//! Matches MATLAB runMultisensorFilters.m functionality.

use clap::Parser;
use prak::common::model::generate_multisensor_model;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::common::ground_truth::generate_multisensor_ground_truth;
use prak::common::rng::SimpleRng;
use prak::multisensor_lmb::parallel_update::{run_parallel_update_lmb_filter, ParallelUpdateMode};
use prak::multisensor_lmb::iterated_corrector::run_ic_lmb_filter;
use prak::multisensor_lmbm::filter::run_multisensor_lmbm_filter;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Random seed for deterministic runs
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Filter type: IC, PU, GA, AA, LMBM
    #[arg(short, long, default_value = "PU")]
    filter_type: String,

    /// Number of sensors
    #[arg(short, long, default_value_t = 3)]
    num_sensors: usize,

    /// Data association method: LBP, LBPFixed, Gibbs, Murty
    #[arg(short = 'a', long, default_value = "LBP")]
    data_association: String,

    /// Scenario type: Fixed or Random
    #[arg(short = 't', long, default_value = "Fixed")]
    scenario_type: String,
}

fn main() {
    let args = Args::parse();

    // Initialize RNG with seed
    let mut rng = SimpleRng::new(args.seed);

    println!("Multi-Sensor {} Filter Example", args.filter_type);
    println!("=====================================");
    println!("Seed: {}", args.seed);
    println!("Number of sensors: {}", args.num_sensors);
    println!("Data association: {}", args.data_association);
    println!("Scenario type: {}", args.scenario_type);
    println!();

    // Parse filter type and update mode
    let (use_lmbm, update_mode) = match args.filter_type.as_str() {
        "IC" => (false, None),
        "PU" => (false, Some(ParallelUpdateMode::PU)),
        "GA" => (false, Some(ParallelUpdateMode::GA)),
        "AA" => (false, Some(ParallelUpdateMode::AA)),
        "LMBM" => (true, None),
        _ => {
            eprintln!("Unknown filter type: {}", args.filter_type);
            eprintln!("Valid options: IC (Iterated Corrector), PU (Parallel Update), GA (Geometric Average), AA (Arithmetic Average), LMBM");
            std::process::exit(1);
        }
    };

    // Parse data association method
    let data_association_method = match args.data_association.as_str() {
        "LBP" => DataAssociationMethod::LBP,
        "LBPFixed" => DataAssociationMethod::LBPFixed,
        "Gibbs" => DataAssociationMethod::Gibbs,
        "Murty" => DataAssociationMethod::Murty,
        _ => {
            eprintln!("Unknown data association method: {}", args.data_association);
            eprintln!("Valid options: LBP, LBPFixed, Gibbs, Murty");
            std::process::exit(1);
        }
    };

    // Parse scenario type
    let scenario_type = match args.scenario_type.as_str() {
        "Fixed" => ScenarioType::Fixed,
        "Random" => ScenarioType::Random,
        _ => {
            eprintln!("Unknown scenario type: {}", args.scenario_type);
            eprintln!("Valid options: Fixed, Random");
            std::process::exit(1);
        }
    };

    // Default parameters (matching MATLAB runMultisensorFilters.m)
    let clutter_rates = vec![5.0; args.num_sensors];
    let detection_probabilities = match args.num_sensors {
        3 => vec![0.67, 0.70, 0.73],
        n => (0..n).map(|i| 0.65 + 0.05 * (i as f64 / (n - 1).max(1) as f64)).collect(),
    };
    let q_values = match args.num_sensors {
        3 => vec![4.0, 3.0, 2.0],
        n => (0..n).map(|i| 4.0 - (i as f64 / (n - 1).max(1) as f64)).collect(),
    };

    let parallel_update_mode = update_mode.unwrap_or(ParallelUpdateMode::PU);

    // Generate model
    println!("Generating multi-sensor model...");
    let model = generate_multisensor_model(
        &mut rng,
        args.num_sensors,
        clutter_rates,
        detection_probabilities,
        q_values,
        parallel_update_mode,
        data_association_method,
        scenario_type,
        None, // Use default birth locations
    );

    // Generate ground truth and measurements
    println!("Generating ground truth and measurements...");
    let ground_truth_output = generate_multisensor_ground_truth(
        &mut rng,
        &model,
        None,
    );

    println!("Simulation length: {} time steps", ground_truth_output.measurements[0].len());
    println!("Number of objects: {}", ground_truth_output.ground_truth.len());

    // Count total measurements per sensor
    for (s, sensor_measurements) in ground_truth_output.measurements.iter().enumerate() {
        let total: usize = sensor_measurements.iter().map(|m| m.len()).sum();
        println!("Sensor {} total measurements: {}", s + 1, total);
    }
    println!();

    // Run filter
    println!("Running {} filter...", args.filter_type);
    let start_time = std::time::Instant::now();

    if use_lmbm {
        let state_estimates = run_multisensor_lmbm_filter(
            &mut rng,
            &model,
            &ground_truth_output.measurements,
            args.num_sensors,
        );
        let elapsed = start_time.elapsed();

        println!("Filter completed in {:.2}s", elapsed.as_secs_f64());
        println!();
        println!("Results:");
        println!("  Estimated trajectories: {}", state_estimates.objects.len());

        // Show cardinality estimates over time
        println!();
        println!("Cardinality estimates (first 10 time steps):");
        for (t, labels) in state_estimates.labels.iter().enumerate().take(10) {
            println!("  t={:3}: {} objects", t + 1, labels.len());
        }
    } else if args.filter_type == "IC" {
        let state_estimates = run_ic_lmb_filter(
            &mut rng,
            &model,
            &ground_truth_output.measurements,
            args.num_sensors,
        );
        let elapsed = start_time.elapsed();

        println!("Filter completed in {:.2}s", elapsed.as_secs_f64());
        println!();
        println!("Results:");
        println!("  Estimated trajectories: {}", state_estimates.objects.len());

        // Show cardinality estimates over time
        println!();
        println!("Cardinality estimates (first 10 time steps):");
        for (t, labels) in state_estimates.labels.iter().enumerate().take(10) {
            println!("  t={:3}: {} objects", t + 1, labels.len());
        }
    } else {
        let state_estimates = run_parallel_update_lmb_filter(
            &mut rng,
            &model,
            &ground_truth_output.measurements,
            args.num_sensors,
            parallel_update_mode,
        );
        let elapsed = start_time.elapsed();

        println!("Filter completed in {:.2}s", elapsed.as_secs_f64());
        println!();
        println!("Results:");
        println!("  Estimated trajectories: {}", state_estimates.objects.len());

        // Show cardinality estimates over time
        println!();
        println!("Cardinality estimates (first 10 time steps):");
        for (t, labels) in state_estimates.labels.iter().enumerate().take(10) {
            println!("  t={:3}: {} objects", t + 1, labels.len());
        }
    }

    println!();
    println!("Example completed successfully!");
}
