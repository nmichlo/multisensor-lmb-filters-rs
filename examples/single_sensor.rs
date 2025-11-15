//! Single-sensor LMB/LMBM filter example
//!
//! Demonstrates running the LMB or LMBM filter on a simulated tracking scenario.
//! Matches MATLAB runFilters.m functionality.

use clap::Parser;
use prak::common::model::generate_model;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::common::ground_truth::generate_ground_truth;
use prak::common::rng::SimpleRng;
use prak::lmb::filter::run_lmb_filter;
use prak::lmbm::filter::run_lmbm_filter;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Random seed for deterministic runs
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Use LMBM filter instead of LMB
    #[arg(short, long)]
    lmbm: bool,

    /// Clutter rate (expected number of false measurements per time step)
    #[arg(short, long, default_value_t = 10.0)]
    clutter_rate: f64,

    /// Detection probability
    #[arg(short = 'p', long, default_value_t = 0.95)]
    detection_probability: f64,

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

    println!("Single-Sensor {} Filter Example", if args.lmbm { "LMBM" } else { "LMB" });
    println!("=====================================");
    println!("Seed: {}", args.seed);
    println!("Clutter rate: {}", args.clutter_rate);
    println!("Detection probability: {}", args.detection_probability);
    println!("Data association: {}", args.data_association);
    println!("Scenario type: {}", args.scenario_type);
    println!();

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

    // Generate model
    println!("Generating model...");
    let model = generate_model(
        &mut rng,
        args.clutter_rate,
        args.detection_probability,
        data_association_method,
        scenario_type,
        None, // Use default birth locations
    );

    // Generate ground truth and measurements
    println!("Generating ground truth and measurements...");
    let ground_truth_output = generate_ground_truth(&mut rng, &model, None);

    println!("Simulation length: {} time steps", ground_truth_output.measurements.len());
    println!("Number of objects: {}", ground_truth_output.ground_truth.len());

    // Count total measurements
    let total_measurements: usize = ground_truth_output.measurements.iter()
        .map(|m| m.len())
        .sum();
    println!("Total measurements: {}", total_measurements);
    println!();

    // Run filter
    println!("Running {} filter...", if args.lmbm { "LMBM" } else { "LMB" });
    let start_time = std::time::Instant::now();

    if args.lmbm {
        let state_estimates = run_lmbm_filter(&mut rng, &model, &ground_truth_output.measurements);
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
        let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth_output.measurements);
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
