//! Minimal benchmark runner for native Rust.
//!
//! Usage:
//!     benchmark_single --scenario <path> --filter <name>
//!
//! Output:
//!     Prints elapsed time in milliseconds as a single number.
//!     Exit 0 on success, non-zero on error.

use std::fs;
use std::time::Instant;

use clap::Parser;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use serde::Deserialize;

use multisensor_lmb_filters_rs::lmb::*;

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser)]
#[command(name = "benchmark_single")]
#[command(about = "Minimal benchmark runner for native Rust")]
struct Args {
    /// Path to scenario JSON file
    #[arg(long)]
    scenario: String,

    /// Filter name (e.g., LMB-LBP, LMBM-Gibbs, AA-LMB-LBP)
    #[arg(long)]
    filter: String,
}

// =============================================================================
// JSON Schema
// =============================================================================

#[derive(Deserialize)]
struct ScenarioJson {
    model: ModelJson,
    bounds: [f64; 4],
    num_sensors: usize,
    steps: Vec<StepJson>,
}

#[derive(Deserialize)]
struct ModelJson {
    dt: f64,
    process_noise_std: f64,
    measurement_noise_std: f64,
    detection_probability: f64,
    survival_probability: f64,
    clutter_rate: f64,
    birth_locations: Vec<[f64; 4]>,
}

#[derive(Deserialize)]
struct StepJson {
    step: usize,
    sensor_readings: Option<Vec<Vec<[f64; 2]>>>,
}

// =============================================================================
// Preprocessing
// =============================================================================

struct PreprocessedScenario {
    motion: MotionModel,
    sensor: SensorModel,
    sensors_config: MultisensorConfig,
    birth: BirthModel,
    steps: Vec<(usize, Vec<DVector<f64>>)>, // Single-sensor: (timestep, measurements)
    multi_steps: Vec<Vec<Vec<DVector<f64>>>>, // Multi-sensor: step -> sensor -> measurements
    num_sensors: usize,
}

fn preprocess(scenario: &ScenarioJson) -> PreprocessedScenario {
    // Motion model
    let motion = MotionModel::constant_velocity_2d(
        scenario.model.dt,
        scenario.model.process_noise_std,
        scenario.model.survival_probability,
    );

    // Observation volume
    let obs_vol =
        (scenario.bounds[1] - scenario.bounds[0]) * (scenario.bounds[3] - scenario.bounds[2]);

    // Sensor model
    let sensor = SensorModel::position_sensor_2d(
        scenario.model.measurement_noise_std,
        scenario.model.detection_probability,
        scenario.model.clutter_rate,
        obs_vol,
    );

    // Multi-sensor config
    let sensors_config = MultisensorConfig::new(vec![sensor.clone(); scenario.num_sensors]);

    // Birth model
    let birth_locs: Vec<_> = scenario
        .model
        .birth_locations
        .iter()
        .enumerate()
        .map(|(i, &loc)| {
            BirthLocation::new(
                i,
                DVector::from_vec(vec![loc[0], loc[1], loc[2], loc[3]]),
                DMatrix::from_diagonal(&DVector::from_vec(vec![2500.0, 2500.0, 100.0, 100.0])),
            )
        })
        .collect();
    let birth = BirthModel::new(birth_locs, 0.01, 0.001);

    // Single-sensor steps (use first sensor)
    let steps: Vec<_> = scenario
        .steps
        .iter()
        .map(|step| {
            let readings = step.sensor_readings.as_ref();
            let single_meas: Vec<DVector<f64>> = if let Some(rss) = readings {
                if !rss.is_empty() && !rss[0].is_empty() {
                    rss[0]
                        .iter()
                        .map(|m| DVector::from_vec(vec![m[0], m[1]]))
                        .collect()
                } else {
                    vec![]
                }
            } else {
                vec![]
            };
            (step.step, single_meas)
        })
        .collect();

    // Multi-sensor steps
    let multi_steps: Vec<_> = scenario
        .steps
        .iter()
        .map(|step| {
            let readings = step.sensor_readings.as_ref();
            if let Some(rss) = readings {
                rss.iter()
                    .map(|r| {
                        r.iter()
                            .map(|m| DVector::from_vec(vec![m[0], m[1]]))
                            .collect()
                    })
                    .collect()
            } else {
                vec![vec![]; scenario.num_sensors]
            }
        })
        .collect();

    PreprocessedScenario {
        motion,
        sensor,
        sensors_config,
        birth,
        steps,
        multi_steps,
        num_sensors: scenario.num_sensors,
    }
}

// =============================================================================
// Thresholds (must match Python)
// =============================================================================

const GM_WEIGHT_THRESHOLD: f64 = 1e-4;
const MAX_GM_COMPONENTS: usize = 100;
const GM_MERGE_THRESHOLD: f64 = f64::INFINITY;

// =============================================================================
// Filter Runners
// =============================================================================

fn run_lmb_filter(prep: &PreprocessedScenario, assoc_config: AssociationConfig) -> f64 {
    let associator = DynamicAssociator::from_config(&assoc_config);
    let mut filter = LmbFilter::with_associator_type(
        prep.motion.clone(),
        prep.sensor.clone(),
        prep.birth.clone(),
        assoc_config,
        associator,
    )
    .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
    .with_gm_merge_threshold(GM_MERGE_THRESHOLD);

    let mut rng = StdRng::seed_from_u64(42);
    let start = Instant::now();

    for (t, single_meas) in prep.steps.iter() {
        let _ = filter.step(&mut rng, single_meas, *t);
    }

    start.elapsed().as_micros() as f64 / 1000.0
}

fn run_lmbm_filter(prep: &PreprocessedScenario, assoc_config: AssociationConfig) -> f64 {
    // LMBM uses hypothesis pruning (not GM pruning) via LmbmConfig
    let lmbm_config = LmbmConfig::default();
    let associator = DynamicAssociator::from_config(&assoc_config);
    let mut filter = LmbmFilter::with_associator_type(
        prep.motion.clone(),
        prep.sensor.clone(),
        prep.birth.clone(),
        assoc_config,
        lmbm_config,
        associator,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let start = Instant::now();

    for (t, single_meas) in prep.steps.iter() {
        let _ = filter.step(&mut rng, single_meas, *t);
    }

    start.elapsed().as_micros() as f64 / 1000.0
}

fn run_aa_lmb_filter(prep: &PreprocessedScenario, assoc_config: AssociationConfig) -> f64 {
    let merger = ArithmeticAverageMerger::uniform(prep.num_sensors, 100);
    let associator = DynamicAssociator::from_config(&assoc_config);
    let mut filter = AaLmbFilter::with_associator_type(
        prep.motion.clone(),
        prep.sensors_config.clone(),
        prep.birth.clone(),
        assoc_config,
        merger,
        associator,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let start = Instant::now();

    for (t, meas) in prep.multi_steps.iter().enumerate() {
        let _ = filter.step(&mut rng, meas, t);
    }

    start.elapsed().as_micros() as f64 / 1000.0
}

fn run_ic_lmb_filter(prep: &PreprocessedScenario, assoc_config: AssociationConfig) -> f64 {
    let merger = IteratedCorrectorMerger::new();
    let associator = DynamicAssociator::from_config(&assoc_config);
    let mut filter = IcLmbFilter::with_associator_type(
        prep.motion.clone(),
        prep.sensors_config.clone(),
        prep.birth.clone(),
        assoc_config,
        merger,
        associator,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let start = Instant::now();

    for (t, meas) in prep.multi_steps.iter().enumerate() {
        let _ = filter.step(&mut rng, meas, t);
    }

    start.elapsed().as_micros() as f64 / 1000.0
}

fn run_pu_lmb_filter(prep: &PreprocessedScenario, assoc_config: AssociationConfig) -> f64 {
    let merger = ParallelUpdateMerger::new(Vec::new());
    let associator = DynamicAssociator::from_config(&assoc_config);
    let mut filter = PuLmbFilter::with_associator_type(
        prep.motion.clone(),
        prep.sensors_config.clone(),
        prep.birth.clone(),
        assoc_config,
        merger,
        associator,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let start = Instant::now();

    for (t, meas) in prep.multi_steps.iter().enumerate() {
        let _ = filter.step(&mut rng, meas, t);
    }

    start.elapsed().as_micros() as f64 / 1000.0
}

fn run_ga_lmb_filter(prep: &PreprocessedScenario, assoc_config: AssociationConfig) -> f64 {
    let merger = GeometricAverageMerger::uniform(prep.num_sensors);
    let associator = DynamicAssociator::from_config(&assoc_config);
    let mut filter = GaLmbFilter::with_associator_type(
        prep.motion.clone(),
        prep.sensors_config.clone(),
        prep.birth.clone(),
        assoc_config,
        merger,
        associator,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let start = Instant::now();

    for (t, meas) in prep.multi_steps.iter().enumerate() {
        let _ = filter.step(&mut rng, meas, t);
    }

    start.elapsed().as_micros() as f64 / 1000.0
}

fn run_ms_lmbm_filter(prep: &PreprocessedScenario, assoc_config: AssociationConfig) -> f64 {
    // MS-LMBM uses hypothesis pruning via LmbmConfig
    let lmbm_config = LmbmConfig::default();
    let associator = MultisensorGibbsAssociator;
    let mut filter = MultisensorLmbmFilter::with_associator_type(
        prep.motion.clone(),
        prep.sensors_config.clone(),
        prep.birth.clone(),
        assoc_config,
        lmbm_config,
        associator,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let start = Instant::now();

    for (t, meas) in prep.multi_steps.iter().enumerate() {
        let _ = filter.step(&mut rng, meas, t);
    }

    start.elapsed().as_micros() as f64 / 1000.0
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load scenario
    let content = fs::read_to_string(&args.scenario)?;
    let scenario: ScenarioJson = serde_json::from_str(&content)?;
    let prep = preprocess(&scenario);

    // Parse filter name to get associator config and run appropriate filter
    let elapsed_ms = match args.filter.as_str() {
        // Single-sensor LMB
        "LMB-LBP" => run_lmb_filter(&prep, AssociationConfig::lbp(100, 1e-6)),
        "LMB-Gibbs" => run_lmb_filter(&prep, AssociationConfig::gibbs(1000)),
        "LMB-Murty" => run_lmb_filter(&prep, AssociationConfig::murty(25)),

        // Single-sensor LMBM
        "LMBM-Gibbs" => run_lmbm_filter(&prep, AssociationConfig::gibbs(1000)),
        "LMBM-Murty" => run_lmbm_filter(&prep, AssociationConfig::murty(25)),

        // Multi-sensor LMB variants
        "AA-LMB-LBP" => run_aa_lmb_filter(&prep, AssociationConfig::lbp(100, 1e-6)),
        "IC-LMB-LBP" => run_ic_lmb_filter(&prep, AssociationConfig::lbp(100, 1e-6)),
        "PU-LMB-LBP" => run_pu_lmb_filter(&prep, AssociationConfig::lbp(100, 1e-6)),
        "GA-LMB-LBP" => run_ga_lmb_filter(&prep, AssociationConfig::lbp(100, 1e-6)),

        // Multi-sensor LMBM
        "MS-LMBM-Gibbs" => run_ms_lmbm_filter(&prep, AssociationConfig::gibbs(1000)),

        _ => {
            eprintln!("Unknown filter: {}", args.filter);
            eprintln!(
                "Available: LMB-LBP, LMB-Gibbs, LMB-Murty, LMBM-Gibbs, LMBM-Murty, \
                 AA-LMB-LBP, IC-LMB-LBP, PU-LMB-LBP, GA-LMB-LBP, MS-LMBM-Gibbs"
            );
            std::process::exit(1);
        }
    };

    // Output only the timing
    println!("{:.3}", elapsed_ms);
    Ok(())
}
