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

    /// Print filter configuration JSON (can combine with --skip-run)
    #[arg(long)]
    get_config: bool,

    /// Skip running the benchmark (useful with --get-config)
    #[arg(long)]
    skip_run: bool,
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
// Thresholds (match Python: existence=1e-3, gm_weight=1e-4, max_components=100, gm_merge=inf)
// =============================================================================

const GM_WEIGHT_THRESHOLD: f64 = 1e-4;
const MAX_GM_COMPONENTS: usize = 100;
const GM_MERGE_THRESHOLD: f64 = f64::INFINITY;

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
// Filter Factory - Single source of truth for filter creation
// =============================================================================

/// Enum to hold any filter type for unified handling
enum AnyFilter {
    Lmb(LmbFilter<DynamicAssociator>),
    Lmbm(LmbmFilter<DynamicAssociator>),
    AaLmb(MultisensorLmbFilter<LbpAssociator, ArithmeticAverageMerger>),
    IcLmb(MultisensorLmbFilter<LbpAssociator, IteratedCorrectorMerger>),
    PuLmb(MultisensorLmbFilter<LbpAssociator, ParallelUpdateMerger>),
    GaLmb(MultisensorLmbFilter<LbpAssociator, GeometricAverageMerger>),
    MsLmbm(MultisensorLmbmFilter<MultisensorGibbsAssociator>),
}

impl AnyFilter {
    /// Get config JSON from any filter type
    fn get_config_json(&self) -> String {
        match self {
            AnyFilter::Lmb(f) => f.get_config().to_json_pretty(),
            AnyFilter::Lmbm(f) => f.get_config().to_json_pretty(),
            AnyFilter::AaLmb(f) => f.get_config().to_json_pretty(),
            AnyFilter::IcLmb(f) => f.get_config().to_json_pretty(),
            AnyFilter::PuLmb(f) => f.get_config().to_json_pretty(),
            AnyFilter::GaLmb(f) => f.get_config().to_json_pretty(),
            AnyFilter::MsLmbm(f) => f.get_config().to_json_pretty(),
        }
    }

    /// Run benchmark and return elapsed time in milliseconds
    fn run(&mut self, prep: &PreprocessedScenario) -> f64 {
        let mut rng = SimpleRng::new(42);
        let start = Instant::now();

        match self {
            AnyFilter::Lmb(f) => {
                for (t, meas) in prep.steps.iter() {
                    let _ = f.step(&mut rng, meas, *t);
                }
            }
            AnyFilter::Lmbm(f) => {
                for (t, meas) in prep.steps.iter() {
                    let _ = f.step(&mut rng, meas, *t);
                }
            }
            AnyFilter::AaLmb(f) => {
                for (t, meas) in prep.multi_steps.iter().enumerate() {
                    let _ = f.step(&mut rng, meas, t);
                }
            }
            AnyFilter::IcLmb(f) => {
                for (t, meas) in prep.multi_steps.iter().enumerate() {
                    let _ = f.step(&mut rng, meas, t);
                }
            }
            AnyFilter::PuLmb(f) => {
                for (t, meas) in prep.multi_steps.iter().enumerate() {
                    let _ = f.step(&mut rng, meas, t);
                }
            }
            AnyFilter::GaLmb(f) => {
                for (t, meas) in prep.multi_steps.iter().enumerate() {
                    let _ = f.step(&mut rng, meas, t);
                }
            }
            AnyFilter::MsLmbm(f) => {
                for (t, meas) in prep.multi_steps.iter().enumerate() {
                    let _ = f.step(&mut rng, meas, t);
                }
            }
        }

        start.elapsed().as_micros() as f64 / 1000.0
    }
}

/// LMBM config used by both single and multi-sensor LMBM filters
const LMBM_CONFIG: LmbmConfig = LmbmConfig {
    max_hypotheses: 25,
    hypothesis_weight_threshold: 1e-3,
    use_eap: false,
};

fn create_filter(filter_name: &str, prep: &PreprocessedScenario) -> Result<AnyFilter, String> {
    match filter_name {
        // Single-sensor LMB
        "LMB-LBP" => {
            let assoc = AssociationConfig::lbp(100, 1e-6);
            Ok(AnyFilter::Lmb(
                LmbFilter::with_associator_type(
                    prep.motion.clone(),
                    prep.sensor.clone(),
                    prep.birth.clone(),
                    assoc.clone(),
                    DynamicAssociator::from_config(&assoc),
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }
        "LMB-Gibbs" => {
            let assoc = AssociationConfig::gibbs(1000);
            Ok(AnyFilter::Lmb(
                LmbFilter::with_associator_type(
                    prep.motion.clone(),
                    prep.sensor.clone(),
                    prep.birth.clone(),
                    assoc.clone(),
                    DynamicAssociator::from_config(&assoc),
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }
        "LMB-Murty" => {
            let assoc = AssociationConfig::murty(25);
            Ok(AnyFilter::Lmb(
                LmbFilter::with_associator_type(
                    prep.motion.clone(),
                    prep.sensor.clone(),
                    prep.birth.clone(),
                    assoc.clone(),
                    DynamicAssociator::from_config(&assoc),
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }

        // Single-sensor LMBM
        "LMBM-Gibbs" => {
            let assoc = AssociationConfig::gibbs(1000);
            Ok(AnyFilter::Lmbm(LmbmFilter::with_associator_type(
                prep.motion.clone(),
                prep.sensor.clone(),
                prep.birth.clone(),
                assoc.clone(),
                LMBM_CONFIG,
                DynamicAssociator::from_config(&assoc),
            )))
        }
        "LMBM-Murty" => {
            let assoc = AssociationConfig::murty(25);
            Ok(AnyFilter::Lmbm(LmbmFilter::with_associator_type(
                prep.motion.clone(),
                prep.sensor.clone(),
                prep.birth.clone(),
                assoc.clone(),
                LMBM_CONFIG,
                DynamicAssociator::from_config(&assoc),
            )))
        }

        // Multi-sensor LMB variants
        "AA-LMB-LBP" => {
            let assoc = AssociationConfig::lbp(100, 1e-6);
            Ok(AnyFilter::AaLmb(
                MultisensorLmbFilter::new(
                    prep.motion.clone(),
                    prep.sensors_config.clone(),
                    prep.birth.clone(),
                    assoc,
                    ArithmeticAverageMerger::uniform(prep.num_sensors, MAX_GM_COMPONENTS),
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }
        "IC-LMB-LBP" => {
            let assoc = AssociationConfig::lbp(100, 1e-6);
            Ok(AnyFilter::IcLmb(
                MultisensorLmbFilter::new(
                    prep.motion.clone(),
                    prep.sensors_config.clone(),
                    prep.birth.clone(),
                    assoc,
                    IteratedCorrectorMerger::new(),
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }
        "PU-LMB-LBP" => {
            let assoc = AssociationConfig::lbp(100, 1e-6);
            Ok(AnyFilter::PuLmb(
                MultisensorLmbFilter::new(
                    prep.motion.clone(),
                    prep.sensors_config.clone(),
                    prep.birth.clone(),
                    assoc,
                    ParallelUpdateMerger::new(Vec::new()),
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }
        "GA-LMB-LBP" => {
            let assoc = AssociationConfig::lbp(100, 1e-6);
            Ok(AnyFilter::GaLmb(
                MultisensorLmbFilter::new(
                    prep.motion.clone(),
                    prep.sensors_config.clone(),
                    prep.birth.clone(),
                    assoc,
                    GeometricAverageMerger::uniform(prep.num_sensors),
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }

        // Multi-sensor LMBM
        "MS-LMBM-Gibbs" => {
            let assoc = AssociationConfig::gibbs(1000);
            Ok(AnyFilter::MsLmbm(
                MultisensorLmbmFilter::with_associator_type(
                    prep.motion.clone(),
                    prep.sensors_config.clone(),
                    prep.birth.clone(),
                    assoc,
                    LMBM_CONFIG,
                    MultisensorGibbsAssociator,
                ),
            ))
        }

        _ => Err(format!(
            "Unknown filter: {}. Available: LMB-LBP, LMB-Gibbs, LMB-Murty, \
             LMBM-Gibbs, LMBM-Murty, AA-LMB-LBP, IC-LMB-LBP, PU-LMB-LBP, \
             GA-LMB-LBP, MS-LMBM-Gibbs",
            filter_name
        )),
    }
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

    // Create filter (single source of truth)
    let mut filter = match create_filter(&args.filter, &prep) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    // Output config if requested
    if args.get_config {
        println!("{}", filter.get_config_json());
    }

    // Run benchmark unless --skip-run
    if !args.skip_run {
        let elapsed_ms = filter.run(&prep);
        // Calculate average time per step
        let total_steps = scenario.steps.len() as f64;
        let avg_ms = elapsed_ms / total_steps; // Corrected: elapsed_ms is already in milliseconds
        println!("{:.4}", avg_ms);
    }

    Ok(())
}
