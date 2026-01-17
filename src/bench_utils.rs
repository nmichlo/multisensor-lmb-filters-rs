//! Benchmark utilities shared between Criterion benchmarks and the benchmark_single binary.
//!
//! This module provides:
//! - JSON schema for scenario files
//! - Scenario preprocessing
//! - Filter factory functions
//! - Common benchmark thresholds

use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use std::fs;

use crate::lmb::*;

// =============================================================================
// Benchmark Thresholds
// =============================================================================

/// GM component weight threshold for pruning
pub const GM_WEIGHT_THRESHOLD: f64 = 1e-4;

/// Maximum number of GM components per track
pub const MAX_GM_COMPONENTS: usize = 100;

/// GM merge threshold (infinity = no merging, for MATLAB equivalence)
pub const GM_MERGE_THRESHOLD: f64 = f64::INFINITY;

/// LMBM configuration for hypothesis management
pub const LMBM_CONFIG: LmbmConfig = LmbmConfig {
    max_hypotheses: 25,
    hypothesis_weight_threshold: 1e-3,
    use_eap: false,
};

// =============================================================================
// JSON Schema for Scenario Files
// =============================================================================

/// Root structure for scenario JSON files
#[derive(Deserialize, Clone)]
pub struct ScenarioJson {
    pub model: ModelJson,
    pub bounds: [f64; 4],
    pub num_sensors: usize,
    pub steps: Vec<StepJson>,
}

/// Model parameters from scenario JSON
#[derive(Deserialize, Clone)]
pub struct ModelJson {
    pub dt: f64,
    pub process_noise_std: f64,
    pub measurement_noise_std: f64,
    pub detection_probability: f64,
    pub survival_probability: f64,
    pub clutter_rate: f64,
    pub birth_locations: Vec<[f64; 4]>,
}

/// Single timestep data from scenario JSON
#[derive(Deserialize, Clone)]
pub struct StepJson {
    pub step: usize,
    pub sensor_readings: Option<Vec<Vec<[f64; 2]>>>,
}

// =============================================================================
// Preprocessed Scenario
// =============================================================================

/// Preprocessed scenario data ready for filter execution
#[derive(Clone)]
pub struct PreprocessedScenario {
    pub motion: MotionModel,
    pub sensor: SensorModel,
    pub sensors_config: MultisensorConfig,
    pub birth: BirthModel,
    /// Single-sensor steps: (timestep, measurements)
    pub steps: Vec<(usize, Vec<DVector<f64>>)>,
    /// Multi-sensor steps: step -> sensor -> measurements
    pub multi_steps: Vec<Vec<Vec<DVector<f64>>>>,
    pub num_sensors: usize,
}

/// Load a scenario from a JSON file
pub fn load_scenario(path: &str) -> ScenarioJson {
    let content = fs::read_to_string(path).expect("Failed to read scenario file");
    serde_json::from_str(&content).expect("Failed to parse scenario JSON")
}

/// Preprocess a scenario into filter-ready data structures
pub fn preprocess(scenario: &ScenarioJson) -> PreprocessedScenario {
    let motion = MotionModel::constant_velocity_2d(
        scenario.model.dt,
        scenario.model.process_noise_std,
        scenario.model.survival_probability,
    );

    let obs_vol =
        (scenario.bounds[1] - scenario.bounds[0]) * (scenario.bounds[3] - scenario.bounds[2]);

    let sensor = SensorModel::position_sensor_2d(
        scenario.model.measurement_noise_std,
        scenario.model.detection_probability,
        scenario.model.clutter_rate,
        obs_vol,
    );

    let sensors_config = MultisensorConfig::new(vec![sensor.clone(); scenario.num_sensors]);

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
// Filter Factory
// =============================================================================

/// Enum to hold any filter type for unified handling.
///
/// Uses concrete `LmbFilterCore`/`LmbmFilterCore` types for dynamic dispatch at runtime.
pub enum AnyFilter {
    Lmb(LmbFilterCore<DynamicAssociator, SingleSensorScheduler>),
    Lmbm(LmbmFilterCore<SingleSensorLmbmStrategy<DynamicAssociator>>),
    AaLmb(AaLmbFilter),
    IcLmb(IcLmbFilter),
    PuLmb(PuLmbFilter),
    GaLmb(GaLmbFilter),
    MsLmbm(MultisensorLmbmFilter),
}

impl AnyFilter {
    /// Get config JSON from any filter type
    pub fn get_config_json(&self) -> String {
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

    /// Run benchmark on single-sensor steps, returning (mean_ms, std_ms)
    pub fn run_single_sensor(&mut self, prep: &PreprocessedScenario) -> (f64, f64) {
        use std::time::Instant;
        let mut rng = SimpleRng::new(42);
        let mut step_times: Vec<f64> = Vec::new();

        match self {
            AnyFilter::Lmb(f) => {
                for (t, meas) in prep.steps.iter() {
                    let start = Instant::now();
                    let _ = f.step(&mut rng, meas, *t);
                    step_times.push(start.elapsed().as_micros() as f64 / 1000.0);
                }
            }
            AnyFilter::Lmbm(f) => {
                for (t, meas) in prep.steps.iter() {
                    let start = Instant::now();
                    let _ = f.step(&mut rng, meas, *t);
                    step_times.push(start.elapsed().as_micros() as f64 / 1000.0);
                }
            }
            _ => panic!("run_single_sensor called on multi-sensor filter"),
        }

        compute_stats(&step_times)
    }

    /// Run benchmark on multi-sensor steps, returning (mean_ms, std_ms)
    pub fn run_multi_sensor(&mut self, prep: &PreprocessedScenario) -> (f64, f64) {
        use std::time::Instant;
        let mut rng = SimpleRng::new(42);
        let mut step_times: Vec<f64> = Vec::new();

        match self {
            AnyFilter::AaLmb(f) => {
                for (t, meas) in prep.multi_steps.iter().enumerate() {
                    let start = Instant::now();
                    let _ = f.step(&mut rng, meas, t);
                    step_times.push(start.elapsed().as_micros() as f64 / 1000.0);
                }
            }
            AnyFilter::IcLmb(f) => {
                for (t, meas) in prep.multi_steps.iter().enumerate() {
                    let start = Instant::now();
                    let _ = f.step(&mut rng, meas, t);
                    step_times.push(start.elapsed().as_micros() as f64 / 1000.0);
                }
            }
            AnyFilter::PuLmb(f) => {
                for (t, meas) in prep.multi_steps.iter().enumerate() {
                    let start = Instant::now();
                    let _ = f.step(&mut rng, meas, t);
                    step_times.push(start.elapsed().as_micros() as f64 / 1000.0);
                }
            }
            AnyFilter::GaLmb(f) => {
                for (t, meas) in prep.multi_steps.iter().enumerate() {
                    let start = Instant::now();
                    let _ = f.step(&mut rng, meas, t);
                    step_times.push(start.elapsed().as_micros() as f64 / 1000.0);
                }
            }
            AnyFilter::MsLmbm(f) => {
                for (t, meas) in prep.multi_steps.iter().enumerate() {
                    let start = Instant::now();
                    let _ = f.step(&mut rng, meas, t);
                    step_times.push(start.elapsed().as_micros() as f64 / 1000.0);
                }
            }
            _ => panic!("run_multi_sensor called on single-sensor filter"),
        }

        compute_stats(&step_times)
    }

    /// Run benchmark (auto-detects single vs multi-sensor)
    pub fn run(&mut self, prep: &PreprocessedScenario) -> (f64, f64) {
        match self {
            AnyFilter::Lmb(_) | AnyFilter::Lmbm(_) => self.run_single_sensor(prep),
            _ => self.run_multi_sensor(prep),
        }
    }
}

fn compute_stats(step_times: &[f64]) -> (f64, f64) {
    let n = step_times.len() as f64;
    let mean = step_times.iter().sum::<f64>() / n;
    let variance = step_times.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    (mean, std)
}

/// Create a filter by name
pub fn create_filter(filter_name: &str, prep: &PreprocessedScenario) -> Result<AnyFilter, String> {
    match filter_name {
        // Single-sensor LMB (using LmbFilterCore for dynamic associator)
        "LMB-LBP" => {
            let assoc = AssociationConfig::lbp(100, 1e-6);
            Ok(AnyFilter::Lmb(
                LmbFilterCore::with_scheduler(
                    prep.motion.clone(),
                    prep.sensor.clone().into(),
                    prep.birth.clone(),
                    assoc.clone(),
                    DynamicAssociator::from_config(&assoc),
                    SingleSensorScheduler::new(),
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }
        "LMB-Gibbs" => {
            let assoc = AssociationConfig::gibbs(1000);
            Ok(AnyFilter::Lmb(
                LmbFilterCore::with_scheduler(
                    prep.motion.clone(),
                    prep.sensor.clone().into(),
                    prep.birth.clone(),
                    assoc.clone(),
                    DynamicAssociator::from_config(&assoc),
                    SingleSensorScheduler::new(),
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }
        "LMB-Murty" => {
            let assoc = AssociationConfig::murty(25);
            Ok(AnyFilter::Lmb(
                LmbFilterCore::with_scheduler(
                    prep.motion.clone(),
                    prep.sensor.clone().into(),
                    prep.birth.clone(),
                    assoc.clone(),
                    DynamicAssociator::from_config(&assoc),
                    SingleSensorScheduler::new(),
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }

        // Single-sensor LMBM (using LmbmFilterCore for dynamic associator)
        "LMBM-Gibbs" => {
            let assoc = AssociationConfig::gibbs(1000);
            let strategy = SingleSensorLmbmStrategy::new(DynamicAssociator::from_config(&assoc));
            Ok(AnyFilter::Lmbm(LmbmFilterCore::with_strategy(
                prep.motion.clone(),
                prep.sensor.clone().into(),
                prep.birth.clone(),
                assoc.clone(),
                LMBM_CONFIG,
                strategy,
            )))
        }
        "LMBM-Murty" => {
            let assoc = AssociationConfig::murty(25);
            let strategy = SingleSensorLmbmStrategy::new(DynamicAssociator::from_config(&assoc));
            Ok(AnyFilter::Lmbm(LmbmFilterCore::with_strategy(
                prep.motion.clone(),
                prep.sensor.clone().into(),
                prep.birth.clone(),
                assoc.clone(),
                LMBM_CONFIG,
                strategy,
            )))
        }

        // Multi-sensor LMB variants (using factory functions)
        "AA-LMB-LBP" => {
            let assoc = AssociationConfig::lbp(100, 1e-6);
            Ok(AnyFilter::AaLmb(
                aa_lmb_filter(
                    prep.motion.clone(),
                    prep.sensors_config.clone(),
                    prep.birth.clone(),
                    assoc,
                    MAX_GM_COMPONENTS,
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }
        "IC-LMB-LBP" => {
            let assoc = AssociationConfig::lbp(100, 1e-6);
            Ok(AnyFilter::IcLmb(
                ic_lmb_filter(
                    prep.motion.clone(),
                    prep.sensors_config.clone(),
                    prep.birth.clone(),
                    assoc,
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }
        "PU-LMB-LBP" => {
            let assoc = AssociationConfig::lbp(100, 1e-6);
            Ok(AnyFilter::PuLmb(
                pu_lmb_filter(
                    prep.motion.clone(),
                    prep.sensors_config.clone(),
                    prep.birth.clone(),
                    assoc,
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }
        "GA-LMB-LBP" => {
            let assoc = AssociationConfig::lbp(100, 1e-6);
            Ok(AnyFilter::GaLmb(
                ga_lmb_filter(
                    prep.motion.clone(),
                    prep.sensors_config.clone(),
                    prep.birth.clone(),
                    assoc,
                )
                .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                .with_gm_merge_threshold(GM_MERGE_THRESHOLD),
            ))
        }

        // Multi-sensor LMBM (using factory functions)
        "MS-LMBM-Gibbs" => {
            let assoc = AssociationConfig::gibbs(1000);
            Ok(AnyFilter::MsLmbm(multisensor_lmbm_filter(
                prep.motion.clone(),
                prep.sensors_config.clone(),
                prep.birth.clone(),
                assoc,
                LMBM_CONFIG,
            )))
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
// Scenario Configurations for Benchmarks
// =============================================================================

/// Configuration for a benchmark scenario
#[derive(Clone, Copy)]
pub struct ScenarioConfig {
    pub name: &'static str,
    pub path: &'static str,
}

/// Single-sensor scenarios for benchmarking
pub const SINGLE_SENSOR_SCENARIOS: &[ScenarioConfig] = &[
    ScenarioConfig {
        name: "n5_s1",
        path: "tests/fixtures/scenario_n5_s1.json",
    },
    ScenarioConfig {
        name: "n10_s1",
        path: "tests/fixtures/scenario_n10_s1.json",
    },
    ScenarioConfig {
        name: "n20_s1",
        path: "tests/fixtures/scenario_n20_s1.json",
    },
];

/// Multi-sensor scenarios for benchmarking
pub const MULTI_SENSOR_SCENARIOS: &[ScenarioConfig] = &[
    ScenarioConfig {
        name: "n5_s2",
        path: "tests/fixtures/scenario_n5_s2.json",
    },
    ScenarioConfig {
        name: "n10_s2",
        path: "tests/fixtures/scenario_n10_s2.json",
    },
    ScenarioConfig {
        name: "n10_s4",
        path: "tests/fixtures/scenario_n10_s4.json",
    },
    ScenarioConfig {
        name: "n20_s2",
        path: "tests/fixtures/scenario_n20_s2.json",
    },
    ScenarioConfig {
        name: "n20_s4",
        path: "tests/fixtures/scenario_n20_s4.json",
    },
    ScenarioConfig {
        name: "n20_s8",
        path: "tests/fixtures/scenario_n20_s8.json",
    },
    ScenarioConfig {
        name: "n50_s8",
        path: "tests/fixtures/scenario_n50_s8.json",
    },
];
