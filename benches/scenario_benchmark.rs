//! Scenario-based benchmarks for LMB filters
//!
//! Loads pre-generated JSON scenarios and benchmarks filter performance.
//! Run with: cargo bench --bench scenario_benchmark

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use std::fs;
use std::path::Path;

use multisensor_lmb_filters_rs::lmb::{
    config::{AssociationConfig, BirthLocation, BirthModel, MotionModel, SensorModel},
    traits::{Filter, GibbsAssociator},
    LmbFilter, SimpleRng,
};

// =============================================================================
// JSON SCENARIO TYPES
// =============================================================================

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Scenario {
    #[serde(rename = "type")]
    scenario_type: String,
    measurement_format: String,
    seed: u64,
    num_objects: usize,
    num_sensors: usize,
    num_steps: usize,
    bounds: [f64; 4],
    init_velocity_std: f64,
    model: ModelConfig,
    steps: Vec<Step>,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    dt: f64,
    process_noise_std: f64,
    measurement_noise_std: f64,
    detection_probability: f64,
    survival_probability: f64,
    clutter_rate: f64,
    birth_locations: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct Step {
    step: usize,
    sensor_readings: Vec<Vec<Vec<f64>>>,
}

// =============================================================================
// SCENARIO LOADING
// =============================================================================

fn load_scenarios() -> Vec<(String, Scenario)> {
    let scenario_dir = Path::new("benchmarks/scenarios");
    if !scenario_dir.exists() {
        eprintln!("Warning: benchmarks/scenarios not found. Run generate_scenarios.py first.");
        return vec![];
    }

    let mut scenarios = Vec::new();
    for entry in fs::read_dir(scenario_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "json") {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            let data = fs::read_to_string(&path).unwrap();
            match serde_json::from_str::<Scenario>(&data) {
                Ok(scenario) => scenarios.push((name, scenario)),
                Err(e) => eprintln!("Warning: Failed to parse {}: {}", path.display(), e),
            }
        }
    }
    scenarios.sort_by(|a, b| a.0.cmp(&b.0));
    scenarios
}

fn extract_measurements(readings: &[Vec<f64>]) -> Vec<DVector<f64>> {
    // Extract [x, y] from [x, y, id] format
    readings
        .iter()
        .map(|r| DVector::from_vec(vec![r[0], r[1]]))
        .collect()
}

fn build_motion_model(config: &ModelConfig) -> MotionModel {
    MotionModel::constant_velocity_2d(
        config.dt,
        config.process_noise_std,
        config.survival_probability,
    )
}

fn build_sensor_model(config: &ModelConfig, bounds: &[f64; 4]) -> SensorModel {
    let obs_volume = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]);
    SensorModel::position_sensor_2d(
        config.measurement_noise_std,
        config.detection_probability,
        config.clutter_rate,
        obs_volume,
    )
}

fn build_birth_model(config: &ModelConfig) -> BirthModel {
    let locations: Vec<BirthLocation> = config
        .birth_locations
        .iter()
        .enumerate()
        .map(|(idx, loc)| {
            let mean = DVector::from_vec(loc.clone());
            let cov = DMatrix::identity(4, 4) * 100.0;
            BirthLocation::new(idx, mean, cov)
        })
        .collect();
    BirthModel::new(locations, 0.01, 0.001)
}

// =============================================================================
// BENCHMARKS
// =============================================================================

fn bench_lmb_lbp(c: &mut Criterion) {
    let scenarios = load_scenarios();
    if scenarios.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("lmb_lbp");
    group.sample_size(10);

    for (name, scenario) in &scenarios {
        // Only benchmark single-sensor scenarios
        if scenario.num_sensors > 1 {
            continue;
        }

        let motion = build_motion_model(&scenario.model);
        let sensor = build_sensor_model(&scenario.model, &scenario.bounds);
        let birth = build_birth_model(&scenario.model);
        let assoc = AssociationConfig::lbp(100, 1e-6);

        group.bench_with_input(BenchmarkId::new("run", name), &scenario, |b, scenario| {
            b.iter(|| {
                let mut rng = SimpleRng::new(42);
                let mut filter =
                    LmbFilter::new(motion.clone(), sensor.clone(), birth.clone(), assoc.clone());

                for step in &scenario.steps {
                    let meas = extract_measurements(&step.sensor_readings[0]);
                    let _ = filter.step(&mut rng, &meas, step.step);
                }
            });
        });
    }

    group.finish();
}

fn bench_lmb_gibbs(c: &mut Criterion) {
    let scenarios = load_scenarios();
    if scenarios.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("lmb_gibbs");
    group.sample_size(10);

    for (name, scenario) in &scenarios {
        if scenario.num_sensors > 1 {
            continue;
        }
        // Skip large scenarios for Gibbs (slower)
        if scenario.num_objects > 20 {
            continue;
        }

        let motion = build_motion_model(&scenario.model);
        let sensor = build_sensor_model(&scenario.model, &scenario.bounds);
        let birth = build_birth_model(&scenario.model);
        let assoc = AssociationConfig::gibbs(1000);

        group.bench_with_input(BenchmarkId::new("run", name), &scenario, |b, scenario| {
            b.iter(|| {
                let mut rng = SimpleRng::new(42);
                let mut filter = LmbFilter::with_associator_type(
                    motion.clone(),
                    sensor.clone(),
                    birth.clone(),
                    assoc.clone(),
                    GibbsAssociator,
                );

                for step in &scenario.steps {
                    let meas = extract_measurements(&step.sensor_readings[0]);
                    let _ = filter.step(&mut rng, &meas, step.step);
                }
            });
        });
    }

    group.finish();
}

// =============================================================================
// CRITERION SETUP
// =============================================================================

criterion_group!(
    name = scenario_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_lmb_lbp, bench_lmb_gibbs
);

criterion_main!(scenario_benches);
