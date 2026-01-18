//! Data-driven fixture tests - ALL configuration comes from fixture files.
//!
//! Run with: `cargo test --test benchmark_fixture_equivalence`
//!
//! Fixtures are fully self-describing: model params, filter config, expected outputs.

use multisensor_lmb_filters_rs::lmb::config::{
    AssociationConfig, BirthLocation, BirthModel, MotionModel, SensorConfig, SensorModel,
};
use multisensor_lmb_filters_rs::lmb::strategy::{
    AaLmbStrategyLbp, CommonPruneConfig, GaLmbStrategyLbp, IcLmbStrategyLbp, LmbPruneConfig,
    LmbStrategyLbp, LmbmPruneConfig, LmbmStrategyGibbs, MultisensorLmbmStrategyGibbs,
    PuLmbStrategyLbp,
};
use multisensor_lmb_filters_rs::lmb::traits::Filter;
use multisensor_lmb_filters_rs::lmb::unified::UnifiedFilter;

use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use std::fs;
use std::path::Path;

const MEAN_TOLERANCE: f64 = 5.0;
const TRACK_COUNT_TOLERANCE: i32 = 1;

// ============================================================================
// Fixture Schema (matches MATLAB output exactly)
// ============================================================================

#[derive(Deserialize)]
struct Fixture {
    scenario_file: String,
    num_sensors: usize,
    num_steps: usize,
    #[allow(dead_code)]
    seed: u64,
    filter: FilterConfig,
    model: ModelConfig,
    thresholds: ThresholdsConfig,
    steps: Vec<FixtureStep>,
}

#[derive(Deserialize)]
struct FilterConfig {
    name: String,
    #[serde(rename = "type")]
    filter_type: String,
    associator: AssociatorFixtureConfig,
    #[allow(dead_code)]
    update_mode: String,
}

#[derive(Deserialize)]
struct AssociatorFixtureConfig {
    #[serde(rename = "type")]
    assoc_type: String,
    params: AssociatorParams,
}

#[derive(Deserialize)]
struct AssociatorParams {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    num_samples: Option<usize>,
    num_assignments: Option<usize>,
}

#[derive(Deserialize)]
struct ModelConfig {
    dt: f64,
    process_noise_std: f64,
    measurement_noise_std: f64,
    detection_probability: f64,
    survival_probability: f64,
    clutter_rate: f64,
    bounds: Vec<f64>,
    birth_locations: Vec<Vec<f64>>,
    birth_existence: f64,
    birth_covariance: Vec<f64>,
}

#[derive(Deserialize)]
struct ThresholdsConfig {
    existence: f64,
    gm_weight: f64,
    max_components: usize,
    #[allow(dead_code)]
    gm_merge: serde_json::Value, // Can be number or "Inf"
}

#[derive(Deserialize)]
struct FixtureStep {
    #[allow(dead_code)]
    step: usize,
    num_tracks: usize,
    #[serde(default)]
    tracks: Vec<FixtureTrack>,
}

#[derive(Deserialize)]
struct FixtureTrack {
    #[allow(dead_code)]
    label: i64,
    mean: Vec<f64>,
}

// ============================================================================
// Scenario Schema
// ============================================================================

#[derive(Deserialize)]
struct Scenario {
    #[allow(dead_code)]
    num_sensors: usize,
    #[allow(dead_code)]
    num_steps: usize,
    steps: Vec<ScenarioStep>,
}

#[derive(Deserialize)]
struct ScenarioStep {
    sensor_readings: Vec<Vec<Vec<f64>>>,
}

// ============================================================================
// Filter Builder (entirely from fixture config)
// ============================================================================

enum AnyFilter {
    Lmb(UnifiedFilter<LmbStrategyLbp>),
    Lmbm(UnifiedFilter<LmbmStrategyGibbs>),
    AaLmb(UnifiedFilter<AaLmbStrategyLbp>),
    GaLmb(UnifiedFilter<GaLmbStrategyLbp>),
    PuLmb(UnifiedFilter<PuLmbStrategyLbp>),
    IcLmb(UnifiedFilter<IcLmbStrategyLbp>),
    MsLmbm(UnifiedFilter<MultisensorLmbmStrategyGibbs>),
}

fn build_filter(fixture: &Fixture) -> (AnyFilter, bool) {
    let m = &fixture.model;
    let t = &fixture.thresholds;
    let f = &fixture.filter;

    // Motion model from fixture
    let motion =
        MotionModel::constant_velocity_2d(m.dt, m.process_noise_std, m.survival_probability);

    // Sensor model from fixture
    let obs_vol = (m.bounds[1] - m.bounds[0]) * (m.bounds[3] - m.bounds[2]);
    let sensor = SensorModel::position_sensor_2d(
        m.measurement_noise_std,
        m.detection_probability,
        m.clutter_rate,
        obs_vol,
    );

    // Birth model from fixture
    // State ordering: [x, y, vx, vy] (matches MATLAB)
    let birth_locs: Vec<BirthLocation> = m
        .birth_locations
        .iter()
        .enumerate()
        .map(|(i, loc)| {
            BirthLocation::new(
                i,
                DVector::from_vec(loc.clone()),
                DMatrix::from_diagonal(&DVector::from_vec(m.birth_covariance.clone())),
            )
        })
        .collect();
    let birth = BirthModel::new(birth_locs, m.birth_existence, 0.001);

    // Associator from fixture
    let assoc = match f.associator.assoc_type.as_str() {
        "LBP" => AssociationConfig::lbp(
            f.associator.params.max_iterations.unwrap_or(100),
            f.associator.params.tolerance.unwrap_or(1e-6),
        ),
        "Gibbs" => AssociationConfig::gibbs(f.associator.params.num_samples.unwrap_or(1000)),
        "Murty" => AssociationConfig::murty(f.associator.params.num_assignments.unwrap_or(25)),
        other => panic!("Unknown associator: {}", other),
    };

    // Common prune config
    let common_prune = CommonPruneConfig {
        existence_threshold: t.existence,
        min_trajectory_length: 3,
    };

    // LMB-specific prune config
    let lmb_prune = LmbPruneConfig {
        gm_weight_threshold: t.gm_weight,
        max_gm_components: t.max_components,
        gm_merge_threshold: f64::INFINITY,
    };

    // LMBM-specific prune config
    let lmbm_prune = LmbmPruneConfig {
        max_hypotheses: 25,
        hypothesis_weight_threshold: 1e-3,
        use_eap: false,
    };

    // Build filter based on type
    match f.filter_type.as_str() {
        "LMB" => {
            let filter = multisensor_lmb_filters_rs::lmb::lmb_filter(
                motion,
                sensor,
                birth,
                assoc,
                common_prune,
                lmb_prune,
            );
            (AnyFilter::Lmb(filter), false)
        }
        "LMBM" => {
            let filter = multisensor_lmb_filters_rs::lmb::lmbm_filter(
                motion,
                sensor,
                birth,
                assoc,
                common_prune,
                lmbm_prune,
            );
            (AnyFilter::Lmbm(filter), false)
        }
        "AA-LMB" => {
            let sensors = SensorConfig::new(vec![sensor.clone(); fixture.num_sensors]);
            let filter = multisensor_lmb_filters_rs::lmb::aa_lmb_filter(
                motion,
                sensors,
                birth,
                assoc,
                common_prune,
                lmb_prune,
                t.max_components,
            );
            (AnyFilter::AaLmb(filter), true)
        }
        "GA-LMB" => {
            let sensors = SensorConfig::new(vec![sensor.clone(); fixture.num_sensors]);
            let filter = multisensor_lmb_filters_rs::lmb::ga_lmb_filter(
                motion,
                sensors,
                birth,
                assoc,
                common_prune,
                lmb_prune,
            );
            (AnyFilter::GaLmb(filter), true)
        }
        "PU-LMB" => {
            let sensors = SensorConfig::new(vec![sensor.clone(); fixture.num_sensors]);
            let filter = multisensor_lmb_filters_rs::lmb::pu_lmb_filter(
                motion,
                sensors,
                birth,
                assoc,
                common_prune,
                lmb_prune,
            );
            (AnyFilter::PuLmb(filter), true)
        }
        "IC-LMB" => {
            let sensors = SensorConfig::new(vec![sensor.clone(); fixture.num_sensors]);
            let filter = multisensor_lmb_filters_rs::lmb::ic_lmb_filter(
                motion,
                sensors,
                birth,
                assoc,
                common_prune,
                lmb_prune,
            );
            (AnyFilter::IcLmb(filter), true)
        }
        "MS-LMBM" => {
            let sensors = SensorConfig::new(vec![sensor.clone(); fixture.num_sensors]);
            let filter = multisensor_lmb_filters_rs::lmb::multisensor_lmbm_filter(
                motion,
                sensors,
                birth,
                assoc,
                common_prune,
                lmbm_prune,
            );
            (AnyFilter::MsLmbm(filter), true)
        }
        other => panic!("Unknown filter type: {}", other),
    }
}

// ============================================================================
// Test Runner
// ============================================================================

struct StepResult {
    num_tracks: usize,
    means: Vec<Vec<f64>>,
}

fn convert_meas(readings: &[Vec<f64>]) -> Vec<DVector<f64>> {
    readings
        .iter()
        .map(|m| DVector::from_vec(m.clone()))
        .collect()
}

fn run_and_compare(fixture: &Fixture, scenario: &Scenario) -> Vec<String> {
    let (filter, _is_multi) = build_filter(fixture);
    let mut rng = rand::thread_rng();
    let mut results: Vec<StepResult> = Vec::new();

    // Run filter
    match filter {
        AnyFilter::Lmb(mut f) => {
            for t in 0..fixture.num_steps {
                let meas = convert_meas(&scenario.steps[t].sensor_readings[0]);
                let est = f.step(&mut rng, &meas, t).unwrap();
                results.push(StepResult {
                    num_tracks: est.tracks.len(),
                    means: est
                        .tracks
                        .iter()
                        .map(|tr| tr.mean.as_slice().to_vec())
                        .collect(),
                });
            }
        }
        AnyFilter::Lmbm(mut f) => {
            for t in 0..fixture.num_steps {
                let meas = convert_meas(&scenario.steps[t].sensor_readings[0]);
                let est = f.step(&mut rng, &meas, t).unwrap();
                results.push(StepResult {
                    num_tracks: est.tracks.len(),
                    means: est
                        .tracks
                        .iter()
                        .map(|tr| tr.mean.as_slice().to_vec())
                        .collect(),
                });
            }
        }
        AnyFilter::AaLmb(mut f) => {
            run_multi(&mut f, &mut rng, scenario, fixture.num_steps, &mut results)
        }
        AnyFilter::GaLmb(mut f) => {
            run_multi(&mut f, &mut rng, scenario, fixture.num_steps, &mut results)
        }
        AnyFilter::PuLmb(mut f) => {
            run_multi(&mut f, &mut rng, scenario, fixture.num_steps, &mut results)
        }
        AnyFilter::IcLmb(mut f) => {
            run_multi(&mut f, &mut rng, scenario, fixture.num_steps, &mut results)
        }
        AnyFilter::MsLmbm(mut f) => {
            for t in 0..fixture.num_steps {
                let meas: Vec<Vec<DVector<f64>>> = scenario.steps[t]
                    .sensor_readings
                    .iter()
                    .map(|s| convert_meas(s))
                    .collect();
                let est = f.step(&mut rng, &meas, t).unwrap();
                results.push(StepResult {
                    num_tracks: est.tracks.len(),
                    means: est
                        .tracks
                        .iter()
                        .map(|tr| tr.mean.as_slice().to_vec())
                        .collect(),
                });
            }
        }
    }

    // Compare results
    compare(&results, fixture)
}

fn run_multi<F>(
    filter: &mut F,
    rng: &mut impl rand::Rng,
    scenario: &Scenario,
    num_steps: usize,
    results: &mut Vec<StepResult>,
) where
    F: Filter<Measurements = Vec<Vec<DVector<f64>>>>,
{
    for t in 0..num_steps {
        let meas: Vec<Vec<DVector<f64>>> = scenario.steps[t]
            .sensor_readings
            .iter()
            .map(|s| convert_meas(s))
            .collect();
        let est = filter.step(rng, &meas, t).unwrap();
        results.push(StepResult {
            num_tracks: est.tracks.len(),
            means: est
                .tracks
                .iter()
                .map(|tr| tr.mean.as_slice().to_vec())
                .collect(),
        });
    }
}

fn compare(results: &[StepResult], fixture: &Fixture) -> Vec<String> {
    let mut errors = Vec::new();

    for t in 0..results.len().min(fixture.num_steps) {
        let rust = &results[t];
        let matlab = &fixture.steps[t];

        // Track count
        let diff = (rust.num_tracks as i32 - matlab.num_tracks as i32).abs();
        if diff > TRACK_COUNT_TOLERANCE {
            errors.push(format!(
                "Step {}: count Rust={} MATLAB={}",
                t, rust.num_tracks, matlab.num_tracks
            ));
        }

        // Means
        if rust.num_tracks > 0 && matlab.num_tracks > 0 {
            let mut matlab_means: Vec<_> = matlab.tracks.iter().map(|t| t.mean.clone()).collect();

            for r_mean in &rust.means {
                if matlab_means.is_empty() {
                    break;
                }
                let mut min_dist = f64::INFINITY;
                let mut min_idx = 0;
                for (i, m_mean) in matlab_means.iter().enumerate() {
                    let dist: f64 = r_mean
                        .iter()
                        .zip(m_mean.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = i;
                    }
                }
                if min_dist > MEAN_TOLERANCE {
                    errors.push(format!(
                        "Step {}: unmatched Rust track at [{:.1}, {:.1}]",
                        t, r_mean[0], r_mean[1]
                    ));
                }
                matlab_means.remove(min_idx);
            }
        }
    }

    errors
}

// ============================================================================
// Test Entry Point
// ============================================================================

fn discover_fixtures() -> Vec<String> {
    let dir = Path::new("tests/fixtures");
    if !dir.exists() {
        return Vec::new();
    }
    fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            path.extension().map_or(false, |x| x == "json")
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map_or(false, |n| n.starts_with("bench_"))
        })
        .map(|e| e.path().to_string_lossy().to_string())
        .collect()
}

#[test]
fn test_all_fixtures() {
    let fixtures = discover_fixtures();
    if fixtures.is_empty() {
        eprintln!("No fixtures found - run MATLAB generator first");
        return;
    }

    let mut failures = Vec::new();

    for fixture_path in &fixtures {
        let fixture_data = fs::read_to_string(fixture_path).unwrap();
        let fixture: Fixture = serde_json::from_str(&fixture_data).unwrap();

        let scenario_path = format!("tests/fixtures/{}", fixture.scenario_file);
        let scenario_data = match fs::read_to_string(&scenario_path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping {}: scenario not found", fixture.filter.name);
                continue;
            }
        };
        let scenario: Scenario = serde_json::from_str(&scenario_data).unwrap();

        let errors = run_and_compare(&fixture, &scenario);
        if !errors.is_empty() {
            failures.push((fixture.filter.name.clone(), errors));
        }
    }

    if !failures.is_empty() {
        let mut msg = String::from("Fixture test failures:\n");
        for (name, errors) in &failures {
            msg.push_str(&format!("\n{}:\n", name));
            for e in errors.iter().take(5) {
                msg.push_str(&format!("  {}\n", e));
            }
        }
        panic!("{}", msg);
    }
}
