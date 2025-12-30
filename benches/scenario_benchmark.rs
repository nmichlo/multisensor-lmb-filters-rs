//! Scenario benchmarks for LMB filters.
//! Run with: cargo bench --bench scenario_benchmark
//!
//! Focus: Timing performance only. For accuracy evaluation, use separate tools.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::Duration;

use multisensor_lmb_filters_rs::lmb::{
    config::{
        AssociationConfig, BirthLocation, BirthModel, LmbmConfig, MotionModel, MultisensorConfig,
        SensorModel,
    },
    traits::{Associator, Filter, GibbsAssociator, Merger, MurtyAssociator},
    ArithmeticAverageMerger, GeometricAverageMerger, IteratedCorrectorMerger, LbpAssociator,
    LmbFilter, LmbmFilter, MultisensorGibbsAssociator, MultisensorLmbFilter, MultisensorLmbmFilter,
    ParallelUpdateMerger, SimpleRng,
};

// =============================================================================
// JSON TYPES
// =============================================================================

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
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

#[derive(Debug, Deserialize, Clone)]
struct ModelConfig {
    dt: f64,
    process_noise_std: f64,
    measurement_noise_std: f64,
    detection_probability: f64,
    survival_probability: f64,
    clutter_rate: f64,
    birth_locations: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize, Clone)]
struct Step {
    step: usize,
    sensor_readings: Vec<Vec<Vec<f64>>>,
}

// =============================================================================
// PREPROCESSED DATA
// =============================================================================

struct PreprocessedScenario {
    motion: MotionModel,
    sensor: SensorModel,
    multi_sensor: MultisensorConfig,
    birth: BirthModel,
    steps: Vec<(usize, Vec<DVector<f64>>, Vec<Vec<DVector<f64>>>)>,
}

fn preprocess(scenario: &Scenario) -> PreprocessedScenario {
    let m = &scenario.model;
    let bounds = &scenario.bounds;

    let motion =
        MotionModel::constant_velocity_2d(m.dt, m.process_noise_std, m.survival_probability);
    let obs_vol = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]);
    let sensor = SensorModel::position_sensor_2d(
        m.measurement_noise_std,
        m.detection_probability,
        m.clutter_rate,
        obs_vol,
    );

    let locations: Vec<_> = m
        .birth_locations
        .iter()
        .enumerate()
        .map(|(i, loc)| {
            BirthLocation::new(
                i,
                DVector::from_vec(loc.clone()),
                DMatrix::identity(4, 4) * 100.0,
            )
        })
        .collect();
    let birth = BirthModel::new(locations, 0.01, 0.001);

    let multi_sensor = MultisensorConfig::new(vec![sensor.clone(); scenario.num_sensors]);

    let steps: Vec<_> = scenario
        .steps
        .iter()
        .map(|step| {
            let multi_meas: Vec<Vec<DVector<f64>>> = step
                .sensor_readings
                .iter()
                .map(|readings| {
                    readings
                        .iter()
                        .map(|r| DVector::from_vec(vec![r[0], r[1]]))
                        .collect()
                })
                .collect();
            let single_meas = multi_meas.first().cloned().unwrap_or_default();
            (step.step, single_meas, multi_meas)
        })
        .collect();

    PreprocessedScenario {
        motion,
        sensor,
        multi_sensor,
        birth,
        steps,
    }
}

fn load_scenarios() -> Vec<(String, Scenario)> {
    let dir = Path::new("benchmarks/scenarios");
    if !dir.exists() {
        return vec![];
    }

    let mut scenarios: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |x| x == "json"))
        .filter_map(|e| {
            let name = e.path().file_stem()?.to_string_lossy().to_string();
            let data = fs::read_to_string(e.path()).ok()?;
            let scenario: Scenario = serde_json::from_str(&data).ok()?;
            Some((name, scenario))
        })
        .collect();
    scenarios.sort_by(|a, b| a.0.cmp(&b.0));
    scenarios
}

// =============================================================================
// UNIFIED FILTER RUNNERS
// =============================================================================

fn run_lmb<A: Associator>(pre: &PreprocessedScenario, mut filter: LmbFilter<A>) {
    let mut rng = SimpleRng::new(42);
    for (t, meas, _) in &pre.steps {
        let _ = filter.step(&mut rng, meas, *t);
    }
}

fn run_lmbm<A: Associator>(pre: &PreprocessedScenario, mut filter: LmbmFilter<A>) {
    let mut rng = SimpleRng::new(42);
    for (t, meas, _) in &pre.steps {
        let _ = filter.step(&mut rng, meas, *t);
    }
}

fn run_multi<A: Associator, M: Merger>(
    pre: &PreprocessedScenario,
    mut filter: MultisensorLmbFilter<A, M>,
) {
    let mut rng = SimpleRng::new(42);
    for (t, _, meas) in &pre.steps {
        let _ = filter.step(&mut rng, meas, *t);
    }
}

fn run_multi_lmbm(
    pre: &PreprocessedScenario,
    mut filter: MultisensorLmbmFilter<MultisensorGibbsAssociator>,
) {
    let mut rng = SimpleRng::new(42);
    for (t, _, meas) in &pre.steps {
        let _ = filter.step(&mut rng, meas, *t);
    }
}

// =============================================================================
// BENCHMARK MACROS (reduces duplication)
// =============================================================================

macro_rules! bench_lmb {
    ($group:expr, $name:expr, $pre:expr, $assoc_name:expr, $assoc:expr, $config:expr) => {
        $group.bench_with_input(
            BenchmarkId::new(concat!("LMB-", $assoc_name), $name),
            $pre,
            |b, p| {
                b.iter(|| {
                    let filter = LmbFilter::with_associator_type(
                        p.motion.clone(),
                        p.sensor.clone(),
                        p.birth.clone(),
                        $config,
                        $assoc,
                    );
                    run_lmb(p, filter)
                });
            },
        );
    };
}

macro_rules! bench_lmbm {
    ($group:expr, $name:expr, $pre:expr, $assoc_name:expr, $assoc:expr, $config:expr, $lmbm_config:expr) => {
        $group.bench_with_input(
            BenchmarkId::new(concat!("LMBM-", $assoc_name), $name),
            $pre,
            |b, p| {
                b.iter(|| {
                    let filter = LmbmFilter::with_associator_type(
                        p.motion.clone(),
                        p.sensor.clone(),
                        p.birth.clone(),
                        $config,
                        $lmbm_config.clone(),
                        $assoc,
                    );
                    run_lmbm(p, filter)
                });
            },
        );
    };
}

macro_rules! bench_multi {
    ($group:expr, $name:expr, $pre:expr, $filter_name:expr, $assoc_name:expr, $merger:expr, $assoc:expr, $config:expr) => {
        $group.bench_with_input(
            BenchmarkId::new(concat!($filter_name, "-", $assoc_name), $name),
            $pre,
            |b, p| {
                b.iter(|| {
                    let filter = MultisensorLmbFilter::with_associator_type(
                        p.motion.clone(),
                        p.multi_sensor.clone(),
                        p.birth.clone(),
                        $config,
                        $merger,
                        $assoc,
                    );
                    run_multi(p, filter)
                });
            },
        );
    };
}

// =============================================================================
// MAIN BENCHMARK
// =============================================================================

fn bench_all_filters(c: &mut Criterion) {
    let scenarios = load_scenarios();

    let mut group = c.benchmark_group("lmb_filters");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for (name, scenario) in &scenarios {
        let pre = preprocess(scenario);
        let n = scenario.num_sensors;
        let lmbm_cfg = LmbmConfig::default();

        // LMB × {LBP, Gibbs, Murty}
        bench_lmb!(
            group,
            name,
            &pre,
            "LBP",
            LbpAssociator,
            AssociationConfig::lbp(100, 1e-6)
        );
        bench_lmb!(
            group,
            name,
            &pre,
            "Gibbs",
            GibbsAssociator,
            AssociationConfig::gibbs(1000)
        );
        bench_lmb!(
            group,
            name,
            &pre,
            "Murty",
            MurtyAssociator,
            AssociationConfig::murty(25)
        );

        // LMBM × {Gibbs, Murty}
        bench_lmbm!(
            group,
            name,
            &pre,
            "Gibbs",
            GibbsAssociator,
            AssociationConfig::gibbs(1000),
            lmbm_cfg
        );
        bench_lmbm!(
            group,
            name,
            &pre,
            "Murty",
            MurtyAssociator,
            AssociationConfig::murty(25),
            lmbm_cfg
        );

        // AA-LMB × {LBP, Gibbs, Murty}
        bench_multi!(
            group,
            name,
            &pre,
            "AA-LMB",
            "LBP",
            ArithmeticAverageMerger::uniform(n, 100),
            LbpAssociator,
            AssociationConfig::lbp(100, 1e-6)
        );
        bench_multi!(
            group,
            name,
            &pre,
            "AA-LMB",
            "Gibbs",
            ArithmeticAverageMerger::uniform(n, 100),
            GibbsAssociator,
            AssociationConfig::gibbs(1000)
        );
        bench_multi!(
            group,
            name,
            &pre,
            "AA-LMB",
            "Murty",
            ArithmeticAverageMerger::uniform(n, 100),
            MurtyAssociator,
            AssociationConfig::murty(25)
        );

        // GA-LMB × {LBP, Gibbs, Murty}
        bench_multi!(
            group,
            name,
            &pre,
            "GA-LMB",
            "LBP",
            GeometricAverageMerger::uniform(n),
            LbpAssociator,
            AssociationConfig::lbp(100, 1e-6)
        );
        bench_multi!(
            group,
            name,
            &pre,
            "GA-LMB",
            "Gibbs",
            GeometricAverageMerger::uniform(n),
            GibbsAssociator,
            AssociationConfig::gibbs(1000)
        );
        bench_multi!(
            group,
            name,
            &pre,
            "GA-LMB",
            "Murty",
            GeometricAverageMerger::uniform(n),
            MurtyAssociator,
            AssociationConfig::murty(25)
        );

        // PU-LMB × {LBP, Gibbs, Murty}
        bench_multi!(
            group,
            name,
            &pre,
            "PU-LMB",
            "LBP",
            ParallelUpdateMerger::new(Vec::new()),
            LbpAssociator,
            AssociationConfig::lbp(100, 1e-6)
        );
        bench_multi!(
            group,
            name,
            &pre,
            "PU-LMB",
            "Gibbs",
            ParallelUpdateMerger::new(Vec::new()),
            GibbsAssociator,
            AssociationConfig::gibbs(1000)
        );
        bench_multi!(
            group,
            name,
            &pre,
            "PU-LMB",
            "Murty",
            ParallelUpdateMerger::new(Vec::new()),
            MurtyAssociator,
            AssociationConfig::murty(25)
        );

        // IC-LMB × {LBP, Gibbs, Murty}
        bench_multi!(
            group,
            name,
            &pre,
            "IC-LMB",
            "LBP",
            IteratedCorrectorMerger::new(),
            LbpAssociator,
            AssociationConfig::lbp(100, 1e-6)
        );
        bench_multi!(
            group,
            name,
            &pre,
            "IC-LMB",
            "Gibbs",
            IteratedCorrectorMerger::new(),
            GibbsAssociator,
            AssociationConfig::gibbs(1000)
        );
        bench_multi!(
            group,
            name,
            &pre,
            "IC-LMB",
            "Murty",
            IteratedCorrectorMerger::new(),
            MurtyAssociator,
            AssociationConfig::murty(25)
        );

        // MS-LMBM × {Gibbs} (only Gibbs supported)
        group.bench_with_input(BenchmarkId::new("MS-LMBM-Gibbs", name), &pre, |b, p| {
            b.iter(|| {
                let filter = MultisensorLmbmFilter::with_associator_type(
                    p.motion.clone(),
                    p.multi_sensor.clone(),
                    p.birth.clone(),
                    AssociationConfig::gibbs(1000),
                    lmbm_cfg.clone(),
                    MultisensorGibbsAssociator,
                );
                run_multi_lmbm(p, filter)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_all_filters);
criterion_main!(benches);
