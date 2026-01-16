//! Criterion benchmarks for LMB filter algorithms.
//!
//! Run with: cargo bench
//! Run specific group: cargo bench -- single_sensor
//! Run specific algorithm: cargo bench -- lmb_lbp

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::time::Duration;

use multisensor_lmb_filters_rs::bench_utils::{
    create_filter, load_scenario, preprocess, AnyFilter, PreprocessedScenario,
    MULTI_SENSOR_SCENARIOS, SINGLE_SENSOR_SCENARIOS,
};
use multisensor_lmb_filters_rs::lmb::{Filter, SimpleRng};

// =============================================================================
// Helper: Run all steps for a filter
// =============================================================================

fn run_single_sensor_steps(filter: &mut AnyFilter, prep: &PreprocessedScenario) {
    let mut rng = SimpleRng::new(42);
    match filter {
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
        _ => panic!("Expected single-sensor filter"),
    }
}

fn run_multi_sensor_steps(filter: &mut AnyFilter, prep: &PreprocessedScenario) {
    let mut rng = SimpleRng::new(42);
    match filter {
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
        _ => panic!("Expected multi-sensor filter"),
    }
}

// =============================================================================
// Single-Sensor LMB Benchmarks
// =============================================================================

fn bench_lmb_lbp(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_sensor/lmb_lbp");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for scenario_cfg in SINGLE_SENSOR_SCENARIOS {
        let scenario = load_scenario(scenario_cfg.path);
        let prep = preprocess(&scenario);

        group.bench_function(BenchmarkId::new("step", scenario_cfg.name), |b| {
            b.iter_batched(
                || create_filter("LMB-LBP", &prep).unwrap(),
                |mut filter| run_single_sensor_steps(&mut filter, &prep),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_lmb_gibbs(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_sensor/lmb_gibbs");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for scenario_cfg in SINGLE_SENSOR_SCENARIOS {
        let scenario = load_scenario(scenario_cfg.path);
        let prep = preprocess(&scenario);

        group.bench_function(BenchmarkId::new("step", scenario_cfg.name), |b| {
            b.iter_batched(
                || create_filter("LMB-Gibbs", &prep).unwrap(),
                |mut filter| run_single_sensor_steps(&mut filter, &prep),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_lmb_murty(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_sensor/lmb_murty");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    // Murty is slow - limit to smallest scenario
    let scenario_cfg = &SINGLE_SENSOR_SCENARIOS[0]; // n5_s1 only
    let scenario = load_scenario(scenario_cfg.path);
    let prep = preprocess(&scenario);

    group.bench_function(BenchmarkId::new("step", scenario_cfg.name), |b| {
        b.iter_batched(
            || create_filter("LMB-Murty", &prep).unwrap(),
            |mut filter| run_single_sensor_steps(&mut filter, &prep),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// =============================================================================
// Single-Sensor LMBM Benchmarks
// =============================================================================

fn bench_lmbm_gibbs(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_sensor/lmbm_gibbs");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    // LMBM is slow - only benchmark smallest scenario
    let scenario_cfg = &SINGLE_SENSOR_SCENARIOS[0]; // n5_s1
    let scenario = load_scenario(scenario_cfg.path);
    let prep = preprocess(&scenario);

    group.bench_function(BenchmarkId::new("step", scenario_cfg.name), |b| {
        b.iter_batched(
            || create_filter("LMBM-Gibbs", &prep).unwrap(),
            |mut filter| run_single_sensor_steps(&mut filter, &prep),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// =============================================================================
// Multi-Sensor LMB Benchmarks
// =============================================================================

fn bench_ga_lmb(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_sensor/ga_lmb");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for scenario_cfg in MULTI_SENSOR_SCENARIOS {
        let scenario = load_scenario(scenario_cfg.path);
        let prep = preprocess(&scenario);

        group.bench_function(BenchmarkId::new("step", scenario_cfg.name), |b| {
            b.iter_batched(
                || create_filter("GA-LMB-LBP", &prep).unwrap(),
                |mut filter| run_multi_sensor_steps(&mut filter, &prep),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_aa_lmb(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_sensor/aa_lmb");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    // AA is slower - only benchmark smaller scenarios
    let aa_scenarios = &MULTI_SENSOR_SCENARIOS[..3]; // n5_s2, n10_s2, n10_s4

    for scenario_cfg in aa_scenarios {
        let scenario = load_scenario(scenario_cfg.path);
        let prep = preprocess(&scenario);

        group.bench_function(BenchmarkId::new("step", scenario_cfg.name), |b| {
            b.iter_batched(
                || create_filter("AA-LMB-LBP", &prep).unwrap(),
                |mut filter| run_multi_sensor_steps(&mut filter, &prep),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_ic_lmb(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_sensor/ic_lmb");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    // IC is moderately slow - benchmark up to n20_s4
    let ic_scenarios = &MULTI_SENSOR_SCENARIOS[..5];

    for scenario_cfg in ic_scenarios {
        let scenario = load_scenario(scenario_cfg.path);
        let prep = preprocess(&scenario);

        group.bench_function(BenchmarkId::new("step", scenario_cfg.name), |b| {
            b.iter_batched(
                || create_filter("IC-LMB-LBP", &prep).unwrap(),
                |mut filter| run_multi_sensor_steps(&mut filter, &prep),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_pu_lmb(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_sensor/pu_lmb");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    // PU has exponential complexity - only benchmark smallest scenarios
    let pu_scenarios = &MULTI_SENSOR_SCENARIOS[..2]; // n5_s2, n10_s2

    for scenario_cfg in pu_scenarios {
        let scenario = load_scenario(scenario_cfg.path);
        let prep = preprocess(&scenario);

        group.bench_function(BenchmarkId::new("step", scenario_cfg.name), |b| {
            b.iter_batched(
                || create_filter("PU-LMB-LBP", &prep).unwrap(),
                |mut filter| run_multi_sensor_steps(&mut filter, &prep),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_ms_lmbm(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_sensor/ms_lmbm");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    // MS-LMBM is very slow - only benchmark smallest scenario
    let scenario_cfg = &MULTI_SENSOR_SCENARIOS[0]; // n5_s2
    let scenario = load_scenario(scenario_cfg.path);
    let prep = preprocess(&scenario);

    group.bench_function(BenchmarkId::new("step", scenario_cfg.name), |b| {
        b.iter_batched(
            || create_filter("MS-LMBM-Gibbs", &prep).unwrap(),
            |mut filter| run_multi_sensor_steps(&mut filter, &prep),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    single_sensor_benches,
    bench_lmb_lbp,
    bench_lmb_gibbs,
    bench_lmb_murty,
    bench_lmbm_gibbs,
);

criterion_group!(
    multi_sensor_benches,
    bench_ga_lmb,
    bench_aa_lmb,
    bench_ic_lmb,
    bench_pu_lmb,
    bench_ms_lmbm,
);

criterion_main!(single_sensor_benches, multi_sensor_benches);
