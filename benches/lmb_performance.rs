//! Performance benchmarks for LMB algorithms
//!
//! Run with: cargo bench
//! Run specific benchmark: cargo bench -- association
//! Compare against baseline: cargo bench -- --save-baseline main

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use multisensor_lmb_filters_rs::common::{
    association::lbp::{loopy_belief_propagation, AssociationMatrices},
    ground_truth::generate_ground_truth,
    model::generate_model,
    types::{DataAssociationMethod, ScenarioType},
    rng::{Rng, SimpleRng},
};
use multisensor_lmb_filters_rs::lmb::{
    association::generate_lmb_association_matrices,
    cardinality::elementary_symmetric_function,
    filter::run_lmb_filter,
    prediction::lmb_prediction_step,
};
use nalgebra::{DMatrix, DVector};

/// Create test data for benchmarks
fn create_test_objects(rng: &mut SimpleRng, n_objects: usize, n_gm: usize) -> Vec<multisensor_lmb_filters_rs::common::types::Object> {
    let model = generate_model(rng, 5.0, 0.9, DataAssociationMethod::LBP, ScenarioType::Fixed, None);
    let mut objects = model.birth_parameters.clone();

    // Limit to n_objects
    objects.truncate(n_objects);

    // Ensure each object has n_gm components
    for obj in &mut objects {
        obj.r = 0.5;
        while obj.number_of_gm_components < n_gm {
            obj.w.push(1.0 / n_gm as f64);
            obj.mu.push(obj.mu[0].clone());
            obj.sigma.push(obj.sigma[0].clone());
            obj.number_of_gm_components += 1;
        }
        // Normalize weights
        let sum: f64 = obj.w.iter().sum();
        for w in &mut obj.w {
            *w /= sum;
        }
    }

    objects
}

fn create_test_measurements(rng: &mut SimpleRng, n_measurements: usize, z_dim: usize) -> Vec<DVector<f64>> {
    (0..n_measurements)
        .map(|_| DVector::from_fn(z_dim, |_, _| rng.randn() * 50.0))
        .collect()
}

// =============================================================================
// ASSOCIATION MATRIX BENCHMARKS
// =============================================================================

fn bench_association_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("association_matrices");
    group.sample_size(20);

    for (n_obj, n_meas) in [(5, 5), (10, 10), (20, 20), (50, 50)] {
        let mut rng = SimpleRng::new(42);
        let model = generate_model(&mut rng, 5.0, 0.9, DataAssociationMethod::LBP, ScenarioType::Fixed, None);
        let objects = create_test_objects(&mut rng, n_obj, 3);
        let measurements = create_test_measurements(&mut rng, n_meas, model.z_dimension);

        group.bench_with_input(
            BenchmarkId::new("generate", format!("{}obj_{}meas", n_obj, n_meas)),
            &(&objects, &measurements, &model),
            |b, (obj, meas, mdl)| {
                b.iter(|| generate_lmb_association_matrices(obj, meas, mdl))
            },
        );
    }

    group.finish();
}

// =============================================================================
// LOOPY BELIEF PROPAGATION BENCHMARKS
// =============================================================================

fn create_association_matrices(n_obj: usize, n_meas: usize) -> AssociationMatrices {
    // Create random but valid association matrices
    let psi = DMatrix::from_fn(n_obj, n_meas, |i, j| 0.1 + ((i * 7 + j * 13) % 100) as f64 / 100.0);
    let phi = DVector::from_fn(n_obj, |i, _| 0.1 + ((i * 11) % 100) as f64 / 100.0);
    let eta = DVector::from_fn(n_obj, |i, _| 0.5 + ((i * 17) % 50) as f64 / 100.0);

    AssociationMatrices { psi, phi, eta }
}

fn bench_lbp(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbp");
    group.sample_size(50);

    for (n_obj, n_meas) in [(5, 5), (10, 10), (20, 20), (50, 50)] {
        let matrices = create_association_matrices(n_obj, n_meas);

        group.bench_with_input(
            BenchmarkId::new("convergence", format!("{}x{}", n_obj, n_meas)),
            &matrices,
            |b, m| {
                b.iter(|| loopy_belief_propagation(m, 1e-6, 100))
            },
        );
    }

    group.finish();
}

// =============================================================================
// ELEMENTARY SYMMETRIC FUNCTION BENCHMARKS
// =============================================================================

fn bench_esf(c: &mut Criterion) {
    let mut group = c.benchmark_group("esf");
    group.sample_size(100);

    for n in [5, 10, 20, 50, 100] {
        let z: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64) * 0.01).collect();

        group.bench_with_input(
            BenchmarkId::new("compute", format!("n={}", n)),
            &z,
            |b, z| {
                b.iter(|| elementary_symmetric_function(z))
            },
        );
    }

    group.finish();
}

// =============================================================================
// PREDICTION STEP BENCHMARKS
// =============================================================================

fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");
    group.sample_size(50);

    for n_obj in [5, 10, 20, 50] {
        let mut rng = SimpleRng::new(42);
        let model = generate_model(&mut rng, 5.0, 0.9, DataAssociationMethod::LBP, ScenarioType::Fixed, None);
        let objects = create_test_objects(&mut rng, n_obj, 3);

        group.bench_with_input(
            BenchmarkId::new("step", format!("{}obj", n_obj)),
            &(&objects, &model),
            |b, (obj, mdl)| {
                b.iter(|| lmb_prediction_step(obj.to_vec(), mdl, 1))
            },
        );
    }

    group.finish();
}

// =============================================================================
// FULL FILTER BENCHMARKS
// =============================================================================

fn bench_full_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_filter");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    // Use only a portion of the measurements for faster benchmarking
    for sim_length in [10, 25, 50] {
        let mut rng = SimpleRng::new(42);
        let model = generate_model(&mut rng, 5.0, 0.9, DataAssociationMethod::LBP, ScenarioType::Fixed, None);
        let ground_truth = generate_ground_truth(&mut rng, &model, None);

        // Take only first sim_length timesteps
        let measurements: Vec<Vec<DVector<f64>>> = ground_truth.measurements
            .into_iter()
            .take(sim_length)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("lmb_lbp", format!("t={}", sim_length)),
            &(&model, &measurements),
            |b, (mdl, meas)| {
                let mut rng = SimpleRng::new(42);
                b.iter(|| run_lmb_filter(&mut rng, mdl, meas))
            },
        );
    }

    group.finish();
}

// =============================================================================
// GROUP DEFINITIONS
// =============================================================================

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        bench_association_matrices,
        bench_lbp,
        bench_esf,
        bench_prediction,
        bench_full_filter
);

criterion_main!(benches);
