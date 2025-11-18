//! Test RNG sequence in ground truth generation
use prak::common::model::generate_model;
use prak::common::rng::{Rng, SimpleRng};
use prak::common::types::{DataAssociationMethod, ScenarioType};
use nalgebra::{Cholesky, DVector};

#[test]
fn test_rng_ground_truth_seed42() {
    // Generate model with seed 0
    let mut model_rng = SimpleRng::new(0);
    let model = generate_model(
        &mut model_rng,
        2.0,  // clutterRate
        0.95, // detectionProbability
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    // Initialize ground truth RNG with seed 42
    let mut rng = SimpleRng::new(42);

    let simulation_length = 3;  // Only first 3 timesteps
    let mut measurements: Vec<Vec<DVector<f64>>> = vec![Vec::new(); simulation_length];

    let q_chol = Cholesky::new(model.q.clone()).expect("Q must be positive definite");

    println!("\n=== Rust RNG Trace for Seed 42, First 3 Timesteps ===\n");

    // Clutter generation
    println!("CLUTTER GENERATION:");
    for t in 0..simulation_length {
        println!("  t={}:", t + 1);
        let num_clutter = rng.poissrnd(model.clutter_rate);
        println!("    Poisson draw: {} clutter measurements", num_clutter);

        for j in 0..num_clutter {
            let mut rand_vec = Vec::new();
            let mut z = DVector::zeros(model.z_dimension);
            for d in 0..model.z_dimension {
                let r = rng.rand();
                rand_vec.push(r);
                let range = model.observation_space_limits[(d, 1)] - model.observation_space_limits[(d, 0)];
                z[d] = model.observation_space_limits[(d, 0)] + range * r;
            }
            println!("      Clutter {}: rand()=[{:.6}, {:.6}]", j + 1, rand_vec[0], rand_vec[1]);
            println!("                z=[{:.4}, {:.4}]", z[0], z[1]);
            measurements[t].push(z);
        }
        println!("    Total clutter at t={}: {}", t + 1, measurements[t].len());
    }

    // Object detection would go here, but skipping for simplicity
    // since the test is focused on clutter RNG

    println!("\n=== SUMMARY ===");
    for (t, meas) in measurements.iter().enumerate() {
        println!("t={}: {} total measurements", t + 1, meas.len());
    }

    println!("\n=== RNG State ===");
    // To get the state, we need to add a method or just compare the next value
    let next_val = rng.rand();
    println!("Next rand() after 3 timesteps: {:.10}", next_val);
}
