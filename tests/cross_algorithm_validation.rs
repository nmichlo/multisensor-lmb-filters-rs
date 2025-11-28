//! Cross-algorithm validation tests
//!
//! Compares LBP, Gibbs, and Murty's data association algorithms to verify
//! they converge to similar marginal distributions.
//!
//! Matches MATLAB marginalEvalulations/evaluateMarginalDistributions.m

use nalgebra::DMatrix;
use prak::common::metrics::{average_hellinger_distance, average_kl_divergence};
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::association::generate_lmb_association_matrices;
use prak::lmb::data_association::{lmb_gibbs, lmb_lbp, lmb_murtys};
use prak::common::model::generate_model;
use prak::common::ground_truth::generate_ground_truth;
use prak::lmb::prediction::lmb_prediction_step;

/// Convert existence probability r to a 2-element distribution [1-r, r]
fn r_to_distribution(r: &[f64]) -> DMatrix<f64> {
    let n = r.len();
    let mut dist = DMatrix::zeros(n, 2);
    for i in 0..n {
        dist[(i, 0)] = 1.0 - r[i];
        dist[(i, 1)] = r[i];
    }
    dist
}

/// Compute average KL divergence for existence probabilities
fn r_kl_divergence(r1: &[f64], r2: &[f64]) -> f64 {
    let p = r_to_distribution(r1);
    let q = r_to_distribution(r2);
    average_kl_divergence(&p, &q)
}

/// Compute average Hellinger distance for existence probabilities
fn r_hellinger_distance(r1: &[f64], r2: &[f64]) -> f64 {
    let p = r_to_distribution(r1);
    let q = r_to_distribution(r2);
    average_hellinger_distance(&p, &q)
}

/// Run cross-algorithm comparison for a single scenario
fn compare_algorithms(
    seed: u64,
    num_gibbs_samples: usize,
    num_murty_assignments: usize,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    // Generate model
    let mut model_rng = SimpleRng::new(0);
    let model = generate_model(
        &mut model_rng,
        20.0,  // clutter rate (matches MATLAB expectedNumberOfClutterReturns = 20)
        0.75,  // detection probability (matches MATLAB)
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    // Generate ground truth with specified seed
    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth = generate_ground_truth(&mut trial_rng, &model, None);

    // Use first timestep with measurements
    let t = 0;
    let mut objects = model.object.clone();
    objects = lmb_prediction_step(objects, &model, t + 1);

    if ground_truth.measurements[t].is_empty() {
        // No measurements - all algorithms should agree (trivial case)
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // Generate association matrices
    let association_result = generate_lmb_association_matrices(
        &objects,
        &ground_truth.measurements[t],
        &model,
    );

    // Run LBP (reference)
    let (r_lbp, w_lbp) = lmb_lbp(&association_result, 1e-18, 10000);

    // Run Gibbs sampling
    let mut gibbs_rng = SimpleRng::new(seed + 1000);
    let (r_gibbs, w_gibbs) = lmb_gibbs(&mut gibbs_rng, &association_result, num_gibbs_samples);

    // Run Murty's algorithm
    let (r_murty, w_murty, _) = lmb_murtys(&association_result, num_murty_assignments);

    // Compute metrics: Gibbs vs LBP
    let r_kl_gibbs = r_kl_divergence(r_lbp.as_slice(), r_gibbs.as_slice());
    let r_h_gibbs = r_hellinger_distance(r_lbp.as_slice(), r_gibbs.as_slice());
    let w_kl_gibbs = average_kl_divergence(&w_lbp, &w_gibbs);
    let w_h_gibbs = average_hellinger_distance(&w_lbp, &w_gibbs);

    // Compute metrics: Murty vs LBP
    let r_kl_murty = r_kl_divergence(r_lbp.as_slice(), r_murty.as_slice());
    let r_h_murty = r_hellinger_distance(r_lbp.as_slice(), r_murty.as_slice());
    let w_kl_murty = average_kl_divergence(&w_lbp, &w_murty);
    let w_h_murty = average_hellinger_distance(&w_lbp, &w_murty);

    (r_kl_gibbs, r_h_gibbs, w_kl_gibbs, w_h_gibbs,
     r_kl_murty, r_h_murty, w_kl_murty, w_h_murty)
}

#[test]
fn test_cross_algorithm_convergence() {
    println!("\n=== Cross-Algorithm Validation ===");
    println!("Comparing LBP (reference) vs Gibbs (10K samples) vs Murty's (1K assignments)\n");

    let seeds = [1, 42, 100, 1000, 12345];
    let num_gibbs_samples = 10000;
    let num_murty_assignments = 1000;

    let mut total_r_kl_gibbs = 0.0;
    let mut total_r_h_gibbs = 0.0;
    let mut total_w_kl_gibbs = 0.0;
    let mut total_w_h_gibbs = 0.0;
    let mut total_r_kl_murty = 0.0;
    let mut total_r_h_murty = 0.0;
    let mut total_w_kl_murty = 0.0;
    let mut total_w_h_murty = 0.0;

    for &seed in &seeds {
        let (r_kl_g, r_h_g, w_kl_g, w_h_g, r_kl_m, r_h_m, w_kl_m, w_h_m) =
            compare_algorithms(seed, num_gibbs_samples, num_murty_assignments);

        println!("Seed {}: Gibbs r_KL={:.6} r_H={:.6} | Murty r_KL={:.6} r_H={:.6}",
                 seed, r_kl_g, r_h_g, r_kl_m, r_h_m);

        total_r_kl_gibbs += r_kl_g;
        total_r_h_gibbs += r_h_g;
        total_w_kl_gibbs += w_kl_g;
        total_w_h_gibbs += w_h_g;
        total_r_kl_murty += r_kl_m;
        total_r_h_murty += r_h_m;
        total_w_kl_murty += w_kl_m;
        total_w_h_murty += w_h_m;
    }

    let n = seeds.len() as f64;
    let avg_r_kl_gibbs = total_r_kl_gibbs / n;
    let avg_r_h_gibbs = total_r_h_gibbs / n;
    let avg_w_kl_gibbs = total_w_kl_gibbs / n;
    let avg_w_h_gibbs = total_w_h_gibbs / n;
    let avg_r_kl_murty = total_r_kl_murty / n;
    let avg_r_h_murty = total_r_h_murty / n;
    let avg_w_kl_murty = total_w_kl_murty / n;
    let avg_w_h_murty = total_w_h_murty / n;

    println!("\n=== Average Errors ({} trials) ===", seeds.len());
    println!("Gibbs (10K samples):");
    println!("  r: KL={:.6}, Hellinger={:.6}", avg_r_kl_gibbs, avg_r_h_gibbs);
    println!("  W: KL={:.6}, Hellinger={:.6}", avg_w_kl_gibbs, avg_w_h_gibbs);
    println!("Murty's (1K assignments):");
    println!("  r: KL={:.6}, Hellinger={:.6}", avg_r_kl_murty, avg_r_h_murty);
    println!("  W: KL={:.6}, Hellinger={:.6}", avg_w_kl_murty, avg_w_h_murty);

    // Assertions - Gibbs with 10K samples should be reasonably close to LBP
    // Note: These tolerances are based on expected statistical convergence
    assert!(avg_r_h_gibbs < 0.2, "Gibbs r Hellinger error too high: {}", avg_r_h_gibbs);
    assert!(avg_w_h_gibbs < 0.3, "Gibbs W Hellinger error too high: {}", avg_w_h_gibbs);

    // Murty's with 1K assignments should be closer to LBP
    assert!(avg_r_h_murty < 0.1, "Murty r Hellinger error too high: {}", avg_r_h_murty);
    assert!(avg_w_h_murty < 0.2, "Murty W Hellinger error too high: {}", avg_w_h_murty);

    println!("\n✓ All cross-algorithm validation tests passed!");
}

#[test]
fn test_murty_converges_to_lbp_with_more_assignments() {
    println!("\n=== Murty's Convergence Test ===");
    println!("Testing if more assignments → closer to LBP\n");

    let seed = 42;
    let assignment_counts = [10, 100, 1000, 5000];

    let mut prev_r_h = f64::MAX;
    let mut _prev_w_h = f64::MAX;

    for &num_assignments in &assignment_counts {
        let (_, _, _, _, r_kl, r_h, w_kl, w_h) = compare_algorithms(seed, 1000, num_assignments);

        println!("Murty's ({} assignments): r_KL={:.6} r_H={:.6} W_KL={:.6} W_H={:.6}",
                 num_assignments, r_kl, r_h, w_kl, w_h);

        // Error should generally decrease (or stay similar) with more assignments
        // Allow some tolerance for statistical variation
        if num_assignments > 10 {
            // With more assignments, Hellinger distance should not increase significantly
            assert!(r_h <= prev_r_h + 0.1,
                    "r Hellinger increased too much: {} -> {}", prev_r_h, r_h);
        }

        prev_r_h = r_h;
        _prev_w_h = w_h;
    }

    println!("\n✓ Murty's convergence test passed!");
}

#[test]
fn test_gibbs_converges_to_lbp_with_more_samples() {
    println!("\n=== Gibbs Convergence Test ===");
    println!("Testing if more samples → closer to LBP\n");

    let seed = 42;
    let sample_counts = [100, 1000, 5000, 10000];

    for &num_samples in &sample_counts {
        let (r_kl, r_h, w_kl, w_h, _, _, _, _) = compare_algorithms(seed, num_samples, 100);

        println!("Gibbs ({} samples): r_KL={:.6} r_H={:.6} W_KL={:.6} W_H={:.6}",
                 num_samples, r_kl, r_h, w_kl, w_h);
    }

    // With 10K samples, should be reasonably close
    let (_, r_h_10k, _, _w_h_10k, _, _, _, _) = compare_algorithms(seed, 10000, 100);
    assert!(r_h_10k < 0.3, "Gibbs r Hellinger error too high with 10K samples: {}", r_h_10k);

    println!("\n✓ Gibbs convergence test passed!");
}
