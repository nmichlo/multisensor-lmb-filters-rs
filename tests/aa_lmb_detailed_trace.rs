// Detailed trace to find where AA-LMB divergence starts
use prak::common::ground_truth::generate_multisensor_ground_truth;
use prak::common::model::generate_multisensor_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ParallelUpdateMode, ScenarioType};
use prak::multisensor_lmb::parallel_update::run_parallel_update_lmb_filter;
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
struct FilterVariant {
    name: String,
    #[serde(rename = "stateEstimates")]
    state_estimates: StateEstimates,
}

#[derive(Debug, Deserialize)]
struct StateEstimates {
    #[allow(dead_code)]
    labels: serde_json::Value,
    mu: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "Sigma")]
    #[allow(dead_code)]
    sigma: Vec<Vec<Vec<Vec<f64>>>>,
    #[allow(dead_code)]
    objects: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    #[allow(dead_code)]
    seed: u64,
    #[serde(rename = "filterVariants")]
    filter_variants: Vec<FilterVariant>,
}

#[test]
fn detailed_trace_seed_1() {
    let seed = 1;

    let fixture_path = format!(
        "{}/../multisensor-lmb-filters/fixtures/numerical_equivalence/multi_sensor_seed{}.json",
        env!("CARGO_MANIFEST_DIR"),
        seed
    );
    let json = fs::read_to_string(&fixture_path).unwrap();
    let fixture: Fixture = serde_json::from_str(&json).unwrap();

    let aa_variant = fixture.filter_variants
        .iter()
        .find(|v| v.name == "AA-LMB")
        .expect("AA-LMB variant not found");

    let mut model_rng = SimpleRng::new(0);
    let mut model = generate_multisensor_model(
        &mut model_rng,
        3,
        vec![5.0, 5.0, 5.0],
        vec![0.67, 0.70, 0.73],
        vec![4.0, 3.0, 2.0],
        ParallelUpdateMode::AA,
        DataAssociationMethod::LBP,  // MUST match MATLAB fixture which uses LBP
        ScenarioType::Fixed,
        None,
    );
    model.lmb_parallel_update_mode = Some(ParallelUpdateMode::AA);

    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth = generate_multisensor_ground_truth(&mut trial_rng, &model, None);

    let mut filter_rng = SimpleRng::new(seed + 1000);
    let rust_estimates = run_parallel_update_lmb_filter(
        &mut filter_rng,
        &model,
        &ground_truth.measurements,
        3,
        ParallelUpdateMode::AA,
    );

    println!("\n=== DETAILED AA-LMB TRACE (Seed {}) ===\n", seed);

    // Check first 25 timesteps in detail
    for t in 0..25.min(rust_estimates.mu.len()) {
        let rust_n = rust_estimates.mu[t].len();
        let matlab_n = aa_variant.state_estimates.mu[t].len();

        if rust_n == 0 && matlab_n == 0 {
            println!("t={}: Both empty", t);
            continue;
        }

        if rust_n != matlab_n {
            println!("t={}: CARDINALITY MISMATCH - Rust={}, MATLAB={}", t, rust_n, matlab_n);
            continue;
        }

        // Compute max difference
        let mut max_diff = 0.0;
        let mut max_diff_obj = 0;
        let mut max_diff_elem = 0;

        for (obj_idx, (rust_mu, matlab_mu)) in rust_estimates.mu[t].iter().zip(&aa_variant.state_estimates.mu[t]).enumerate() {
            for (elem_idx, (&r, &m)) in rust_mu.iter().zip(matlab_mu).enumerate() {
                let diff = (r - m).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_obj = obj_idx;
                    max_diff_elem = elem_idx;
                }
            }
        }

        if max_diff > 1e-10 {
            println!("t={}: max_diff={:.6e} at obj[{}][{}]", t, max_diff, max_diff_obj, max_diff_elem);
            println!("       Rust  : {:.10e}", rust_estimates.mu[t][max_diff_obj][max_diff_elem]);
            println!("       MATLAB: {:.10e}", aa_variant.state_estimates.mu[t][max_diff_obj][max_diff_elem]);

            if max_diff > 0.1 {
                println!("       ⚠️ SIGNIFICANT DIVERGENCE");
                // Print all objects at this timestep to check ordering
                println!("       --- Rust objects (birth_t,birth_loc,x,y) ---");
                for (i, mu) in rust_estimates.mu[t].iter().enumerate() {
                    let label = &rust_estimates.labels[t];
                    let birth_time = if i < label.ncols() { label[(0, i)] } else { 0 };
                    let birth_loc = if i < label.ncols() { label[(1, i)] } else { 0 };
                    println!("         obj[{}]: (t={},loc={}) ({:.2}, {:.2})", i, birth_time, birth_loc, mu[0], mu[1]);
                }
                println!("       --- MATLAB objects (x,y) ---");
                for (i, mu) in aa_variant.state_estimates.mu[t].iter().enumerate() {
                    println!("         obj[{}]: ({:.2}, {:.2})", i, mu[0], mu[1]);
                }
            }
        } else {
            println!("t={}: ✓ MATCH (max_diff={:.2e})", t, max_diff);
        }
    }

    // Now check where it fails
    println!("\n=== CHECKING REMAINING TIMESTEPS ===\n");

    let mut first_divergence_t = None;
    for t in 25..rust_estimates.mu.len() {
        let rust_n = rust_estimates.mu[t].len();
        let matlab_n = aa_variant.state_estimates.mu[t].len();

        if rust_n != matlab_n {
            println!("t={}: FIRST CARDINALITY FAILURE - Rust={}, MATLAB={}", t, rust_n, matlab_n);
            first_divergence_t = Some(t);
            break;
        }

        let mut max_diff: f64 = 0.0;
        let mut max_diff_obj = 0;
        let mut max_diff_elem = 0;
        for (obj_idx, (rust_mu, matlab_mu)) in rust_estimates.mu[t].iter().zip(&aa_variant.state_estimates.mu[t]).enumerate() {
            for (elem_idx, (&r, &m)) in rust_mu.iter().zip(matlab_mu).enumerate() {
                let diff = (r - m).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_obj = obj_idx;
                    max_diff_elem = elem_idx;
                }
            }
        }

        if max_diff > 1.0 {
            println!("t={}: FIRST MAJOR DIVERGENCE - max_diff={:.2e} at obj[{}][{}]", t, max_diff, max_diff_obj, max_diff_elem);
            first_divergence_t = Some(t);
            break;
        } else if max_diff > 1e-6 {
            println!("t={}: max_diff={:.6e}", t, max_diff);
        }
    }

    // Print detailed comparison around divergence point
    if let Some(dt) = first_divergence_t {
        let start_t = if dt >= 10 { dt - 10 } else { 0 };
        println!("\n=== DETAILED COMPARISON (t={} to t={}) ===\n", start_t, dt);

        for t in start_t..=dt.min(rust_estimates.mu.len() - 1) {
            let rust_n = rust_estimates.mu[t].len();
            let matlab_n = aa_variant.state_estimates.mu[t].len();
            println!("t={}: Rust has {} objects, MATLAB has {}", t, rust_n, matlab_n);

            for i in 0..rust_n.min(matlab_n) {
                let rust_mu = &rust_estimates.mu[t][i];
                let matlab_mu = &aa_variant.state_estimates.mu[t][i];
                let label = &rust_estimates.labels[t];
                let birth_time = if i < label.ncols() { label[(0, i)] } else { 0 };
                let birth_loc = if i < label.ncols() { label[(1, i)] } else { 0 };

                let diff_x = (rust_mu[0] - matlab_mu[0]).abs();
                let diff_y = (rust_mu[1] - matlab_mu[1]).abs();
                let max_diff = diff_x.max(diff_y);

                // Always show object 6 (the diverging one) if it exists
                if max_diff > 0.01 || (i == 6 && t >= start_t) {
                    let marker = if max_diff > 1.0 { "***" } else if max_diff > 0.01 { "  !" } else { "   " };
                    println!("{} obj[{}]: (t={},loc={}) Rust=({:.4},{:.4}) MATLAB=({:.4},{:.4}) DIFF=({:.6},{:.6})",
                        marker, i, birth_time, birth_loc, rust_mu[0], rust_mu[1], matlab_mu[0], matlab_mu[1], diff_x, diff_y);
                }
            }
        }
    }
}
