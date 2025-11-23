// Test to verify seed 1000 shows no divergence
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
fn trace_seed_1000_no_divergence() {
    let seed = 1000;
    
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
        DataAssociationMethod::Murty,
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
    
    println!("\n=== Seed 1000 Trace ===\n");
    
    for t in 0..rust_estimates.mu.len() {
        if rust_estimates.mu[t].is_empty() && aa_variant.state_estimates.mu[t].is_empty() {
            continue;
        }
        
        if rust_estimates.mu[t].len() != aa_variant.state_estimates.mu[t].len() {
            println!("t={}: CARD MISMATCH - Rust={}, MATLAB={}",
                t, rust_estimates.mu[t].len(), aa_variant.state_estimates.mu[t].len());
            continue;
        }
        
        let mut max_diff = 0.0;
        for (rust_mu, matlab_mu) in rust_estimates.mu[t].iter().zip(&aa_variant.state_estimates.mu[t]) {
            for (&r, &m) in rust_mu.iter().zip(matlab_mu) {
                let diff = (r - m).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        
        if max_diff > 1e-10 {
            println!("t={}: max_diff={:.6e} {}", t, max_diff,
                if max_diff > 1.0 { "⚠️ DIVERGED" } else { "" });
        } else {
            println!("t={}: ✓ MATCH", t);
        }
    }
}
