use prak::common::ground_truth::generate_ground_truth;
use prak::common::metrics::ospa;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::filter::run_lmb_filter;
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FilterVariantMetrics {
    name: String,
    #[serde(rename = "eOspa")]
    e_ospa: Vec<f64>,
    #[serde(rename = "hOspa")]
    h_ospa: Vec<f64>,
    cardinality: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrialFixture {
    seed: u64,
    filter_variants: Vec<FilterVariantMetrics>,
}

fn main() {
    // Load fixture
    let json_str = fs::read_to_string("tests/data/single_trial_42.json").unwrap();
    let fixture: TrialFixture = serde_json::from_str(&json_str).unwrap();

    // Find Murty fixture
    let murty_fixture = fixture.filter_variants.iter()
        .find(|v| v.name == "LMB-Murty")
        .expect("LMB-Murty not found");

    // Run Rust Murty
    let seed = 42;
    let mut model_rng = SimpleRng::new(0);
    let mut model = generate_model(
        &mut model_rng,
        2.0,
        0.95,
        DataAssociationMethod::Murty,
        ScenarioType::Fixed,
        None,
    );
    model.data_association_method = DataAssociationMethod::Murty;

    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth_output = generate_ground_truth(&mut trial_rng, &model, None);

    let mut filter_rng = SimpleRng::new(seed + 1000);
    let state_estimates = run_lmb_filter(&mut filter_rng, &model, &ground_truth_output.measurements);

    // Compare timestep by timestep
    println!("Timestep-by-timestep comparison (LMB-Murty):");
    println!("{:>4} | {:>12} | {:>12} | {:>12} | {:>5} | {:>5}",
             "T", "Rust E-OSPA", "MATLAB E-OSPA", "Diff", "R-Card", "M-Card");
    println!("{}", "-".repeat(75));

    for t in 0..murty_fixture.e_ospa.len() {
        let metrics = ospa(
            &ground_truth_output.ground_truth_rfs.x[t],
            &ground_truth_output.ground_truth_rfs.mu[t],
            &ground_truth_output.ground_truth_rfs.sigma[t],
            &state_estimates.mu[t],
            &state_estimates.sigma[t],
            &model.ospa_parameters,
        );

        let rust_eospa = metrics.e_ospa.total;
        let matlab_eospa = murty_fixture.e_ospa[t];
        let diff = (rust_eospa - matlab_eospa).abs();
        let rust_card = state_estimates.mu[t].len();
        let matlab_card = murty_fixture.cardinality[t];

        let marker = if diff > 1e-6 { "***" } else { "" };

        println!("{:4} | {:12.6} | {:12.6} | {:12.6} | {:5} | {:5}  {}",
                 t, rust_eospa, matlab_eospa, diff, rust_card, matlab_card, marker);

        // Stop at first major divergence
        if diff > 0.1 {
            println!("\n=== DIVERGENCE DETECTED AT TIMESTEP {} ===", t);
            println!("Rust cardinality: {}, MATLAB cardinality: {}", rust_card, matlab_card);
            println!("Rust E-OSPA: {:.10}, MATLAB E-OSPA: {:.10}", rust_eospa, matlab_eospa);
            break;
        }
    }
}
