use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::association::generate_lmb_association_matrices;
use prak::lmb::cardinality::lmb_map_cardinality_estimate;
use prak::lmb::data_association::{lmb_gibbs, lmb_lbp, lmb_murtys};
use prak::lmb::prediction::lmb_prediction_step;
use prak::lmb::update::compute_posterior_lmb_spatial_distributions;

fn run_to_timestep(method: DataAssociationMethod, target_t: usize) -> (Vec<f64>, usize) {
    let seed = 42;
    let mut model_rng = SimpleRng::new(0);
    let mut model = generate_model(
        &mut model_rng,
        2.0,
        0.95,
        method,
        ScenarioType::Fixed,
        None,
    );
    model.data_association_method = method;

    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth_output = generate_ground_truth(&mut trial_rng, &model, None);

    let mut filter_rng = SimpleRng::new(seed + 1000);
    let mut objects = model.birth_parameters.clone();

    for t in 0..=target_t {
        objects = lmb_prediction_step(objects, &model, t + 1);

        if !ground_truth_output.measurements[t].is_empty() {
            let association_result =
                generate_lmb_association_matrices(&objects, &ground_truth_output.measurements[t], &model);

            let (r, w) = match method {
                DataAssociationMethod::LBP => {
                    lmb_lbp(&association_result, model.lbp_convergence_tolerance, model.maximum_number_of_lbp_iterations)
                }
                DataAssociationMethod::Gibbs => {
                    lmb_gibbs(&mut filter_rng, &association_result, model.number_of_samples)
                }
                DataAssociationMethod::Murty => {
                    let (r, w, _v) = lmb_murtys(&association_result, model.number_of_assignments);
                    (r, w)
                }
                _ => panic!("Unsupported method"),
            };

            objects = compute_posterior_lmb_spatial_distributions(
                objects,
                &r,
                &w,
                &association_result.posterior_parameters,
                &model,
            );

            objects = objects.into_iter()
                .filter(|obj| obj.r > model.existence_threshold)
                .collect();
        }
    }

    let existence_probs: Vec<f64> = objects.iter().map(|obj| obj.r).collect();
    let (n_map, _) = lmb_map_cardinality_estimate(&existence_probs);
    (existence_probs, n_map)
}

fn main() {
    println!("Comparing data association methods at timestep 25:\n");

    let (lbp_probs, lbp_card) = run_to_timestep(DataAssociationMethod::LBP, 25);
    let (gibbs_probs, gibbs_card) = run_to_timestep(DataAssociationMethod::Gibbs, 25);
    let (murty_probs, murty_card) = run_to_timestep(DataAssociationMethod::Murty, 25);

    println!("LBP:");
    println!("  MAP cardinality: {}", lbp_card);
    println!("  Existence probs: {}", lbp_probs.iter()
        .map(|r| format!("{:.6}", r))
        .collect::<Vec<_>>()
        .join(", "));

    println!("\nGibbs:");
    println!("  MAP cardinality: {}", gibbs_card);
    println!("  Existence probs: {}", gibbs_probs.iter()
        .map(|r| format!("{:.6}", r))
        .collect::<Vec<_>>()
        .join(", "));

    println!("\nMurty:");
    println!("  MAP cardinality: {}", murty_card);
    println!("  Existence probs: {}", murty_probs.iter()
        .map(|r| format!("{:.6}", r))
        .collect::<Vec<_>>()
        .join(", "));

    println!("\n=== Comparison ===");
    println!("Expected MAP cardinality (from MATLAB): 4");
    println!("Rust MAP cardinalities: LBP={}, Gibbs={}, Murty={}", lbp_card, gibbs_card, murty_card);

    if lbp_card == gibbs_card && lbp_card != murty_card {
        println!("\n⚠️  ISSUE: Murty differs from LBP/Gibbs!");
        println!("This suggests a bug in the Murty implementation.");
    } else if lbp_card == murty_card {
        println!("\n✓ All methods agree on cardinality.");
        println!("Issue may be in MAP cardinality estimation, not data association.");
    }
}
