use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::association::generate_lmb_association_matrices;
use prak::lmb::cardinality::lmb_map_cardinality_estimate;
use prak::lmb::data_association::lmb_murtys;
use prak::lmb::prediction::lmb_prediction_step;
use prak::lmb::update::compute_posterior_lmb_spatial_distributions;

fn main() {
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

    // Initialize with birth objects
    let mut objects = model.birth_parameters.clone();

    // Run filter up to timestep 26 (just after the divergence)
    for t in 0..26 {
        println!("\n=== Timestep {} ===", t);

        // Prediction
        objects = lmb_prediction_step(objects, &model, t + 1);
        println!("After prediction: {} objects", objects.len());

        // Measurement update
        if !ground_truth_output.measurements[t].is_empty() {
            let num_meas = ground_truth_output.measurements[t].len();
            println!("Number of measurements: {}", num_meas);

            // Generate association matrices
            let association_result =
                generate_lmb_association_matrices(&objects, &ground_truth_output.measurements[t], &model);

            // Run Murty's
            let (r, w, _v) = lmb_murtys(&association_result, model.number_of_assignments);

            // Print existence probabilities BEFORE gating
            println!("Existence probabilities (before gating):");
            for (i, &r_i) in r.iter().enumerate() {
                println!("  Object {}: r = {:.6} {}", i, r_i,
                    if r_i > model.existence_threshold { "(KEPT)" } else { "(PRUNED)" });
            }

            // Compute posterior
            objects = compute_posterior_lmb_spatial_distributions(
                objects,
                &r,
                &w,
                &association_result.posterior_parameters,
                &model,
            );

            // Gate by existence
            let objects_to_keep: Vec<bool> = objects.iter()
                .map(|obj| obj.r > model.existence_threshold)
                .collect();

            // Print gating results
            let num_kept = objects_to_keep.iter().filter(|&&keep| keep).count();
            let num_pruned = objects_to_keep.iter().filter(|&&keep| !keep).count();
            println!("Gating: keeping {}, pruning {}", num_kept, num_pruned);

            // Apply gating
            objects = objects.into_iter()
                .enumerate()
                .filter_map(|(i, obj)| if objects_to_keep[i] { Some(obj) } else { None })
                .collect();
        }

        // MAP cardinality
        let existence_probs: Vec<f64> = objects.iter().map(|obj| obj.r).collect();
        let (n_map, map_indices) = lmb_map_cardinality_estimate(&existence_probs);
        println!("After gating: {} objects, MAP cardinality: {}", objects.len(), n_map);

        // Stop after timestep 25 (where divergence occurs)
        if t == 25 {
            println!("\n=== FINAL STATE AT TIMESTEP 25 ===");
            println!("Number of tracked objects: {}", objects.len());
            println!("MAP cardinality: {}", n_map);
            println!("MAP indices: {:?}", map_indices);
            println!("\nFinal existence probabilities:");
            for (i, &r) in existence_probs.iter().enumerate() {
                let is_map = map_indices.contains(&i);
                println!("  Object {}: r = {:.6} {}", i, r, if is_map { "(IN MAP)" } else { "" });
            }
            break;
        }
    }
}
