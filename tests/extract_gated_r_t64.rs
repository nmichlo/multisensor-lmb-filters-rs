//! Extract r values AFTER gating at t=64
use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::association::generate_lmb_association_matrices;
use prak::lmb::cardinality::lmb_map_cardinality_estimate;
use prak::lmb::data_association::lmb_murtys;
use prak::lmb::prediction::lmb_prediction_step;
use prak::lmb::update::compute_posterior_lmb_spatial_distributions;

#[test]
fn extract_gated_r_t64() {
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

    let mut trial_rng = SimpleRng::new(42);
    let ground_truth_output = generate_ground_truth(&mut trial_rng, &model, None);

    let mut objects = model.object.clone();

    for t in 0..65 {
        objects = lmb_prediction_step(objects, &model, t + 1);

        if !ground_truth_output.measurements[t].is_empty() {
            let association_result = generate_lmb_association_matrices(
                &objects,
                &ground_truth_output.measurements[t],
                &model,
            );

            let (r, w, _v) = lmb_murtys(&association_result, model.number_of_assignments);

            if t == 64 {
                println!("\n=== t=64 MURTY OUTPUT (before posterior update) ===");
                for (i, &ri) in r.iter().enumerate().take(12) {
                    println!("r_murty[{}] = {:.15} (bits: {:064b})", i, ri, ri.to_bits());
                }
            }

            objects = compute_posterior_lmb_spatial_distributions(
                objects,
                &r,
                &w,
                &association_result.posterior_parameters,
                &model,
            );

            if t == 64 {
                println!("\n=== t=64 AFTER POSTERIOR UPDATE ===");
                for (i, obj) in objects.iter().enumerate().take(12) {
                    println!("r_post[{}] = {:.15} (bits: {:064b})", i, obj.r, obj.r.to_bits());
                }
            }
        }

        // Gate objects
        let existence_probs: Vec<f64> = objects.iter().map(|obj| obj.r).collect();
        let objects_likely_to_exist = prak::common::utils::gate_objects_by_existence(
            &existence_probs,
            model.existence_threshold,
        );

        objects = objects
            .into_iter()
            .enumerate()
            .filter_map(|(i, obj)| {
                if objects_likely_to_exist[i] {
                    Some(obj)
                } else {
                    None
                }
            })
            .collect();

        if t == 64 {
            println!("\n=== t=64 (0-indexed) AFTER GATING ===");
            println!("Number of objects after gating: {}", objects.len());
            println!("Existence threshold: {:.10}", model.existence_threshold);

            println!("\nGated r values (passed to MAP):");
            for (i, obj) in objects.iter().enumerate() {
                println!("r[{}] = {:.15} (bits: {:064b})", i, obj.r, obj.r.to_bits());
            }

            // Check which ones are exactly 1.0
            let ones: Vec<usize> = objects.iter().enumerate()
                .filter(|(_, obj)| obj.r == 1.0)
                .map(|(i, _)| i)
                .collect();
            println!("\nObjects with r exactly equal to 1.0: {:?}", ones);

            // Run MAP and show result
            let r_vals: Vec<f64> = objects.iter().map(|obj| obj.r).collect();
            let (n_map, map_indices) = lmb_map_cardinality_estimate(&r_vals);

            println!("\nMAP result:");
            println!("  n_map = {}", n_map);
            print!("  indices (0-indexed) = [");
            for (i, &idx) in map_indices.iter().enumerate() {
                print!("{}", idx);
                if i < map_indices.len() - 1 {
                    print!(", ");
                }
            }
            println!("]");

            println!("\nTarget estimates:");
            for (i, &idx) in map_indices.iter().enumerate() {
                println!("  Target {} (object {}): mu[1] = {:.10}", i, idx, objects[idx].mu[0][(0, 0)]);
            }

            println!("\n=== Expected from MATLAB ===");
            println!("Number of objects: 12");
            println!("n_map = 9");
            println!("indices = [0, 3, 5, 6, 7, 2, 8, 9, 1]");
            println!("Target 0 (object 0): mu[1] = -84.9372243229");
        }
    }
}
