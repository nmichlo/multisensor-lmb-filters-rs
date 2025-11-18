//! Extract exact r values at t=64 for seed 42
use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::association::generate_lmb_association_matrices;
use prak::lmb::data_association::lmb_murtys;
use prak::lmb::prediction::lmb_prediction_step;
use prak::lmb::update::compute_posterior_lmb_spatial_distributions;

#[test]
fn extract_r_values_seed42_t64() {
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

    for t in 0..64 {
        objects = lmb_prediction_step(objects, &model, t + 1);

        if !ground_truth_output.measurements[t].is_empty() {
            let association_result = generate_lmb_association_matrices(
                &objects,
                &ground_truth_output.measurements[t],
                &model,
            );

            let (r, w, _v) = lmb_murtys(&association_result, model.number_of_assignments);

            if t == 63 {  // t=64 (0-indexed)
                println!("\n=== t=64 Rust r values ===");
                println!("Number of objects: {}", r.len());
                println!("Number of measurements: {}", ground_truth_output.measurements[t].len());

                println!("\nExact r values (all objects):");
                for (i, ri) in r.iter().enumerate() {
                    println!("r[{}] = {:.15}", i, ri);
                }

                let count_above_001 = r.iter().filter(|&&ri| ri > 0.01).count();
                let count_above_099 = r.iter().filter(|&&ri| ri > 0.99).count();
                let count_exactly_1 = r.iter().filter(|&&ri| ri == 1.0).count();
                let max_r = r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min_r = r.iter().cloned().fold(f64::INFINITY, f64::min);

                println!("\nNumber with r > 0.01: {}", count_above_001);
                println!("Number with r > 0.99: {}", count_above_099);
                println!("Number with r == 1.0: {}", count_exactly_1);
                println!("Max r: {:.15}", max_r);
                println!("Min r: {:.15}", min_r);

                println!("\n=== Expected from MATLAB ===");
                println!("Number of objects: 16");
                println!("Number with r > 0.01: 11");
                println!("Number with r > 0.99: 9");
                println!("Number with r == 1.0: 0");
                println!("Max r: 0.999592836467778");
            }

            objects = compute_posterior_lmb_spatial_distributions(
                objects,
                &r,
                &w,
                &association_result.posterior_parameters,
                &model,
            );
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
    }
}
