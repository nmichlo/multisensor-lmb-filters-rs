//! Debug test to extract Murty intermediate values at failure point
//!
//! Seed 42, timestep 64 - where LMB-Murty diverges from MATLAB

use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::association::generate_lmb_association_matrices;
use prak::lmb::data_association::lmb_murtys;
use prak::lmb::prediction::lmb_prediction_step;

#[test]
fn debug_murty_t64_seed42() {
    // Generate model with fixed seed 0
    let mut model_rng = SimpleRng::new(0);
    let mut model = generate_model(
        &mut model_rng,
        2.0,  // clutterRate
        0.95, // detectionProbability
        DataAssociationMethod::Murty,
        ScenarioType::Fixed,
        None,
    );
    model.data_association_method = DataAssociationMethod::Murty;

    // Generate ground truth with seed 42
    let mut trial_rng = SimpleRng::new(42);
    let ground_truth_output = generate_ground_truth(&mut trial_rng, &model, None);

    // Run filter up to timestep 63 to get state before failure
    let mut filter_rng = SimpleRng::new(42 + 1000);
    let mut objects = model.object.clone();

    println!("\n=== Running LMB-Murty for seed 42 up to t=63 ===");

    for t in 0..64 {
        // Prediction
        objects = lmb_prediction_step(objects, &model, t + 1);

        // Measurement update
        if !ground_truth_output.measurements[t].is_empty() {
            let association_result = generate_lmb_association_matrices(
                &objects,
                &ground_truth_output.measurements[t],
                &model,
            );

            let (r, w, v) = lmb_murtys(&association_result, model.number_of_assignments);

            if t == 63 {
                println!("\n=== Timestep 63 (before failure) ===");
                println!("Number of objects: {}", objects.len());
                println!("Number of measurements: {}", ground_truth_output.measurements[t].len());
                println!("Association matrix dimensions: {} × {}", association_result.cost.nrows(), association_result.cost.ncols());
                println!("\nExistence probabilities r:");
                for (i, ri) in r.iter().enumerate() {
                    println!("  Object {}: r = {:.6}", i, ri);
                }
                println!("\nAssociation weights W (first 5 objects if more):");
                let n_show = w.nrows().min(5);
                for i in 0..n_show {
                    print!("  Object {}: [", i);
                    for j in 0..w.ncols() {
                        print!("{:.4}", w[(i, j)]);
                        if j < w.ncols() - 1 {
                            print!(", ");
                        }
                    }
                    println!("]");
                }
                println!("\nAssignment matrix V (first 5 rows):");
                let k_show = v.nrows().min(5);
                for i in 0..k_show {
                    print!("  Event {}: [", i);
                    for j in 0..v.ncols() {
                        print!("{}", v[(i, j)]);
                        if j < v.ncols() - 1 {
                            print!(", ");
                        }
                    }
                    println!("]");
                }
            }

            // Update posterior (this is where the bug manifests)
            objects = prak::lmb::update::compute_posterior_lmb_spatial_distributions(
                objects,
                &r,
                &w,
                &association_result.posterior_parameters,
                &model,
            );

            if t == 63 {
                println!("\n=== After posterior update at t=63 ===");
                for (i, obj) in objects.iter().enumerate() {
                    if obj.r > 0.01 {
                        println!("Object {}: r={:.6}, num_comp={}, mu[0]={:?}",
                                 i, obj.r, obj.number_of_gm_components,
                                 if !obj.mu.is_empty() { format!("{:.4}", obj.mu[0][0]) } else { "N/A".to_string() });
                    }
                }
            }
        }

        // Gate low-probability objects
        let objects_likely_to_exist = prak::common::utils::gate_objects_by_existence(
            &objects.iter().map(|obj| obj.r).collect::<Vec<_>>(),
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

        if t == 63 {
            println!("\n=== After gating at t=63 ===");
            println!("Objects remaining: {}", objects.len());
        }
    }

    // Now run timestep 64 with detailed logging
    println!("\n\n=== CRITICAL TIMESTEP 64 (FAILURE POINT) ===");
    let t = 64;

    // Prediction
    println!("\n1. Prediction step");
    objects = lmb_prediction_step(objects, &model, t + 1);
    println!("   Objects after prediction: {}", objects.len());

    // Measurement update
    println!("\n2. Generating association matrices");
    println!("   Measurements at t=64: {}", ground_truth_output.measurements[t].len());

    let association_result = generate_lmb_association_matrices(
        &objects,
        &ground_truth_output.measurements[t],
        &model,
    );

    println!("   Association matrix L dimensions: {} × {}",
             association_result.gibbs.l.nrows(), association_result.gibbs.l.ncols());
    println!("   Cost matrix dimensions: {} × {}",
             association_result.cost.nrows(), association_result.cost.ncols());

    println!("\n3. Running Murty's algorithm (K={})", model.number_of_assignments);
    let (r, w, v) = lmb_murtys(&association_result, model.number_of_assignments);

    println!("\n4. Murty results:");
    println!("   Existence probabilities r:");
    for (i, ri) in r.iter().enumerate() {
        if *ri > 0.01 {
            println!("     Object {}: r = {:.6}", i, ri);
        }
    }

    println!("\n   Association weights W (objects with r > 0.01):");
    for i in 0..r.len() {
        if r[i] > 0.01 {
            print!("     Object {}: [", i);
            for j in 0..w.ncols().min(10) {  // Show first 10 columns
                print!("{:.4}", w[(i, j)]);
                if j < w.ncols().min(10) - 1 {
                    print!(", ");
                }
            }
            if w.ncols() > 10 {
                print!(", ...");
            }
            println!("]");
        }
    }

    println!("\n   K-best assignments V:");
    for i in 0..v.nrows().min(10) {
        print!("     Event {}: [", i);
        for j in 0..v.ncols().min(10) {
            print!("{}", v[(i, j)]);
            if j < v.ncols().min(10) - 1 {
                print!(", ");
            }
        }
        if v.ncols() > 10 {
            print!(", ...");
        }
        println!("]");
    }

    println!("\n5. Computing posterior spatial distributions");
    objects = prak::lmb::update::compute_posterior_lmb_spatial_distributions(
        objects,
        &r,
        &w,
        &association_result.posterior_parameters,
        &model,
    );

    println!("\n6. State estimates after update:");
    for (i, obj) in objects.iter().enumerate() {
        if obj.r > 0.01 {
            println!("   Object {}: r={:.6}, num_comp={}", i, obj.r, obj.number_of_gm_components);
            if !obj.mu.is_empty() {
                println!("     mu[0]: [{:.4}, {:.4}, {:.4}, {:.4}]",
                         obj.mu[0][0], obj.mu[0][1], obj.mu[0][2], obj.mu[0][3]);
            }
        }
    }

    // Extract MAP estimate (like the filter does)
    let existence_probs: Vec<f64> = objects.iter().map(|obj| obj.r).collect();
    let (n_map, map_indices) = prak::lmb::cardinality::lmb_map_cardinality_estimate(&existence_probs);

    println!("\n7. MAP cardinality estimate:");
    println!("   Estimated cardinality: {}", n_map);
    println!("   Selected objects: {:?}", map_indices);

    println!("\n8. Final state estimate mu:");
    for (idx_in_est, &obj_idx) in map_indices.iter().enumerate() {
        let obj = &objects[obj_idx];
        if !obj.mu.is_empty() {
            println!("   Target {} (object {}): mu[0] = [{:.6}, {:.6}, {:.6}, {:.6}]",
                     idx_in_est, obj_idx,
                     obj.mu[0][0], obj.mu[0][1], obj.mu[0][2], obj.mu[0][3]);
        }
    }

    println!("\n=== Debug test complete ===");
    println!("\nExpected MATLAB value for target 0, mu[0]: -84.94");
    println!("Compare with Rust value above to identify discrepancy source");
}
