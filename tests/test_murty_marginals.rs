//! Test Murty marginal computation

use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::association::generate_lmb_association_matrices;
use prak::lmb::data_association::lmb_murtys;
use prak::lmb::prediction::lmb_prediction_step;

#[test]
fn test_murty_marginals_seed42_first_timestep() {
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

    // Start with birth objects
    let mut objects = model.object.clone();

    println!("\n=== Testing Murty Marginals at t=1 (first measurement update) ===");

    // First timestep
    let t = 0;
    objects = lmb_prediction_step(objects, &model, t + 1);

    println!("\nAfter prediction:");
    println!("  {} objects", objects.len());

    if !ground_truth_output.measurements[t].is_empty() {
        let association_result = generate_lmb_association_matrices(
            &objects,
            &ground_truth_output.measurements[t],
            &model,
        );

        println!("\nMeasurements: {}", ground_truth_output.measurements[t].len());
        println!("Association matrix L: {} × {}", association_result.gibbs.l.nrows(), association_result.gibbs.l.ncols());

        let (r, w, v) = lmb_murtys(&association_result, model.number_of_assignments);

        println!("\nMurty results:");
        println!("  K-best assignments: {} events", v.nrows());
        println!("  Objects: {}", r.len());

        println!("\nExistence probabilities r:");
        for (i, ri) in r.iter().enumerate() {
            if *ri > 0.001 {
                println!("  Object {}: r = {:.10}", i, ri);
            }
        }

        println!("\nAssignment matrix V (first 10 events):");
        for k in 0..v.nrows().min(10) {
            print!("  Event {}: [", k);
            for j in 0..v.ncols().min(10) {
                print!("{}", v[(k, j)]);
                if j < v.ncols().min(10) - 1 {
                    print!(", ");
                }
            }
            println!("]");
        }

        println!("\nAssociation weights W (objects with r > 0.001):");
        for (i, ri) in r.iter().enumerate() {
            if *ri > 0.001 {
                print!("  Object {}: [", i);
                for j in 0..w.ncols().min(5) {
                    print!("{:.6}", w[(i, j)]);
                    if j < w.ncols().min(5) - 1 {
                        print!(", ");
                    }
                }
                println!("]");
            }
        }

        // Check for any r > 1.0 or r < 0
        let mut invalid_count = 0;
        for (i, ri) in r.iter().enumerate() {
            if *ri > 1.0 || *ri < 0.0 {
                println!("\n⚠ WARNING: Invalid r value at object {}: r = {}", i, ri);
                invalid_count += 1;
            }
            if (*ri - 1.0).abs() < 1e-10 && *ri > 0.999 {
                println!("  Note: Object {} has r very close to 1.0: r = {:.15}", i, ri);
            }
        }

        if invalid_count == 0 {
            println!("\n✓ All r values are valid (0 ≤ r ≤ 1)");
        }
    }
}
