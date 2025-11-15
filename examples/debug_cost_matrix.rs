use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::association::generate_lmb_association_matrices;
use prak::lmb::prediction::lmb_prediction_step;
use prak::lmb::data_association::lmb_murtys;
use prak::lmb::update::compute_posterior_lmb_spatial_distributions;
use prak::common::association::murtys::murtys_algorithm_wrapper;

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
    let mut objects = model.birth_parameters.clone();

    // Run to timestep 25
    for t in 0..=25 {
        objects = lmb_prediction_step(objects, &model, t + 1);

        if !ground_truth_output.measurements[t].is_empty() {
            let association_result =
                generate_lmb_association_matrices(&objects, &ground_truth_output.measurements[t], &model);

            if t == 25 {
                println!("=== Timestep 25 ===");
                println!("Number of objects: {}", objects.len());
                println!("Number of measurements: {}", ground_truth_output.measurements[t].len());
                println!("\nCost matrix (n={}, m={}):", association_result.cost.nrows(), association_result.cost.ncols());

                for i in 0..association_result.cost.nrows() {
                    print!("Object {}: ", i);
                    for j in 0..association_result.cost.ncols() {
                        let c = association_result.cost[(i, j)];
                        if c > 100.0 {
                            print!("  {:>10.2} (INF)", c);
                        } else {
                            print!("  {:>10.6}", c);
                        }
                    }
                    println!();
                }

                println!("\nRunning Murty's algorithm...");
                let murty_result = murtys_algorithm_wrapper(&association_result.cost, model.number_of_assignments);
                println!("Number of assignments returned: {}", murty_result.assignments.nrows());
                println!("\nFirst 5 assignments:");
                for k in 0..murty_result.assignments.nrows().min(5) {
                    print!("  Event {}: ", k);
                    for obj_idx in 0..murty_result.assignments.ncols() {
                        print!("{} ", murty_result.assignments[(k, obj_idx)]);
                    }
                    println!();
                }
            }

            let (r, w, _v) = lmb_murtys(&association_result, model.number_of_assignments);

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
}
