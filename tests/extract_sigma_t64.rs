//! Extract Sigma matrix at t=64 for debugging
use prak::common::ground_truth::generate_ground_truth;
use prak::common::model::generate_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::association::generate_lmb_association_matrices;
use prak::lmb::prediction::lmb_prediction_step;
use prak::lmb::update::compute_posterior_lmb_spatial_distributions;
use prak::common::association::murtys::murtys_algorithm_wrapper;
use prak::common::types::DMatrix;

#[test]
fn extract_sigma_t64() {
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

            if t == 64 {
                println!("\n=== Rust t=64 (0-indexed) ===");
                println!("Number of objects before gating: {}", objects.len());
                println!("Number of measurements: {}", ground_truth_output.measurements[t].len());

                // Re-run Murty's core computation to extract Sigma
                let n = association_result.cost.nrows();
                let m = association_result.cost.ncols();

                // Run Murty's algorithm
                let murtys_result = murtys_algorithm_wrapper(&association_result.cost, model.number_of_assignments);
                let v = murtys_result.assignments;
                let k = v.nrows();

                // Determine marginal distributions (from data_association.rs lines 98-150)
                let mut w_indicator = vec![DMatrix::zeros((k, n)); m + 1];
                for meas_idx in 0..=m {
                    for i in 0..k {
                        for j in 0..n {
                            if v[(i, j)] == meas_idx {
                                w_indicator[meas_idx][(i, j)] = 1.0;
                            }
                        }
                    }
                }

                let mut j_matrix = DMatrix::zeros((k, n));
                for i in 0..k {
                    for obj_idx in 0..n {
                        let meas_idx = v[(i, obj_idx)];
                        j_matrix[(i, obj_idx)] = association_result.gibbs.l[(obj_idx, meas_idx)];
                    }
                }

                let mut l_marg = Vec::with_capacity(m + 1);
                for meas_idx in 0..=m {
                    let mut l_col = DVector::zeros(n);
                    for obj_idx in 0..n {
                        let mut sum = 0.0;
                        for event_idx in 0..k {
                            let mut prod = 1.0;
                            for j in 0..n {
                                prod *= j_matrix[(event_idx, j)];
                            }
                            sum += prod * w_indicator[meas_idx][(event_idx, obj_idx)];
                        }
                        l_col[obj_idx] = sum;
                    }
                    l_marg.push(l_col);
                }

                let mut sigma = DMatrix::zeros(n, m + 1);
                for obj_idx in 0..n {
                    for meas_idx in 0..=m {
                        sigma[(obj_idx, meas_idx)] = l_marg[meas_idx][obj_idx];
                    }
                }

                println!("\n=== Sigma matrix (first 5 objects, all measurements) ===");
                for i in 0..n.min(5) {
                    print!("Sigma[{},:] = [", i);
                    for j in 0..=m {
                        print!("{:.17}", sigma[(i, j)]);
                        if j < m {
                            print!(", ");
                        }
                    }
                    println!("]");
                }

                println!("\n=== R matrix (first 5 objects) ===");
                for i in 0..n.min(5) {
                    print!("R[{},:] = [", i);
                    for j in 0..=m {
                        print!("{:.17}", association_result.gibbs.r[(i, j)]);
                        if j < m {
                            print!(", ");
                        }
                    }
                    println!("]");
                }

                println!("\n=== Tau = (Sigma .* R) ./ sum(Sigma, 2) ===");
                let mut tau = DMatrix::zeros(n, m + 1);
                for obj_idx in 0..n {
                    let row_sum: f64 = sigma.row(obj_idx).sum();
                    if row_sum > 1e-15 {
                        for meas_idx in 0..=m {
                            tau[(obj_idx, meas_idx)] =
                                (sigma[(obj_idx, meas_idx)] * association_result.gibbs.r[(obj_idx, meas_idx)])
                                    / row_sum;
                        }
                    }
                }
                for i in 0..n.min(5) {
                    print!("Tau[{},:] = [", i);
                    for j in 0..=m {
                        print!("{:.17}", tau[(i, j)]);
                        if j < m {
                            print!(", ");
                        }
                    }
                    println!("]");
                }

                println!("\n=== r = sum(Tau, 2) ===");
                for i in 0..n.min(5) {
                    let r_val: f64 = tau.row(i).sum();
                    println!("r[{}] = {:.17} (r == 1.0: {})", i, r_val, r_val == 1.0);
                }
            }

            // Continue with normal filter flow
            let (r, w, _v) = prak::lmb::data_association::lmb_murtys(&association_result, model.number_of_assignments);
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
