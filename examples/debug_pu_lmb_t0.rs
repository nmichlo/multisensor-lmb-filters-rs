/// Debug PU-LMB for seed 42 at t=0 - Match Octave fixture
use prak::common::ground_truth::generate_multisensor_ground_truth;
use prak::common::model::generate_multisensor_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ParallelUpdateMode, ScenarioType};
use prak::lmb::cardinality::lmb_map_cardinality_estimate;
use prak::lmb::prediction::lmb_prediction_step;
use prak::multisensor_lmb::association::generate_lmb_sensor_association_matrices;
use prak::multisensor_lmb::parallel_update::compute_posterior_lmb_spatial_distributions_multisensor;
use prak::multisensor_lmb::merging::pu_lmb_track_merging;
use prak::common::association::lbp::{loopy_belief_propagation, AssociationMatrices};

fn main() {
    println!("\n=== RUST DEBUG: PU-LMB seed 42 ===");

    // Generate model with seed 0
    let mut model_rng = SimpleRng::new(0);
    let number_of_sensors = 3;
    let model = generate_multisensor_model(
        &mut model_rng,
        number_of_sensors,
        vec![5.0, 5.0, 5.0],
        vec![0.67, 0.70, 0.73],
        vec![4.0, 3.0, 2.0],
        ParallelUpdateMode::PU,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    // Generate ground truth with trial seed
    let mut trial_rng = SimpleRng::new(42);
    let ground_truth_output = generate_multisensor_ground_truth(&mut trial_rng, &model, None);

    // Run filter with filter seed (not needed for LBP)
    let mut _filter_rng = SimpleRng::new(1042);
    let measurements = &ground_truth_output.measurements;

    // Initialize
    let mut objects = model.birth_parameters.clone();
    println!("Initial objects: {}", objects.len());

    // Run t=0 only (0-indexed in Rust, 1-indexed in Octave)
    let t = 0;
    println!("\n==================================");
    println!("TIMESTEP t={} (Octave t={})", t, t+1);
    println!("==================================");

    // Prediction
    objects = lmb_prediction_step(objects, &model, t + 1);

    println!("\n=== AFTER PREDICTION ===");
    println!("Number of objects: {}", objects.len());
    for (i, obj) in objects.iter().enumerate() {
        println!("obj {}: r_pred = {:.15}", i, obj.r);
    }

    // Save prior for PU merging
    let prior_objects = objects.clone();

    // Measurement update for each sensor
    let mut sensor_objects = Vec::with_capacity(number_of_sensors);

    for s in 0..number_of_sensors {
        let meas = &measurements[s][t];
        println!("\n=== SENSOR {} UPDATE ===", s + 1);
        println!("Number of measurements: {}", meas.len());

        if !meas.is_empty() {
            // Generate association matrices
            let (association_matrices, posterior_parameters) =
                generate_lmb_sensor_association_matrices(&objects, meas, &model, s);

            // LBP data association - convert to AssociationMatrices type
            let lbp_matrices = AssociationMatrices {
                psi: association_matrices.psi.clone(),
                phi: association_matrices.phi.clone(),
                eta: association_matrices.eta.clone(),
            };
            let lbp_result = loopy_belief_propagation(
                &lbp_matrices,
                model.lbp_convergence_tolerance,
                model.maximum_number_of_lbp_iterations,
            );
            let r = lbp_result.r.as_slice().to_vec();
            let w = lbp_result.w;

            println!("Posterior existence (r) from LBP:");
            for (i, &r_val) in r.iter().enumerate() {
                println!("  obj {}: r = {:.15}", i, r_val);
            }

            // Compute posterior spatial distributions
            let updated = compute_posterior_lmb_spatial_distributions_multisensor(
                objects.clone(),
                &r,
                &w,
                &posterior_parameters,
                &model,
            );

            println!("Measurement-updated distributions:");
            for (i, obj) in updated.iter().enumerate() {
                println!(
                    "  obj {}: r = {:.15}, GM components = {}",
                    i, obj.r, obj.number_of_gm_components
                );
                print!("    GM weights: [");
                for (j, &weight) in obj.w.iter().enumerate() {
                    print!("{:.6e}", weight);
                    if j < obj.w.len() - 1 {
                        print!(", ");
                    }
                }
                println!("]");
            }

            sensor_objects.push(updated);
        } else {
            // No measurements - missed detection update
            let mut updated = objects.clone();
            let pd_vec = model.detection_probability_multisensor.as_ref()
                .expect("Multi-sensor model should have detection_probability_multisensor");
            let pd = pd_vec[s];
            for obj in &mut updated {
                obj.r = (obj.r * (1.0 - pd)) / (1.0 - obj.r * pd);
            }
            sensor_objects.push(updated);
        }
    }

    println!("\n=== BEFORE PU MERGE ===");
    for (s, sensor_objs) in sensor_objects.iter().enumerate() {
        println!("Sensor {}:", s + 1);
        for (i, obj) in sensor_objs.iter().enumerate() {
            println!("  obj {}: r = {:.15}", i, obj.r);
        }
    }

    // PU Track Merging
    println!("\n=== PU TRACK MERGING ===");
    println!("Prior objects (for decorrelation):");
    for (i, obj) in prior_objects.iter().enumerate() {
        println!("  prior obj {}: r = {:.15}", i, obj.r);
    }

    let objects_merged = pu_lmb_track_merging(&sensor_objects, &prior_objects, number_of_sensors);

    println!("\n=== AFTER PU MERGE ===");
    for (i, obj) in objects_merged.iter().enumerate() {
        println!("obj {}: r_fused = {:.15}", i, obj.r);
    }

    // Gate tracks
    let existence_threshold = model.existence_threshold;
    let objects_gated: Vec<_> = objects_merged
        .into_iter()
        .filter(|obj| obj.r > existence_threshold)
        .collect();

    println!("\n=== AFTER GATING (threshold={:.3}) ===", existence_threshold);
    for (i, obj) in objects_gated.iter().enumerate() {
        println!("obj {}: r = {:.15}", i, obj.r);
    }

    // MAP cardinality extraction
    let r_values: Vec<f64> = objects_gated.iter().map(|obj| obj.r).collect();
    let (n_map, map_indices) = lmb_map_cardinality_estimate(&r_values);

    println!("\n=== MAP CARDINALITY ===");
    println!("nMap = {}", n_map);
    println!("Indices: {:?}", map_indices);

    println!("\n=== RUST DEBUG COMPLETE ===");
}
