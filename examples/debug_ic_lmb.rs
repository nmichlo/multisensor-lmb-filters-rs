use prak::common::ground_truth::generate_multisensor_ground_truth;
use prak::common::metrics::ospa;
use prak::common::model::generate_multisensor_model;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ParallelUpdateMode, ScenarioType};
use prak::multisensor_lmb::iterated_corrector::run_ic_lmb_filter;

fn main() {
    let seed = 42;
    
    // Generate model
    let mut model_rng = SimpleRng::new(0);
    const NUMBER_OF_SENSORS: usize = 3;
    let model = generate_multisensor_model(
        &mut model_rng,
        NUMBER_OF_SENSORS,
        vec![5.0, 5.0, 5.0],
        vec![0.67, 0.70, 0.73],
        vec![4.0, 3.0, 2.0],
        ParallelUpdateMode::PU,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );
    
    println!("Model detection probs: {:?}", model.detection_probability_multisensor);
    println!("Birth parameters:");
    println!("  Number of birth objects: {}", model.birth_parameters.len());
    for (i, obj) in model.birth_parameters.iter().enumerate() {
        println!("    Birth {}: r={:.3}, birthTime={}, birthLoc={}",
            i, obj.r, obj.birth_time, obj.birth_location);
        if !obj.mu.is_empty() {
            println!("      mu[0]: [{:.3}, {:.3}, {:.3}, {:.3}]",
                obj.mu[0][0], obj.mu[0][1], obj.mu[0][2], obj.mu[0][3]);
        }
    }
    
    // Generate ground truth
    let mut trial_rng = SimpleRng::new(seed);
    let ground_truth_output = generate_multisensor_ground_truth(&mut trial_rng, &model, None);
    
    println!("\nGround truth at t=0:");
    println!("  Number of objects: {}", ground_truth_output.ground_truth_rfs.x[0].len());
    for (i, x) in ground_truth_output.ground_truth_rfs.x[0].iter().enumerate() {
        println!("    Object {}: [{:.3}, {:.3}, {:.3}, {:.3}]", i, x[0], x[1], x[2], x[3]);
    }
    
    println!("\nMeasurements at t=0:");
    for s in 0..NUMBER_OF_SENSORS {
        println!("  Sensor {}: {} measurements", s, ground_truth_output.measurements[s][0].len());
        for (i, m) in ground_truth_output.measurements[s][0].iter().enumerate() {
            println!("    Meas {}: [{:.3}, {:.3}]", i, m[0], m[1]);
        }
    }
    
    // Run IC-LMB filter
    let mut filter_rng = SimpleRng::new(seed + 1000);
    let state_estimates = run_ic_lmb_filter(
        &mut filter_rng,
        &model,
        &ground_truth_output.measurements,
        NUMBER_OF_SENSORS,
    );
    
    println!("\nState estimates at t=0:");
    println!("  Number of estimates: {}", state_estimates.mu[0].len());
    for (i, mu) in state_estimates.mu[0].iter().enumerate() {
        println!("    Est {}: [{:.3}, {:.3}, {:.3}, {:.3}]", i, mu[0], mu[1], mu[2], mu[3]);
    }
    
    // Compute OSPA for first few timesteps
    for t in 0..3 {
        let metrics = ospa(
            &ground_truth_output.ground_truth_rfs.x[t],
            &ground_truth_output.ground_truth_rfs.mu[t],
            &ground_truth_output.ground_truth_rfs.sigma[t],
            &state_estimates.mu[t],
            &state_estimates.sigma[t],
            &model.ospa_parameters,
        );

        println!("\nOSPA at t={}:", t);
        println!("  E-OSPA: {:.6}", metrics.e_ospa.total);
        println!("  H-OSPA: {:.6}", metrics.h_ospa.total);
        println!("  Ground truth objects: {}", ground_truth_output.ground_truth_rfs.x[t].len());
        println!("  State estimates: {}", state_estimates.mu[t].len());
    }
    println!("\nExpected (from MATLAB):");
    println!("  t=0: E-OSPA=4.057993");
    println!("  t=1: E-OSPA=4.829382");
    println!("  t=2: E-OSPA=5.000000");
}
