//! Accuracy trial tests - Compare filter variants
//!
//! Simplified version focused on verifying filter correctness and OSPA metrics.
//! Based on MATLAB singleSensorAccuracyTrial.m and multiSensorAccuracyTrial.m.

use prak::common::ground_truth::{generate_ground_truth, generate_multisensor_ground_truth};
use prak::common::metrics::ospa;
use prak::common::model::{generate_model, generate_multisensor_model};
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ParallelUpdateMode, ScenarioType};
use prak::lmb::filter::run_lmb_filter;
use prak::lmbm::filter::run_lmbm_filter;
use prak::multisensor_lmb::iterated_corrector::run_ic_lmb_filter;
use prak::multisensor_lmb::parallel_update::run_parallel_update_lmb_filter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_sensor_lmb_lbp_small() {
        // Small accuracy trial for LMB with LBP
        let num_trials = 3;
        let mut rng = SimpleRng::new(42);

        for trial in 0..num_trials {
            println!("LMB-LBP Trial {}/{}", trial + 1, num_trials);

            let model = generate_model(
                &mut rng,
                10.0,
                0.95,
                DataAssociationMethod::LBP,
                ScenarioType::Fixed,
                None,
            );

            let ground_truth_output = generate_ground_truth(&mut rng, &model, None);
            let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth_output.measurements);

            // Compute OSPA for each timestep
            let simulation_length = ground_truth_output.measurements.len();
            for t in 0..simulation_length {
                let metrics = ospa(
                    &ground_truth_output.ground_truth_rfs.x[t],
                    &ground_truth_output.ground_truth_rfs.mu[t],
                    &ground_truth_output.ground_truth_rfs.sigma[t],
                    &state_estimates.mu[t],
                    &state_estimates.sigma[t],
                    &model.ospa_parameters,
                );

                // Verify OSPA is reasonable
                assert!(
                    metrics.e_ospa.total.is_finite() && metrics.e_ospa.total >= 0.0,
                    "E-OSPA at t={} is invalid: {}",
                    t,
                    metrics.e_ospa.total
                );
                assert!(
                    metrics.e_ospa.total <= model.ospa_parameters.e_c,
                    "E-OSPA at t={} exceeds cutoff: {} > {}",
                    t,
                    metrics.e_ospa.total,
                    model.ospa_parameters.e_c
                );
            }
        }
    }

    #[test]
    fn test_single_sensor_determinism() {
        // Verify same seed produces same OSPA
        let run_trial = |seed: u64| -> Vec<f64> {
            let mut rng = SimpleRng::new(seed);
            let model = generate_model(
                &mut rng,
                10.0,
                0.95,
                DataAssociationMethod::LBP,
                ScenarioType::Fixed,
                None,
            );

            let ground_truth_output = generate_ground_truth(&mut rng, &model, None);
            let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth_output.measurements);

            // Compute E-OSPA for each timestep
            let mut e_ospa = Vec::new();
            let simulation_length = ground_truth_output.measurements.len();
            for t in 0..simulation_length {
                let metrics = ospa(
                    &ground_truth_output.ground_truth_rfs.x[t],
                    &ground_truth_output.ground_truth_rfs.mu[t],
                    &ground_truth_output.ground_truth_rfs.sigma[t],
                    &state_estimates.mu[t],
                    &state_estimates.sigma[t],
                    &model.ospa_parameters,
                );
                e_ospa.push(metrics.e_ospa.total);
            }
            e_ospa
        };

        let e1 = run_trial(12345);
        let e2 = run_trial(12345);

        assert_eq!(e1.len(), e2.len());
        for t in 0..e1.len() {
            assert!(
                (e1[t] - e2[t]).abs() < 1e-10,
                "E-OSPA differs at t={}: {} vs {}",
                t,
                e1[t],
                e2[t]
            );
        }
    }

    #[test]
    #[ignore] // Computationally expensive
    fn test_single_sensor_all_lmb_methods() {
        // Test all LMB data association methods
        let methods = vec![
            DataAssociationMethod::LBP,
            DataAssociationMethod::Gibbs,
            DataAssociationMethod::Murty,
        ];

        for method in &methods {
            println!("\nTesting LMB with {:?}", method);
            let mut rng = SimpleRng::new(42);

            let model = generate_model(&mut rng, 10.0, 0.95, *method, ScenarioType::Fixed, None);
            let ground_truth_output = generate_ground_truth(&mut rng, &model, None);
            let state_estimates = run_lmb_filter(&mut rng, &model, &ground_truth_output.measurements);

            // Check a few timesteps
            for t in 0..10.min(ground_truth_output.measurements.len()) {
                let metrics = ospa(
                    &ground_truth_output.ground_truth_rfs.x[t],
                    &ground_truth_output.ground_truth_rfs.mu[t],
                    &ground_truth_output.ground_truth_rfs.sigma[t],
                    &state_estimates.mu[t],
                    &state_estimates.sigma[t],
                    &model.ospa_parameters,
                );

                assert!(
                    metrics.e_ospa.total.is_finite() && metrics.e_ospa.total >= 0.0,
                    "{:?} LMB: E-OSPA at t={} is invalid: {}",
                    method,
                    t,
                    metrics.e_ospa.total
                );
            }
        }
    }

    #[test]
    #[ignore] // Computationally expensive
    fn test_single_sensor_lmbm_methods() {
        // Test LMBM data association methods
        let methods = vec![DataAssociationMethod::Gibbs, DataAssociationMethod::Murty];

        for method in &methods {
            println!("\nTesting LMBM with {:?}", method);
            let mut rng = SimpleRng::new(42);

            let model = generate_model(&mut rng, 10.0, 0.95, *method, ScenarioType::Fixed, None);
            let ground_truth_output = generate_ground_truth(&mut rng, &model, None);
            let state_estimates = run_lmbm_filter(&mut rng, &model, &ground_truth_output.measurements);

            // Check a few timesteps
            for t in 0..10.min(ground_truth_output.measurements.len()) {
                let metrics = ospa(
                    &ground_truth_output.ground_truth_rfs.x[t],
                    &ground_truth_output.ground_truth_rfs.mu[t],
                    &ground_truth_output.ground_truth_rfs.sigma[t],
                    &state_estimates.mu[t],
                    &state_estimates.sigma[t],
                    &model.ospa_parameters,
                );

                assert!(
                    metrics.e_ospa.total.is_finite() && metrics.e_ospa.total >= 0.0,
                    "{:?} LMBM: E-OSPA at t={} is invalid: {}",
                    method,
                    t,
                    metrics.e_ospa.total
                );
            }
        }
    }

    #[test]
    fn test_multisensor_ic_lmb_small() {
        // Test multi-sensor IC-LMB
        let mut rng = SimpleRng::new(42);

        let num_sensors = 3;
        let clutter_rates = vec![5.0, 5.0, 5.0];
        let detection_probs = vec![0.67, 0.7, 0.73];
        let measurement_noise_scales = vec![4.0, 3.0, 2.0];

        let model = generate_multisensor_model(
            &mut rng,
            num_sensors,
            clutter_rates,
            detection_probs,
            measurement_noise_scales,
            ParallelUpdateMode::PU,
            DataAssociationMethod::LBP,
            ScenarioType::Fixed,
            None,
        );

        let ground_truth_output = generate_multisensor_ground_truth(&mut rng, &model, None);
        let state_estimates = run_ic_lmb_filter(&mut rng, &model, &ground_truth_output.measurements, 0);

        // Check a few timesteps
        for t in 0..10.min(ground_truth_output.measurements[0].len()) {
            let metrics = ospa(
                &ground_truth_output.ground_truth_rfs.x[t],
                &ground_truth_output.ground_truth_rfs.mu[t],
                &ground_truth_output.ground_truth_rfs.sigma[t],
                &state_estimates.mu[t],
                &state_estimates.sigma[t],
                &model.ospa_parameters,
            );

            assert!(
                metrics.e_ospa.total.is_finite() && metrics.e_ospa.total >= 0.0,
                "IC-LMB: E-OSPA at t={} is invalid: {}",
                t,
                metrics.e_ospa.total
            );
        }

        println!("IC-LMB test completed successfully");
    }

    #[test]
    #[ignore] // Computationally expensive
    fn test_multisensor_all_lmb_methods() {
        // Test all multi-sensor LMB update modes
        let modes = vec![
            ("IC", true),
            ("PU", false),
            ("GA", false),
            ("AA", false),
        ];

        for (name, is_ic) in &modes {
            println!("\nTesting multi-sensor LMB with {}", name);
            let mut rng = SimpleRng::new(42);

            let num_sensors = 3;
            let clutter_rates = vec![5.0, 5.0, 5.0];
            let detection_probs = vec![0.67, 0.7, 0.73];
            let measurement_noise_scales = vec![4.0, 3.0, 2.0];

            let update_mode = match *name {
                "PU" => ParallelUpdateMode::PU,
                "GA" => ParallelUpdateMode::GA,
                "AA" => ParallelUpdateMode::AA,
                _ => ParallelUpdateMode::PU,
            };

            let model = generate_multisensor_model(
                &mut rng,
                num_sensors,
                clutter_rates,
                detection_probs,
                measurement_noise_scales,
                update_mode,
                DataAssociationMethod::LBP,
                ScenarioType::Fixed,
                None,
            );

            let ground_truth_output = generate_multisensor_ground_truth(&mut rng, &model, None);

            let state_estimates = if *is_ic {
                run_ic_lmb_filter(&mut rng, &model, &ground_truth_output.measurements, 0)
            } else {
                run_parallel_update_lmb_filter(&mut rng, &model, &ground_truth_output.measurements, num_sensors, update_mode)
            };

            // Check a few timesteps
            for t in 0..10.min(ground_truth_output.measurements[0].len()) {
                let metrics = ospa(
                    &ground_truth_output.ground_truth_rfs.x[t],
                    &ground_truth_output.ground_truth_rfs.mu[t],
                    &ground_truth_output.ground_truth_rfs.sigma[t],
                    &state_estimates.mu[t],
                    &state_estimates.sigma[t],
                    &model.ospa_parameters,
                );

                assert!(
                    metrics.e_ospa.total.is_finite() && metrics.e_ospa.total >= 0.0,
                    "{} multi-LMB: E-OSPA at t={} is invalid: {}",
                    name,
                    t,
                    metrics.e_ospa.total
                );
            }
        }
    }
}
