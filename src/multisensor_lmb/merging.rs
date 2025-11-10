//! LMB track merging algorithms
//!
//! Implements three fusion strategies for combining multi-sensor LMB estimates:
//! - AA (Arithmetic Average): Simple weighted combination
//! - GA (Geometric Average): Covariance intersection with canonical form
//! - PU (Parallel Update): Information form fusion with decorrelation
//!
//! Matches MATLAB aaLmbTrackMerging.m, gaLmbTrackMerging.m, and puLmbTrackMerging.m exactly.

use crate::common::types::Object;
use nalgebra::{DMatrix, DVector};

/// Arithmetic Average (AA) track merging
///
/// Fuses multi-sensor LMB estimates using simple weighted averaging.
///
/// # Arguments
/// * `sensor_objects` - Vector of LMB object sets from each sensor
/// * `number_of_sensors` - Number of sensors
///
/// # Returns
/// Fused LMB object set
///
/// # Implementation Notes
/// Matches MATLAB aaLmbTrackMerging.m exactly:
/// 1. Average existence probabilities: r_fused = mean(r_sensors)
/// 2. Average means: mu_fused = mean(mu_sensors)
/// 3. Average covariances: sigma_fused = mean(sigma_sensors)
/// 4. Keep GM components from first sensor
pub fn aa_lmb_track_merging(
    sensor_objects: &[Vec<Object>],
    number_of_sensors: usize,
) -> Vec<Object> {
    if sensor_objects.is_empty() || sensor_objects[0].is_empty() {
        return vec![];
    }

    let number_of_objects = sensor_objects[0].len();
    let mut fused_objects = sensor_objects[0].clone();

    // For each object
    for i in 0..number_of_objects {
        let num_gm = sensor_objects[0][i].number_of_gm_components;

        // Average existence probabilities
        let mut r_sum = 0.0;
        for s in 0..number_of_sensors {
            r_sum += sensor_objects[s][i].r;
        }
        fused_objects[i].r = r_sum / (number_of_sensors as f64);

        // Average means and covariances for each GM component
        for j in 0..num_gm {
            // Average means
            let mut mu_sum = DVector::zeros(sensor_objects[0][i].mu[j].len());
            for s in 0..number_of_sensors {
                mu_sum += &sensor_objects[s][i].mu[j];
            }
            fused_objects[i].mu[j] = mu_sum / (number_of_sensors as f64);

            // Average covariances
            let dim = sensor_objects[0][i].sigma[j].nrows();
            let mut sigma_sum = DMatrix::zeros(dim, dim);
            for s in 0..number_of_sensors {
                sigma_sum += &sensor_objects[s][i].sigma[j];
            }
            fused_objects[i].sigma[j] = sigma_sum / (number_of_sensors as f64);
        }
    }

    fused_objects
}

/// Geometric Average (GA) track merging
///
/// Fuses multi-sensor LMB estimates using covariance intersection via canonical form.
///
/// # Arguments
/// * `sensor_objects` - Vector of LMB object sets from each sensor
/// * `number_of_sensors` - Number of sensors
///
/// # Returns
/// Fused LMB object set
///
/// # Implementation Notes
/// Matches MATLAB gaLmbTrackMerging.m exactly:
/// 1. Convert to canonical form: K = Sigma^-1, h = K*mu, g = -0.5*mu'*K*mu
/// 2. Exponentiate canonical parameters by 1/S
/// 3. Convert back: Sigma = K^-1, mu = K^-1*h
/// 4. Geometric average of existence: r_fused = prod(r_sensors)^(1/S)
pub fn ga_lmb_track_merging(
    sensor_objects: &[Vec<Object>],
    number_of_sensors: usize,
) -> Vec<Object> {
    if sensor_objects.is_empty() || sensor_objects[0].is_empty() {
        return vec![];
    }

    let number_of_objects = sensor_objects[0].len();
    let mut fused_objects = sensor_objects[0].clone();
    let weight = 1.0 / (number_of_sensors as f64);

    // For each object
    for i in 0..number_of_objects {
        let num_gm = sensor_objects[0][i].number_of_gm_components;

        // Geometric average of existence probabilities
        let mut r_prod = 1.0;
        for s in 0..number_of_sensors {
            r_prod *= sensor_objects[s][i].r;
        }
        fused_objects[i].r = r_prod.powf(weight);

        // For each GM component
        for j in 0..num_gm {
            let dim = sensor_objects[0][i].sigma[j].nrows();

            // Initialize canonical form sums
            let mut k_sum = DMatrix::zeros(dim, dim);
            let mut h_sum = DVector::zeros(dim);
            let mut _g_sum = 0.0;

            // Convert each sensor to canonical form and weight
            for s in 0..number_of_sensors {
                // K = Sigma^-1
                let k = sensor_objects[s][i].sigma[j]
                    .clone()
                    .try_inverse()
                    .unwrap_or_else(|| {
                        let svd = sensor_objects[s][i].sigma[j].clone().svd(true, true);
                        svd.pseudo_inverse(1e-10).unwrap()
                    });

                // h = K * mu
                let h = &k * &sensor_objects[s][i].mu[j];

                // g = -0.5 * mu' * K * mu
                let g = -0.5 * sensor_objects[s][i].mu[j].dot(&h);

                // Weight by 1/S and accumulate
                k_sum += k * weight;
                h_sum += h * weight;
                _g_sum += g * weight;
            }

            // Convert back from canonical form
            // Sigma = K^-1
            let sigma_fused = k_sum.clone().try_inverse().unwrap_or_else(|| {
                let svd = k_sum.svd(true, true);
                svd.pseudo_inverse(1e-10).unwrap()
            });

            // mu = Sigma * h = K^-1 * h
            let mu_fused = &sigma_fused * &h_sum;

            fused_objects[i].mu[j] = mu_fused;
            fused_objects[i].sigma[j] = sigma_fused;
        }
    }

    fused_objects
}

/// Parallel Update (PU) track merging
///
/// Fuses multi-sensor LMB estimates using information form with decorrelation.
///
/// # Arguments
/// * `sensor_objects` - Vector of LMB object sets from each sensor
/// * `prior_objects` - Prior LMB object set (for decorrelation)
/// * `number_of_sensors` - Number of sensors
///
/// # Returns
/// Fused LMB object set
///
/// # Implementation Notes
/// Matches MATLAB puLmbTrackMerging.m exactly:
/// 1. Convert prior and posteriors to canonical form
/// 2. Information fusion: K_fused = sum(K_sensor) - (S-1)*K_prior
/// 3. Convert back to moment form
/// 4. Existence fusion: r_fused = 1 - prod(1-r_sensor)
pub fn pu_lmb_track_merging(
    sensor_objects: &[Vec<Object>],
    prior_objects: &[Object],
    number_of_sensors: usize,
) -> Vec<Object> {
    if sensor_objects.is_empty() || sensor_objects[0].is_empty() {
        return vec![];
    }

    let number_of_objects = sensor_objects[0].len();
    let mut fused_objects = sensor_objects[0].clone();

    // For each object
    for i in 0..number_of_objects {
        let num_gm = sensor_objects[0][i].number_of_gm_components;

        // Existence fusion: r_fused = 1 - prod(1 - r_sensor)
        let mut r_product = 1.0;
        for s in 0..number_of_sensors {
            r_product *= 1.0 - sensor_objects[s][i].r;
        }
        fused_objects[i].r = 1.0 - r_product;

        // For each GM component
        for j in 0..num_gm {
            let dim = sensor_objects[0][i].sigma[j].nrows();

            // Prior canonical form
            let k_prior = prior_objects[i].sigma[j]
                .clone()
                .try_inverse()
                .unwrap_or_else(|| {
                    let svd = prior_objects[i].sigma[j].clone().svd(true, true);
                    svd.pseudo_inverse(1e-10).unwrap()
                });
            let h_prior = &k_prior * &prior_objects[i].mu[j];

            // Initialize fused canonical parameters
            let mut k_fused = DMatrix::zeros(dim, dim);
            let mut h_fused = DVector::zeros(dim);

            // Accumulate sensor information
            for s in 0..number_of_sensors {
                // Sensor canonical form
                let k_sensor = sensor_objects[s][i].sigma[j]
                    .clone()
                    .try_inverse()
                    .unwrap_or_else(|| {
                        let svd = sensor_objects[s][i].sigma[j].clone().svd(true, true);
                        svd.pseudo_inverse(1e-10).unwrap()
                    });
                let h_sensor = &k_sensor * &sensor_objects[s][i].mu[j];

                k_fused += &k_sensor;
                h_fused += &h_sensor;
            }

            // Decorrelation: subtract (S-1) * prior
            let decorr = (number_of_sensors - 1) as f64;
            k_fused -= &k_prior * decorr;
            h_fused -= &h_prior * decorr;

            // Convert back to moment form
            let sigma_fused = k_fused.clone().try_inverse().unwrap_or_else(|| {
                let svd = k_fused.svd(true, true);
                svd.pseudo_inverse(1e-10).unwrap()
            });
            let mu_fused = &sigma_fused * &h_fused;

            fused_objects[i].mu[j] = mu_fused;
            fused_objects[i].sigma[j] = sigma_fused;
        }
    }

    fused_objects
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::model::generate_model;
    use crate::common::types::{DataAssociationMethod, ScenarioType};

    #[test]
    fn test_aa_lmb_track_merging() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let objects1 = model.birth_parameters.clone();
        let mut objects2 = model.birth_parameters.clone();

        // Modify second sensor slightly
        for obj in &mut objects2 {
            obj.r *= 0.9;
        }

        let sensor_objects = vec![objects1.clone(), objects2];
        let fused = aa_lmb_track_merging(&sensor_objects, 2);

        // Check dimensions
        assert_eq!(fused.len(), objects1.len());

        // Check that existence probabilities are averaged
        for i in 0..fused.len() {
            let expected_r = (objects1[i].r + objects1[i].r * 0.9) / 2.0;
            assert!((fused[i].r - expected_r).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ga_lmb_track_merging() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let objects1 = model.birth_parameters.clone();
        let mut objects2 = model.birth_parameters.clone();

        // Modify second sensor
        for obj in &mut objects2 {
            obj.r *= 0.8;
        }

        let sensor_objects = vec![objects1.clone(), objects2];
        let fused = ga_lmb_track_merging(&sensor_objects, 2);

        // Check dimensions
        assert_eq!(fused.len(), objects1.len());

        // Check that existence probabilities use geometric average
        for i in 0..fused.len() {
            let expected_r = (objects1[i].r * objects1[i].r * 0.8).sqrt();
            assert!((fused[i].r - expected_r).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pu_lmb_track_merging() {
        let model = generate_model(
            10.0,
            0.9,
            DataAssociationMethod::Gibbs,
            ScenarioType::Fixed,
            None,
        );

        let prior_objects = model.birth_parameters.clone();
        let mut objects1 = model.birth_parameters.clone();
        let mut objects2 = model.birth_parameters.clone();

        // Modify sensor estimates
        for obj in &mut objects1 {
            obj.r = 0.7;
        }
        for obj in &mut objects2 {
            obj.r = 0.6;
        }

        let sensor_objects = vec![objects1, objects2];
        let fused = pu_lmb_track_merging(&sensor_objects, &prior_objects, 2);

        // Check dimensions
        assert_eq!(fused.len(), prior_objects.len());

        // Check existence fusion: r = 1 - (1-0.7)*(1-0.6) = 1 - 0.12 = 0.88
        for obj in &fused {
            let expected_r = 1.0 - (1.0 - 0.7) * (1.0 - 0.6);
            assert!((obj.r - expected_r).abs() < 1e-10);
        }
    }
}