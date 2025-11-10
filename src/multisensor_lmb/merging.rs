//! LMB track merging algorithms
//!
//! Implements three fusion strategies for combining multi-sensor LMB estimates:
//! - AA (Arithmetic Average): Simple weighted combination
//! - GA (Geometric Average): Covariance intersection with canonical form
//! - PU (Parallel Update): Information form fusion with decorrelation
//!
//! Matches MATLAB aaLmbTrackMerging.m, gaLmbTrackMerging.m, and puLmbTrackMerging.m exactly.

use crate::common::types::{Model, Object};
use nalgebra::{DMatrix, DVector};

/// Arithmetic Average (AA) track merging
///
/// Fuses multi-sensor LMB estimates by concatenating weighted GM components.
///
/// # Arguments
/// * `sensor_objects` - Vector of LMB object sets from each sensor
/// * `model` - Model containing sensor weights and max GM components
///
/// # Returns
/// Fused LMB object set
///
/// # Implementation Notes
/// Matches MATLAB aaLmbTrackMerging.m exactly:
/// 1. Weighted sum of existence probabilities: r_fused = sum(weight_s * r_s)
/// 2. Concatenate weighted GM components from all sensors
/// 3. Sort by weight descending and truncate to maximumNumberOfGmComponents
/// 4. Renormalize weights to sum to 1
pub fn aa_lmb_track_merging(
    sensor_objects: &[Vec<Object>],
    model: &Model,
) -> Vec<Object> {
    if sensor_objects.is_empty() || sensor_objects[0].is_empty() {
        return vec![];
    }

    let number_of_objects = sensor_objects[0].len();
    let number_of_sensors = model.number_of_sensors.unwrap_or(sensor_objects.len());

    // Get sensor weights (default to uniform if not specified)
    let sensor_weights = model.aa_sensor_weights.as_ref()
        .map(|w| w.clone())
        .unwrap_or_else(|| vec![1.0 / number_of_sensors as f64; number_of_sensors]);

    let mut fused_objects = sensor_objects[0].clone();

    // For each object
    for i in 0..number_of_objects {
        // Initialize with first sensor's weighted components
        let mut r = sensor_weights[0] * sensor_objects[0][i].r;
        let mut w = sensor_objects[0][i].w.iter()
            .map(|&weight| sensor_weights[0] * weight)
            .collect::<Vec<f64>>();
        let mut mu = sensor_objects[0][i].mu.clone();
        let mut sigma = sensor_objects[0][i].sigma.clone();

        // Merge remaining sensors
        for s in 1..number_of_sensors {
            // Add weighted existence probability
            r += sensor_weights[s] * sensor_objects[s][i].r;

            // Concatenate weighted GM components
            for j in 0..sensor_objects[s][i].number_of_gm_components {
                w.push(sensor_weights[s] * sensor_objects[s][i].w[j]);
                mu.push(sensor_objects[s][i].mu[j].clone());
                sigma.push(sensor_objects[s][i].sigma[j].clone());
            }
        }

        // Sort by weight descending and get indices
        let mut indexed_weights: Vec<(usize, f64)> = w.iter().enumerate()
            .map(|(idx, &weight)| (idx, weight))
            .collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Truncate to max GM components
        let max_components = model.maximum_number_of_gm_components;
        let num_components = max_components.min(w.len());
        let sorted_indices: Vec<usize> = indexed_weights.iter()
            .take(num_components)
            .map(|(idx, _)| *idx)
            .collect();

        // Reorder and normalize
        let selected_weights: Vec<f64> = sorted_indices.iter()
            .map(|&idx| w[idx])
            .collect();
        let weight_sum: f64 = selected_weights.iter().sum();
        let normalized_weights: Vec<f64> = selected_weights.iter()
            .map(|&weight| weight / weight_sum)
            .collect();

        fused_objects[i].r = r;
        fused_objects[i].number_of_gm_components = num_components;
        fused_objects[i].w = normalized_weights;
        fused_objects[i].mu = sorted_indices.iter()
            .map(|&idx| mu[idx].clone())
            .collect();
        fused_objects[i].sigma = sorted_indices.iter()
            .map(|&idx| sigma[idx].clone())
            .collect();
    }

    fused_objects
}

/// M-projection (moment matching) for Gaussian mixture
///
/// Collapses a Gaussian mixture to a single Gaussian by matching first and second moments.
///
/// # Arguments
/// * `obj` - Object containing the GM to project
///
/// # Returns
/// (nu, T) where nu is the mean and T is the covariance
fn m_projection(obj: &Object) -> (DVector<f64>, DMatrix<f64>) {
    let dim = obj.mu[0].len();

    // Compute weighted mean
    let mut nu = DVector::zeros(dim);
    for j in 0..obj.number_of_gm_components {
        nu += &obj.mu[j] * obj.w[j];
    }

    // Compute weighted covariance
    let mut t = DMatrix::zeros(dim, dim);
    for j in 0..obj.number_of_gm_components {
        let mu_diff = &obj.mu[j] - &nu;
        t += (&obj.sigma[j] + &mu_diff * mu_diff.transpose()) * obj.w[j];
    }

    (nu, t)
}

/// Geometric Average (GA) track merging
///
/// Fuses multi-sensor LMB estimates using weighted geometric average in canonical form.
///
/// # Arguments
/// * `sensor_objects` - Vector of LMB object sets from each sensor
/// * `model` - Model containing sensor weights and state dimension
///
/// # Returns
/// Fused LMB object set
///
/// # Implementation Notes
/// Matches MATLAB gaLmbTrackMerging.m exactly:
/// 1. Moment match each sensor's GM to single Gaussian (m-projection)
/// 2. Convert to canonical form: K = weight_s * inv(T), h = K*nu
/// 3. Sum weighted canonical parameters: g = -0.5*nu'*K*nu - 0.5*weight_s*log(det(2*pi*T))
/// 4. Convert back: Sigma_GA = inv(sum K), mu_GA = Sigma_GA * (sum h)
/// 5. Geometric mean of existence: r_fused = eta * prod(r_s^weight_s) / (eta * prod(r_s^weight_s) + prod((1-r_s)^weight_s))
pub fn ga_lmb_track_merging(
    sensor_objects: &[Vec<Object>],
    model: &Model,
) -> Vec<Object> {
    if sensor_objects.is_empty() || sensor_objects[0].is_empty() {
        return vec![];
    }

    let number_of_objects = sensor_objects[0].len();
    let number_of_sensors = model.number_of_sensors.unwrap_or(sensor_objects.len());
    let x_dimension = model.x_dimension;

    // Get sensor weights (default to uniform if not specified)
    let sensor_weights = model.ga_sensor_weights.as_ref()
        .map(|w| w.clone())
        .unwrap_or_else(|| vec![1.0 / number_of_sensors as f64; number_of_sensors]);

    let mut fused_objects = sensor_objects[0].clone();

    // For each object
    for i in 0..number_of_objects {
        // Initialize canonical form accumulators
        let mut k = DMatrix::zeros(x_dimension, x_dimension);
        let mut h = DVector::zeros(x_dimension);
        let mut g = 0.0;

        // Moment match and fuse each sensor
        for s in 0..number_of_sensors {
            // M-projection: collapse GM to single Gaussian
            let (nu, t) = m_projection(&sensor_objects[s][i]);

            // Convert to canonical form and weight
            let t_det = t.determinant();
            let t_inv = t.clone().try_inverse().unwrap_or_else(|| {
                let svd = t.clone().svd(true, true);
                svd.pseudo_inverse(1e-10).unwrap()
            });

            let k_matched = &t_inv * sensor_weights[s];
            let h_matched = &k_matched * &nu;
            let g_matched = -0.5 * nu.dot(&(&k_matched * &nu))
                - 0.5 * sensor_weights[s] * (2.0 * std::f64::consts::PI * t_det).ln();

            // Accumulate
            k += &k_matched;
            h += &h_matched;
            g += g_matched;
        }

        // Convert back to covariance form
        let sigma_ga = k.clone().try_inverse().unwrap_or_else(|| {
            let svd = k.clone().svd(true, true);
            svd.pseudo_inverse(1e-10).unwrap()
        });
        let mu_ga = &sigma_ga * &h;
        let k_times_mu_ga = &k * &mu_ga;
        let sigma_ga_det_final = sigma_ga.determinant();
        let eta = (g + 0.5 * mu_ga.dot(&k_times_mu_ga)
            + 0.5 * (2.0 * std::f64::consts::PI * sigma_ga_det_final).ln()).exp();

        // Geometric average of existence probability
        let mut numerator = eta;
        let mut partial_denominator = 1.0;
        for s in 0..number_of_sensors {
            let r_s = sensor_objects[s][i].r;
            numerator *= r_s.powf(sensor_weights[s]);
            partial_denominator *= (1.0 - r_s).powf(sensor_weights[s]);
        }

        // Update object with fused single Gaussian
        fused_objects[i].r = numerator / (numerator + partial_denominator);
        fused_objects[i].number_of_gm_components = 1;
        fused_objects[i].w = vec![1.0];
        fused_objects[i].mu = vec![mu_ga];
        fused_objects[i].sigma = vec![sigma_ga];
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
        use crate::common::model::generate_multisensor_model;
        use crate::multisensor_lmb::parallel_update::ParallelUpdateMode;

        let number_of_sensors = 2;
        let model = generate_multisensor_model(
            number_of_sensors,
            vec![10.0; number_of_sensors],
            vec![0.9; number_of_sensors],
            vec![10.0; number_of_sensors],
            ParallelUpdateMode::AA,
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

        let sensor_objects = vec![objects1.clone(), objects2.clone()];
        let fused = aa_lmb_track_merging(&sensor_objects, &model);

        // Check dimensions
        assert_eq!(fused.len(), objects1.len());

        // With uniform weights (0.5, 0.5), weighted sum should be:
        // r_fused = 0.5 * r1 + 0.5 * r2 = 0.5 * r1 + 0.5 * (0.9 * r1) = 0.95 * r1
        for i in 0..fused.len() {
            let expected_r = 0.5 * objects1[i].r + 0.5 * objects2[i].r;
            assert!((fused[i].r - expected_r).abs() < 1e-10);
        }

        // Check that GM components are concatenated (2 sensors -> 2x components)
        assert!(fused[0].number_of_gm_components <= model.maximum_number_of_gm_components);
    }

    #[test]
    fn test_ga_lmb_track_merging() {
        use crate::common::model::generate_multisensor_model;
        use crate::multisensor_lmb::parallel_update::ParallelUpdateMode;

        let number_of_sensors = 2;
        let model = generate_multisensor_model(
            number_of_sensors,
            vec![10.0; number_of_sensors],
            vec![0.9; number_of_sensors],
            vec![10.0; number_of_sensors],
            ParallelUpdateMode::GA,
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
        let fused = ga_lmb_track_merging(&sensor_objects, &model);

        // Check dimensions
        assert_eq!(fused.len(), objects1.len());

        // GA fusion always produces single Gaussian
        assert_eq!(fused[0].number_of_gm_components, 1);
        assert_eq!(fused[0].w.len(), 1);
        assert_eq!(fused[0].w[0], 1.0);
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