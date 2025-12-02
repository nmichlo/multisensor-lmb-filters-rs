//! Association matrix building
//!
//! This module provides unified construction of association matrices
//! used by data association algorithms.

use nalgebra::{DMatrix, DVector};

use crate::types::{SensorModel, Track};

use super::likelihood::{compute_likelihood, LikelihoodWorkspace};

/// Grid of posterior parameters for all (track, measurement) pairs
#[derive(Debug, Clone)]
pub struct PosteriorGrid {
    /// Number of tracks
    pub num_tracks: usize,
    /// Number of measurements
    pub num_measurements: usize,
    /// Posterior means: indexed as [track_idx][measurement_idx]
    pub means: Vec<Vec<DVector<f64>>>,
    /// Posterior covariances
    pub covariances: Vec<Vec<DMatrix<f64>>>,
    /// Kalman gains
    pub kalman_gains: Vec<Vec<DMatrix<f64>>>,
}

impl PosteriorGrid {
    /// Create an empty posterior grid
    pub fn new(num_tracks: usize, num_measurements: usize) -> Self {
        Self {
            num_tracks,
            num_measurements,
            means: Vec::with_capacity(num_tracks),
            covariances: Vec::with_capacity(num_tracks),
            kalman_gains: Vec::with_capacity(num_tracks),
        }
    }

    /// Get posterior mean for (track, measurement) pair
    pub fn get_mean(&self, track_idx: usize, measurement_idx: usize) -> Option<&DVector<f64>> {
        self.means.get(track_idx)?.get(measurement_idx)
    }

    /// Get posterior covariance for (track, measurement) pair
    pub fn get_covariance(
        &self,
        track_idx: usize,
        measurement_idx: usize,
    ) -> Option<&DMatrix<f64>> {
        self.covariances.get(track_idx)?.get(measurement_idx)
    }
}

/// Association matrices used by data association algorithms
#[derive(Debug, Clone)]
pub struct AssociationMatrices {
    // LBP matrices
    /// Likelihood ratios: psi[i,j] = L[i,j] normalized
    pub psi: DMatrix<f64>,
    /// Miss probability factor: phi[i] = (1-p_D) / eta[i]
    pub phi: DVector<f64>,
    /// Normalization: eta[i] = 1 - p_D × r[i]
    pub eta: DVector<f64>,

    // Cost matrix for assignment algorithms
    /// Negative log-likelihood for assignment: cost[i,j] = -log(L[i,j])
    pub cost: DMatrix<f64>,

    // Gibbs sampling probabilities
    /// Sampling probabilities (row-normalized)
    pub sampling_prob: DMatrix<f64>,

    /// Posterior parameters for each (track, measurement) pair
    pub posteriors: PosteriorGrid,

    /// Log-likelihood ratios (raw values)
    pub log_likelihood_ratios: DMatrix<f64>,
}

impl AssociationMatrices {
    /// Get number of tracks
    #[inline]
    pub fn num_tracks(&self) -> usize {
        self.psi.nrows()
    }

    /// Get number of measurements
    #[inline]
    pub fn num_measurements(&self) -> usize {
        self.psi.ncols()
    }
}

/// Builder for constructing association matrices
///
/// Handles both single-sensor and multi-sensor cases.
pub struct AssociationBuilder<'a> {
    tracks: &'a [Track],
    sensor: &'a SensorModel,
    workspace: LikelihoodWorkspace,
}

impl<'a> AssociationBuilder<'a> {
    /// Create a new association builder
    pub fn new(tracks: &'a [Track], sensor: &'a SensorModel) -> Self {
        let x_dim = tracks.first().map(|t| t.x_dim()).unwrap_or(4);
        let z_dim = sensor.z_dim();
        Self {
            tracks,
            sensor,
            workspace: LikelihoodWorkspace::new(x_dim, z_dim),
        }
    }

    /// Build association matrices for single-sensor case
    pub fn build(&mut self, measurements: &[DVector<f64>]) -> AssociationMatrices {
        let n = self.tracks.len();
        let m = measurements.len();

        // Initialize matrices
        let mut log_likelihood_ratios = DMatrix::from_element(n, m, f64::NEG_INFINITY);
        let mut cost = DMatrix::from_element(n, m, f64::INFINITY);
        let mut posteriors = PosteriorGrid::new(n, m);

        let p_d = self.sensor.detection_probability;

        // Compute likelihoods for all (track, measurement) pairs
        for (i, track) in self.tracks.iter().enumerate() {
            let mut track_means = Vec::with_capacity(m);
            let mut track_covs = Vec::with_capacity(m);
            let mut track_gains = Vec::with_capacity(m);

            // Use the primary (first) component for likelihood computation
            let prior_mean = &track.components[0].mean;
            let prior_cov = &track.components[0].covariance;

            for (j, measurement) in measurements.iter().enumerate() {
                let result = compute_likelihood(
                    prior_mean,
                    prior_cov,
                    measurement,
                    self.sensor,
                    &mut self.workspace,
                );

                log_likelihood_ratios[(i, j)] = result.log_likelihood_ratio;
                cost[(i, j)] = -result.log_likelihood_ratio;

                track_means.push(result.posterior_mean);
                track_covs.push(result.posterior_covariance);
                track_gains.push(result.kalman_gain);
            }

            posteriors.means.push(track_means);
            posteriors.covariances.push(track_covs);
            posteriors.kalman_gains.push(track_gains);
        }

        // Compute LBP matrices
        let mut eta = DVector::zeros(n);
        let mut phi = DVector::zeros(n);
        let mut psi = DMatrix::zeros(n, m);

        for (i, track) in self.tracks.iter().enumerate() {
            let r = track.existence;

            // eta[i] = 1 - p_D × r[i]
            eta[i] = 1.0 - p_d * r;

            // phi[i] = r[i] × (1 - p_D) / eta[i]
            if eta[i].abs() > 1e-15 {
                phi[i] = r * (1.0 - p_d) / eta[i];
            } else {
                phi[i] = 0.0;
            }

            // psi[i,j] = r[i] × L[i,j] / eta[i]
            for j in 0..m {
                let lik_ratio = log_likelihood_ratios[(i, j)].exp();
                if eta[i].abs() > 1e-15 {
                    psi[(i, j)] = r * lik_ratio / eta[i];
                }
            }
        }

        // Compute sampling probabilities (for Gibbs)
        let mut sampling_prob = DMatrix::zeros(n, m + 1); // +1 for miss
        for i in 0..n {
            let miss_prob = phi[i];
            let mut row_sum = miss_prob;

            for j in 0..m {
                let prob = psi[(i, j)];
                sampling_prob[(i, j)] = prob;
                row_sum += prob;
            }

            // Normalize
            if row_sum > 1e-15 {
                for j in 0..m {
                    sampling_prob[(i, j)] /= row_sum;
                }
                sampling_prob[(i, m)] = miss_prob / row_sum; // miss column
            } else {
                sampling_prob[(i, m)] = 1.0; // If nothing likely, assume miss
            }
        }

        AssociationMatrices {
            psi,
            phi,
            eta,
            cost,
            sampling_prob,
            posteriors,
            log_likelihood_ratios,
        }
    }

    /// Build matrices for one sensor in multi-sensor case
    pub fn build_for_sensor(
        &mut self,
        measurements: &[DVector<f64>],
        _sensor_idx: usize,
    ) -> AssociationMatrices {
        // Same computation, sensor_idx can be used for logging/debugging
        self.build(measurements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrackLabel;
    use nalgebra::DMatrix;

    fn create_test_track() -> Track {
        Track::new(
            TrackLabel::new(0, 0),
            0.9,
            DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0]),
            DMatrix::identity(4, 4) * 10.0,
        )
    }

    fn create_test_sensor() -> SensorModel {
        SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0)
    }

    #[test]
    fn test_association_builder() {
        let tracks = vec![create_test_track()];
        let sensor = create_test_sensor();
        let mut builder = AssociationBuilder::new(&tracks, &sensor);

        let measurements = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![10.0, 10.0]),
        ];

        let matrices = builder.build(&measurements);

        assert_eq!(matrices.num_tracks(), 1);
        assert_eq!(matrices.num_measurements(), 2);

        // Close measurement should have higher psi
        assert!(matrices.psi[(0, 0)] > matrices.psi[(0, 1)]);
    }

    #[test]
    fn test_posterior_grid() {
        let tracks = vec![create_test_track()];
        let sensor = create_test_sensor();
        let mut builder = AssociationBuilder::new(&tracks, &sensor);

        let measurements = vec![DVector::from_vec(vec![0.5, 0.5])];

        let matrices = builder.build(&measurements);

        // Should have posterior for (0, 0)
        let posterior_mean = matrices.posteriors.get_mean(0, 0);
        assert!(posterior_mean.is_some());
    }

    #[test]
    fn test_sampling_probabilities_sum_to_one() {
        let tracks = vec![create_test_track()];
        let sensor = create_test_sensor();
        let mut builder = AssociationBuilder::new(&tracks, &sensor);

        let measurements = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![5.0, 5.0]),
        ];

        let matrices = builder.build(&measurements);

        // Each row of sampling_prob should sum to 1
        for i in 0..matrices.num_tracks() {
            let row_sum: f64 = (0..=matrices.num_measurements())
                .map(|j| matrices.sampling_prob[(i, j)])
                .sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sums to {}", i, row_sum);
        }
    }
}
