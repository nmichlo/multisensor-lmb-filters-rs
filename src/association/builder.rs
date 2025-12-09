//! Association matrix construction for data association algorithms.
//!
//! This module builds the matrices needed by data association algorithms (LBP, Gibbs, Murty).
//! The key insight is that all algorithms need similar inputs: likelihood ratios between
//! track-measurement pairs, miss probabilities, and Kalman-updated posteriors.
//!
//! The [`AssociationBuilder`] computes these once, avoiding redundant likelihood evaluations.
//! The resulting [`AssociationMatrices`] can be passed to any [`Associator`](crate::filter::Associator).

use nalgebra::{DMatrix, DVector};

use crate::lmb::{SensorModel, Track};

use super::likelihood::{compute_likelihood, LikelihoodWorkspace};

/// Pre-computed Kalman posteriors for all (track, measurement) pairs.
///
/// When a track detects a measurement, the posterior state is computed via
/// Kalman update. Since we don't know which associations are correct, we
/// pre-compute posteriors for all possible pairings. The data association
/// algorithm then selects or weights these posteriors.
///
/// This grid stores `n_tracks × n_measurements × n_components` posterior means
/// and covariances, indexed as `[track_idx][measurement_idx][component_idx]`.
#[derive(Debug, Clone)]
pub struct PosteriorGrid {
    pub num_tracks: usize,
    pub num_measurements: usize,
    /// Kalman-updated posterior means for each (track, measurement, component) triple.
    /// Indexed as means[track][measurement][component].
    pub means: Vec<Vec<Vec<DVector<f64>>>>,
    /// Kalman-updated posterior covariances.
    /// Indexed as covariances[track][measurement][component].
    pub covariances: Vec<Vec<Vec<DMatrix<f64>>>>,
    /// Kalman gains (useful for some update strategies).
    /// Indexed as kalman_gains[track][measurement][component].
    pub kalman_gains: Vec<Vec<Vec<DMatrix<f64>>>>,
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

    /// Get posterior mean for (track, measurement) pair - returns first component for backwards compat.
    pub fn get_mean(&self, track_idx: usize, measurement_idx: usize) -> Option<&DVector<f64>> {
        self.means.get(track_idx)?.get(measurement_idx)?.first()
    }

    /// Get posterior covariance for (track, measurement) pair - returns first component.
    pub fn get_covariance(
        &self,
        track_idx: usize,
        measurement_idx: usize,
    ) -> Option<&DMatrix<f64>> {
        self.covariances
            .get(track_idx)?
            .get(measurement_idx)?
            .first()
    }

    /// Get posterior mean for specific (track, measurement, component) triple.
    pub fn get_mean_for_component(
        &self,
        track_idx: usize,
        measurement_idx: usize,
        component_idx: usize,
    ) -> Option<&DVector<f64>> {
        self.means
            .get(track_idx)?
            .get(measurement_idx)?
            .get(component_idx)
    }

    /// Get posterior covariance for specific (track, measurement, component) triple.
    pub fn get_covariance_for_component(
        &self,
        track_idx: usize,
        measurement_idx: usize,
        component_idx: usize,
    ) -> Option<&DMatrix<f64>> {
        self.covariances
            .get(track_idx)?
            .get(measurement_idx)?
            .get(component_idx)
    }

    /// Get number of components for a given track.
    pub fn num_components(&self, track_idx: usize) -> usize {
        self.means
            .get(track_idx)
            .and_then(|track| track.first())
            .map(|meas| meas.len())
            .unwrap_or(0)
    }
}

/// All matrices needed by data association algorithms.
///
/// Different association algorithms need the likelihood information in different forms:
///
/// - **LBP** uses `psi`, `phi`, `eta` for message passing on the factor graph
/// - **Gibbs** uses `sampling_prob` for conditional sampling
/// - **Murty** uses `cost` (negative log-likelihood) for the Hungarian algorithm
///
/// All algorithms use `posteriors` to get the Kalman-updated states.
///
/// The matrices encode the same underlying likelihood information, just transformed
/// for each algorithm's needs. Computing them together avoids redundant work.
#[derive(Debug, Clone)]
pub struct AssociationMatrices {
    /// LBP message matrix: `psi[i,j] = r[i] × L[i,j] / eta[i]`
    /// where L is the likelihood ratio and eta is a normalization factor.
    pub psi: DMatrix<f64>,

    /// LBP miss factor: `phi[i] = r[i] × (1 - p_D) / eta[i]`
    /// representing the relative probability of track i missing detection.
    pub phi: DVector<f64>,

    /// LBP normalization: `eta[i] = 1 - p_D × r[i]`
    /// used to normalize the LBP messages.
    pub eta: DVector<f64>,

    /// Cost matrix for assignment algorithms: `cost[i,j] = -log(L[i,j])`
    /// Murty/Hungarian minimize cost, which maximizes likelihood.
    pub cost: DMatrix<f64>,

    /// Row-normalized sampling probabilities for Gibbs (n × (m+1)).
    /// Last column is miss probability. Each row sums to 1.
    pub sampling_prob: DMatrix<f64>,

    /// Pre-computed Kalman posteriors for all (track, measurement) pairs.
    pub posteriors: PosteriorGrid,

    /// Raw log-likelihood ratios: `log(L[i,j])` for track i and measurement j.
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

/// Builds association matrices from tracks and measurements.
///
/// The builder pre-allocates workspace for likelihood computations and constructs
/// all the matrices needed by association algorithms in a single pass over the
/// (track, measurement) pairs.
///
/// For each pair, it computes:
/// 1. The likelihood ratio (how well the measurement fits the track prediction)
/// 2. The Kalman posterior (state estimate if this association is correct)
///
/// These are then transformed into the various matrix formats needed by LBP,
/// Gibbs, and Murty algorithms.
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

    /// Build association matrices from current tracks and given measurements.
    ///
    /// This is the main computation that evaluates all (track, measurement) pairs,
    /// computing likelihoods and Kalman posteriors. The results are structured
    /// for use by any association algorithm.
    ///
    /// For multi-component GM tracks, the likelihood is computed as a weighted sum
    /// over all components, matching MATLAB's generateLmbAssociationMatrices.m.
    pub fn build(&mut self, measurements: &[DVector<f64>]) -> AssociationMatrices {
        let n = self.tracks.len();
        let m = measurements.len();

        // Initialize matrices
        // L matrix accumulates likelihood ratios (sum over GM components)
        let mut l_matrix = DMatrix::zeros(n, m);
        let mut cost = DMatrix::from_element(n, m, f64::INFINITY);
        let mut posteriors = PosteriorGrid::new(n, m);

        let p_d = self.sensor.detection_probability;
        let _clutter_density = self.sensor.clutter_density();

        // Compute likelihoods for all (track, measurement) pairs
        // Matching MATLAB: iterate over ALL GM components and sum weighted likelihoods
        for (i, track) in self.tracks.iter().enumerate() {
            let num_components = track.components.len();

            // track_means[j][k] = posterior mean for measurement j, component k
            let mut track_means: Vec<Vec<DVector<f64>>> = Vec::with_capacity(m);
            let mut track_covs: Vec<Vec<DMatrix<f64>>> = Vec::with_capacity(m);
            let mut track_gains: Vec<Vec<DMatrix<f64>>> = Vec::with_capacity(m);

            // Initialize L values to 0 for accumulation
            for j in 0..m {
                l_matrix[(i, j)] = 0.0;
            }

            // Iterate over ALL GM components to compute marginal likelihood
            // Matching MATLAB generateLmbAssociationMatrices.m exactly:
            // L[i,j] = sum_k( r * p_D * w[k] / lambda * gaussian_likelihood[k] )
            let r = track.existence;

            // For each measurement, compute posteriors for ALL components
            for (j, measurement) in measurements.iter().enumerate() {
                let mut meas_means = Vec::with_capacity(num_components);
                let mut meas_covs = Vec::with_capacity(num_components);
                let mut meas_gains = Vec::with_capacity(num_components);

                for component in track.components.iter() {
                    let prior_mean = &component.mean;
                    let prior_cov = &component.covariance;
                    let comp_weight = component.weight;

                    let result = compute_likelihood(
                        prior_mean,
                        prior_cov,
                        measurement,
                        self.sensor,
                        &mut self.workspace,
                    );

                    // compute_likelihood returns: log(p_D / lambda * gaussian)
                    // We need: r * p_D * w[k] / lambda * gaussian
                    // So: r * w[k] * exp(log_likelihood_ratio)
                    let component_likelihood = r * comp_weight * result.log_likelihood_ratio.exp();
                    l_matrix[(i, j)] += component_likelihood;

                    // Store posterior for this component
                    meas_means.push(result.posterior_mean);
                    meas_covs.push(result.posterior_covariance);
                    meas_gains.push(result.kalman_gain);
                }

                track_means.push(meas_means);
                track_covs.push(meas_covs);
                track_gains.push(meas_gains);
            }

            // Convert L to cost: cost = -log(L)
            for j in 0..m {
                if l_matrix[(i, j)] > 0.0 {
                    cost[(i, j)] = -l_matrix[(i, j)].ln();
                }
            }

            posteriors.means.push(track_means);
            posteriors.covariances.push(track_covs);
            posteriors.kalman_gains.push(track_gains);
        }

        // Also need log_likelihood_ratios matrix
        let mut log_likelihood_ratios = DMatrix::from_element(n, m, f64::NEG_INFINITY);
        for i in 0..n {
            for j in 0..m {
                if l_matrix[(i, j)] > 0.0 {
                    log_likelihood_ratios[(i, j)] = l_matrix[(i, j)].ln();
                }
            }
        }

        // Compute LBP matrices
        let mut eta = DVector::zeros(n);
        let mut phi = DVector::zeros(n);
        let mut psi = DMatrix::zeros(n, m);

        for (i, track) in self.tracks.iter().enumerate() {
            let r = track.existence;

            // eta[i] = 1 - p_D × r[i]
            eta[i] = 1.0 - p_d * r;

            // phi[i] = (1 - p_D) × r[i]
            // Note: LBP phi does NOT divide by eta (unlike Gibbs R matrix which stores phi/eta)
            phi[i] = (1.0 - p_d) * r;

            // psi[i,j] = L[i,j] / eta[i]
            // Note: L already includes r (existence) in its computation
            for j in 0..m {
                let l_val = log_likelihood_ratios[(i, j)].exp(); // L[i,j] already has r in it
                if eta[i].abs() > 1e-15 {
                    psi[(i, j)] = l_val / eta[i];
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
    use crate::lmb::TrackLabel;
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
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Row {} sums to {}",
                i,
                row_sum
            );
        }
    }
}
