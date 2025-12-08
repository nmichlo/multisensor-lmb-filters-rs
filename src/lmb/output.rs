//! Output types for filter state estimates and trajectories.
//!
//! After processing measurements, filters produce state estimates: the inferred
//! positions, velocities, and uncertainties of tracked objects. This module
//! defines the types used to represent these outputs.
//!
//! - [`EstimatedTrack`] - A single track's state at one timestep
//! - [`StateEstimate`] - All tracks' states at one timestep
//! - [`Trajectory`] - A single track's complete history across timesteps
//! - [`FilterOutput`] - Complete output: all estimates and all trajectories

use nalgebra::{DMatrix, DVector};

use super::types::TrackLabel;

/// Estimated state of a single track at one timestep.
///
/// This represents the filter's best estimate of an object's state (position,
/// velocity, etc.) along with uncertainty. The track label provides identity
/// across timesteps.
#[derive(Debug, Clone)]
pub struct EstimatedTrack {
    /// Unique track identifier (birth time + birth location).
    pub label: TrackLabel,
    /// Estimated state vector (e.g., [x, vx, y, vy] for 2D constant velocity).
    pub mean: DVector<f64>,
    /// Uncertainty in the state estimate.
    pub covariance: DMatrix<f64>,
}

impl EstimatedTrack {
    /// Create a new estimated track
    pub fn new(label: TrackLabel, mean: DVector<f64>, covariance: DMatrix<f64>) -> Self {
        Self {
            label,
            mean,
            covariance,
        }
    }

    /// Get state dimension
    #[inline]
    pub fn x_dim(&self) -> usize {
        self.mean.len()
    }
}

/// All track state estimates at a single timestep.
///
/// This is the primary output from each filter step: the set of objects
/// believed to exist at this time, along with their states and uncertainties.
/// The number of tracks varies as objects appear and disappear.
#[derive(Debug, Clone)]
pub struct StateEstimate {
    /// Timestep index (0-based).
    pub timestamp: usize,
    /// All estimated tracks at this timestep.
    pub tracks: Vec<EstimatedTrack>,
}

impl StateEstimate {
    /// Create a new state estimate
    pub fn new(timestamp: usize, tracks: Vec<EstimatedTrack>) -> Self {
        Self { timestamp, tracks }
    }

    /// Create an empty state estimate
    pub fn empty(timestamp: usize) -> Self {
        Self {
            timestamp,
            tracks: Vec::new(),
        }
    }

    /// Number of estimated tracks
    #[inline]
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }
}

/// Complete trajectory of a single track across multiple timesteps.
///
/// A trajectory accumulates states over time for a single tracked object,
/// from when it first appeared (birth) until it disappears or the filter ends.
/// Useful for smoothing, visualization, and performance evaluation.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Unique track identifier.
    pub label: TrackLabel,
    /// State vector at each recorded timestep.
    pub states: Vec<DVector<f64>>,
    /// Covariance matrix at each recorded timestep.
    pub covariances: Vec<DMatrix<f64>>,
    /// Timestep indices corresponding to each state.
    pub timestamps: Vec<usize>,
}

impl Trajectory {
    /// Create a new trajectory
    pub fn new(label: TrackLabel) -> Self {
        Self {
            label,
            states: Vec::new(),
            covariances: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Add a state to the trajectory
    pub fn add_state(&mut self, state: DVector<f64>, covariance: DMatrix<f64>, timestamp: usize) {
        self.states.push(state);
        self.covariances.push(covariance);
        self.timestamps.push(timestamp);
    }

    /// Length of the trajectory
    #[inline]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if trajectory is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Get state at index
    pub fn get_state(&self, index: usize) -> Option<&DVector<f64>> {
        self.states.get(index)
    }

    /// Get covariance at index
    pub fn get_covariance(&self, index: usize) -> Option<&DMatrix<f64>> {
        self.covariances.get(index)
    }

    /// Get timestamp at index
    pub fn get_timestamp(&self, index: usize) -> Option<usize> {
        self.timestamps.get(index).copied()
    }
}

/// Complete output from running a filter over a sequence of measurements.
///
/// Contains both per-timestep estimates (for online use) and complete
/// trajectories (for offline analysis/evaluation). The estimates give
/// what the filter believed at each moment; trajectories give the full
/// history of each tracked object.
#[derive(Debug, Clone)]
pub struct FilterOutput {
    /// State estimates at each timestep (in chronological order).
    pub estimates: Vec<StateEstimate>,
    /// Complete trajectories for all tracks that were ever estimated.
    pub trajectories: Vec<Trajectory>,
}

impl FilterOutput {
    /// Create a new filter output
    pub fn new(estimates: Vec<StateEstimate>, trajectories: Vec<Trajectory>) -> Self {
        Self {
            estimates,
            trajectories,
        }
    }

    /// Create an empty filter output
    pub fn empty() -> Self {
        Self {
            estimates: Vec::new(),
            trajectories: Vec::new(),
        }
    }

    /// Number of timesteps
    #[inline]
    pub fn num_timesteps(&self) -> usize {
        self.estimates.len()
    }

    /// Total number of unique tracks
    #[inline]
    pub fn num_trajectories(&self) -> usize {
        self.trajectories.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimated_track() {
        let label = TrackLabel::new(0, 1);
        let mean = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let cov = DMatrix::identity(4, 4);

        let track = EstimatedTrack::new(label, mean, cov);
        assert_eq!(track.x_dim(), 4);
    }

    #[test]
    fn test_state_estimate() {
        let est = StateEstimate::empty(5);
        assert_eq!(est.timestamp, 5);
        assert_eq!(est.num_tracks(), 0);
    }

    #[test]
    fn test_trajectory() {
        let label = TrackLabel::new(0, 0);
        let mut traj = Trajectory::new(label);

        for t in 0..5 {
            let state = DVector::from_vec(vec![t as f64; 4]);
            let cov = DMatrix::identity(4, 4);
            traj.add_state(state, cov, t);
        }

        assert_eq!(traj.len(), 5);
        assert!(!traj.is_empty());
        assert_eq!(traj.get_timestamp(2), Some(2));
    }

    #[test]
    fn test_filter_output() {
        let output = FilterOutput::empty();
        assert_eq!(output.num_timesteps(), 0);
        assert_eq!(output.num_trajectories(), 0);
    }
}
