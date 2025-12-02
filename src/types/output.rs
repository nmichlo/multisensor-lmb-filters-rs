//! Output types for filter results
//!
//! This module defines the output types returned by filters.

use nalgebra::{DMatrix, DVector};

use super::TrackLabel;

/// Single track estimate at a timestep
#[derive(Debug, Clone)]
pub struct EstimatedTrack {
    /// Track label (unique identifier)
    pub label: TrackLabel,
    /// Estimated mean state
    pub mean: DVector<f64>,
    /// Estimated covariance
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

/// State estimates at a single timestep
#[derive(Debug, Clone)]
pub struct StateEstimate {
    /// Timestep index
    pub timestamp: usize,
    /// Estimated tracks at this timestep
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

/// Complete trajectory of a single track
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Track label
    pub label: TrackLabel,
    /// States at each timestep (each DVector is a state)
    pub states: Vec<DVector<f64>>,
    /// Covariances at each timestep
    pub covariances: Vec<DMatrix<f64>>,
    /// Timestamps corresponding to states
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

/// Complete output from a filter run
#[derive(Debug, Clone)]
pub struct FilterOutput {
    /// State estimates at each timestep
    pub estimates: Vec<StateEstimate>,
    /// Complete trajectories for all tracks
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
