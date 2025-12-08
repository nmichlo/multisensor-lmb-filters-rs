//! Track and Gaussian component types
//!
//! This module defines the core track types used throughout the tracking library.
//! Uses runtime dimensions (DVector/DMatrix) for Python binding compatibility.

use nalgebra::{DMatrix, DVector};
use smallvec::SmallVec;

/// Track label uniquely identifies a track
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackLabel {
    /// Time step when the track was born
    pub birth_time: usize,
    /// Birth location index
    pub birth_location: usize,
}

impl TrackLabel {
    /// Create a new track label
    pub fn new(birth_time: usize, birth_location: usize) -> Self {
        Self {
            birth_time,
            birth_location,
        }
    }
}

/// Gaussian component with runtime dimensions
#[derive(Debug, Clone)]
pub struct GaussianComponent {
    /// Component weight (must be positive, typically sums to 1 within a track)
    pub weight: f64,
    /// Mean vector (state estimate)
    pub mean: DVector<f64>,
    /// Covariance matrix (uncertainty)
    pub covariance: DMatrix<f64>,
}

impl GaussianComponent {
    /// Create a new Gaussian component
    pub fn new(weight: f64, mean: DVector<f64>, covariance: DMatrix<f64>) -> Self {
        Self {
            weight,
            mean,
            covariance,
        }
    }

    /// Get state dimension from mean vector
    #[inline]
    pub fn x_dim(&self) -> usize {
        self.mean.len()
    }

    /// Create a zero-weighted component (for initialization)
    pub fn zero(x_dim: usize) -> Self {
        Self {
            weight: 0.0,
            mean: DVector::zeros(x_dim),
            covariance: DMatrix::zeros(x_dim, x_dim),
        }
    }
}

/// Core track type - uses DVector/DMatrix for Python binding compatibility
///
/// A track represents a potential object being tracked. It has:
/// - A unique label identifying when/where it was born
/// - An existence probability (how likely the track represents a real object)
/// - One or more Gaussian components (for Gaussian mixture representation)
/// - Optional trajectory history for state extraction
#[derive(Debug, Clone)]
pub struct Track {
    /// Unique track identifier
    pub label: TrackLabel,
    /// Existence probability (0.0 to 1.0)
    pub existence: f64,
    /// Gaussian mixture components (SmallVec avoids heap for typical 1-4 components)
    pub components: SmallVec<[GaussianComponent; 4]>,
    /// Optional trajectory history
    pub trajectory: Option<TrajectoryHistory>,
}

impl Track {
    /// Create a new track with a single Gaussian component
    pub fn new(
        label: TrackLabel,
        existence: f64,
        mean: DVector<f64>,
        covariance: DMatrix<f64>,
    ) -> Self {
        let component = GaussianComponent::new(1.0, mean, covariance);
        let mut components = SmallVec::new();
        components.push(component);
        Self {
            label,
            existence,
            components,
            trajectory: None,
        }
    }

    /// Create a new birth track
    pub fn new_birth(
        birth_location: usize,
        birth_time: usize,
        existence: f64,
        mean: DVector<f64>,
        covariance: DMatrix<f64>,
    ) -> Self {
        Self::new(
            TrackLabel::new(birth_time, birth_location),
            existence,
            mean,
            covariance,
        )
    }

    /// Get state dimension from first component
    #[inline]
    pub fn x_dim(&self) -> usize {
        self.components.first().map(|c| c.x_dim()).unwrap_or(0)
    }

    /// Get the number of Gaussian mixture components
    #[inline]
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Get the primary (highest weight) component's mean
    pub fn primary_mean(&self) -> Option<&DVector<f64>> {
        self.components
            .iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
            .map(|c| &c.mean)
    }

    /// Get the primary (highest weight) component's covariance
    pub fn primary_covariance(&self) -> Option<&DMatrix<f64>> {
        self.components
            .iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
            .map(|c| &c.covariance)
    }

    /// Compute weighted mean across all components
    pub fn weighted_mean(&self) -> DVector<f64> {
        if self.components.is_empty() {
            return DVector::zeros(0);
        }

        let x_dim = self.x_dim();
        let total_weight: f64 = self.components.iter().map(|c| c.weight).sum();

        if total_weight == 0.0 {
            return self.components[0].mean.clone();
        }

        self.components
            .iter()
            .fold(DVector::zeros(x_dim), |acc, c| {
                acc + &c.mean * (c.weight / total_weight)
            })
    }

    /// Add a component to the track
    pub fn add_component(&mut self, component: GaussianComponent) {
        self.components.push(component);
    }

    /// Initialize trajectory history
    pub fn init_trajectory(&mut self, max_length: usize) {
        let x_dim = self.x_dim();
        self.trajectory = Some(TrajectoryHistory::new(x_dim, max_length));
    }

    /// Record current state to trajectory
    pub fn record_state(&mut self, timestamp: usize) {
        // Clone the mean first to avoid borrow conflict
        let mean = self.primary_mean().cloned();
        if let (Some(ref mut traj), Some(m)) = (&mut self.trajectory, mean) {
            traj.add_state(m, timestamp);
        }
    }
}

/// Trajectory history for a track
#[derive(Debug, Clone)]
pub struct TrajectoryHistory {
    /// Historical state estimates (each column is a timestep)
    pub states: DMatrix<f64>,
    /// Time indices corresponding to each state
    pub timestamps: Vec<usize>,
    /// Current length of stored trajectory
    pub length: usize,
    /// Maximum capacity
    capacity: usize,
}

impl TrajectoryHistory {
    /// Create a new trajectory history with given capacity
    pub fn new(x_dim: usize, capacity: usize) -> Self {
        Self {
            states: DMatrix::zeros(x_dim, capacity),
            timestamps: Vec::with_capacity(capacity),
            length: 0,
            capacity,
        }
    }

    /// Add a state to the trajectory
    pub fn add_state(&mut self, state: DVector<f64>, timestamp: usize) {
        if self.length < self.capacity {
            self.states.set_column(self.length, &state);
            self.timestamps.push(timestamp);
            self.length += 1;
        }
    }

    /// Get the state at a specific index
    pub fn get_state(&self, index: usize) -> Option<DVector<f64>> {
        if index < self.length {
            Some(self.states.column(index).into_owned())
        } else {
            None
        }
    }

    /// Get all states as a matrix (columns are timesteps)
    pub fn get_states(&self) -> DMatrix<f64> {
        self.states.columns(0, self.length).into_owned()
    }
}

/// Cardinality estimation result from MAP cardinality extraction.
///
/// This is the result of applying the LMB cardinality estimation algorithm,
/// which determines how many objects exist and which tracks represent them.
#[derive(Debug, Clone)]
pub struct CardinalityEstimate {
    /// MAP estimate of the number of objects.
    pub n_estimated: usize,
    /// Indices of the tracks selected for the MAP estimate.
    pub map_indices: Vec<usize>,
}

impl CardinalityEstimate {
    /// Create a new cardinality estimate.
    pub fn new(n_estimated: usize, map_indices: Vec<usize>) -> Self {
        Self {
            n_estimated,
            map_indices,
        }
    }

    /// Create an empty cardinality estimate.
    pub fn empty() -> Self {
        Self {
            n_estimated: 0,
            map_indices: Vec::new(),
        }
    }
}

/// Detailed output from a single filter step, exposing all intermediate data.
///
/// This is used for fixture validation and testing. It contains the state
/// after each major step of the filter algorithm:
///
/// 1. **Predicted tracks** - after prediction step (motion model + birth)
/// 2. **Association matrices** - likelihood ratios, costs, sampling probs
/// 3. **Association result** - marginal weights from data association
/// 4. **Updated tracks** - after measurement update step
/// 5. **Cardinality estimate** - MAP cardinality extraction
/// 6. **Final estimate** - extracted state estimates
#[derive(Debug, Clone)]
pub struct StepDetailedOutput {
    /// Tracks after prediction step (before measurement update).
    pub predicted_tracks: Vec<Track>,
    /// Association matrices from the association builder (None if no measurements).
    pub association_matrices: Option<crate::association::AssociationMatrices>,
    /// Association result from the data association algorithm (None if no measurements).
    pub association_result: Option<super::traits::AssociationResult>,
    /// Tracks after measurement update step.
    pub updated_tracks: Vec<Track>,
    /// Cardinality estimation result.
    pub cardinality: CardinalityEstimate,
    /// Final state estimate after gating.
    pub final_estimate: super::output::StateEstimate,
}

/// LMBM Hypothesis - represents a single hypothesis in LMBM filter
///
/// Unlike LMB which maintains a single track set with Gaussian mixtures,
/// LMBM maintains multiple weighted hypotheses, each with single-component tracks.
#[derive(Debug, Clone)]
pub struct LmbmHypothesis {
    /// Hypothesis weight (in log space for numerical stability)
    pub log_weight: f64,
    /// Tracks in this hypothesis (single component per track)
    pub tracks: Vec<Track>,
}

impl LmbmHypothesis {
    /// Create a new hypothesis
    pub fn new(log_weight: f64, tracks: Vec<Track>) -> Self {
        Self { log_weight, tracks }
    }

    /// Create an empty hypothesis
    pub fn empty() -> Self {
        Self {
            log_weight: 0.0,
            tracks: Vec::new(),
        }
    }

    /// Get the linear (non-log) weight
    pub fn weight(&self) -> f64 {
        self.log_weight.exp()
    }

    /// Number of tracks in this hypothesis
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_creation() {
        let label = TrackLabel::new(0, 1);
        let mean = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let cov = DMatrix::identity(4, 4);

        let track = Track::new(label, 0.9, mean.clone(), cov);

        assert_eq!(track.x_dim(), 4);
        assert_eq!(track.existence, 0.9);
        assert_eq!(track.num_components(), 1);
        assert_eq!(track.label.birth_time, 0);
        assert_eq!(track.label.birth_location, 1);
    }

    #[test]
    fn test_trajectory_history() {
        let mut traj = TrajectoryHistory::new(4, 100);

        for t in 0..10 {
            let state = DVector::from_vec(vec![t as f64; 4]);
            traj.add_state(state, t);
        }

        assert_eq!(traj.length, 10);
        assert_eq!(traj.timestamps.len(), 10);

        let state_5 = traj.get_state(5).unwrap();
        assert_eq!(state_5[0], 5.0);
    }

    #[test]
    fn test_weighted_mean() {
        let label = TrackLabel::new(0, 0);
        let mut track = Track::new(
            label,
            0.9,
            DVector::from_vec(vec![1.0, 0.0]),
            DMatrix::identity(2, 2),
        );

        // Add another component
        track.add_component(GaussianComponent::new(
            1.0,
            DVector::from_vec(vec![3.0, 0.0]),
            DMatrix::identity(2, 2),
        ));

        let weighted = track.weighted_mean();
        // With equal weights, mean should be (1+3)/2 = 2
        assert!((weighted[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_lmbm_hypothesis() {
        let hyp = LmbmHypothesis::new(0.0, vec![]);
        assert_eq!(hyp.weight(), 1.0); // exp(0) = 1
        assert_eq!(hyp.num_tracks(), 0);
    }
}
