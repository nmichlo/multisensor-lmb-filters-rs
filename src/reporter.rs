//! Observability for LMB filter execution.
//!
//! This module provides the [`StepReporter`] trait for debugging and research
//! instrumentation. Reporters receive callbacks at key points during filter
//! execution without polluting the core algorithm logic.
//!
//! # Zero-Cost Abstraction
//!
//! The default [`NoOpReporter`] compiles to zero overhead - all callback
//! methods are empty and will be optimized away by the compiler.
//!
//! # Use Cases
//!
//! - **Debugging**: Capture intermediate states to diagnose filter issues
//! - **Research**: Collect association matrices, track evolution, etc.
//! - **Logging**: Emit structured events for monitoring
//! - **Visualization**: Build real-time filter state displays
//!
//! # Example
//!
//! ```ignore
//! use multisensor_lmb_filters_rs::lmb::{StepReporter, DebugReporter, Track};
//!
//! // Use DebugReporter to capture all events
//! let mut reporter = DebugReporter::new();
//!
//! // ... run filter with reporter ...
//!
//! // Access captured data
//! println!("Captured {} prediction events", reporter.predictions().len());
//! ```
//!
//! # Future Integration
//!
//! Reporter hooks will be integrated into filters in Phase 8-9.
//! Currently, this module provides the abstraction layer.

use crate::association::AssociationMatrices;
use crate::traits::AssociationResult;
use crate::types::Track;

// ============================================================================
// StepReporter Trait
// ============================================================================

/// Observability trait for filter step execution.
///
/// Implement this trait to receive callbacks during filter execution.
/// All methods have default empty implementations, so you only need
/// to override the events you care about.
///
/// # Thread Safety
///
/// Reporters use `&mut self` for callbacks, so they are NOT required
/// to be `Send + Sync`. If you need thread-safe reporting, use interior
/// mutability (e.g., `Mutex<Vec<...>>`) in your implementation.
///
/// # Performance
///
/// Callbacks receive references to avoid cloning overhead. If you need
/// to store the data, clone it within your callback implementation.
///
/// # Example
///
/// ```
/// use multisensor_lmb_filters_rs::lmb::{StepReporter, Track};
///
/// struct CountingReporter {
///     prediction_count: usize,
///     update_count: usize,
/// }
///
/// impl StepReporter for CountingReporter {
///     fn on_prediction(&mut self, _tracks: &[Track]) {
///         self.prediction_count += 1;
///     }
///
///     fn on_update_complete(&mut self, _tracks: &[Track]) {
///         self.update_count += 1;
///     }
/// }
/// ```
pub trait StepReporter {
    /// Called after prediction step (motion model applied).
    ///
    /// At this point, tracks have been propagated forward in time
    /// but birth tracks have not yet been added.
    fn on_prediction(&mut self, _tracks: &[Track]) {}

    /// Called after birth tracks are added.
    ///
    /// The `new_tracks` parameter contains only the newly birthed tracks,
    /// not the full track set.
    fn on_birth(&mut self, _new_tracks: &[Track]) {}

    /// Called after association matrices are computed for a sensor.
    ///
    /// This provides access to the raw likelihood ratios, cost matrices,
    /// and sampling probabilities before the association algorithm runs.
    fn on_association_matrices(&mut self, _sensor_idx: usize, _matrices: &AssociationMatrices) {}

    /// Called after data association is solved for a sensor.
    ///
    /// The result contains the association weights/assignments that will
    /// be used to update tracks.
    fn on_association_result(&mut self, _sensor_idx: usize, _result: &AssociationResult) {}

    /// Called after tracks are updated from a single sensor.
    ///
    /// For multi-sensor filters, this is called once per sensor before fusion.
    /// For single-sensor filters, this is the final update result.
    fn on_sensor_update(&mut self, _sensor_idx: usize, _tracks: &[Track]) {}

    /// Called after multi-sensor fusion is complete.
    ///
    /// This is only called for multi-sensor parallel fusion filters (AA, GA, PU).
    /// Sequential filters (IC) don't have a separate fusion step.
    fn on_fusion(&mut self, _tracks: &[Track]) {}

    /// Called after track pruning/gating.
    ///
    /// Provides both the removed tracks and the surviving tracks.
    fn on_pruning(&mut self, _removed: &[Track], _kept: &[Track]) {}

    /// Called after the complete update cycle is finished.
    ///
    /// This is called after all sensors have been processed and
    /// any fusion/pruning is complete.
    fn on_update_complete(&mut self, _tracks: &[Track]) {}

    /// Called when a track's trajectory is archived (track deleted but history saved).
    fn on_trajectory_archived(&mut self, _track: &Track) {}
}

// ============================================================================
// NoOpReporter
// ============================================================================

/// Zero-cost reporter that does nothing.
///
/// This is the default reporter used when no observability is needed.
/// All callbacks are empty and will be optimized away by the compiler.
///
/// # Example
///
/// ```
/// use multisensor_lmb_filters_rs::lmb::{StepReporter, NoOpReporter, Track};
///
/// let mut reporter = NoOpReporter;
///
/// // These calls are zero-cost
/// reporter.on_prediction(&[]);
/// reporter.on_update_complete(&[]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOpReporter;

impl NoOpReporter {
    /// Create a new no-op reporter.
    pub fn new() -> Self {
        Self
    }
}

impl StepReporter for NoOpReporter {
    // All methods use default empty implementations
}

// ============================================================================
// DebugReporter
// ============================================================================

/// Reporter that captures all events for debugging.
///
/// This reporter clones and stores all data passed to callbacks,
/// allowing post-hoc analysis of filter execution.
///
/// # Memory Usage
///
/// Be aware that this reporter stores clones of all track data,
/// which can consume significant memory for long runs or large
/// track sets. Consider using a sampling strategy for production
/// debugging.
///
/// # Example
///
/// ```
/// use multisensor_lmb_filters_rs::lmb::{StepReporter, DebugReporter, Track};
///
/// let mut reporter = DebugReporter::new();
///
/// // Simulate some callbacks
/// reporter.on_prediction(&[]);
/// reporter.on_birth(&[]);
/// reporter.on_update_complete(&[]);
///
/// // Check captured data
/// assert_eq!(reporter.prediction_events().len(), 1);
/// assert_eq!(reporter.birth_events().len(), 1);
/// assert_eq!(reporter.update_complete_events().len(), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct DebugReporter {
    /// Captured prediction events (tracks after prediction)
    predictions: Vec<Vec<Track>>,

    /// Captured birth events (newly birthed tracks)
    births: Vec<Vec<Track>>,

    /// Captured sensor update events (sensor_idx, tracks)
    sensor_updates: Vec<(usize, Vec<Track>)>,

    /// Captured fusion events (tracks after fusion)
    fusions: Vec<Vec<Track>>,

    /// Captured pruning events (removed, kept)
    prunings: Vec<(Vec<Track>, Vec<Track>)>,

    /// Captured update complete events (tracks after full update)
    update_completes: Vec<Vec<Track>>,

    /// Captured archived trajectories
    archived_trajectories: Vec<Track>,
}

impl DebugReporter {
    /// Create a new debug reporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all captured events.
    pub fn clear(&mut self) {
        self.predictions.clear();
        self.births.clear();
        self.sensor_updates.clear();
        self.fusions.clear();
        self.prunings.clear();
        self.update_completes.clear();
        self.archived_trajectories.clear();
    }

    /// Get captured prediction events.
    pub fn prediction_events(&self) -> &[Vec<Track>] {
        &self.predictions
    }

    /// Get captured birth events.
    pub fn birth_events(&self) -> &[Vec<Track>] {
        &self.births
    }

    /// Get captured sensor update events.
    pub fn sensor_update_events(&self) -> &[(usize, Vec<Track>)] {
        &self.sensor_updates
    }

    /// Get captured fusion events.
    pub fn fusion_events(&self) -> &[Vec<Track>] {
        &self.fusions
    }

    /// Get captured pruning events.
    pub fn pruning_events(&self) -> &[(Vec<Track>, Vec<Track>)] {
        &self.prunings
    }

    /// Get captured update complete events.
    pub fn update_complete_events(&self) -> &[Vec<Track>] {
        &self.update_completes
    }

    /// Get captured archived trajectories.
    pub fn archived_trajectories(&self) -> &[Track] {
        &self.archived_trajectories
    }

    /// Total number of captured events across all types.
    pub fn total_events(&self) -> usize {
        self.predictions.len()
            + self.births.len()
            + self.sensor_updates.len()
            + self.fusions.len()
            + self.prunings.len()
            + self.update_completes.len()
            + self.archived_trajectories.len()
    }
}

impl StepReporter for DebugReporter {
    fn on_prediction(&mut self, tracks: &[Track]) {
        self.predictions.push(tracks.to_vec());
    }

    fn on_birth(&mut self, new_tracks: &[Track]) {
        self.births.push(new_tracks.to_vec());
    }

    fn on_sensor_update(&mut self, sensor_idx: usize, tracks: &[Track]) {
        self.sensor_updates.push((sensor_idx, tracks.to_vec()));
    }

    fn on_fusion(&mut self, tracks: &[Track]) {
        self.fusions.push(tracks.to_vec());
    }

    fn on_pruning(&mut self, removed: &[Track], kept: &[Track]) {
        self.prunings.push((removed.to_vec(), kept.to_vec()));
    }

    fn on_update_complete(&mut self, tracks: &[Track]) {
        self.update_completes.push(tracks.to_vec());
    }

    fn on_trajectory_archived(&mut self, track: &Track) {
        self.archived_trajectories.push(track.clone());
    }
}

// ============================================================================
// LoggingReporter
// ============================================================================

/// Reporter that logs events to stderr using the log crate.
///
/// This reporter emits log messages at configurable levels for each
/// event type. Useful for debugging without storing large amounts of data.
///
/// # Log Levels
///
/// By default:
/// - `on_prediction`, `on_update_complete`: INFO
/// - `on_birth`, `on_fusion`, `on_pruning`: DEBUG
/// - `on_sensor_update`, `on_association_*`: TRACE
///
/// # Example
///
/// ```
/// use multisensor_lmb_filters_rs::lmb::{StepReporter, LoggingReporter, Track};
///
/// // Enable debug logging (requires log crate setup)
/// let mut reporter = LoggingReporter::new();
///
/// // Events will be logged according to configured levels
/// reporter.on_prediction(&[]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LoggingReporter {
    /// Whether to include track details in log messages
    verbose: bool,
}

impl LoggingReporter {
    /// Create a new logging reporter.
    pub fn new() -> Self {
        Self { verbose: false }
    }

    /// Create a verbose logging reporter that includes track details.
    pub fn verbose() -> Self {
        Self { verbose: true }
    }
}

impl StepReporter for LoggingReporter {
    fn on_prediction(&mut self, tracks: &[Track]) {
        if self.verbose {
            log::info!("Prediction complete: {} tracks", tracks.len());
            for (i, t) in tracks.iter().enumerate() {
                log::debug!(
                    "  Track {}: label={:?}, existence={:.4}, components={}",
                    i,
                    t.label,
                    t.existence,
                    t.components.len()
                );
            }
        } else {
            log::info!("Prediction complete: {} tracks", tracks.len());
        }
    }

    fn on_birth(&mut self, new_tracks: &[Track]) {
        log::debug!("Birth: {} new tracks", new_tracks.len());
    }

    fn on_association_matrices(&mut self, sensor_idx: usize, matrices: &AssociationMatrices) {
        log::trace!(
            "Association matrices for sensor {}: {} tracks × {} measurements",
            sensor_idx,
            matrices.num_tracks(),
            matrices.num_measurements()
        );
    }

    fn on_association_result(&mut self, sensor_idx: usize, result: &AssociationResult) {
        log::trace!(
            "Association result for sensor {}: {} tracks, converged={}",
            sensor_idx,
            result.marginal_weights.nrows(),
            result.converged
        );
    }

    fn on_sensor_update(&mut self, sensor_idx: usize, tracks: &[Track]) {
        log::trace!(
            "Sensor {} update complete: {} tracks",
            sensor_idx,
            tracks.len()
        );
    }

    fn on_fusion(&mut self, tracks: &[Track]) {
        log::debug!("Fusion complete: {} tracks", tracks.len());
    }

    fn on_pruning(&mut self, removed: &[Track], kept: &[Track]) {
        log::debug!(
            "Pruning: removed {} tracks, kept {} tracks",
            removed.len(),
            kept.len()
        );
    }

    fn on_update_complete(&mut self, tracks: &[Track]) {
        log::info!("Update complete: {} tracks", tracks.len());
    }

    fn on_trajectory_archived(&mut self, track: &Track) {
        log::debug!("Trajectory archived: label={:?}", track.label);
    }
}

// ============================================================================
// CompositeReporter
// ============================================================================

/// Reporter that forwards events to multiple child reporters.
///
/// Useful when you need both logging and debugging, or want to
/// combine multiple specialized reporters.
///
/// # Example
///
/// ```
/// use multisensor_lmb_filters_rs::lmb::{
///     StepReporter, CompositeReporter, DebugReporter, LoggingReporter, Track
/// };
///
/// let debug = DebugReporter::new();
/// let logging = LoggingReporter::new();
///
/// let mut composite = CompositeReporter::new(debug, logging);
///
/// // Events go to both reporters
/// composite.on_prediction(&[]);
///
/// // Access the debug reporter's data
/// assert_eq!(composite.first().prediction_events().len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct CompositeReporter<A: StepReporter, B: StepReporter> {
    first: A,
    second: B,
}

impl<A: StepReporter, B: StepReporter> CompositeReporter<A, B> {
    /// Create a new composite reporter.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }

    /// Get a reference to the first reporter.
    pub fn first(&self) -> &A {
        &self.first
    }

    /// Get a mutable reference to the first reporter.
    pub fn first_mut(&mut self) -> &mut A {
        &mut self.first
    }

    /// Get a reference to the second reporter.
    pub fn second(&self) -> &B {
        &self.second
    }

    /// Get a mutable reference to the second reporter.
    pub fn second_mut(&mut self) -> &mut B {
        &mut self.second
    }

    /// Consume and return both reporters.
    pub fn into_parts(self) -> (A, B) {
        (self.first, self.second)
    }
}

impl<A: StepReporter, B: StepReporter> StepReporter for CompositeReporter<A, B> {
    fn on_prediction(&mut self, tracks: &[Track]) {
        self.first.on_prediction(tracks);
        self.second.on_prediction(tracks);
    }

    fn on_birth(&mut self, new_tracks: &[Track]) {
        self.first.on_birth(new_tracks);
        self.second.on_birth(new_tracks);
    }

    fn on_association_matrices(&mut self, sensor_idx: usize, matrices: &AssociationMatrices) {
        self.first.on_association_matrices(sensor_idx, matrices);
        self.second.on_association_matrices(sensor_idx, matrices);
    }

    fn on_association_result(&mut self, sensor_idx: usize, result: &AssociationResult) {
        self.first.on_association_result(sensor_idx, result);
        self.second.on_association_result(sensor_idx, result);
    }

    fn on_sensor_update(&mut self, sensor_idx: usize, tracks: &[Track]) {
        self.first.on_sensor_update(sensor_idx, tracks);
        self.second.on_sensor_update(sensor_idx, tracks);
    }

    fn on_fusion(&mut self, tracks: &[Track]) {
        self.first.on_fusion(tracks);
        self.second.on_fusion(tracks);
    }

    fn on_pruning(&mut self, removed: &[Track], kept: &[Track]) {
        self.first.on_pruning(removed, kept);
        self.second.on_pruning(removed, kept);
    }

    fn on_update_complete(&mut self, tracks: &[Track]) {
        self.first.on_update_complete(tracks);
        self.second.on_update_complete(tracks);
    }

    fn on_trajectory_archived(&mut self, track: &Track) {
        self.first.on_trajectory_archived(track);
        self.second.on_trajectory_archived(track);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_reporter() {
        let mut reporter = NoOpReporter::new();

        // These should all compile and do nothing
        reporter.on_prediction(&[]);
        reporter.on_birth(&[]);
        reporter.on_sensor_update(0, &[]);
        reporter.on_fusion(&[]);
        reporter.on_pruning(&[], &[]);
        reporter.on_update_complete(&[]);
    }

    #[test]
    fn test_debug_reporter_captures_events() {
        let mut reporter = DebugReporter::new();

        // Initially empty
        assert_eq!(reporter.total_events(), 0);

        // Add some events
        reporter.on_prediction(&[]);
        reporter.on_birth(&[]);
        reporter.on_sensor_update(0, &[]);
        reporter.on_sensor_update(1, &[]);
        reporter.on_fusion(&[]);
        reporter.on_pruning(&[], &[]);
        reporter.on_update_complete(&[]);

        // Verify counts
        assert_eq!(reporter.prediction_events().len(), 1);
        assert_eq!(reporter.birth_events().len(), 1);
        assert_eq!(reporter.sensor_update_events().len(), 2);
        assert_eq!(reporter.fusion_events().len(), 1);
        assert_eq!(reporter.pruning_events().len(), 1);
        assert_eq!(reporter.update_complete_events().len(), 1);
        assert_eq!(reporter.total_events(), 7);

        // Clear
        reporter.clear();
        assert_eq!(reporter.total_events(), 0);
    }

    #[test]
    fn test_debug_reporter_captures_track_data() {
        use crate::lmb::{GaussianComponent, TrackLabel};
        use nalgebra::{DMatrix, DVector};
        use smallvec::smallvec;

        let mut reporter = DebugReporter::new();

        // Create a test track
        let track = Track {
            label: TrackLabel::new(1, 0),
            existence: 0.9,
            components: smallvec![GaussianComponent {
                weight: 1.0,
                mean: DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
                covariance: DMatrix::identity(4, 4),
            }],
            trajectory: None,
        };

        reporter.on_prediction(&[track.clone()]);

        // Verify track data is captured
        assert_eq!(reporter.prediction_events().len(), 1);
        assert_eq!(reporter.prediction_events()[0].len(), 1);
        assert_eq!(reporter.prediction_events()[0][0].label, track.label);
        assert!((reporter.prediction_events()[0][0].existence - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_debug_reporter_sensor_update_indices() {
        let mut reporter = DebugReporter::new();

        reporter.on_sensor_update(0, &[]);
        reporter.on_sensor_update(1, &[]);
        reporter.on_sensor_update(2, &[]);

        let events = reporter.sensor_update_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].0, 0);
        assert_eq!(events[1].0, 1);
        assert_eq!(events[2].0, 2);
    }

    #[test]
    fn test_logging_reporter() {
        // Just verify it compiles and doesn't panic
        let mut reporter = LoggingReporter::new();
        reporter.on_prediction(&[]);
        reporter.on_update_complete(&[]);

        let mut verbose = LoggingReporter::verbose();
        verbose.on_prediction(&[]);
    }

    #[test]
    fn test_composite_reporter() {
        let debug = DebugReporter::new();
        let noop = NoOpReporter::new();

        let mut composite = CompositeReporter::new(debug, noop);

        // Events go to both
        composite.on_prediction(&[]);
        composite.on_birth(&[]);
        composite.on_update_complete(&[]);

        // Verify debug reporter captured events
        assert_eq!(composite.first().prediction_events().len(), 1);
        assert_eq!(composite.first().birth_events().len(), 1);
        assert_eq!(composite.first().update_complete_events().len(), 1);
    }

    #[test]
    fn test_composite_reporter_into_parts() {
        let debug = DebugReporter::new();
        let logging = LoggingReporter::new();

        let mut composite = CompositeReporter::new(debug, logging);
        composite.on_prediction(&[]);

        let (debug, _logging) = composite.into_parts();
        assert_eq!(debug.prediction_events().len(), 1);
    }

    #[test]
    fn test_reporter_default_implementations() {
        // Verify all default implementations compile
        struct MinimalReporter;
        impl StepReporter for MinimalReporter {}

        let mut reporter = MinimalReporter;
        reporter.on_prediction(&[]);
        reporter.on_birth(&[]);
        reporter.on_association_matrices(0, &create_dummy_matrices());
        reporter.on_association_result(0, &create_dummy_result());
        reporter.on_sensor_update(0, &[]);
        reporter.on_fusion(&[]);
        reporter.on_pruning(&[], &[]);
        reporter.on_update_complete(&[]);
        reporter.on_trajectory_archived(&create_dummy_track());
    }

    // Helper functions for tests
    fn create_dummy_matrices() -> AssociationMatrices {
        use crate::association::PosteriorGrid;
        use nalgebra::{DMatrix, DVector};

        AssociationMatrices {
            psi: DMatrix::zeros(1, 1),
            phi: DVector::zeros(1),
            eta: DVector::from_element(1, 1.0),
            cost: DMatrix::zeros(1, 1),
            sampling_prob: DMatrix::zeros(1, 2),
            posteriors: PosteriorGrid::new(1, 1),
            log_likelihood_ratios: DMatrix::zeros(1, 1),
        }
    }

    fn create_dummy_result() -> AssociationResult {
        use nalgebra::{DMatrix, DVector};

        AssociationResult {
            marginal_weights: DMatrix::zeros(1, 1),
            miss_weights: DVector::zeros(1),
            posterior_existence: DVector::from_element(1, 0.5),
            sampled_associations: None,
            iterations: 1,
            converged: true,
        }
    }

    fn create_dummy_track() -> Track {
        use crate::lmb::TrackLabel;
        use smallvec::smallvec;

        Track {
            label: TrackLabel::new(0, 0),
            existence: 0.5,
            components: smallvec![],
            trajectory: None,
        }
    }
}
