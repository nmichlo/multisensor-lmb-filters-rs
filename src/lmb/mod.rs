//! Labeled Multi-Bernoulli (LMB) tracking algorithms
//!
//! This module contains all LMB-family multi-object tracking implementations:
//!
//! # Single-Sensor Filters
//!
//! - [`LmbFilter`] - LMB filter with marginal association (Gaussian mixture posteriors)
//! - [`LmbmFilter`] - LMBM filter with hypothesis management (discrete association samples)
//!
//! # Multi-Sensor Filters
//!
//! - [`AaLmbFilter`] - Arithmetic Average fusion (fast, simple)
//! - [`GaLmbFilter`] - Geometric Average fusion (conservative covariance)
//! - [`PuLmbFilter`] - Parallel Update fusion (optimal for independent sensors)
//! - [`IcLmbFilter`] - Iterated Corrector fusion (sequential sensor updates)
//! - [`MultisensorLmbmFilter`] - Multi-sensor LMBM with joint association
//!
//! # Core Types
//!
//! - [`Track`] - Core track type with Gaussian mixture components
//! - [`GaussianComponent`] - Single Gaussian component
//! - [`TrackLabel`] - Unique track identifier
//! - [`LmbmHypothesis`] - LMBM hypothesis
//!
//! # Configuration
//!
//! - [`MotionModel`] - Prediction model (transition matrix, process noise)
//! - [`SensorModel`] - Observation model (observation matrix, measurement noise)
//! - [`BirthModel`] - Birth model for new tracks
//! - [`FilterParams`] - Complete filter configuration
//!
//! # Traits
//!
//! - [`Filter`] - Core filter interface implemented by all filters
//! - [`Associator`] - Data association algorithms (LBP, Gibbs, Murty)
//! - [`Updater`] - Track update strategies (marginal vs hard assignment)
//! - [`Merger`] - Multi-sensor track fusion strategies

// Core types
pub mod config;
pub mod output;
pub mod types;

// Filter infrastructure
pub mod builder;
pub mod common_ops;
pub mod core;
pub mod core_lmbm;
pub mod errors;
pub mod factory;
pub mod reporter;
pub mod scheduler;
pub mod traits;

// Filter implementations
pub mod multisensor;

// Utilities
pub mod cardinality;
pub mod measurements;

// Re-export all public types from submodules

// Types
pub use types::{GaussianComponent, LmbmHypothesis, Track, TrackLabel, TrajectoryHistory};

// Configuration
pub use config::{
    AssociationConfig, BirthLocation, BirthModel, DataAssociationMethod, FilterParams,
    FilterThresholds, LmbmConfig, MotionModel, MultisensorConfig, SensorModel,
};

// Model behavior traits (Phase 3 - Extensibility)
pub use config::{MotionModelBehavior, SensorModelBehavior};

// Type-safe filter configurations (Phase 2)
pub use config::{CommonConfig, LmbFilterConfig, LmbmFilterConfig};

// Configuration snapshots (for debugging)
pub use config::{
    AssociationConfigSnapshot, BirthLocationSnapshot, BirthModelSnapshot, FilterConfigSnapshot,
    LmbmConfigSnapshot, MotionModelSnapshot, SensorModelSnapshot, ThresholdsSnapshot,
};

// Output
pub use output::{EstimatedTrack, FilterOutput, StateEstimate, Trajectory};

// Errors
pub use errors::{AssociationError, FilterError};

// Traits and implementations
pub use traits::{
    AssociationResult, Associator, DynamicAssociator, Filter, GibbsAssociator,
    HardAssignmentUpdater, LbpAssociator, MarginalUpdater, Merger, MurtyAssociator, Updater,
};

// Update schedulers (Phase 4 - Strategy Pattern)
pub use scheduler::{
    DynamicScheduler, FusionCapable, IcLmbScheduler, LmbScheduler, MultisensorCapable,
    ParallelScheduler, SequentialScheduler, SingleSensorScheduler, UpdateScheduler,
};

// Observability (Phase 5 - StepReporter)
pub use reporter::{CompositeReporter, DebugReporter, LoggingReporter, NoOpReporter, StepReporter};

// Builder traits
pub use builder::{FilterBuilder, LmbFilterBuilder};

// Single-sensor filters (from unified cores)
pub use core::{LmbFilter, LmbFilterCore};
pub use core_lmbm::LmbmFilter;

// Multi-sensor LMB filters (from unified core)
pub use core::{AaLmbFilter, GaLmbFilter, IcLmbFilter, PuLmbFilter};

// Multi-sensor LMBM filter (from unified core)
pub use core_lmbm::MultisensorLmbmFilter;

// Fusion mergers and multi-sensor traits
pub use multisensor::{
    ArithmeticAverageMerger, GeometricAverageMerger, IteratedCorrectorMerger,
    MultisensorAssociationResult, MultisensorAssociator, MultisensorGibbsAssociator,
    MultisensorMeasurements, ParallelUpdateMerger,
};

// Utilities (re-exported from common)
pub use crate::common::rng::{SimpleRng, Uniform01};

// Measurement sources (zero-copy abstractions)
pub use measurements::{
    MeasurementSource, SingleSensorMeasurements, SliceOfSlicesMeasurements, VecOfVecsMeasurements,
};

// Unified sensor configuration (Phase 7C - moved to config.rs)
pub use config::{SensorSet, SensorSetIter};

// Unified LMBM filter core (Phase 9)
pub use core_lmbm::{
    LmbmAssociationIntermediate, LmbmAssociator, LmbmFilterCore, MultisensorLmbmStrategy,
    SingleSensorLmbmStrategy,
};

// Factory functions for common filter configurations (Phase 7C)
pub use factory::{
    aa_lmb_filter, ga_lmb_filter, ic_lmb_filter, lmb_filter, lmbm_filter, multisensor_lmbm_filter,
    pu_lmb_filter,
};

// ============================================================================
// Default Filter Constants
// ============================================================================

/// Default existence probability threshold for track gating.
/// Tracks with existence below this threshold are pruned.
pub const DEFAULT_EXISTENCE_THRESHOLD: f64 = 1e-3;

/// Default minimum trajectory length to save when pruning tracks.
/// Short-lived tracks are discarded without saving their trajectory.
pub const DEFAULT_MIN_TRAJECTORY_LENGTH: usize = 3;

/// Default weight threshold for GM component pruning.
/// Components with weight below this threshold are pruned.
pub const DEFAULT_GM_WEIGHT_THRESHOLD: f64 = 1e-4;

/// Default maximum number of GM components per track.
pub const DEFAULT_MAX_GM_COMPONENTS: usize = 100;

/// Default Mahalanobis distance threshold for GM component merging.
/// Components closer than this threshold are merged.
/// Set to `f64::INFINITY` to disable merging (matches MATLAB behavior).
/// MATLAB does not perform Mahalanobis merging by default - it uses
/// weight-based pruning only. To match MATLAB exactly, use infinity.
pub const DEFAULT_GM_MERGE_THRESHOLD: f64 = f64::INFINITY;

/// Default maximum number of hypotheses for LMBM filters.
pub const DEFAULT_LMBM_MAX_HYPOTHESES: usize = 100;

/// Default hypothesis weight threshold for LMBM filters.
/// Hypotheses with weight below this threshold are pruned.
pub const DEFAULT_LMBM_WEIGHT_THRESHOLD: f64 = 1e-5;

/// Default maximum trajectory length for track history recording.
pub const DEFAULT_MAX_TRAJECTORY_LENGTH: usize = 1000;

/// Numerical zero threshold for avoiding division by zero and log(0).
/// Values below this threshold are treated as effectively zero.
pub const NUMERICAL_ZERO: f64 = 1e-15;

/// Threshold for detecting underflow in likelihood computations.
/// Values below this threshold use LOG_UNDERFLOW instead of ln().
pub const UNDERFLOW_THRESHOLD: f64 = 1e-300;
