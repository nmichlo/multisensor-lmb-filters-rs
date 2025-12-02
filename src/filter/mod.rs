//! Filter implementations for multi-object tracking.
//!
//! This module provides the unified filter trait and implementations:
//!
//! - [`Filter`] - Core trait implemented by all filters
//! - [`Associator`] - Trait for data association algorithms
//! - [`Merger`] - Trait for multi-sensor track merging
//! - [`Updater`] - Trait for track update strategies
//!
//! # Filter Types
//!
//! ## Single-Sensor Filters
//!
//! - [`LmbFilter`] - Single-sensor LMB filter with marginal association
//! - [`LmbmFilter`] - Single-sensor LMBM filter with hypothesis management
//!
//! ## Multi-Sensor Filters
//!
//! - [`MultisensorLmbFilter`] - Multi-sensor LMB with configurable fusion
//! - [`AaLmbFilter`] - Arithmetic Average fusion (fast, simple)
//! - [`GaLmbFilter`] - Geometric Average fusion (conservative covariance)
//! - [`PuLmbFilter`] - Parallel Update fusion (optimal for independent sensors)
//! - [`IcLmbFilter`] - Iterated Corrector (sequential sensor updates)
//! - [`MultisensorLmbmFilter`] - Multi-sensor LMBM with joint association

pub mod common_ops;
pub mod errors;
pub mod lmb;
pub mod lmbm;
pub mod multisensor_lmb;
pub mod multisensor_lmbm;
pub mod multisensor_traits;
pub mod traits;

pub use errors::{AssociationError, FilterError};
pub use lmb::LmbFilter;
pub use lmbm::LmbmFilter;
pub use multisensor_lmb::{
    AaLmbFilter, ArithmeticAverageMerger, GaLmbFilter, GeometricAverageMerger, IcLmbFilter,
    IteratedCorrectorMerger, MultisensorLmbFilter, MultisensorMeasurements, ParallelUpdateMerger,
    PuLmbFilter,
};
pub use multisensor_lmbm::MultisensorLmbmFilter;
pub use multisensor_traits::{MultisensorAssociationResult, MultisensorAssociator, MultisensorGibbsAssociator};
pub use traits::{Associator, Filter, Merger, Updater};

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

/// Default maximum number of hypotheses for LMBM filters.
pub const DEFAULT_LMBM_MAX_HYPOTHESES: usize = 100;

/// Default hypothesis weight threshold for LMBM filters.
/// Hypotheses with weight below this threshold are pruned.
pub const DEFAULT_LMBM_WEIGHT_THRESHOLD: f64 = 1e-5;
