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
//! - `MultisensorLmbmFilter` - Multi-sensor LMBM (coming soon)

pub mod errors;
pub mod lmb;
pub mod lmbm;
pub mod multisensor_lmb;
pub mod traits;

pub use errors::{AssociationError, FilterError};
pub use lmb::LmbFilter;
pub use lmbm::LmbmFilter;
pub use multisensor_lmb::{
    AaLmbFilter, ArithmeticAverageMerger, GaLmbFilter, GeometricAverageMerger, IcLmbFilter,
    IteratedCorrectorMerger, MultisensorLmbFilter, MultisensorMeasurements, ParallelUpdateMerger,
    PuLmbFilter,
};
pub use traits::{Associator, Filter, Merger, Updater};
