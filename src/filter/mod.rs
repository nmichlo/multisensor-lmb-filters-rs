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
//! - [`LmbFilter`] - Single-sensor LMB filter
//! - [`LmbmFilter`] - Single-sensor LMBM filter
//! - `MultisensorLmbFilter<M: Merger>` - Multi-sensor LMB (coming soon)
//! - `MultisensorLmbmFilter` - Multi-sensor LMBM (coming soon)

pub mod errors;
pub mod lmb;
pub mod lmbm;
pub mod traits;

pub use errors::{AssociationError, FilterError};
pub use lmb::LmbFilter;
pub use lmbm::LmbmFilter;
pub use traits::{Associator, Filter, Merger, Updater};
