//! Filter implementations
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
//! - `LmbFilter` - Single-sensor LMB filter (coming soon)
//! - `LmbmFilter` - Single-sensor LMBM filter (coming soon)
//! - `MultisensorLmbFilter<M: Merger>` - Multi-sensor LMB (coming soon)
//! - `MultisensorLmbmFilter` - Multi-sensor LMBM (coming soon)

pub mod errors;
pub mod traits;

pub use errors::{AssociationError, FilterError};
pub use traits::{Associator, Filter, Merger, Updater};
