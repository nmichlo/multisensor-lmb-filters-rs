//! Single-sensor LMB and LMBM filter implementations.
//!
//! This module contains the standard single-sensor variants:
//!
//! - [`LmbFilter`][super::core::LmbFilter] - Labeled Multi-Bernoulli filter (now in core.rs)
//! - [`LmbmFilter`][super::core_lmbm::LmbmFilter] - Labeled Multi-Bernoulli Mixture filter (now in core_lmbm.rs)

pub mod lmbm;

// Legacy re-export for backward compatibility (now in core_lmbm.rs)
pub use lmbm::LmbmFilter;
