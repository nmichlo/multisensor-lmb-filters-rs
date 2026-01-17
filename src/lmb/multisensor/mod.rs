//! Multi-sensor LMB and LMBM filter implementations.
//!
//! This module contains multi-sensor variants of the LMB tracking algorithms:
//!
//! # Multi-sensor LMB
//!
//! Multi-sensor LMB filters are now in `core.rs` - use type aliases:
//! - [`AaLmbFilter`][super::core::AaLmbFilter] - Arithmetic Average fusion
//! - [`GaLmbFilter`][super::core::GaLmbFilter] - Geometric Average fusion
//! - [`PuLmbFilter`][super::core::PuLmbFilter] - Parallel Update fusion
//! - [`IcLmbFilter`][super::core::IcLmbFilter] - Iterated Corrector fusion
//!
//! # Multi-sensor LMBM
//!
//! Multi-sensor LMBM filter is now in `core_lmbm.rs`:
//! - [`MultisensorLmbmFilter`][super::core_lmbm::MultisensorLmbmFilter]
//!
//! # Fusion Strategies
//!
//! - [`ArithmeticAverageMerger`] - Simple weighted average
//! - [`GeometricAverageMerger`] - Covariance intersection
//! - [`ParallelUpdateMerger`] - Information-form fusion
//! - [`IteratedCorrectorMerger`] - Sequential sensor updates

use nalgebra::DVector;

pub mod fusion;
pub mod traits;

// Re-export fusion strategies
pub use fusion::{
    ArithmeticAverageMerger, GeometricAverageMerger, IteratedCorrectorMerger, ParallelUpdateMerger,
};

// Re-export from traits.rs
pub use traits::{MultisensorAssociationResult, MultisensorAssociator, MultisensorGibbsAssociator};

/// Multi-sensor measurements: one measurement set per sensor.
pub type MultisensorMeasurements = Vec<Vec<DVector<f64>>>;
