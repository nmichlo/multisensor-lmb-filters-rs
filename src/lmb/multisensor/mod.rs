//! Multi-sensor LMB and LMBM filter implementations.
//!
//! This module contains multi-sensor variants of the LMB tracking algorithms:
//!
//! # Multi-sensor LMB
//!
//! - [`MultisensorLmbFilter`] - Generic multi-sensor LMB with configurable fusion
//! - [`AaLmbFilter`] - Arithmetic Average fusion
//! - [`GaLmbFilter`] - Geometric Average fusion
//! - [`PuLmbFilter`] - Parallel Update fusion
//! - [`IcLmbFilter`] - Iterated Corrector fusion
//!
//! # Multi-sensor LMBM
//!
//! - [`MultisensorLmbmFilter`] - Multi-sensor LMBM with joint association
//!
//! # Fusion Strategies
//!
//! - [`ArithmeticAverageMerger`] - Simple weighted average
//! - [`GeometricAverageMerger`] - Covariance intersection
//! - [`ParallelUpdateMerger`] - Information-form fusion
//! - [`IteratedCorrectorMerger`] - Sequential sensor updates

pub mod lmb;
pub mod lmbm;
pub mod traits;

// Re-export from lmb.rs
pub use lmb::{
    AaLmbFilter,
    ArithmeticAverageMerger,
    GaLmbFilter,
    GeometricAverageMerger,
    IcLmbFilter,
    IteratedCorrectorMerger,
    MultisensorLmbFilter,
    MultisensorMeasurements,
    ParallelUpdateMerger,
    PuLmbFilter,
};

// Re-export from lmbm.rs
pub use lmbm::MultisensorLmbmFilter;

// Re-export from traits.rs
pub use traits::{
    MultisensorAssociationResult,
    MultisensorAssociator,
    MultisensorGibbsAssociator,
};
