//! Core algorithmic components
//!
//! This module provides the shared algorithmic building blocks used by all filters:
//!
//! - [`prediction`] - Track prediction (Chapman-Kolmogorov)
//! - [`update`] - Existence probability updates

pub mod prediction;
pub mod update;

pub use prediction::predict_tracks;
pub use update::{update_existence_no_detection, update_existence_no_detection_multisensor};
