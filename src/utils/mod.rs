//! Common utilities and shared components for tracking algorithms.
//!
//! This module contains data association algorithms, linear algebra utilities,
//! and numerical constants used by filter implementations.

pub mod association;
pub mod constants;
pub mod linalg;
pub mod rng;
pub mod utils;
pub mod update;
pub mod prediction;
pub mod gibbs;
pub mod hungarian;
pub mod lbp;
pub mod murtys;
// Filter infrastructure
pub mod common_ops;
/// Benchmark utilities (scenario loading, filter factory)
pub mod bench;