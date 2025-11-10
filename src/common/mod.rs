/*!
Common utilities and shared components for tracking algorithms.

This module contains data structures, model generation, ground truth simulation,
data association algorithms, and evaluation metrics used across all filter implementations.
*/

pub mod types;
pub mod model;
pub mod ground_truth;
pub mod linalg;
pub mod metrics;
pub mod association;
pub mod utils;
pub mod simple_rng;
