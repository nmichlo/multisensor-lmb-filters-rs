//! Data association algorithms and likelihood computation
//!
//! This module provides:
//! - [`likelihood`] - Core likelihood computation
//! - [`builder`] - Association matrix construction
//! - [`lbp`] - Loopy Belief Propagation
//! - [`gibbs`] - Gibbs sampling
//! - [`murtys`] - Murty's k-best algorithm
//! - [`hungarian`] - Hungarian algorithm for assignment

pub mod builder;
pub mod likelihood;

// Re-export existing algorithms from common (will be moved later)
pub use crate::common::association::{
    gibbs, hungarian, lbp, murtys,
};

pub use likelihood::{compute_likelihood, LikelihoodResult, LikelihoodWorkspace};
pub use builder::{AssociationBuilder, AssociationMatrices, PosteriorGrid};
