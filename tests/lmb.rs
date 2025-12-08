//! LMB filter integration tests.
//!
//! Tests for the LMB (Labeled Multi-Bernoulli) filter family including
//! MATLAB equivalence, marginal evaluations, and algorithm validation.

#[path = "lmb/utils.rs"]
mod utils;

#[path = "lmb/matlab_equivalence.rs"]
mod matlab_equivalence;

#[path = "lmb/marginal_evaluations.rs"]
mod marginal_evaluations;

#[path = "lmb/gibbs_frequency.rs"]
mod gibbs_frequency;

#[path = "lmb/map_cardinality.rs"]
mod map_cardinality;
