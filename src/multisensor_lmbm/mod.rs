/*!
Multi-sensor LMBM filter implementation.

Exact solution for multi-sensor tracking, memory-intensive but accurate.
*/

pub mod association;
pub mod filter;
pub mod gibbs;
pub mod hypothesis;
pub mod update;

pub use association::{generate_multisensor_lmbm_association_matrices, MultisensorLmbmPosteriorParameters};
pub use filter::{run_multisensor_lmbm_filter, MultisensorLmbmStateEstimates};
pub use gibbs::multisensor_lmbm_gibbs_sampling;
pub use hypothesis::determine_multisensor_posterior_hypothesis_parameters;
