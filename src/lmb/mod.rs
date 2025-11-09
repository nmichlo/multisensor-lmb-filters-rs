/*!
Single-sensor Labelled Multi-Bernoulli (LMB) filter implementation.

Supports multiple data association methods:
- Loopy Belief Propagation (LBP)
- Gibbs sampling
- Murty's algorithm
*/

pub mod prediction;
pub mod update;
pub mod gibbs_sampling;
pub mod murtys;

// Main filter entry point
// pub fn run_lmb_filter() { todo!() }
