/*!
Multi-sensor LMB filter variants.

Implements various sensor fusion strategies:
- PU-LMB (Parallel Update) - most accurate
- IC-LMB (Iterated Corrector) - sequential updates
- GA-LMB (Geometric Average) - good localization
- AA-LMB (Arithmetic Average) - better cardinality
*/

pub mod parallel_update;
pub mod iterated_corrector;
pub mod merging;
pub mod association;
pub mod utils;

// Main filter entry points
// pub fn run_parallel_update_lmb_filter() { todo!() }
// pub fn run_ic_lmb_filter() { todo!() }
