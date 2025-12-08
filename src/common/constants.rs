//! Numerical constants used throughout the tracking algorithms
//!
//! These constants are chosen to match MATLAB behavior and ensure numerical stability.
//! They are intentionally separate from the Model parameters which are user-configurable.

/// Epsilon for existence probability comparisons
///
/// Used when checking if a probability is effectively zero (e.g., for normalization).
/// This is smaller than typical floating-point epsilon to avoid premature cutoff.
///
/// # MATLAB Reference
/// Matches implicit threshold used in MATLAB for division by near-zero values.
pub const EPSILON_EXISTENCE: f64 = 1e-15;

/// Epsilon for ESF (Elementary Symmetric Function) adjustment
///
/// Used in MAP cardinality estimation to avoid unit existence probabilities
/// which would cause log(0) = -Inf in ESF computation.
///
/// # MATLAB Reference
/// Matches MATLAB lmbMapCardinalityEstimate.m line 19:
/// `r = r - 1e-6;  % Does not work with unit existence probabilities`
pub const ESF_ADJUSTMENT: f64 = 1e-6;

/// Tolerance for SVD pseudo-inverse computation
///
/// Singular values below this threshold are treated as zero.
/// Used in robust_inverse() as last resort fallback.
pub const SVD_TOLERANCE: f64 = 1e-10;

/// Default LBP convergence tolerance
///
/// Used when checking if loopy belief propagation has converged.
/// This is the default value; can be overridden via Model.lbp_convergence_tolerance.
pub const DEFAULT_LBP_TOLERANCE: f64 = 1e-6;

/// Default GM (Gaussian Mixture) weight threshold
///
/// Components with weights below this are pruned during mixture reduction.
/// This is the default value; can be overridden via Model.gm_weight_threshold.
pub const DEFAULT_GM_WEIGHT_THRESHOLD: f64 = 1e-6;
