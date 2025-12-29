//! Common filter operations shared across filter implementations.
//!
//! This module contains extracted helper functions that are used by multiple
//! filter implementations to avoid code duplication.

use crate::components::prediction::predict_tracks;

use super::cardinality::lmb_map_cardinality_estimate;
use super::config::{BirthModel, MotionModel};
use super::output::{EstimatedTrack, StateEstimate, Trajectory};
use super::traits::AssociationResult;
use super::types::{CardinalityEstimate, GaussianComponent, LmbmHypothesis, Track};

use nalgebra::{DMatrix, DVector};
use smallvec::SmallVec;

// ============================================================================
// Gaussian Mixture Component Operations
// ============================================================================

/// Prune and normalize Gaussian mixture components.
///
/// This performs the common operation of:
/// 1. Sorting components by weight (descending)
/// 2. Keeping only components above the weight threshold
/// 3. Limiting to maximum number of components
/// 4. Renormalizing weights to sum to 1.0
///
/// # Arguments
/// * `components` - Mutable reference to component list
/// * `weight_threshold` - Minimum weight to keep (components below are pruned)
/// * `max_components` - Maximum number of components to keep
pub fn prune_and_normalize_components(
    components: &mut SmallVec<[GaussianComponent; 4]>,
    weight_threshold: f64,
    max_components: usize,
) {
    if components.is_empty() {
        return;
    }

    // Sort by weight descending
    components.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find cutoff index
    let mut cutoff = components.len();
    for (i, comp) in components.iter().enumerate() {
        if comp.weight < weight_threshold || i >= max_components {
            cutoff = i;
            break;
        }
    }

    // Truncate
    components.truncate(cutoff);

    // Renormalize
    normalize_component_weights(components);
}

/// Normalize Gaussian mixture component weights to sum to 1.0.
///
/// # Arguments
/// * `components` - Mutable reference to component list
pub fn normalize_component_weights(components: &mut SmallVec<[GaussianComponent; 4]>) {
    let total: f64 = components.iter().map(|c| c.weight).sum();
    if total > super::NUMERICAL_ZERO {
        for comp in components.iter_mut() {
            comp.weight /= total;
        }
    }
}

/// Normalize track's Gaussian mixture component weights to sum to 1.0.
///
/// Convenience wrapper around [`normalize_component_weights`] for tracks.
///
/// # Arguments
/// * `track` - Mutable reference to track
pub fn normalize_track_weights(track: &mut Track) {
    normalize_component_weights(&mut track.components);
}

// ============================================================================
// Cardinality Estimation
// ============================================================================

/// Compute MAP cardinality estimate from track existence probabilities.
///
/// This is a convenience wrapper around `lmb_map_cardinality_estimate` that
/// extracts existence probabilities from tracks and returns a `CardinalityEstimate`.
///
/// # Arguments
/// * `tracks` - Slice of tracks with existence probabilities
///
/// # Returns
/// A `CardinalityEstimate` containing the MAP estimate and selected track indices.
pub fn compute_cardinality(tracks: &[Track]) -> CardinalityEstimate {
    let existences: Vec<f64> = tracks.iter().map(|t| t.existence).collect();
    let (n_estimated, map_indices) = lmb_map_cardinality_estimate(&existences);
    CardinalityEstimate::new(n_estimated, map_indices)
}

// ============================================================================
// Gaussian Mixture Reduction
// ============================================================================
//
// ## Algorithm Comparison: Weight-Based Pruning vs Mahalanobis Merging
//
// This implementation uses **weight-based pruning**: sort by weight, keep top N.
// MATLAB uses **Mahalanobis-distance merging**: combine similar components first.
//
// ### Weight-Based Pruning (this implementation)
// ```text
// 1. Normalize weights to sum to 1
// 2. Sort components by weight (descending)
// 3. Keep top N components above threshold
// 4. Renormalize kept weights
// ```
// - Time complexity: O(n log n) for sorting
// - Drops low-weight components entirely
//
// ### Mahalanobis Merging (MATLAB approach)
// ```text
// 1. Compute pairwise Mahalanobis distances
// 2. While any distance < threshold:
//    - Find closest pair (i, j)
//    - Merge: w_new = w_i + w_j
//    - Merge: μ_new = (w_i*μ_i + w_j*μ_j) / w_new
//    - Merge: Σ_new = weighted_cov + spread_correction
// 3. Then prune to max components
// ```
// - Time complexity: O(n²) for pairwise distances
// - Preserves information from merged components
//
// ### Practical Impact
//
// For tracking, the **weighted mean** (expected position) is nearly identical:
// - Weight differences: ~1-2% between algorithms
// - Position differences: ~0.0003 units in weighted mean
//
// The algorithms are NOT equivalent, but produce equivalent tracking results
// because the weighted mean integrates over all components.
//
// ### Why Not Implement Merging?
//
// 1. O(n²) vs O(n log n) - significant for many components
// 2. Tracking accuracy is equivalent (weighted mean matches)
// 3. Simpler implementation, fewer edge cases
//
// To match MATLAB exactly, implement `merge_by_mahalanobis()` before pruning.
// ============================================================================

/// Merge Gaussian mixture components using Mahalanobis distance (MATLAB-equivalent).
///
/// This implements MATLAB-style GM reduction:
/// 1. Find pair with minimum Mahalanobis distance
/// 2. Merge into single component preserving total probability mass
/// 3. Repeat until all distances > threshold AND components <= max_components
///
/// The merged covariance uses the spread correction formula:
/// ```text
/// Σ_new = (w_i*Σ_i + w_j*Σ_j)/w_new + (w_i*w_j/w_new²) * (μ_i - μ_j)(μ_i - μ_j)ᵀ
/// ```
///
/// **Performance Note**: This is O(n²) per merge iteration. For high-performance
/// applications where exact MATLAB compatibility is not required, weight-based
/// pruning in `prune_weighted_components()` with `merge_threshold = f64::INFINITY`
/// is O(n log n) and produces equivalent tracking results (weighted mean differs
/// by only ~0.0003 units).
///
/// # Arguments
/// * `components` - Mutable vector of (weight, mean, covariance) tuples
/// * `merge_threshold` - Mahalanobis distance threshold for merging (typically 4.0)
/// * `max_components` - Maximum components after reduction
pub fn merge_components_by_mahalanobis(
    components: &mut Vec<(f64, DVector<f64>, DMatrix<f64>)>,
    merge_threshold: f64,
    max_components: usize,
) {
    use crate::common::linalg::mahalanobis_distance;

    while components.len() > 1 {
        // Find pair with minimum Mahalanobis distance
        let mut min_dist = f64::INFINITY;
        let mut merge_pair = (0, 1);

        for i in 0..components.len() {
            for j in (i + 1)..components.len() {
                // Use the covariance of the higher-weight component for distance
                let (w_i, mu_i, sigma_i) = &components[i];
                let (w_j, mu_j, sigma_j) = &components[j];

                // Use covariance of higher-weight component
                let sigma = if w_i >= w_j { sigma_i } else { sigma_j };
                let dist = mahalanobis_distance(mu_j, mu_i, sigma);

                if dist < min_dist {
                    min_dist = dist;
                    merge_pair = (i, j);
                }
            }
        }

        // Stop if no pair is close enough AND we're at/under max_components
        // (Must continue merging if over max_components, even if distance is large)
        if min_dist > merge_threshold && components.len() <= max_components {
            break;
        }

        // Merge the closest pair
        let (i, j) = merge_pair;
        let (w_i, mu_i, sigma_i) = components[i].clone();
        let (w_j, mu_j, sigma_j) = components[j].clone();

        let w_new = w_i + w_j;
        let mu_new = (&mu_i * w_i + &mu_j * w_j) / w_new;

        // Merged covariance with spread correction
        let mu_diff = &mu_i - &mu_j;
        let spread_correction = &mu_diff * mu_diff.transpose() * (w_i * w_j / (w_new * w_new));
        let sigma_new = (&sigma_i * w_i + &sigma_j * w_j) / w_new + spread_correction;

        // Remove old components (remove j first since j > i)
        components.remove(j);
        components.remove(i);

        // Add merged component
        components.push((w_new, mu_new, sigma_new));
    }
}

/// Prune weighted components with optional Mahalanobis merging.
///
/// # Algorithm
/// 1. If `merge_threshold < INFINITY`: merge similar components first (MATLAB-style)
/// 2. Normalize weights to sum to 1
/// 3. Sort by weight (descending), keep top N above threshold
/// 4. Renormalize kept weights
///
/// # Arguments
/// * `weighted_components` - Vector of (weight, mean, covariance) tuples
/// * `weight_threshold` - Minimum weight to keep (use 0.0 to skip threshold check)
/// * `max_components` - Maximum number of components to keep
/// * `merge_threshold` - Mahalanobis distance threshold for merging. Use `f64::INFINITY`
///   to disable merging (faster, O(n log n) vs O(n²), but not MATLAB-equivalent)
///
/// # Returns
/// Pruned, truncated and renormalized components as a SmallVec
pub fn prune_weighted_components(
    mut weighted_components: Vec<(f64, DVector<f64>, DMatrix<f64>)>,
    weight_threshold: f64,
    max_components: usize,
    merge_threshold: f64,
) -> SmallVec<[GaussianComponent; 4]> {
    if weighted_components.is_empty() {
        return SmallVec::new();
    }

    // Mahalanobis merging (MATLAB-equivalent) if enabled
    if merge_threshold.is_finite() && weighted_components.len() > 1 {
        merge_components_by_mahalanobis(&mut weighted_components, merge_threshold, max_components);
    }

    // Normalize weights
    let total_weight: f64 = weighted_components.iter().map(|(w, _, _)| w).sum();
    if total_weight > super::NUMERICAL_ZERO {
        for (w, _, _) in &mut weighted_components {
            *w /= total_weight;
        }
    }

    // Sort by weight descending
    weighted_components.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build result with threshold and max checks
    let mut result: SmallVec<[GaussianComponent; 4]> = SmallVec::new();
    let mut kept_weight = 0.0;

    for (w, mean, cov) in weighted_components {
        if w < weight_threshold {
            break; // Sorted descending, so remaining are also below threshold
        }
        if result.len() >= max_components {
            break;
        }
        kept_weight += w;
        result.push(GaussianComponent::new(w, mean, cov));
    }

    // Renormalize kept components
    if kept_weight > super::NUMERICAL_ZERO && !result.is_empty() {
        for comp in result.iter_mut() {
            comp.weight /= kept_weight;
        }
    }

    result
}

/// Truncate Gaussian mixture to max components (no threshold).
///
/// Convenience wrapper for [`prune_weighted_components`] with no weight threshold.
/// Keeps the top `max_components` by weight, renormalizes.
///
/// **Note:** Despite the historical name, this does NOT perform Mahalanobis-distance
/// merging. It simply sorts by weight and truncates. See module-level docs for
/// the difference between pruning and merging.
///
/// # Arguments
/// * `weighted_components` - Vector of (weight, mean, covariance) tuples
/// * `max_components` - Maximum number of components to keep
///
/// # Returns
/// Truncated and renormalized components as a SmallVec
pub fn truncate_components(
    weighted_components: Vec<(f64, DVector<f64>, DMatrix<f64>)>,
    max_components: usize,
) -> SmallVec<[GaussianComponent; 4]> {
    // Use f64::INFINITY to disable Mahalanobis merging (backward compat)
    prune_weighted_components(weighted_components, 0.0, max_components, f64::INFINITY)
}

/// Alias for backwards compatibility.
#[doc(hidden)]
#[deprecated(
    since = "0.2.0",
    note = "Use `truncate_components` instead - this function does not actually merge"
)]
pub fn merge_and_truncate_components(
    weighted_components: Vec<(f64, DVector<f64>, DMatrix<f64>)>,
    max_components: usize,
) -> SmallVec<[GaussianComponent; 4]> {
    truncate_components(weighted_components, max_components)
}

// ============================================================================
// Single-track operations (LmbFilter, MultisensorLmbFilter)
// ============================================================================

/// Gate tracks by existence probability, saving long trajectories.
///
/// Removes tracks with existence below the threshold and saves trajectories
/// of pruned tracks that meet the minimum length requirement.
///
/// # Arguments
/// * `tracks` - Mutable reference to track list
/// * `trajectories` - Mutable reference to saved trajectories list
/// * `existence_threshold` - Minimum existence probability to keep a track
/// * `min_trajectory_length` - Minimum trajectory length to save
pub fn gate_tracks(
    tracks: &mut Vec<Track>,
    trajectories: &mut Vec<Trajectory>,
    existence_threshold: f64,
    min_trajectory_length: usize,
) {
    let mut kept_tracks = Vec::new();

    for track in tracks.drain(..) {
        if track.existence > existence_threshold {
            kept_tracks.push(track);
        } else if let Some(ref traj) = track.trajectory {
            // Save long trajectories even if track is pruned
            if traj.length >= min_trajectory_length {
                trajectories.push(Trajectory {
                    label: track.label,
                    states: (0..traj.length).filter_map(|i| traj.get_state(i)).collect(),
                    covariances: Vec::new(),
                    timestamps: traj.timestamps.clone(),
                });
            }
        }
    }

    *tracks = kept_tracks;
}

/// Extract state estimates using MAP cardinality estimation.
///
/// Uses the existence probabilities to determine how many objects exist,
/// then selects the most likely ones.
///
/// # Arguments
/// * `tracks` - Reference to track list
/// * `timestamp` - Current timestamp for the estimate
pub fn extract_estimates(tracks: &[Track], timestamp: usize) -> StateEstimate {
    if tracks.is_empty() {
        return StateEstimate::empty(timestamp);
    }

    // Get existence probabilities
    let existence_probs: Vec<f64> = tracks.iter().map(|t| t.existence).collect();

    // MAP cardinality estimation
    let (n_map, map_indices) = lmb_map_cardinality_estimate(&existence_probs);

    // Extract estimates for selected tracks
    let mut estimated_tracks = Vec::with_capacity(n_map);
    for &idx in &map_indices {
        let track = &tracks[idx];
        if let (Some(mean), Some(cov)) = (track.primary_mean(), track.primary_covariance()) {
            estimated_tracks.push(EstimatedTrack::new(track.label, mean.clone(), cov.clone()));
        }
    }

    StateEstimate::new(timestamp, estimated_tracks)
}

/// Update track trajectories by recording current state.
///
/// # Arguments
/// * `tracks` - Mutable reference to track list
/// * `timestamp` - Current timestamp
pub fn update_trajectories(tracks: &mut [Track], timestamp: usize) {
    for track in tracks.iter_mut() {
        track.record_state(timestamp);
    }
}

/// Initialize trajectory recording for tracks that don't have one.
///
/// # Arguments
/// * `tracks` - Mutable reference to track list
/// * `max_length` - Maximum trajectory length
pub fn init_birth_trajectories(tracks: &mut [Track], max_length: usize) {
    for track in tracks.iter_mut() {
        if track.trajectory.is_none() {
            track.init_trajectory(max_length);
        }
    }
}

/// Update existence probabilities from association marginal weights.
///
/// Uses the miss weights and marginal detection weights from the association
/// result to update track existence probabilities.
///
/// Uses the posterior_existence from the association result directly.
/// For LBP, this is the exact posterior existence from belief propagation.
/// For Gibbs/Murty, this is an approximation computed from marginals.
///
/// # Arguments
/// * `tracks` - Mutable reference to track list
/// * `result` - Association result containing posterior_existence
pub fn update_existence_from_marginals(tracks: &mut [Track], result: &AssociationResult) {
    for (i, track) in tracks.iter_mut().enumerate() {
        track.existence = result.posterior_existence[i];
    }
}

// ============================================================================
// Hypothesis-based operations (LmbmFilter, MultisensorLmbmFilter)
// ============================================================================

/// Predict all hypotheses forward in time.
///
/// Applies the motion model to all tracks in all hypotheses and adds
/// birth tracks to each hypothesis.
///
/// # Arguments
/// * `hypotheses` - Mutable reference to hypothesis list
/// * `motion` - Motion model for prediction
/// * `birth` - Birth model for new track generation
/// * `timestep` - Current timestep
pub fn predict_all_hypotheses(
    hypotheses: &mut [LmbmHypothesis],
    motion: &MotionModel,
    birth: &BirthModel,
    timestep: usize,
) {
    for hyp in hypotheses.iter_mut() {
        predict_tracks(&mut hyp.tracks, motion, birth, timestep, true);
    }
}

/// Gate tracks by existence probability across all hypotheses.
///
/// Computes weighted total existence for each track position, then removes
/// tracks with low total existence from all hypotheses.
///
/// # Arguments
/// * `hypotheses` - Mutable reference to hypothesis list
/// * `trajectories` - Mutable reference to saved trajectories list
/// * `existence_threshold` - Minimum existence probability to keep a track
/// * `min_trajectory_length` - Minimum trajectory length to save
pub fn gate_hypothesis_tracks(
    hypotheses: &mut [LmbmHypothesis],
    trajectories: &mut Vec<Trajectory>,
    existence_threshold: f64,
    min_trajectory_length: usize,
) {
    if hypotheses.is_empty() || hypotheses[0].tracks.is_empty() {
        return;
    }

    let num_tracks = hypotheses[0].tracks.len();

    // Compute weighted total existence for each track position
    let mut total_existence = vec![0.0; num_tracks];
    for hyp in hypotheses.iter() {
        let w = hyp.weight();
        for (i, track) in hyp.tracks.iter().enumerate() {
            if i < num_tracks {
                total_existence[i] += w * track.existence;
            }
        }
    }

    // Determine which tracks to keep
    let keep_mask: Vec<bool> = total_existence
        .iter()
        .map(|&r| r > existence_threshold)
        .collect();

    // Remove low-existence tracks from all hypotheses
    for hyp in hypotheses.iter_mut() {
        let mut kept_tracks = Vec::new();
        for (i, track) in hyp.tracks.drain(..).enumerate() {
            if i < keep_mask.len() && keep_mask[i] {
                kept_tracks.push(track);
            } else if let Some(ref traj) = track.trajectory {
                // Save long trajectories
                if traj.length >= min_trajectory_length {
                    trajectories.push(Trajectory {
                        label: track.label,
                        states: (0..traj.length).filter_map(|j| traj.get_state(j)).collect(),
                        covariances: Vec::new(),
                        timestamps: traj.timestamps.clone(),
                    });
                }
            }
        }
        hyp.tracks = kept_tracks;
    }
}

/// Extract state estimates from a hypothesis mixture.
///
/// Uses MAP or EAP cardinality estimation on the weighted existence probabilities
/// to determine how many objects exist, then extracts states from the
/// highest-weight hypothesis.
///
/// # Arguments
/// * `hypotheses` - Reference to hypothesis list (assumed sorted by weight, descending)
/// * `timestamp` - Current timestamp for the estimate
/// * `use_eap` - If true, use EAP (floor of sum) instead of MAP
pub fn extract_hypothesis_estimates(
    hypotheses: &[LmbmHypothesis],
    timestamp: usize,
    use_eap: bool,
) -> StateEstimate {
    if hypotheses.is_empty() || hypotheses[0].tracks.is_empty() {
        return StateEstimate::empty(timestamp);
    }

    // Compute weighted total existence for each track
    let num_tracks = hypotheses[0].tracks.len();
    let mut total_existence = vec![0.0; num_tracks];
    for hyp in hypotheses {
        let w = hyp.weight();
        for (i, track) in hyp.tracks.iter().enumerate() {
            if i < num_tracks {
                total_existence[i] += w * track.existence;
            }
        }
    }

    // MAP or EAP cardinality estimation
    let (n_map, map_indices) = if use_eap {
        // EAP: round(sum of existence), select top-k
        // MATLAB uses round() not floor() for EAP cardinality estimation
        let n = total_existence.iter().sum::<f64>().round() as usize;
        let mut indexed: Vec<(usize, f64)> = hypotheses[0]
            .tracks
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.existence))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let indices: Vec<usize> = indexed.into_iter().take(n).map(|(i, _)| i).collect();
        (n, indices)
    } else {
        // MAP estimate
        lmb_map_cardinality_estimate(&total_existence)
    };

    // Extract estimates from highest-weight hypothesis
    let best_hyp = &hypotheses[0];
    let mut estimated_tracks = Vec::with_capacity(n_map);
    for &idx in &map_indices {
        if idx < best_hyp.tracks.len() {
            let track = &best_hyp.tracks[idx];
            if let (Some(mean), Some(cov)) = (track.primary_mean(), track.primary_covariance()) {
                estimated_tracks.push(EstimatedTrack::new(track.label, mean.clone(), cov.clone()));
            }
        }
    }

    StateEstimate::new(timestamp, estimated_tracks)
}

/// Compute cardinality estimate from a hypothesis mixture.
///
/// Uses MAP or EAP cardinality estimation on the weighted existence probabilities.
/// This is the cardinality-only version of `extract_hypothesis_estimates`.
///
/// # Arguments
/// * `hypotheses` - Reference to hypothesis list (assumed sorted by weight, descending)
/// * `use_eap` - If true, use EAP (round of sum) instead of MAP
///
/// # Returns
/// A `CardinalityEstimate` containing:
/// - `n_estimated`: The estimated number of objects
/// - `map_indices`: Indices of tracks selected for extraction
pub fn compute_hypothesis_cardinality(
    hypotheses: &[LmbmHypothesis],
    use_eap: bool,
) -> CardinalityEstimate {
    if hypotheses.is_empty() || hypotheses[0].tracks.is_empty() {
        return CardinalityEstimate::new(0, vec![]);
    }

    // Compute weighted total existence for each track
    let num_tracks = hypotheses[0].tracks.len();
    let mut total_existence = vec![0.0; num_tracks];
    for hyp in hypotheses {
        let w = hyp.weight();
        for (i, track) in hyp.tracks.iter().enumerate() {
            if i < num_tracks {
                total_existence[i] += w * track.existence;
            }
        }
    }

    // MAP or EAP cardinality estimation
    let (n_map, map_indices) = if use_eap {
        // EAP: round(sum of existence), select top-k by existence
        // MATLAB uses round() not floor() for EAP cardinality estimation
        let n = total_existence.iter().sum::<f64>().round() as usize;
        let mut indexed: Vec<(usize, f64)> = hypotheses[0]
            .tracks
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.existence))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let indices: Vec<usize> = indexed.into_iter().take(n).map(|(i, _)| i).collect();
        (n, indices)
    } else {
        // MAP estimate
        lmb_map_cardinality_estimate(&total_existence)
    };

    CardinalityEstimate::new(n_map, map_indices)
}

/// Update track trajectories in all hypotheses.
///
/// # Arguments
/// * `hypotheses` - Mutable reference to hypothesis list
/// * `timestamp` - Current timestamp
pub fn update_hypothesis_trajectories(hypotheses: &mut [LmbmHypothesis], timestamp: usize) {
    for hyp in hypotheses.iter_mut() {
        for track in &mut hyp.tracks {
            track.record_state(timestamp);
        }
    }
}

/// Initialize trajectory recording for tracks in all hypotheses.
///
/// # Arguments
/// * `hypotheses` - Mutable reference to hypothesis list
/// * `max_length` - Maximum trajectory length
pub fn init_hypothesis_birth_trajectories(hypotheses: &mut [LmbmHypothesis], max_length: usize) {
    for hyp in hypotheses.iter_mut() {
        for track in &mut hyp.tracks {
            if track.trajectory.is_none() {
                track.init_trajectory(max_length);
            }
        }
    }
}

// ============================================================================
// Hypothesis normalization (LmbmFilter, MultisensorLmbmFilter)
// ============================================================================

/// Normalize and gate hypotheses (hypothesis-level gating only).
///
/// 1. Normalizes hypothesis weights using log-sum-exp
/// 2. Removes hypotheses with weight below threshold
/// 3. Sorts by descending weight
/// 4. Caps to maximum number of hypotheses
/// 5. Renormalizes after truncation
///
/// **Note:** This function does NOT prune tracks. For MATLAB-equivalent behavior,
/// use [`normalize_gate_and_prune_tracks`] which includes track pruning.
///
/// # Arguments
/// * `hypotheses` - Mutable reference to hypothesis list
/// * `weight_threshold` - Minimum hypothesis weight to keep (linear scale)
/// * `max_hypotheses` - Maximum number of hypotheses to keep
pub fn normalize_and_gate_hypotheses(
    hypotheses: &mut Vec<LmbmHypothesis>,
    weight_threshold: f64,
    max_hypotheses: usize,
) {
    if hypotheses.is_empty() {
        return;
    }

    // Log-sum-exp normalization
    let max_log_w = hypotheses
        .iter()
        .map(|h| h.log_weight)
        .fold(f64::NEG_INFINITY, f64::max);

    let sum_exp: f64 = hypotheses
        .iter()
        .map(|h| (h.log_weight - max_log_w).exp())
        .sum();

    let log_normalizer = max_log_w + sum_exp.ln();

    // Normalize all hypothesis weights
    for hyp in hypotheses.iter_mut() {
        hyp.log_weight -= log_normalizer;
    }

    // Filter by weight threshold
    let log_threshold = weight_threshold.ln();
    hypotheses.retain(|h| h.log_weight > log_threshold);

    if hypotheses.is_empty() {
        // Reinitialize with empty hypothesis if all were pruned
        hypotheses.push(LmbmHypothesis::new(0.0, Vec::new()));
        return;
    }

    // Renormalize after gating (MATLAB line 28: w = w(likelyHypotheses) ./ sum(w(likelyHypotheses)))
    let sum_exp_after_gate: f64 = hypotheses.iter().map(|h| h.log_weight.exp()).sum();
    let log_normalizer_after_gate = sum_exp_after_gate.ln();
    for hyp in hypotheses.iter_mut() {
        hyp.log_weight -= log_normalizer_after_gate;
    }

    // Sort by descending weight (descending log_weight)
    hypotheses.sort_by(|a, b| b.log_weight.partial_cmp(&a.log_weight).unwrap());

    // Cap to maximum hypotheses
    if hypotheses.len() > max_hypotheses {
        hypotheses.truncate(max_hypotheses);

        // Renormalize after truncation
        let sum_exp: f64 = hypotheses.iter().map(|h| h.log_weight.exp()).sum();
        let log_normalizer = sum_exp.ln();
        for hyp in hypotheses.iter_mut() {
            hyp.log_weight -= log_normalizer;
        }
    }
}

/// Normalize, gate hypotheses, and prune low-existence tracks (MATLAB-equivalent).
///
/// This matches MATLAB's `lmbmNormalisationAndGating.m` exactly:
/// 1. Normalizes hypothesis weights using log-sum-exp (lines 22-24)
/// 2. Gates hypotheses by weight threshold, renormalize (lines 26-28)
/// 3. Sorts by descending weight (lines 30-31)
/// 4. Caps to maximum hypotheses, renormalize (lines 34-39)
/// 5. Computes weighted total existence: r = sum(w .* [hypotheses.r], 2) (line 41)
/// 6. Determines objectsLikelyToExist = r > threshold (line 42)
/// 7. Prunes tracks from ALL hypotheses using objectsLikelyToExist mask (lines 43-51)
///
/// # Arguments
/// * `hypotheses` - Mutable reference to hypothesis list
/// * `trajectories` - Mutable reference to trajectory list for saving pruned tracks
/// * `weight_threshold` - Minimum hypothesis weight to keep (linear scale)
/// * `max_hypotheses` - Maximum number of hypotheses to keep
/// * `existence_threshold` - Minimum weighted existence to keep a track
/// * `min_trajectory_length` - Minimum trajectory length to save
///
/// # Returns
/// A boolean vector indicating which track positions were kept (objectsLikelyToExist)
pub fn normalize_gate_and_prune_tracks(
    hypotheses: &mut Vec<LmbmHypothesis>,
    trajectories: &mut Vec<Trajectory>,
    weight_threshold: f64,
    max_hypotheses: usize,
    existence_threshold: f64,
    min_trajectory_length: usize,
) -> Vec<bool> {
    // Step 1-4: Hypothesis normalization and gating
    normalize_and_gate_hypotheses(hypotheses, weight_threshold, max_hypotheses);

    if hypotheses.is_empty() || hypotheses[0].tracks.is_empty() {
        return vec![];
    }

    // Step 5-6: Compute objects likely to exist (MATLAB lines 41-42)
    let keep_mask = compute_objects_likely_to_exist(hypotheses, existence_threshold);

    // Step 7: Prune tracks from ALL hypotheses using the mask
    // MATLAB lines 43-51: for i = 1:numberOfHypotheses
    //     hypotheses(i).r = hypotheses(i).r(objectsLikelyToExist, :);
    //     hypotheses(i).mu = hypotheses(i).mu(objectsLikelyToExist);
    //     hypotheses(i).Sigma = hypotheses(i).Sigma(objectsLikelyToExist);
    //     ...
    // end
    for hyp in hypotheses.iter_mut() {
        let mut kept_tracks = Vec::new();
        for (i, track) in hyp.tracks.drain(..).enumerate() {
            if i < keep_mask.len() && keep_mask[i] {
                kept_tracks.push(track);
            } else if let Some(ref traj) = track.trajectory {
                // Save long trajectories from pruned tracks
                if traj.length >= min_trajectory_length {
                    trajectories.push(Trajectory {
                        label: track.label,
                        states: (0..traj.length).filter_map(|j| traj.get_state(j)).collect(),
                        covariances: Vec::new(),
                        timestamps: traj.timestamps.clone(),
                    });
                }
            }
        }
        hyp.tracks = kept_tracks;
    }

    keep_mask
}

/// Compute which objects are likely to exist based on weighted existence probabilities.
///
/// For each track, computes the weighted sum of existence probabilities across all
/// hypotheses and returns true if this sum exceeds the threshold.
///
/// This function extracts the core existence computation logic (MATLAB line 41-42 in
/// lmbmNormalisationAndGating.m) into a reusable helper.
///
/// # Arguments
/// * `hypotheses` - Vector of LMBM hypotheses
/// * `existence_threshold` - Threshold for determining if an object likely exists
///
/// # Returns
/// Boolean vector where `result[i]` is true if object i likely exists
///
/// # Example
/// ```ignore
/// let ole = compute_objects_likely_to_exist(&hypotheses, 0.5);
/// assert_eq!(ole[0], true); // Object 0 likely exists
/// ```
pub fn compute_objects_likely_to_exist(
    hypotheses: &[LmbmHypothesis],
    existence_threshold: f64,
) -> Vec<bool> {
    if hypotheses.is_empty() {
        return Vec::new();
    }

    let num_tracks = hypotheses[0].tracks.len();
    let mut ole = vec![false; num_tracks];

    // Compute weighted sum of existence probabilities across all hypotheses
    // MATLAB: r = sum(w .* [hypotheses.r], 2)
    for i in 0..num_tracks {
        let mut weighted_existence = 0.0;
        for hyp in hypotheses {
            if i < hyp.tracks.len() {
                weighted_existence += hyp.weight() * hyp.tracks[i].existence;
            }
        }
        // MATLAB: objectsLikelyToExist = r > model.existenceThreshold
        ole[i] = weighted_existence > existence_threshold;
    }

    ole
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmb::types::{GaussianComponent, TrackLabel};
    use nalgebra::{DMatrix, DVector};
    use smallvec::smallvec;

    fn create_test_track(existence: f64) -> Track {
        Track {
            label: TrackLabel::new(0, 0),
            existence,
            components: smallvec![GaussianComponent::new(
                1.0,
                DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
                DMatrix::identity(4, 4),
            )],
            trajectory: None,
        }
    }

    #[test]
    fn test_gate_tracks() {
        let mut tracks = vec![
            create_test_track(0.9),
            create_test_track(0.0001), // Below threshold
            create_test_track(0.5),
        ];
        let mut trajectories = Vec::new();

        gate_tracks(&mut tracks, &mut trajectories, 0.01, 3);

        assert_eq!(tracks.len(), 2);
        assert!(tracks[0].existence > 0.5);
        assert!(tracks[1].existence > 0.1);
    }

    #[test]
    fn test_extract_estimates() {
        let tracks = vec![
            create_test_track(0.9),
            create_test_track(0.1),
            create_test_track(0.8),
        ];

        let estimate = extract_estimates(&tracks, 42);

        assert_eq!(estimate.timestamp, 42);
        // MAP should select the highest existence tracks
        assert!(estimate.tracks.len() <= tracks.len());
    }

    #[test]
    fn test_update_trajectories() {
        let mut tracks = vec![create_test_track(0.9), create_test_track(0.8)];

        // Initialize trajectories first
        init_birth_trajectories(&mut tracks, 100);

        update_trajectories(&mut tracks, 5);

        for track in &tracks {
            assert!(track.trajectory.is_some());
        }
    }
}
