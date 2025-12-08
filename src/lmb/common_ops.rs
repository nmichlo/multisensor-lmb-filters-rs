//! Common filter operations shared across filter implementations.
//!
//! This module contains extracted helper functions that are used by multiple
//! filter implementations to avoid code duplication.

use crate::components::prediction::predict_tracks;

use super::cardinality::lmb_map_cardinality_estimate;
use super::config::{BirthModel, MotionModel};
use super::output::{EstimatedTrack, StateEstimate, Trajectory};
use super::types::{GaussianComponent, LmbmHypothesis, Track};
use super::traits::AssociationResult;

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
    components.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));

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

/// Prune weighted components by threshold, truncate to max, and normalize.
///
/// This is the general form used when building new component mixtures
/// from (weight, mean, covariance) tuples.
///
/// # Arguments
/// * `weighted_components` - Vector of (weight, mean, covariance) tuples
/// * `weight_threshold` - Minimum weight to keep (use 0.0 to skip threshold check)
/// * `max_components` - Maximum number of components to keep
///
/// # Returns
/// Pruned, truncated and renormalized components as a SmallVec
pub fn prune_weighted_components(
    mut weighted_components: Vec<(f64, DVector<f64>, DMatrix<f64>)>,
    weight_threshold: f64,
    max_components: usize,
) -> SmallVec<[GaussianComponent; 4]> {
    if weighted_components.is_empty() {
        return SmallVec::new();
    }

    // First normalize weights
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

/// Merge Gaussian mixture components by sorting and truncating.
///
/// Convenience wrapper for [`prune_weighted_components`] with no threshold.
///
/// # Arguments
/// * `weighted_components` - Vector of (weight, mean, covariance) tuples
/// * `max_components` - Maximum number of components to keep
///
/// # Returns
/// Truncated and renormalized components as a SmallVec
pub fn merge_and_truncate_components(
    weighted_components: Vec<(f64, DVector<f64>, DMatrix<f64>)>,
    max_components: usize,
) -> SmallVec<[GaussianComponent; 4]> {
    prune_weighted_components(weighted_components, 0.0, max_components)
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
                    states: (0..traj.length)
                        .filter_map(|i| traj.get_state(i))
                        .collect(),
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
/// result to update track existence probabilities. If strongly associated with
/// measurements, existence is boosted; if mostly miss, existence is reduced.
///
/// # Arguments
/// * `tracks` - Mutable reference to track list
/// * `result` - Association result containing miss_weights and marginal_weights
pub fn update_existence_from_marginals(tracks: &mut [Track], result: &AssociationResult) {
    for (i, track) in tracks.iter_mut().enumerate() {
        let miss_weight = result.miss_weights[i];
        let detection_weight: f64 = (0..result.marginal_weights.ncols())
            .map(|j| result.marginal_weights[(i, j)])
            .sum();

        let total = miss_weight + detection_weight;
        if total > super::NUMERICAL_ZERO {
            // Weighted update: detection increases confidence
            let detection_ratio = detection_weight / total;
            // Interpolate between current existence and boosted value based on detection
            track.existence = track.existence * (1.0 - detection_ratio * 0.5)
                + detection_ratio * 0.5 * 1.0_f64.min(track.existence * 2.0);
        }
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
                        states: (0..traj.length)
                            .filter_map(|j| traj.get_state(j))
                            .collect(),
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
        // EAP: floor(sum of existence), select top-k
        let n = total_existence.iter().sum::<f64>().floor() as usize;
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

/// Normalize and gate hypotheses.
///
/// 1. Normalizes hypothesis weights using log-sum-exp
/// 2. Removes hypotheses with weight below threshold
/// 3. Sorts by descending weight
/// 4. Caps to maximum number of hypotheses
/// 5. Renormalizes after truncation
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
