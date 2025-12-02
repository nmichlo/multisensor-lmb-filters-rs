//! Common filter operations shared across filter implementations.
//!
//! This module contains extracted helper functions that are used by multiple
//! filter implementations to avoid code duplication.

use crate::lmb::cardinality::lmb_map_cardinality_estimate;
use crate::types::{EstimatedTrack, LmbmHypothesis, StateEstimate, Track, Trajectory};

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

// ============================================================================
// Hypothesis-based operations (LmbmFilter, MultisensorLmbmFilter)
// ============================================================================

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GaussianComponent, TrackLabel};
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
