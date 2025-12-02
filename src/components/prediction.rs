//! Track prediction using Chapman-Kolmogorov equation
//!
//! This module provides unified prediction for all filter types.
//! The prediction step propagates tracks forward in time using the motion model.

use crate::types::{BirthModel, GaussianComponent, MotionModel, Track};

/// Predict a single Gaussian component forward in time
///
/// Implements the Chapman-Kolmogorov prediction:
/// - `μ' = A × μ + u`
/// - `Σ' = A × Σ × Aᵀ + R`
#[inline]
pub fn predict_component(component: &mut GaussianComponent, motion: &MotionModel) {
    // Mean prediction: μ' = A × μ + u
    component.mean = &motion.transition_matrix * &component.mean + &motion.control_input;

    // Covariance prediction: Σ' = A × Σ × Aᵀ + R
    component.covariance = &motion.transition_matrix
        * &component.covariance
        * motion.transition_matrix.transpose()
        + &motion.process_noise;
}

/// Predict a single track forward in time
///
/// Updates existence probability and all Gaussian components.
#[inline]
pub fn predict_track(track: &mut Track, motion: &MotionModel) {
    // Existence prediction: r' = p_S × r
    track.existence *= motion.survival_probability;

    // Predict all components
    track
        .components
        .iter_mut()
        .for_each(|c| predict_component(c, motion));
}

/// Predict all tracks and add birth tracks
///
/// This is the main prediction function used by all filters.
/// It handles both existing track prediction and birth track addition.
///
/// # Arguments
/// * `tracks` - Mutable slice of tracks to predict
/// * `motion` - Motion model parameters
/// * `birth` - Birth model parameters
/// * `timestep` - Current timestep (for birth track labeling)
/// * `use_lmbm` - If true, use LMBM birth existence probability
pub fn predict_tracks(
    tracks: &mut Vec<Track>,
    motion: &MotionModel,
    birth: &BirthModel,
    timestep: usize,
    use_lmbm: bool,
) {
    // Predict existing tracks
    tracks.iter_mut().for_each(|t| predict_track(t, motion));

    // Add birth tracks
    let birth_existence = if use_lmbm {
        birth.lmbm_existence
    } else {
        birth.lmb_existence
    };

    let new_tracks = birth.locations.iter().map(|loc| {
        Track::new_birth(
            loc.label,
            timestep,
            birth_existence,
            loc.mean.clone(),
            loc.covariance.clone(),
        )
    });

    tracks.extend(new_tracks);
}

/// Predict LMBM hypotheses
///
/// For LMBM, we predict all tracks within each hypothesis.
/// Birth tracks are added to all hypotheses.
pub fn predict_hypotheses(
    hypotheses: &mut Vec<crate::types::LmbmHypothesis>,
    motion: &MotionModel,
    birth: &BirthModel,
    timestep: usize,
) {
    for hyp in hypotheses.iter_mut() {
        predict_tracks(&mut hyp.tracks, motion, birth, timestep, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrackLabel;
    use nalgebra::{DMatrix, DVector};

    fn create_test_motion() -> MotionModel {
        MotionModel::constant_velocity_2d(1.0, 0.1, 0.99)
    }

    #[test]
    fn test_predict_component() {
        let motion = create_test_motion();

        let mut comp = GaussianComponent::new(
            1.0,
            DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0]),
            DMatrix::identity(4, 4),
        );

        predict_component(&mut comp, &motion);

        // After prediction with dt=1, position should increase by velocity
        assert!((comp.mean[0] - 1.0).abs() < 1e-10); // x + vx*dt = 0 + 1*1
        assert!((comp.mean[2] - 1.0).abs() < 1e-10); // y + vy*dt = 0 + 1*1
    }

    #[test]
    fn test_predict_track() {
        let motion = create_test_motion();

        let mut track = Track::new(
            TrackLabel::new(0, 0),
            0.9,
            DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0]),
            DMatrix::identity(4, 4),
        );

        predict_track(&mut track, &motion);

        // Existence should decrease by survival probability
        assert!((track.existence - 0.9 * 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_predict_tracks_with_birth() {
        let motion = create_test_motion();

        let birth_loc = crate::types::BirthLocation::new(
            0,
            DVector::from_vec(vec![10.0, 0.0, 10.0, 0.0]),
            DMatrix::identity(4, 4) * 100.0,
        );
        let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);

        let mut tracks = vec![Track::new(
            TrackLabel::new(0, 0),
            0.9,
            DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0]),
            DMatrix::identity(4, 4),
        )];

        predict_tracks(&mut tracks, &motion, &birth, 1, false);

        // Should have original track + 1 birth track
        assert_eq!(tracks.len(), 2);

        // Birth track should have correct existence
        assert!((tracks[1].existence - 0.1).abs() < 1e-10);
    }
}
