"""Real numerical equivalence tests against MATLAB fixtures.

These tests verify that Python bindings produce identical results to Rust,
which was validated against MATLAB-generated fixtures.

All tests use TOLERANCE = 1e-10 for floating-point comparison.
"""

import json
from pathlib import Path

import numpy as np
import pytest

# Tolerance for floating point comparisons (matches Rust tests)
TOLERANCE = 1e-10


def get_fixture_path(name: str) -> Path:
    """Get path to a test fixture file."""
    return Path(__file__).parent / "data" / name


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    path = get_fixture_path(name)
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def lmb_fixture():
    """Load the LMB step-by-step fixture."""
    return load_fixture("step_by_step/lmb_step_by_step_seed42.json")


class TestPredictionEquivalence:
    """Test prediction step produces same results as fixture."""

    def test_predict_component_mean_covariance(self, lmb_fixture):
        """Test prediction of component mean and covariance matches fixture."""
        from multisensor_lmb_filters_rs._multisensor_lmb_filters_rs import (
            _predict_component,
        )

        model = lmb_fixture["model"]
        A = np.array(model["A"])
        R = np.array(model["R"])

        # Get prior objects from fixture
        prior_objects = lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        predicted_objects = lmb_fixture["step1_prediction"]["output"]["predicted_objects"]

        # Control input is zero for these fixtures
        u = np.zeros(A.shape[0])

        for i, (prior, expected) in enumerate(zip(prior_objects, predicted_objects)):
            # Skip birth objects (they have different handling)
            # Birth objects are the last 5 in the list
            if i >= len(prior_objects) - 5:
                continue

            # Get first component (most fixtures use single-component tracks)
            prior_mean = np.array(prior["mu"][0])
            prior_cov = np.array(prior["Sigma"][0])

            # Run prediction
            pred_mean, pred_cov = _predict_component(
                mean=prior_mean,
                covariance=prior_cov,
                transition_matrix=A,
                process_noise=R,
                control_input=u,
            )

            expected_mean = np.array(expected["mu"][0])
            expected_cov = np.array(expected["Sigma"][0])

            np.testing.assert_allclose(
                pred_mean,
                expected_mean,
                atol=TOLERANCE,
                err_msg=f"Mean mismatch for object {i}",
            )
            np.testing.assert_allclose(
                pred_cov,
                expected_cov,
                atol=TOLERANCE,
                err_msg=f"Covariance mismatch for object {i}",
            )

    def test_predict_existence(self, lmb_fixture):
        """Test prediction of existence probability matches fixture."""
        from multisensor_lmb_filters_rs._multisensor_lmb_filters_rs import (
            _predict_existence,
        )

        model = lmb_fixture["model"]
        p_s = model["P_s"]

        prior_objects = lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        predicted_objects = lmb_fixture["step1_prediction"]["output"]["predicted_objects"]

        for i, (prior, expected) in enumerate(zip(prior_objects, predicted_objects)):
            # Skip birth objects
            if i >= len(prior_objects) - 5:
                continue

            prior_r = prior["r"]
            expected_r = expected["r"]

            pred_r = _predict_existence(prior_r, p_s)

            assert (
                abs(pred_r - expected_r) < TOLERANCE
            ), f"Existence mismatch for object {i}: {pred_r} != {expected_r}"


class TestModelCreation:
    """Test that models can be created from fixture data."""

    def test_motion_model_from_fixture(self, lmb_fixture):
        """Test creating MotionModel from fixture parameters."""
        from multisensor_lmb_filters_rs import MotionModel

        model = lmb_fixture["model"]
        A = np.array(model["A"], dtype=np.float64)
        R = np.array(model["R"], dtype=np.float64)
        p_s = model["P_s"]

        motion = MotionModel(
            transition_matrix=A,
            process_noise=R,
            control_input=np.zeros(A.shape[0], dtype=np.float64),
            survival_probability=p_s,
        )

        # Verify parameters stored correctly
        assert motion.x_dim == 4
        assert motion.survival_probability == pytest.approx(p_s, rel=TOLERANCE)
        np.testing.assert_allclose(motion.transition_matrix, A, rtol=TOLERANCE)
        np.testing.assert_allclose(motion.process_noise, R, rtol=TOLERANCE)

    def test_sensor_model_from_fixture(self, lmb_fixture):
        """Test creating SensorModel from fixture parameters."""
        from multisensor_lmb_filters_rs import SensorModel

        model = lmb_fixture["model"]
        C = np.array(model["C"], dtype=np.float64)
        Q = np.array(model["Q"], dtype=np.float64)
        p_d = model["P_d"]
        clutter = model["clutter_per_unit_volume"]

        # Assume observation volume for clutter rate conversion
        obs_volume = 10000.0
        clutter_rate = clutter * obs_volume

        sensor = SensorModel(
            observation_matrix=C,
            measurement_noise=Q,
            detection_probability=p_d,
            clutter_rate=clutter_rate,
            observation_volume=obs_volume,
        )

        assert sensor.z_dim == 2
        assert sensor.x_dim == 4
        assert sensor.detection_probability == pytest.approx(p_d, rel=TOLERANCE)


class TestLmbFilterIntegration:
    """Test full LMB filter produces reproducible results."""

    def test_filter_determinism(self, lmb_fixture):
        """Test that same seed produces identical results."""
        from multisensor_lmb_filters_rs import (
            BirthLocation,
            BirthModel,
            LmbFilter,
            MotionModel,
            SensorModel,
        )

        model = lmb_fixture["model"]
        seed = lmb_fixture["seed"]

        # Create motion model
        motion = MotionModel(
            transition_matrix=np.array(model["A"], dtype=np.float64),
            process_noise=np.array(model["R"], dtype=np.float64),
            control_input=np.zeros(4, dtype=np.float64),
            survival_probability=model["P_s"],
        )

        # Create sensor model
        sensor = SensorModel(
            observation_matrix=np.array(model["C"], dtype=np.float64),
            measurement_noise=np.array(model["Q"], dtype=np.float64),
            detection_probability=model["P_d"],
            clutter_rate=model["clutter_per_unit_volume"] * 10000.0,
            observation_volume=10000.0,
        )

        # Create birth model
        birth_loc = BirthLocation(
            label=0,
            mean=np.zeros(4, dtype=np.float64),
            covariance=np.eye(4, dtype=np.float64) * 100.0,
        )
        birth = BirthModel(
            locations=[birth_loc],
            lmb_existence=0.03,
            lmbm_existence=0.01,
        )

        # Create two filters with same seed
        filter1 = LmbFilter(motion=motion, sensor=sensor, birth=birth, seed=seed)
        filter2 = LmbFilter(motion=motion, sensor=sensor, birth=birth, seed=seed)

        # Run with fixture measurements
        measurements = [np.array(m, dtype=np.float64) for m in lmb_fixture["measurements"]]

        est1 = filter1.step(measurements=measurements, timestep=0)
        est2 = filter2.step(measurements=measurements, timestep=0)

        # Results should be identical
        assert est1.num_tracks == est2.num_tracks

        for t1, t2 in zip(est1.tracks, est2.tracks):
            np.testing.assert_array_equal(t1.mean, t2.mean)
            np.testing.assert_array_equal(t1.covariance, t2.covariance)


class TestLmbmFilterIntegration:
    """Test LMBM filter properties."""

    def test_lmbm_has_num_tracks(self):
        """Test that LMBM filter exposes num_tracks property."""
        from multisensor_lmb_filters_rs import (
            BirthLocation,
            BirthModel,
            LmbmFilter,
            MotionModel,
            SensorModel,
        )

        motion = MotionModel.constant_velocity_2d(1.0, 0.1, 0.99)
        sensor = SensorModel.position_sensor_2d(1.0, 0.9, 10.0, 100.0)
        birth_loc = BirthLocation(label=0, mean=np.zeros(4), covariance=np.eye(4) * 100)
        birth = BirthModel(locations=[birth_loc], lmb_existence=0.1, lmbm_existence=0.01)

        lmbm = LmbmFilter(motion=motion, sensor=sensor, birth=birth, seed=42)

        # LMBM should have both num_hypotheses and num_tracks
        assert hasattr(lmbm, "num_hypotheses")
        assert hasattr(lmbm, "num_tracks")
        assert lmbm.num_hypotheses >= 1
        assert lmbm.num_tracks >= 0
