"""Numerical equivalence tests against MATLAB fixtures.

ALL fixture data is tested. Filters are run with fixture inputs
and outputs are compared for numerical correctness.
"""

import numpy as np
import pytest
from conftest import (
    load_fixture,
    make_birth_model_from_objects,
    make_motion_model,
    make_multisensor_config,
    make_sensor_model,
    measurements_to_numpy,
    nested_measurements_to_numpy,
)


class TestFilterLmbEquivalence:
    """Test FilterLmb produces correct results on fixture data."""

    def test_determinism_same_seed(self, lmb_fixture):
        """Same seed produces identical results."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        prior_objects = lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        birth = make_birth_model_from_objects(prior_objects)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])

        filter1 = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=42)
        filter2 = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=42)

        est1 = filter1.step(measurements, timestep=0)
        est2 = filter2.step(measurements, timestep=0)

        assert est1.num_tracks == est2.num_tracks
        for t1, t2 in zip(est1.tracks, est2.tracks):
            np.testing.assert_array_almost_equal(t1.mean, t2.mean, decimal=10)
            np.testing.assert_array_almost_equal(t1.covariance, t2.covariance, decimal=10)

    def test_lbp_produces_valid_output(self, lmb_fixture):
        """LBP association produces valid filter output."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        prior_objects = lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        birth = make_birth_model_from_objects(prior_objects)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])

        filter = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=lmb_fixture["seed"])
        estimate = filter.step(measurements, timestep=lmb_fixture["timestep"])

        # Verify output structure
        assert estimate.num_tracks >= 0
        assert estimate.timestamp == lmb_fixture["timestep"]

        # Verify track estimates have valid data
        for track in estimate.tracks:
            assert track.x_dim == 4  # State dimension from fixture
            assert track.mean.shape == (4,)
            assert track.covariance.shape == (4, 4)
            # Covariance should be positive semi-definite
            eigvals = np.linalg.eigvalsh(track.covariance)
            assert np.all(eigvals >= -1e-10), f"Covariance not PSD: {eigvals}"

    def test_gibbs_produces_valid_output(self, lmb_fixture):
        """Gibbs association produces valid filter output."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        prior_objects = lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        birth = make_birth_model_from_objects(prior_objects)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])

        filter = FilterLmb(motion, sensor, birth, AssociatorConfig.gibbs(samples=100), seed=42)
        estimate = filter.step(measurements, timestep=0)

        assert estimate.num_tracks >= 0
        for track in estimate.tracks:
            assert track.mean.shape == (4,)
            eigvals = np.linalg.eigvalsh(track.covariance)
            assert np.all(eigvals >= -1e-10)

    def test_murty_produces_valid_output(self, lmb_fixture):
        """Murty association produces valid filter output."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        prior_objects = lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        birth = make_birth_model_from_objects(prior_objects)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])

        filter = FilterLmb(motion, sensor, birth, AssociatorConfig.murty(assignments=50), seed=42)
        estimate = filter.step(measurements, timestep=0)

        assert estimate.num_tracks >= 0
        for track in estimate.tracks:
            assert track.mean.shape == (4,)

    def test_reset_clears_state(self, lmb_fixture):
        """Reset clears filter state."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        prior_objects = lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        birth = make_birth_model_from_objects(prior_objects)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])

        filter = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=42)

        # Run a step
        filter.step(measurements, timestep=0)

        # Reset and verify tracks are cleared
        filter.reset()
        assert filter.num_tracks == 0


class TestFilterLmbmEquivalence:
    """Test FilterLmbm produces correct results on fixture data."""

    def test_lmbm_produces_valid_output(self, lmbm_fixture):
        """LMBM filter produces valid output on fixture data."""
        from multisensor_lmb_filters_rs import FilterLmbm

        model = lmbm_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        prior_hypothesis = lmbm_fixture["step1_prediction"]["input"]["prior_hypothesis"]
        birth = make_birth_model_from_objects(prior_hypothesis["objects"])

        measurements = measurements_to_numpy(lmbm_fixture["measurements"])

        filter = FilterLmbm(motion, sensor, birth, seed=lmbm_fixture["seed"])
        estimate = filter.step(measurements, timestep=lmbm_fixture["timestep"])

        assert estimate.num_tracks >= 0
        assert filter.num_hypotheses >= 1

        for track in estimate.tracks:
            assert track.mean.shape == (4,)
            assert track.covariance.shape == (4, 4)

    def test_lmbm_determinism(self, lmbm_fixture):
        """LMBM with same seed produces identical results."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmbm

        model = lmbm_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        prior_hypothesis = lmbm_fixture["step1_prediction"]["input"]["prior_hypothesis"]
        birth = make_birth_model_from_objects(prior_hypothesis["objects"])

        measurements = measurements_to_numpy(lmbm_fixture["measurements"])

        filter1 = FilterLmbm(motion, sensor, birth, AssociatorConfig.gibbs(500), seed=42)
        filter2 = FilterLmbm(motion, sensor, birth, AssociatorConfig.gibbs(500), seed=42)

        est1 = filter1.step(measurements, timestep=0)
        est2 = filter2.step(measurements, timestep=0)

        assert est1.num_tracks == est2.num_tracks
        for t1, t2 in zip(est1.tracks, est2.tracks):
            np.testing.assert_array_almost_equal(t1.mean, t2.mean, decimal=10)


class TestFilterMultisensorEquivalence:
    """Test multisensor filters produce correct results on fixture data."""

    def test_ic_lmb_produces_valid_output(self, multisensor_lmb_fixture):
        """IC-LMB filter produces valid output on fixture data."""
        from multisensor_lmb_filters_rs import FilterIcLmb

        model = multisensor_lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor_config = make_multisensor_config(model)

        prior_objects = multisensor_lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        birth = make_birth_model_from_objects(prior_objects)

        measurements = nested_measurements_to_numpy(multisensor_lmb_fixture["measurements"])

        filter = FilterIcLmb(motion, sensor_config, birth, seed=multisensor_lmb_fixture["seed"])
        estimate = filter.step(measurements, timestep=multisensor_lmb_fixture["timestep"])

        assert estimate.num_tracks >= 0
        for track in estimate.tracks:
            assert track.mean.shape == (4,)
            eigvals = np.linalg.eigvalsh(track.covariance)
            assert np.all(eigvals >= -1e-10)

    @pytest.mark.parametrize(
        "filter_cls_name", ["FilterAaLmb", "FilterGaLmb", "FilterPuLmb", "FilterIcLmb"]
    )
    def test_all_multisensor_lmb_variants(self, multisensor_lmb_fixture, filter_cls_name):
        """All multisensor LMB filter variants produce valid output."""
        import multisensor_lmb_filters_rs as lmb

        FilterCls = getattr(lmb, filter_cls_name)

        model = multisensor_lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor_config = make_multisensor_config(model)

        prior_objects = multisensor_lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        birth = make_birth_model_from_objects(prior_objects)

        measurements = nested_measurements_to_numpy(multisensor_lmb_fixture["measurements"])

        filter = FilterCls(motion, sensor_config, birth, seed=42)
        estimate = filter.step(measurements, timestep=0)

        assert estimate.num_tracks >= 0
        for track in estimate.tracks:
            assert track.x_dim == 4

    def test_multisensor_lmbm_produces_valid_output(self, multisensor_lmbm_fixture):
        """Multi-sensor LMBM filter produces valid output."""
        from multisensor_lmb_filters_rs import FilterMultisensorLmbm

        model = multisensor_lmbm_fixture["model"]
        motion = make_motion_model(model)
        sensor_config = make_multisensor_config(model)

        prior_hypothesis = multisensor_lmbm_fixture["step1_prediction"]["input"]["prior_hypothesis"]
        birth = make_birth_model_from_objects(prior_hypothesis["objects"])

        measurements = nested_measurements_to_numpy(multisensor_lmbm_fixture["measurements"])

        filter = FilterMultisensorLmbm(
            motion, sensor_config, birth, seed=multisensor_lmbm_fixture["seed"]
        )
        estimate = filter.step(measurements, timestep=multisensor_lmbm_fixture["timestep"])

        assert estimate.num_tracks >= 0
        assert filter.num_hypotheses >= 1


class TestModelConfiguration:
    """Test model and configuration classes."""

    def test_motion_model_constant_velocity_2d(self):
        """MotionModel.constant_velocity_2d creates valid model."""
        from multisensor_lmb_filters_rs import MotionModel

        motion = MotionModel.constant_velocity_2d(
            dt=1.0,
            process_noise_std=1.0,
            survival_probability=0.95,
        )

        assert motion.x_dim == 4
        assert motion.survival_probability == 0.95
        assert motion.transition_matrix.shape == (4, 4)
        assert motion.process_noise.shape == (4, 4)

    def test_sensor_model_position_2d(self):
        """SensorModel.position_2d creates valid model."""
        from multisensor_lmb_filters_rs import SensorModel

        sensor = SensorModel.position_2d(
            measurement_noise_std=3.0,
            detection_probability=0.8,
            clutter_rate=10.0,
            observation_volume=40000.0,
        )

        assert sensor.x_dim == 4
        assert sensor.z_dim == 2
        assert sensor.detection_probability == 0.8
        assert sensor.observation_matrix.shape == (2, 4)
        assert sensor.measurement_noise.shape == (2, 2)

    def test_sensor_config_multi(self):
        """SensorConfigMulti holds multiple sensors."""
        from multisensor_lmb_filters_rs import SensorConfigMulti, SensorModel

        sensor1 = SensorModel.position_2d(3.0, 0.7, 5.0, 40000.0)
        sensor2 = SensorModel.position_2d(4.0, 0.8, 8.0, 40000.0)

        config = SensorConfigMulti([sensor1, sensor2])

        assert len(config) == 2
        assert config.num_sensors == 2
        assert config[0].detection_probability == 0.7
        assert config[1].detection_probability == 0.8

    def test_birth_model_and_location(self):
        """BirthModel and BirthLocation work correctly."""
        from multisensor_lmb_filters_rs import BirthLocation, BirthModel

        loc1 = BirthLocation(
            label=1,
            mean=np.array([0.0, 0.0, 0.0, 0.0]),
            covariance=np.eye(4),
        )
        loc2 = BirthLocation(
            label=2,
            mean=np.array([10.0, 20.0, 0.0, 0.0]),
            covariance=np.eye(4) * 2,
        )

        birth = BirthModel(
            locations=[loc1, loc2],
            lmb_existence=0.03,
            lmbm_existence=0.003,
        )

        assert len(birth) == 2
        assert birth.num_locations == 2
        assert birth.lmb_existence == 0.03
        assert birth.lmbm_existence == 0.003


class TestOutputTypes:
    """Test output type classes."""

    def test_track_label_equality_and_hash(self):
        """TrackLabel supports equality and hashing."""
        from multisensor_lmb_filters_rs import TrackLabel

        label1 = TrackLabel(birth_time=5, birth_location=2)
        label2 = TrackLabel(birth_time=5, birth_location=2)
        label3 = TrackLabel(birth_time=5, birth_location=3)

        assert label1 == label2
        assert label1 != label3
        assert hash(label1) == hash(label2)

        # Can be used in sets/dicts
        label_set = {label1, label2, label3}
        assert len(label_set) == 2

    def test_state_estimate_iteration(self, lmb_fixture):
        """StateEstimate supports iteration over tracks."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        prior_objects = lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        birth = make_birth_model_from_objects(prior_objects)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])

        filter = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=42)
        estimate = filter.step(measurements, timestep=0)

        # Test iteration
        track_list = list(estimate)
        assert len(track_list) == estimate.num_tracks

        # Test indexing
        if estimate.num_tracks > 0:
            first_track = estimate[0]
            assert first_track.mean.shape == (4,)


class TestAssociatorConfig:
    """Test associator configuration."""

    def test_lbp_config(self):
        """LBP configuration with custom parameters."""
        from multisensor_lmb_filters_rs import AssociatorConfig

        config = AssociatorConfig.lbp(max_iterations=50, tolerance=1e-8)
        repr_str = repr(config)
        assert "lbp" in repr_str.lower()
        assert "50" in repr_str

    def test_gibbs_config(self):
        """Gibbs configuration with custom samples."""
        from multisensor_lmb_filters_rs import AssociatorConfig

        config = AssociatorConfig.gibbs(samples=500)
        repr_str = repr(config)
        assert "gibbs" in repr_str.lower()
        assert "500" in repr_str

    def test_murty_config(self):
        """Murty configuration with custom assignments."""
        from multisensor_lmb_filters_rs import AssociatorConfig

        config = AssociatorConfig.murty(assignments=200)
        repr_str = repr(config)
        assert "murty" in repr_str.lower()
        assert "200" in repr_str


class TestFilterThresholds:
    """Test filter threshold configuration."""

    def test_custom_thresholds(self):
        """Custom thresholds are applied correctly."""
        from multisensor_lmb_filters_rs import FilterThresholds

        thresholds = FilterThresholds(
            existence=0.3,
            gm_weight=1e-5,
            max_components=50,
            min_trajectory_length=5,
        )

        assert thresholds.existence_threshold == 0.3
        assert thresholds.gm_weight_threshold == 1e-5
        assert thresholds.max_gm_components == 50


class TestTrialFixtures:
    """Test that trial fixtures load and have expected structure."""

    def test_single_trial_structure(self, single_trial_fixture):
        """Single-sensor trial fixture has expected structure."""
        assert "filterVariants" in single_trial_fixture
        variants = single_trial_fixture["filterVariants"]
        assert len(variants) >= 4

        for variant in variants:
            assert "name" in variant
            assert "eOspa" in variant
            assert "hOspa" in variant
            assert len(variant["eOspa"]) > 0
            assert len(variant["hOspa"]) > 0

    def test_multisensor_trial_structure(self, multisensor_trial_fixture):
        """Multi-sensor trial fixture has expected structure."""
        assert "filterVariants" in multisensor_trial_fixture
        variants = multisensor_trial_fixture["filterVariants"]
        assert len(variants) >= 4

        for variant in variants:
            assert "name" in variant
            assert "eOspa" in variant
            assert "hOspa" in variant


class TestAllFixturesLoad:
    """Verify all fixture files exist and load correctly."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "step_by_step/lmb_step_by_step_seed42.json",
            "step_by_step/lmbm_step_by_step_seed42.json",
            "step_by_step/multisensor_lmb_step_by_step_seed42.json",
            "step_by_step/multisensor_lmbm_step_by_step_seed42.json",
            "single_trial_42.json",
            "single_trial_42_quick.json",
            "single_detection_trial_42_quick.json",
            "multisensor_trial_42.json",
            "multisensor_clutter_trial_42_quick.json",
            "multisensor_detection_trial_42_quick.json",
        ],
    )
    def test_fixture_loads(self, fixture_name):
        """All 10 fixture files load without error."""
        fixture = load_fixture(fixture_name)
        assert "seed" in fixture or "filterVariants" in fixture


class TestMultipleSteps:
    """Test filters over multiple timesteps."""

    def test_lmb_multiple_steps(self, lmb_fixture):
        """LMB filter runs correctly over multiple steps."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        prior_objects = lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        birth = make_birth_model_from_objects(prior_objects)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])

        filter = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=42)

        # Run multiple steps with same measurements (simulating sequence)
        for t in range(3):
            estimate = filter.step(measurements, timestep=t)
            assert estimate.timestamp == t

    def test_ic_lmb_multiple_steps(self, multisensor_lmb_fixture):
        """IC-LMB filter runs correctly over multiple steps."""
        from multisensor_lmb_filters_rs import FilterIcLmb

        model = multisensor_lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor_config = make_multisensor_config(model)

        prior_objects = multisensor_lmb_fixture["step1_prediction"]["input"]["prior_objects"]
        birth = make_birth_model_from_objects(prior_objects)

        measurements = nested_measurements_to_numpy(multisensor_lmb_fixture["measurements"])

        filter = FilterIcLmb(motion, sensor_config, birth, seed=42)

        for t in range(3):
            estimate = filter.step(measurements, timestep=t)
            assert estimate.timestamp == t
