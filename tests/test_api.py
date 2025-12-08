"""Tests for the Python API surface area."""

import numpy as np
import pytest


class TestImports:
    """Test that all public API components can be imported."""

    def test_import_types(self):
        from multisensor_lmb_filters_rs import GaussianComponent, Track, TrackLabel

    def test_import_config(self):
        from multisensor_lmb_filters_rs import (
            AssociationConfig,
            BirthLocation,
            BirthModel,
            FilterThresholds,
            LmbmConfig,
            MotionModel,
            MultisensorConfig,
            SensorModel,
        )

    def test_import_output(self):
        from multisensor_lmb_filters_rs import (
            EstimatedTrack,
            FilterOutput,
            StateEstimate,
            Trajectory,
        )

    def test_import_filters(self):
        from multisensor_lmb_filters_rs import (
            AaLmbFilter,
            GaLmbFilter,
            IcLmbFilter,
            LmbFilter,
            LmbmFilter,
            MultisensorLmbmFilter,
            PuLmbFilter,
        )

    def test_import_version(self):
        from multisensor_lmb_filters_rs import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0


class TestTrackLabel:
    """Test TrackLabel class."""

    def test_create(self):
        from multisensor_lmb_filters_rs import TrackLabel

        label = TrackLabel(birth_time=5, birth_location=3)
        assert label.birth_time == 5
        assert label.birth_location == 3

    def test_repr(self):
        from multisensor_lmb_filters_rs import TrackLabel

        label = TrackLabel(birth_time=5, birth_location=3)
        assert "5" in repr(label)
        assert "3" in repr(label)

    def test_equality(self):
        from multisensor_lmb_filters_rs import TrackLabel

        label1 = TrackLabel(birth_time=5, birth_location=3)
        label2 = TrackLabel(birth_time=5, birth_location=3)
        label3 = TrackLabel(birth_time=5, birth_location=4)
        assert label1 == label2
        assert label1 != label3

    def test_hash(self):
        from multisensor_lmb_filters_rs import TrackLabel

        label1 = TrackLabel(birth_time=5, birth_location=3)
        label2 = TrackLabel(birth_time=5, birth_location=3)
        assert hash(label1) == hash(label2)
        # Can use in sets/dicts
        s = {label1, label2}
        assert len(s) == 1


class TestGaussianComponent:
    """Test GaussianComponent class."""

    def test_create(self):
        from multisensor_lmb_filters_rs import GaussianComponent

        comp = GaussianComponent(
            weight=0.5,
            mean=np.array([1.0, 2.0, 3.0, 4.0]),
            covariance=np.eye(4) * 2.0,
        )
        assert comp.weight == 0.5
        assert comp.x_dim == 4
        np.testing.assert_array_equal(comp.mean, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(comp.covariance, np.eye(4) * 2.0)


class TestTrack:
    """Test Track class."""

    def test_create(self):
        from multisensor_lmb_filters_rs import Track, TrackLabel

        label = TrackLabel(birth_time=0, birth_location=0)
        track = Track(
            label=label,
            existence=0.8,
            mean=np.array([1.0, 2.0, 3.0, 4.0]),
            covariance=np.eye(4),
        )
        assert track.existence == 0.8
        assert track.x_dim == 4
        assert track.label == label
        assert track.num_components == 1

    def test_components(self):
        from multisensor_lmb_filters_rs import Track, TrackLabel

        label = TrackLabel(birth_time=0, birth_location=0)
        track = Track(
            label=label,
            existence=0.8,
            mean=np.zeros(4),
            covariance=np.eye(4),
        )
        components = track.components
        assert len(components) == 1
        assert components[0].weight == 1.0

    def test_weighted_mean(self):
        from multisensor_lmb_filters_rs import Track, TrackLabel

        label = TrackLabel(birth_time=0, birth_location=0)
        mean = np.array([1.0, 2.0, 3.0, 4.0])
        track = Track(
            label=label,
            existence=0.8,
            mean=mean,
            covariance=np.eye(4),
        )
        np.testing.assert_array_almost_equal(track.weighted_mean(), mean)


class TestMotionModel:
    """Test MotionModel class."""

    def test_constant_velocity_2d(self):
        from multisensor_lmb_filters_rs import MotionModel

        motion = MotionModel.constant_velocity_2d(
            dt=1.0, process_noise_std=0.1, survival_prob=0.99
        )
        assert motion.x_dim == 4
        assert motion.survival_probability == 0.99
        assert motion.transition_matrix.shape == (4, 4)
        assert motion.process_noise.shape == (4, 4)
        assert motion.control_input.shape == (4,)

    def test_custom_model(self):
        from multisensor_lmb_filters_rs import MotionModel

        F = np.eye(2)
        Q = np.eye(2) * 0.1
        u = np.zeros(2)
        motion = MotionModel(
            transition_matrix=F,
            process_noise=Q,
            control_input=u,
            survival_probability=0.95,
        )
        assert motion.x_dim == 2
        assert motion.survival_probability == 0.95


class TestSensorModel:
    """Test SensorModel class."""

    def test_position_sensor_2d(self):
        from multisensor_lmb_filters_rs import SensorModel

        sensor = SensorModel.position_sensor_2d(
            measurement_noise_std=1.0,
            detection_prob=0.9,
            clutter_rate=10.0,
            obs_volume=100.0,
        )
        assert sensor.z_dim == 2
        assert sensor.x_dim == 4
        assert sensor.detection_probability == 0.9
        assert sensor.clutter_rate == 10.0
        assert sensor.observation_volume == 100.0
        assert sensor.clutter_density == pytest.approx(10.0 / 100.0)

    def test_custom_model(self):
        from multisensor_lmb_filters_rs import SensorModel

        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        R = np.eye(2) * 4.0
        sensor = SensorModel(
            observation_matrix=H,
            measurement_noise=R,
            detection_probability=0.8,
            clutter_rate=5.0,
            observation_volume=50.0,
        )
        assert sensor.z_dim == 2
        assert sensor.x_dim == 4


class TestBirthModel:
    """Test BirthModel and BirthLocation classes."""

    def test_birth_location(self):
        from multisensor_lmb_filters_rs import BirthLocation

        loc = BirthLocation(
            label=0,
            mean=np.zeros(4),
            covariance=np.eye(4) * 100.0,
        )
        assert loc.label == 0
        np.testing.assert_array_equal(loc.mean, np.zeros(4))
        np.testing.assert_array_equal(loc.covariance, np.eye(4) * 100.0)

    def test_birth_model(self):
        from multisensor_lmb_filters_rs import BirthLocation, BirthModel

        loc1 = BirthLocation(label=0, mean=np.zeros(4), covariance=np.eye(4))
        loc2 = BirthLocation(label=1, mean=np.ones(4), covariance=np.eye(4))
        birth = BirthModel(
            locations=[loc1, loc2], lmb_existence=0.1, lmbm_existence=0.01
        )
        assert birth.num_locations == 2
        assert birth.lmb_existence == 0.1
        assert birth.lmbm_existence == 0.01


class TestAssociationConfig:
    """Test AssociationConfig class."""

    def test_lbp(self):
        from multisensor_lmb_filters_rs import AssociationConfig

        config = AssociationConfig.lbp(max_iterations=100, tolerance=1e-8)
        assert config.method == "lbp"

    def test_gibbs(self):
        from multisensor_lmb_filters_rs import AssociationConfig

        config = AssociationConfig.gibbs(samples=500)
        assert config.method == "gibbs"

    def test_murty(self):
        from multisensor_lmb_filters_rs import AssociationConfig

        config = AssociationConfig.murty(assignments=200)
        assert config.method == "murty"

    def test_default(self):
        from multisensor_lmb_filters_rs import AssociationConfig

        config = AssociationConfig()
        assert config.method == "lbp"


class TestFilterThresholds:
    """Test FilterThresholds class."""

    def test_create(self):
        from multisensor_lmb_filters_rs import FilterThresholds

        thresholds = FilterThresholds(
            existence_threshold=0.6,
            gm_weight_threshold=1e-5,
            max_gm_components=50,
            min_trajectory_length=5,
        )
        assert thresholds.existence_threshold == 0.6
        assert thresholds.gm_weight_threshold == 1e-5
        assert thresholds.max_gm_components == 50
        assert thresholds.min_trajectory_length == 5

    def test_defaults(self):
        from multisensor_lmb_filters_rs import FilterThresholds

        thresholds = FilterThresholds()
        assert thresholds.existence_threshold == 0.5
        assert thresholds.gm_weight_threshold == 1e-4
        assert thresholds.max_gm_components == 100
        assert thresholds.min_trajectory_length == 3


class TestLmbmConfig:
    """Test LmbmConfig class."""

    def test_create(self):
        from multisensor_lmb_filters_rs import LmbmConfig

        config = LmbmConfig(
            max_hypotheses=500, hypothesis_weight_threshold=1e-8, use_eap=True
        )
        assert config.max_hypotheses == 500
        assert config.hypothesis_weight_threshold == 1e-8
        assert config.use_eap is True

    def test_defaults(self):
        from multisensor_lmb_filters_rs import LmbmConfig

        config = LmbmConfig()
        assert config.max_hypotheses == 1000
        assert config.hypothesis_weight_threshold == 1e-6
        assert config.use_eap is False


class TestMultisensorConfig:
    """Test MultisensorConfig class."""

    def test_create(self, sensor_model_2d):
        from multisensor_lmb_filters_rs import MultisensorConfig

        config = MultisensorConfig(sensors=[sensor_model_2d, sensor_model_2d])
        assert config.num_sensors == 2
        assert len(config) == 2
        assert config.z_dim == 2
