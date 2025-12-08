"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Fixed random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def motion_model_2d():
    """Standard 2D constant velocity motion model."""
    from multisensor_lmb_filters_rs import MotionModel

    return MotionModel.constant_velocity_2d(
        dt=1.0,
        process_noise_std=0.1,
        survival_prob=0.95,
    )


@pytest.fixture
def sensor_model_2d():
    """Standard 2D position sensor model."""
    from multisensor_lmb_filters_rs import SensorModel

    return SensorModel.position_sensor_2d(
        measurement_noise_std=1.0,
        detection_prob=0.9,
        clutter_rate=10.0,
        obs_volume=100.0,
    )


@pytest.fixture
def birth_model_2d(state_dim=4):
    """Standard birth model with one location at origin."""
    from multisensor_lmb_filters_rs import BirthLocation, BirthModel

    birth_loc = BirthLocation(
        label=0,
        mean=np.zeros(state_dim),
        covariance=np.eye(state_dim) * 100.0,
    )
    return BirthModel(
        locations=[birth_loc],
        lmb_existence=0.1,
        lmbm_existence=0.01,
    )


@pytest.fixture
def multisensor_config_2d(sensor_model_2d):
    """Multi-sensor config with 2 identical sensors."""
    from multisensor_lmb_filters_rs import MultisensorConfig

    return MultisensorConfig(sensors=[sensor_model_2d, sensor_model_2d])


@pytest.fixture
def lmb_filter(motion_model_2d, sensor_model_2d, birth_model_2d):
    """Pre-configured LMB filter with fixed seed."""
    from multisensor_lmb_filters_rs import LmbFilter

    return LmbFilter(
        motion=motion_model_2d,
        sensor=sensor_model_2d,
        birth=birth_model_2d,
        seed=42,
    )


@pytest.fixture
def lmbm_filter(motion_model_2d, sensor_model_2d, birth_model_2d):
    """Pre-configured LMBM filter with fixed seed."""
    from multisensor_lmb_filters_rs import LmbmFilter

    return LmbmFilter(
        motion=motion_model_2d,
        sensor=sensor_model_2d,
        birth=birth_model_2d,
        seed=42,
    )


@pytest.fixture
def sample_measurements():
    """Sample measurements for testing."""
    return [
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])],  # t=0
        [np.array([1.1, 2.1]), np.array([3.1, 4.1])],  # t=1
        [np.array([1.2, 2.2]), np.array([3.2, 4.2])],  # t=2
    ]


@pytest.fixture
def sample_multisensor_measurements():
    """Sample multi-sensor measurements for testing."""
    # Two sensors, 3 timesteps
    return [
        # t=0: sensor 0 sees 2 detections, sensor 1 sees 1
        [
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],  # sensor 0
            [np.array([1.5, 2.5])],  # sensor 1
        ],
        # t=1
        [
            [np.array([1.1, 2.1])],  # sensor 0
            [np.array([1.6, 2.6]), np.array([3.1, 4.1])],  # sensor 1
        ],
        # t=2
        [
            [np.array([1.2, 2.2]), np.array([3.2, 4.2])],  # sensor 0
            [np.array([1.7, 2.7])],  # sensor 1
        ],
    ]
