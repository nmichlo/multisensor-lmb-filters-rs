"""
Multi-object tracking with Labeled Multi-Bernoulli filters.

This package provides Python bindings for the multisensor-lmb-filters-rs Rust library,
implementing LMB (Labeled Multi-Bernoulli) and LMBM (LMB Mixture) filters for
multi-object tracking.

Example
-------
>>> import numpy as np
>>> from multisensor_lmb_filters_rs import (
...     LmbFilter, MotionModel, SensorModel, BirthModel, BirthLocation
... )
>>>
>>> # Create models
>>> motion = MotionModel.constant_velocity_2d(dt=1.0, process_noise_std=0.1, survival_prob=0.99)
>>> sensor = SensorModel.position_sensor_2d(
...     measurement_noise_std=1.0, detection_prob=0.9,
...     clutter_rate=10.0, obs_volume=100.0
... )
>>> birth_loc = BirthLocation(
...     label=0,
...     mean=np.array([0.0, 0.0, 0.0, 0.0]),
...     covariance=np.eye(4) * 100.0
... )
>>> birth = BirthModel(locations=[birth_loc], lmb_existence=0.1, lmbm_existence=0.01)
>>>
>>> # Create filter and process measurements
>>> filter = LmbFilter(motion, sensor, birth)
>>> estimate = filter.step(measurements=[np.array([1.0, 2.0])], timestep=0)
"""

from multisensor_lmb_filters_rs._multisensor_lmb_filters_rs import (
    AaLmbFilter,
    AssociationConfig,
    BirthLocation,
    BirthModel,
    EstimatedTrack,
    FilterOutput,
    FilterThresholds,
    GaLmbFilter,
    GaussianComponent,
    IcLmbFilter,
    LmbFilter,
    LmbmConfig,
    LmbmFilter,
    MotionModel,
    MultisensorConfig,
    MultisensorLmbmFilter,
    PuLmbFilter,
    SensorModel,
    StateEstimate,
    Track,
    TrackLabel,
    Trajectory,
    __version__,
)

__all__ = [
    # Types
    "TrackLabel",
    "GaussianComponent",
    "Track",
    # Configuration
    "MotionModel",
    "SensorModel",
    "MultisensorConfig",
    "BirthLocation",
    "BirthModel",
    "AssociationConfig",
    "FilterThresholds",
    "LmbmConfig",
    # Output
    "EstimatedTrack",
    "StateEstimate",
    "Trajectory",
    "FilterOutput",
    # Single-sensor filters
    "LmbFilter",
    "LmbmFilter",
    # Multi-sensor filters
    "AaLmbFilter",
    "GaLmbFilter",
    "PuLmbFilter",
    "IcLmbFilter",
    "MultisensorLmbmFilter",
    # Version
    "__version__",
]
