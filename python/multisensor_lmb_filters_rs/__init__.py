"""Multi-object tracking with Labeled Multi-Bernoulli filters.

This library provides high-performance implementations of LMB tracking
algorithms with various data association methods and sensor fusion strategies.

Single-Sensor Filters:
    FilterLmb - LMB filter with Gaussian mixture track representation
    FilterLmbm - LMBM filter with hypothesis-based representation

Multi-Sensor Filters:
    FilterAaLmb - Arithmetic Average fusion
    FilterGaLmb - Geometric Average fusion
    FilterPuLmb - Parallel Update fusion
    FilterIcLmb - Iterated Corrector fusion
    FilterMultisensorLmbm - Multi-sensor LMBM

Configuration:
    MotionModel - Motion/prediction model parameters
    SensorModel - Sensor observation model parameters
    SensorConfigMulti - Multi-sensor configuration
    BirthModel - Track birth model
    BirthLocation - Birth location specification
    AssociatorConfig - Data association configuration
    FilterThresholds - Filter pruning thresholds
    FilterLmbmConfig - LMBM-specific configuration

Output Types:
    TrackEstimate - Single track state estimate
    StateEstimate - All tracks at one timestep
    TrackLabel - Unique track identifier
    GaussianComponent - Gaussian mixture component
"""

from multisensor_lmb_filters_rs._multisensor_lmb_filters_rs import (
    # Configuration
    AssociatorConfig,
    BirthLocation,
    # Birth
    BirthModel,
    # Filters - Multi-sensor
    FilterAaLmb,
    FilterGaLmb,
    FilterIcLmb,
    # Filters - Single-sensor
    FilterLmb,
    FilterLmbm,
    FilterLmbmConfig,
    FilterMultisensorLmbm,
    FilterPuLmb,
    FilterThresholds,
    GaussianComponent,
    # Models
    MotionModel,
    SensorConfigMulti,
    SensorModel,
    StateEstimate,
    # Output
    TrackEstimate,
    # Types
    TrackLabel,
    # Version
    __version__,
)

__all__ = [
    # Models
    "MotionModel",
    "SensorModel",
    "SensorConfigMulti",
    # Birth
    "BirthModel",
    "BirthLocation",
    # Configuration
    "AssociatorConfig",
    "FilterThresholds",
    "FilterLmbmConfig",
    # Filters - Single-sensor
    "FilterLmb",
    "FilterLmbm",
    # Filters - Multi-sensor
    "FilterAaLmb",
    "FilterGaLmb",
    "FilterPuLmb",
    "FilterIcLmb",
    "FilterMultisensorLmbm",
    # Output
    "TrackEstimate",
    "StateEstimate",
    # Types
    "TrackLabel",
    "GaussianComponent",
    # Version
    "__version__",
]
