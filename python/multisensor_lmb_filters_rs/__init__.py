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
    AssociatorConfig,
    BirthLocation,
    BirthModel,
    FilterAaLmb,
    FilterGaLmb,
    FilterIcLmb,
    FilterLmb,
    FilterLmbm,
    FilterLmbmConfig,
    FilterMultisensorLmbm,
    FilterPuLmb,
    FilterThresholds,
    GaussianComponent,
    MotionModel,
    SensorConfigMulti,
    SensorModel,
    StateEstimate,
    TrackEstimate,
    TrackLabel,
    __version__,
)

# redundant alias supresses F401
from multisensor_lmb_filters_rs._multisensor_lmb_filters_rs import (
    _AssociationMatrices as _AssociationMatrices,
)
from multisensor_lmb_filters_rs._multisensor_lmb_filters_rs import (
    _AssociationResult as _AssociationResult,
)
from multisensor_lmb_filters_rs._multisensor_lmb_filters_rs import (
    _CardinalityEstimate as _CardinalityEstimate,
)
from multisensor_lmb_filters_rs._multisensor_lmb_filters_rs import (
    _LmbmHypothesis as _LmbmHypothesis,
)
from multisensor_lmb_filters_rs._multisensor_lmb_filters_rs import (
    _StepOutput as _StepOutput,
)
from multisensor_lmb_filters_rs._multisensor_lmb_filters_rs import (
    _TrackData as _TrackData,
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

# Note: Internal types (_TrackData, _AssociationMatrices, etc.) are imported
# but not in __all__. They are accessible via direct import for testing.
