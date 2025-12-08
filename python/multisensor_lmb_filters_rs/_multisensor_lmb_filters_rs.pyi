"""Type stubs for multisensor_lmb_filters_rs Rust bindings."""

from typing import Protocol, Sequence, runtime_checkable
import numpy as np
from numpy.typing import NDArray

__version__: str

# =============================================================================
# Protocols for common interfaces
# =============================================================================


@runtime_checkable
class HasLabel(Protocol):
    """Protocol for objects with a track label."""

    @property
    def label(self) -> "TrackLabel": ...


@runtime_checkable
class HasMean(Protocol):
    """Protocol for objects with a state mean."""

    @property
    def mean(self) -> NDArray[np.float64]: ...


@runtime_checkable
class HasCovariance(Protocol):
    """Protocol for objects with a state covariance."""

    @property
    def covariance(self) -> NDArray[np.float64]: ...


@runtime_checkable
class HasStateDimension(Protocol):
    """Protocol for objects with a state dimension."""

    @property
    def x_dim(self) -> int: ...


@runtime_checkable
class SingleSensorFilter(Protocol):
    """Protocol for single-sensor filters."""

    def step(
        self, measurements: Sequence[NDArray[np.float64]], timestep: int
    ) -> "StateEstimate": ...

    def reset(self) -> None: ...

    @property
    def num_tracks(self) -> int: ...


@runtime_checkable
class MultisensorFilter(Protocol):
    """Protocol for multi-sensor filters."""

    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> "StateEstimate": ...

    def reset(self) -> None: ...

    @property
    def num_tracks(self) -> int: ...


# =============================================================================
# Core types
# =============================================================================


class TrackLabel:
    """Track label uniquely identifies a track by birth time and location."""

    def __init__(self, birth_time: int, birth_location: int) -> None: ...
    @property
    def birth_time(self) -> int: ...
    @property
    def birth_location(self) -> int: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


class GaussianComponent(HasMean, HasCovariance, HasStateDimension):
    """Gaussian component with weight, mean, and covariance."""

    def __init__(
        self,
        weight: float,
        mean: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> None: ...
    @property
    def weight(self) -> float: ...
    @property
    def mean(self) -> NDArray[np.float64]: ...
    @property
    def covariance(self) -> NDArray[np.float64]: ...
    @property
    def x_dim(self) -> int: ...
    def __repr__(self) -> str: ...


class Track(HasLabel, HasStateDimension):
    """Track representing a potential object with Gaussian mixture state."""

    def __init__(
        self,
        label: TrackLabel,
        existence: float,
        mean: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> None: ...
    @property
    def label(self) -> TrackLabel: ...
    @property
    def existence(self) -> float: ...
    @property
    def x_dim(self) -> int: ...
    @property
    def num_components(self) -> int: ...
    @property
    def components(self) -> list[GaussianComponent]: ...
    def weighted_mean(self) -> NDArray[np.float64]: ...
    def primary_mean(self) -> NDArray[np.float64] | None: ...
    def primary_covariance(self) -> NDArray[np.float64] | None: ...
    def __repr__(self) -> str: ...


# =============================================================================
# Configuration types
# =============================================================================


class MotionModel(HasStateDimension):
    """Motion model for prediction step."""

    def __init__(
        self,
        transition_matrix: NDArray[np.float64],
        process_noise: NDArray[np.float64],
        control_input: NDArray[np.float64],
        survival_probability: float,
    ) -> None: ...
    @staticmethod
    def constant_velocity_2d(
        dt: float, process_noise_std: float, survival_prob: float
    ) -> "MotionModel": ...
    @property
    def x_dim(self) -> int: ...
    @property
    def transition_matrix(self) -> NDArray[np.float64]: ...
    @property
    def process_noise(self) -> NDArray[np.float64]: ...
    @property
    def control_input(self) -> NDArray[np.float64]: ...
    @property
    def survival_probability(self) -> float: ...
    def __repr__(self) -> str: ...


class SensorModel(HasStateDimension):
    """Sensor observation model."""

    def __init__(
        self,
        observation_matrix: NDArray[np.float64],
        measurement_noise: NDArray[np.float64],
        detection_probability: float,
        clutter_rate: float,
        observation_volume: float,
    ) -> None: ...
    @staticmethod
    def position_sensor_2d(
        measurement_noise_std: float,
        detection_prob: float,
        clutter_rate: float,
        obs_volume: float,
    ) -> "SensorModel": ...
    @property
    def z_dim(self) -> int: ...
    @property
    def x_dim(self) -> int: ...
    @property
    def observation_matrix(self) -> NDArray[np.float64]: ...
    @property
    def measurement_noise(self) -> NDArray[np.float64]: ...
    @property
    def detection_probability(self) -> float: ...
    @property
    def clutter_rate(self) -> float: ...
    @property
    def observation_volume(self) -> float: ...
    @property
    def clutter_density(self) -> float: ...
    def __repr__(self) -> str: ...


class MultisensorConfig:
    """Multi-sensor configuration."""

    def __init__(self, sensors: Sequence[SensorModel]) -> None: ...
    @property
    def num_sensors(self) -> int: ...
    @property
    def z_dim(self) -> int: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...


class BirthLocation(HasMean, HasCovariance):
    """Birth location parameters."""

    def __init__(
        self,
        label: int,
        mean: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> None: ...
    @property
    def label(self) -> int: ...
    @property
    def mean(self) -> NDArray[np.float64]: ...
    @property
    def covariance(self) -> NDArray[np.float64]: ...
    def __repr__(self) -> str: ...


class BirthModel:
    """Birth model parameters."""

    def __init__(
        self,
        locations: Sequence[BirthLocation],
        lmb_existence: float,
        lmbm_existence: float,
    ) -> None: ...
    @property
    def num_locations(self) -> int: ...
    @property
    def lmb_existence(self) -> float: ...
    @property
    def lmbm_existence(self) -> float: ...
    def __repr__(self) -> str: ...


class AssociationConfig:
    """Data association configuration."""

    def __init__(self) -> None: ...
    @staticmethod
    def lbp(max_iterations: int = 50, tolerance: float = 1e-6) -> "AssociationConfig": ...
    @staticmethod
    def gibbs(samples: int = 1000) -> "AssociationConfig": ...
    @staticmethod
    def murty(assignments: int = 100) -> "AssociationConfig": ...
    @property
    def method(self) -> str: ...
    def __repr__(self) -> str: ...


class FilterThresholds:
    """Filter threshold configuration."""

    def __init__(
        self,
        existence_threshold: float = 0.5,
        gm_weight_threshold: float = 1e-4,
        max_gm_components: int = 100,
        min_trajectory_length: int = 3,
    ) -> None: ...
    @property
    def existence_threshold(self) -> float: ...
    @property
    def gm_weight_threshold(self) -> float: ...
    @property
    def max_gm_components(self) -> int: ...
    @property
    def min_trajectory_length(self) -> int: ...
    def __repr__(self) -> str: ...


class LmbmConfig:
    """LMBM-specific configuration."""

    def __init__(
        self,
        max_hypotheses: int = 1000,
        hypothesis_weight_threshold: float = 1e-6,
        use_eap: bool = False,
    ) -> None: ...
    @property
    def max_hypotheses(self) -> int: ...
    @property
    def hypothesis_weight_threshold(self) -> float: ...
    @property
    def use_eap(self) -> bool: ...
    def __repr__(self) -> str: ...


# =============================================================================
# Output types
# =============================================================================


class EstimatedTrack(HasLabel, HasMean, HasCovariance, HasStateDimension):
    """Estimated state of a single track at one timestep."""

    @property
    def label(self) -> TrackLabel: ...
    @property
    def mean(self) -> NDArray[np.float64]: ...
    @property
    def covariance(self) -> NDArray[np.float64]: ...
    @property
    def x_dim(self) -> int: ...
    def __repr__(self) -> str: ...


class StateEstimate:
    """All track state estimates at a single timestep."""

    @property
    def timestamp(self) -> int: ...
    @property
    def num_tracks(self) -> int: ...
    @property
    def tracks(self) -> list[EstimatedTrack]: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...


class Trajectory(HasLabel):
    """Complete trajectory of a single track across multiple timesteps."""

    @property
    def label(self) -> TrackLabel: ...
    @property
    def states(self) -> list[NDArray[np.float64]]: ...
    @property
    def covariances(self) -> list[NDArray[np.float64]]: ...
    @property
    def timestamps(self) -> list[int]: ...
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def get_state(self, index: int) -> NDArray[np.float64] | None: ...
    def get_covariance(self, index: int) -> NDArray[np.float64] | None: ...
    def get_timestamp(self, index: int) -> int | None: ...
    def __repr__(self) -> str: ...


class FilterOutput:
    """Complete output from running a filter over a sequence of measurements."""

    @property
    def estimates(self) -> list[StateEstimate]: ...
    @property
    def trajectories(self) -> list[Trajectory]: ...
    @property
    def num_timesteps(self) -> int: ...
    @property
    def num_trajectories(self) -> int: ...
    def __repr__(self) -> str: ...


# =============================================================================
# Single-sensor filters
# =============================================================================


class LmbFilter(SingleSensorFilter):
    """Single-sensor LMB (Labeled Multi-Bernoulli) filter."""

    def __init__(
        self,
        motion: MotionModel,
        sensor: SensorModel,
        birth: BirthModel,
        association: AssociationConfig | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[NDArray[np.float64]],
        timestep: int,
    ) -> StateEstimate: ...
    def reset(self) -> None: ...
    @property
    def num_tracks(self) -> int: ...
    def __repr__(self) -> str: ...


class LmbmFilter:
    """Single-sensor LMBM (LMB Mixture) filter."""

    def __init__(
        self,
        motion: MotionModel,
        sensor: SensorModel,
        birth: BirthModel,
        association: AssociationConfig | None = None,
        lmbm_config: LmbmConfig | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[NDArray[np.float64]],
        timestep: int,
    ) -> StateEstimate: ...
    def reset(self) -> None: ...
    @property
    def num_hypotheses(self) -> int: ...
    def __repr__(self) -> str: ...


# =============================================================================
# Multi-sensor filters
# =============================================================================


class AaLmbFilter(MultisensorFilter):
    """Multi-sensor LMB filter with Arithmetic Average fusion."""

    def __init__(
        self,
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association: AssociationConfig | None = None,
        max_components: int = 100,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> StateEstimate: ...
    def reset(self) -> None: ...
    @property
    def num_tracks(self) -> int: ...
    def __repr__(self) -> str: ...


class GaLmbFilter(MultisensorFilter):
    """Multi-sensor LMB filter with Geometric Average fusion."""

    def __init__(
        self,
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association: AssociationConfig | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> StateEstimate: ...
    def reset(self) -> None: ...
    @property
    def num_tracks(self) -> int: ...
    def __repr__(self) -> str: ...


class PuLmbFilter(MultisensorFilter):
    """Multi-sensor LMB filter with Parallel Update fusion."""

    def __init__(
        self,
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association: AssociationConfig | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> StateEstimate: ...
    def reset(self) -> None: ...
    @property
    def num_tracks(self) -> int: ...
    def __repr__(self) -> str: ...


class IcLmbFilter(MultisensorFilter):
    """Multi-sensor LMB filter with Iterated Corrector fusion."""

    def __init__(
        self,
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association: AssociationConfig | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> StateEstimate: ...
    def reset(self) -> None: ...
    @property
    def num_tracks(self) -> int: ...
    def __repr__(self) -> str: ...


class MultisensorLmbmFilter:
    """Multi-sensor LMBM (LMB Mixture) filter."""

    def __init__(
        self,
        motion: MotionModel,
        sensors: MultisensorConfig,
        birth: BirthModel,
        association: AssociationConfig | None = None,
        lmbm_config: LmbmConfig | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> StateEstimate: ...
    def reset(self) -> None: ...
    @property
    def num_hypotheses(self) -> int: ...
    def __repr__(self) -> str: ...
