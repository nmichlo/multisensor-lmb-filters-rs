"""Type stubs for multisensor_lmb_filters_rs native module."""

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

__version__: str

# =============================================================================
# Intermediate Types (for step_detailed output)
# =============================================================================

class _TrackData:
    """Full track data matching fixture format."""

    label: tuple[int, int]
    existence: float
    mu: NDArray[np.float64]  # [n_components x state_dim]
    sigma: NDArray[np.float64]  # [n_components x state_dim x state_dim]
    w: NDArray[np.float64]  # [n_components]

    def __init__(
        self,
        label: tuple[int, int],
        existence: float,
        means: list[list[float]],
        covariances: list[list[list[float]]],
        weights: list[float],
    ) -> None: ...

class _PosteriorParameters:
    """Posterior parameters for a single track."""

    w: NDArray[np.float64]  # [num_meas + 1, num_comp]
    mu: NDArray[np.float64]  # [num_meas * num_comp, state_dim]
    sigma: NDArray[np.float64]  # [num_meas * num_comp, state_dim, state_dim]

class _AssociationMatrices:
    """Matrices from association builder."""

    cost: NDArray[np.float64]  # [n_tracks x n_measurements]
    likelihood: NDArray[np.float64]  # [n_tracks x (n_measurements + 1)]
    miss_prob: NDArray[np.float64]  # [n_tracks x (n_measurements + 1)]
    sampling_prob: NDArray[np.float64]  # [n_tracks x n_measurements]
    eta: NDArray[np.float64]  # [n_tracks]
    posterior_parameters: list[_PosteriorParameters]

class _AssociationResult:
    """Result from data association algorithm."""

    marginal_weights: NDArray[np.float64]  # [n_tracks x n_measurements]
    miss_weights: NDArray[np.float64]  # [n_tracks]
    posterior_existence: NDArray[np.float64]  # [n_tracks]
    assignments: NDArray[np.int32] | None  # [n_samples x n_tracks] (Gibbs/Murty only)

class _CardinalityEstimate:
    """Cardinality extraction result."""

    n_estimated: int
    map_indices: NDArray[np.intp]

class _LmbmHypothesis:
    """LMBM hypothesis for setting filter state."""

    log_weight: float
    weight: float
    num_tracks: int
    r: list[float]
    mu: list[list[float]]
    sigma: list[list[list[float]]]
    birth_time: list[int]
    birth_location: list[int]

    def __init__(
        self,
        log_weight: float,
        tracks: list[_TrackData],
    ) -> None: ...
    @staticmethod
    def from_matlab(
        w: float,
        r: list[float],
        mu: list[list[float]],
        sigma: list[list[list[float]]],
        birth_time: list[int],
        birth_location: list[int],
    ) -> _LmbmHypothesis: ...

class _SensorUpdateOutput:
    """Per-sensor intermediate data from multisensor filter update."""

    sensor_index: int
    input_tracks: list[_TrackData]  # Prior tracks for this sensor's update
    association_matrices: _AssociationMatrices | None
    association_result: _AssociationResult | None
    updated_tracks: list[_TrackData]

class _StepOutput:
    """Full step output for fixture comparison."""

    predicted_tracks: list[_TrackData]
    association_matrices: _AssociationMatrices | None
    association_result: _AssociationResult | None
    updated_tracks: list[_TrackData]
    cardinality: _CardinalityEstimate
    # Multisensor-specific fields (None for single-sensor filters)
    sensor_updates: list[_SensorUpdateOutput] | None
    # LMBM-specific fields (None for LMB filters)
    predicted_hypotheses: list[_LmbmHypothesis] | None
    pre_normalization_hypotheses: list[_LmbmHypothesis] | None
    normalized_hypotheses: list[_LmbmHypothesis] | None
    objects_likely_to_exist: list[bool] | None

# =============================================================================
# Models
# =============================================================================

class MotionModel:
    """Motion model for track state prediction."""

    x_dim: int
    survival_probability: float
    transition_matrix: NDArray[np.float64]
    process_noise: NDArray[np.float64]

    def __init__(
        self,
        transition_matrix: NDArray[np.float64],
        process_noise: NDArray[np.float64],
        control_input: NDArray[np.float64],
        survival_probability: float,
    ) -> None: ...
    @staticmethod
    def constant_velocity_2d(
        dt: float,
        process_noise_std: float,
        survival_probability: float,
    ) -> MotionModel: ...

class SensorModel:
    """Sensor observation model."""

    x_dim: int
    z_dim: int
    detection_probability: float
    clutter_rate: float
    observation_volume: float
    observation_matrix: NDArray[np.float64]
    measurement_noise: NDArray[np.float64]

    def __init__(
        self,
        observation_matrix: NDArray[np.float64],
        measurement_noise: NDArray[np.float64],
        detection_probability: float,
        clutter_rate: float,
        observation_volume: float,
    ) -> None: ...
    @staticmethod
    def position_2d(
        measurement_noise_std: float,
        detection_probability: float,
        clutter_rate: float,
        observation_volume: float,
    ) -> SensorModel: ...

class SensorConfigMulti:
    """Multi-sensor configuration."""

    num_sensors: int

    def __init__(self, sensors: Sequence[SensorModel]) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> SensorModel: ...

# =============================================================================
# Birth
# =============================================================================

class BirthLocation:
    """Birth location specification."""

    label: int
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]

    def __init__(
        self,
        label: int,
        mean: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> None: ...

class BirthModel:
    """Birth model for new tracks."""

    num_locations: int
    lmb_existence: float
    lmbm_existence: float

    def __init__(
        self,
        locations: Sequence[BirthLocation],
        lmb_existence: float,
        lmbm_existence: float,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> BirthLocation: ...

# =============================================================================
# Configuration
# =============================================================================

class AssociatorConfig:
    """Data association configuration."""

    @staticmethod
    def lbp(max_iterations: int = 100, tolerance: float = 1e-6) -> AssociatorConfig: ...
    @staticmethod
    def gibbs(samples: int = 1000) -> AssociatorConfig: ...
    @staticmethod
    def murty(assignments: int = 100) -> AssociatorConfig: ...

class FilterThresholds:
    """Filter pruning thresholds."""

    existence_threshold: float
    gm_weight_threshold: float
    max_gm_components: int
    gm_merge_threshold: float

    def __init__(
        self,
        existence: float = 0.5,
        gm_weight: float = 1e-4,
        max_components: int = 100,
        min_trajectory_length: int = 3,
        gm_merge: float = ...,  # Default is f64::INFINITY
    ) -> None: ...

class FilterLmbmConfig:
    """LMBM-specific configuration."""

    def __init__(
        self,
        max_hypotheses: int = 1000,
        hypothesis_weight_threshold: float = 1e-6,
        use_eap: bool = False,
        existence_threshold: float | None = None,
    ) -> None: ...

# =============================================================================
# Output Types
# =============================================================================

class TrackLabel:
    """Unique track identifier."""

    birth_time: int
    birth_location: int

    def __init__(self, birth_time: int, birth_location: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class GaussianComponent:
    """Gaussian mixture component."""

    weight: float
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    x_dim: int

class TrackEstimate:
    """Single track state estimate."""

    label: TrackLabel
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    x_dim: int

class StateEstimate:
    """All track estimates at one timestep."""

    timestamp: int
    num_tracks: int
    tracks: list[TrackEstimate]

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> TrackEstimate: ...
    def __iter__(self) -> Iterator[TrackEstimate]: ...

# =============================================================================
# Filters - Single-sensor
# =============================================================================

class FilterLmb:
    """LMB filter with Gaussian mixture track representation."""

    num_tracks: int

    def __init__(
        self,
        motion: MotionModel,
        sensor: SensorModel,
        birth: BirthModel,
        association: AssociatorConfig | None = None,
        thresholds: FilterThresholds | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[NDArray[np.float64]],
        timestep: int,
    ) -> StateEstimate: ...
    def step_detailed(
        self,
        measurements: Sequence[NDArray[np.float64]],
        timestep: int,
    ) -> _StepOutput: ...
    def get_tracks(self) -> list[_TrackData]: ...
    def set_tracks(self, tracks: list[_TrackData]) -> None: ...
    def reset(self) -> None: ...

class FilterLmbm:
    """LMBM filter with hypothesis-based representation."""

    num_tracks: int
    num_hypotheses: int

    def __init__(
        self,
        motion: MotionModel,
        sensor: SensorModel,
        birth: BirthModel,
        association: AssociatorConfig | None = None,
        thresholds: FilterThresholds | None = None,
        lmbm_config: FilterLmbmConfig | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[NDArray[np.float64]],
        timestep: int,
    ) -> StateEstimate: ...
    def step_detailed(
        self,
        measurements: Sequence[NDArray[np.float64]],
        timestep: int,
    ) -> _StepOutput: ...
    def get_tracks(self) -> list[_TrackData]: ...
    def set_hypotheses(self, hypotheses: list[_LmbmHypothesis]) -> None: ...
    def reset(self) -> None: ...

# =============================================================================
# Filters - Multi-sensor
# =============================================================================

class FilterAaLmb:
    """Arithmetic Average multi-sensor LMB filter."""

    num_tracks: int

    def __init__(
        self,
        motion: MotionModel,
        sensors: SensorConfigMulti,
        birth: BirthModel,
        association: AssociatorConfig | None = None,
        thresholds: FilterThresholds | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> StateEstimate: ...
    def step_detailed(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> _StepOutput: ...
    def get_tracks(self) -> list[_TrackData]: ...
    def set_tracks(self, tracks: list[_TrackData]) -> None: ...
    def reset(self) -> None: ...

class FilterGaLmb:
    """Geometric Average multi-sensor LMB filter."""

    num_tracks: int

    def __init__(
        self,
        motion: MotionModel,
        sensors: SensorConfigMulti,
        birth: BirthModel,
        association: AssociatorConfig | None = None,
        thresholds: FilterThresholds | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> StateEstimate: ...
    def step_detailed(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> _StepOutput: ...
    def get_tracks(self) -> list[_TrackData]: ...
    def set_tracks(self, tracks: list[_TrackData]) -> None: ...
    def reset(self) -> None: ...

class FilterPuLmb:
    """Parallel Update multi-sensor LMB filter."""

    num_tracks: int

    def __init__(
        self,
        motion: MotionModel,
        sensors: SensorConfigMulti,
        birth: BirthModel,
        association: AssociatorConfig | None = None,
        thresholds: FilterThresholds | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> StateEstimate: ...
    def step_detailed(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> _StepOutput: ...
    def get_tracks(self) -> list[_TrackData]: ...
    def set_tracks(self, tracks: list[_TrackData]) -> None: ...
    def reset(self) -> None: ...

class FilterIcLmb:
    """Iterated Corrector multi-sensor LMB filter."""

    num_tracks: int

    def __init__(
        self,
        motion: MotionModel,
        sensors: SensorConfigMulti,
        birth: BirthModel,
        association: AssociatorConfig | None = None,
        thresholds: FilterThresholds | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> StateEstimate: ...
    def step_detailed(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> _StepOutput: ...
    def get_tracks(self) -> list[_TrackData]: ...
    def set_tracks(self, tracks: list[_TrackData]) -> None: ...
    def reset(self) -> None: ...

class FilterMultisensorLmbm:
    """Multi-sensor LMBM filter."""

    num_tracks: int
    num_hypotheses: int

    def __init__(
        self,
        motion: MotionModel,
        sensors: SensorConfigMulti,
        birth: BirthModel,
        association: AssociatorConfig | None = None,
        thresholds: FilterThresholds | None = None,
        lmbm_config: FilterLmbmConfig | None = None,
        seed: int | None = None,
    ) -> None: ...
    def step(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> StateEstimate: ...
    def step_detailed(
        self,
        measurements: Sequence[Sequence[NDArray[np.float64]]],
        timestep: int,
    ) -> _StepOutput: ...
    def get_tracks(self) -> list[_TrackData]: ...
    def set_hypotheses(self, hypotheses: list[_LmbmHypothesis]) -> None: ...
    def reset(self) -> None: ...
