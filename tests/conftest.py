"""Test fixtures and comparison utilities for multisensor-lmb-filters-rs.

This module provides:
- Fixture loading helpers
- Full-structure comparison functions that compare ENTIRE outputs
- Factory functions for creating models from fixtures

IMPORTANT: Comparison functions compare ENTIRE structures and raise on
FIRST divergence with detailed error messages.
"""

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "data"
TOLERANCE = 1e-10


# =============================================================================
# Fixture Loading
# =============================================================================


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


@pytest.fixture
def lmb_fixture():
    """Load single-sensor LMB step-by-step fixture."""
    return load_fixture("step_by_step/lmb_step_by_step_seed42.json")


@pytest.fixture
def lmbm_fixture():
    """Load single-sensor LMBM step-by-step fixture."""
    return load_fixture("step_by_step/lmbm_step_by_step_seed42.json")


@pytest.fixture
def multisensor_lmb_fixture():
    """Load multi-sensor LMB step-by-step fixture."""
    return load_fixture("step_by_step/multisensor_lmb_step_by_step_seed42.json")


@pytest.fixture
def multisensor_lmbm_fixture():
    """Load multi-sensor LMBM step-by-step fixture."""
    return load_fixture("step_by_step/multisensor_lmbm_step_by_step_seed42.json")


# =============================================================================
# Comparison Functions - Compare ENTIRE structures
# =============================================================================


def compare_scalar(name: str, expected, actual, tol: float = TOLERANCE):
    """Compare scalar values, raise on mismatch."""
    diff = abs(expected - actual)
    if diff > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}, diff={diff}")


def compare_array(name: str, expected: list, actual: np.ndarray, tol: float = TOLERANCE):
    """Compare array values, raise detailed error on first mismatch."""
    expected_arr = np.array(expected, dtype=np.float64)

    # Check shape
    if expected_arr.shape != actual.shape:
        raise AssertionError(
            f"{name}: shape mismatch - expected {expected_arr.shape}, got {actual.shape}"
        )

    # Check values
    diff = np.abs(expected_arr - actual)
    max_diff = np.max(diff)
    if max_diff > tol:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        raise AssertionError(
            f"{name}[{idx}]: expected {expected_arr[idx]}, got {actual[idx]}, diff={max_diff}"
        )


def compare_track_estimate(name: str, expected: dict, actual, tol: float = TOLERANCE):
    """Compare a single track estimate against expected fixture data."""
    # Compare label
    exp_label = expected.get("label", expected.get("birthTime", [0, 0]))
    if isinstance(exp_label, list):
        exp_birth_time, exp_birth_loc = exp_label[0], exp_label[1]
    else:
        exp_birth_time = expected.get("birthTime", 0)
        exp_birth_loc = expected.get("birthLocation", 0)

    if actual.label.birth_time != exp_birth_time:
        raise AssertionError(
            f"{name}.label.birth_time: expected {exp_birth_time}, got {actual.label.birth_time}"
        )
    if actual.label.birth_location != exp_birth_loc:
        raise AssertionError(
            f"{name}.label.birth_location: expected {exp_birth_loc}, got {actual.label.birth_location}"
        )

    # Compare mean (use first component if multiple)
    if "mu" in expected:
        exp_mean = expected["mu"][0] if isinstance(expected["mu"][0], list) else expected["mu"]
        compare_array(f"{name}.mean", exp_mean, actual.mean, tol)


def compare_state_estimate(
    name: str,
    expected_n: int,
    expected_tracks: list | None,
    actual,
    tol: float = TOLERANCE,
):
    """Compare entire StateEstimate against expected values.

    Args:
        name: Name for error messages
        expected_n: Expected number of tracks (cardinality)
        expected_tracks: Expected track data (may be None if not available)
        actual: Actual StateEstimate from filter
        tol: Numerical tolerance
    """
    # Compare cardinality
    if actual.num_tracks != expected_n:
        raise AssertionError(f"{name}.num_tracks: expected {expected_n}, got {actual.num_tracks}")

    # Compare individual tracks if expected data is available
    if expected_tracks is not None and len(expected_tracks) > 0:
        if len(actual.tracks) != len(expected_tracks):
            raise AssertionError(
                f"{name}.tracks count: expected {len(expected_tracks)}, got {len(actual.tracks)}"
            )

        for i, (exp_track, act_track) in enumerate(zip(expected_tracks, actual.tracks)):
            compare_track_estimate(f"{name}.tracks[{i}]", exp_track, act_track, tol)


# =============================================================================
# Model Factory Functions
# =============================================================================


def make_motion_model(model: dict):
    """Create MotionModel from fixture model dict."""
    from multisensor_lmb_filters_rs import MotionModel

    return MotionModel(
        transition_matrix=np.array(model["A"], dtype=np.float64),
        process_noise=np.array(model["R"], dtype=np.float64),
        control_input=np.zeros(len(model["A"]), dtype=np.float64),
        survival_probability=model["P_s"],
    )


def make_sensor_model(model: dict, obs_volume: float = 40000.0):
    """Create SensorModel from fixture model dict (single-sensor)."""
    from multisensor_lmb_filters_rs import SensorModel

    return SensorModel(
        observation_matrix=np.array(model["C"], dtype=np.float64),
        measurement_noise=np.array(model["Q"], dtype=np.float64),
        detection_probability=model["P_d"],
        clutter_rate=model["clutter_per_unit_volume"] * obs_volume,
        observation_volume=obs_volume,
    )


def make_multisensor_config(model: dict, obs_volume: float = 40000.0):
    """Create SensorConfigMulti from fixture model dict (multi-sensor)."""
    from multisensor_lmb_filters_rs import SensorConfigMulti, SensorModel

    num_sensors = model["numberOfSensors"]
    sensors = []
    for i in range(num_sensors):
        sensors.append(
            SensorModel(
                observation_matrix=np.array(model["C"][i], dtype=np.float64),
                measurement_noise=np.array(model["Q"][i], dtype=np.float64),
                detection_probability=model["P_d"][i],
                clutter_rate=model["clutter_per_unit_volume"][i] * obs_volume,
                observation_volume=obs_volume,
            )
        )
    return SensorConfigMulti(sensors)


def make_birth_model(
    prior_objects: list,
    lmb_existence: float = 0.03,
    lmbm_existence: float = 0.003,
):
    """Create BirthModel from prior objects in fixture.

    Uses the first Gaussian component from each object as a birth location.
    """
    from multisensor_lmb_filters_rs import BirthLocation, BirthModel

    locations = []
    for i, obj in enumerate(prior_objects[:5]):  # Take first 5 objects as birth locations
        label = obj.get("label", [0, i])
        birth_loc_idx = label[1] if isinstance(label, list) else i
        locations.append(
            BirthLocation(
                label=birth_loc_idx,
                mean=np.array(obj["mu"][0], dtype=np.float64),
                covariance=np.array(obj["Sigma"][0], dtype=np.float64),
            )
        )

    return BirthModel(
        locations=locations,
        lmb_existence=lmb_existence,
        lmbm_existence=lmbm_existence,
    )


def measurements_to_numpy(measurements: list) -> list:
    """Convert fixture measurements to list of numpy arrays."""
    return [np.array(m, dtype=np.float64) for m in measurements]


def nested_measurements_to_numpy(measurements: list) -> list:
    """Convert nested fixture measurements to list of lists of numpy arrays."""
    return [[np.array(m, dtype=np.float64) for m in sensor_meas] for sensor_meas in measurements]
