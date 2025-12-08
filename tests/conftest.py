"""Test fixtures and helpers for multisensor-lmb-filters-rs Python bindings."""

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "data"
TOLERANCE = 1e-10


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


@pytest.fixture
def single_trial_fixture():
    """Load single-sensor trial fixture."""
    return load_fixture("single_trial_42.json")


@pytest.fixture
def multisensor_trial_fixture():
    """Load multi-sensor trial fixture."""
    return load_fixture("multisensor_trial_42.json")


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


def make_birth_model_from_objects(
    objects: list, lmb_existence: float = 0.03, lmbm_existence: float = 0.003
):
    """Create BirthModel from prior objects in fixture.

    Uses the first Gaussian component from each object as a birth location.
    Birth locations are extracted from the last few objects which typically
    represent potential birth locations in the fixture data.
    """
    from multisensor_lmb_filters_rs import BirthLocation, BirthModel

    # Extract birth locations - use distinct labels
    locations = []
    for i, obj in enumerate(objects[:5]):  # Take first 5 objects as birth locations
        label = obj["label"]
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
