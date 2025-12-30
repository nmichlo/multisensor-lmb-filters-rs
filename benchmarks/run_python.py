"""
Python benchmark runner for multi-object tracking filters.

Loads pre-generated JSON scenarios and runs filters, printing timing to stdout.
"""

import json
import time
from pathlib import Path

import numpy as np
from multisensor_lmb_filters_rs import (
    AssociatorConfig,
    BirthLocation,
    BirthModel,
    FilterAaLmb,
    FilterGaLmb,
    FilterIcLmb,
    FilterLmb,
    FilterLmbm,
    FilterMultisensorLmbm,
    FilterPuLmb,
    FilterThresholds,
    MotionModel,
    SensorConfigMulti,
    SensorModel,
)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# Timeout in seconds - skip scenarios that take longer
TIMEOUT_SECONDS = 10.0


def build_models(scenario: dict):
    """Build filter models from scenario config."""
    model = scenario["model"]

    # Motion model: constant velocity 2D
    motion = MotionModel.constant_velocity_2d(
        dt=model["dt"],
        process_noise_std=model["process_noise_std"],
        survival_probability=model["survival_probability"],
    )

    # Sensor model: 2D position observations
    # observation_volume = area of observation space
    observation_volume = 200.0 * 200.0  # bounds are [-100, 100] x [-100, 100]
    sensor = SensorModel.position_2d(
        measurement_noise_std=model["measurement_noise_std"],
        detection_probability=model["detection_probability"],
        clutter_rate=model["clutter_rate"],
        observation_volume=observation_volume,
    )

    # Birth model: 4 birth locations
    birth_locs = []
    for i, loc in enumerate(model["birth_locations"]):
        mean = np.array(loc, dtype=np.float64)
        covariance = np.eye(4, dtype=np.float64) * 100.0
        birth_locs.append(BirthLocation(i, mean, covariance))

    birth = BirthModel(birth_locs, lmb_existence=0.01, lmbm_existence=0.001)

    return motion, sensor, birth


def extract_measurements(readings: list) -> np.ndarray:
    """Extract [x, y] from [x, y, id] format."""
    if not readings:
        return np.empty((0, 2), dtype=np.float64)
    return np.array([[r[0], r[1]] for r in readings], dtype=np.float64)


def run_single_sensor(scenario: dict, filter_cls, assoc_config) -> float:
    """Run single-sensor filter, return elapsed time in seconds."""
    motion, sensor, birth = build_models(scenario)
    thresholds = FilterThresholds()

    filt = filter_cls(motion, sensor, birth, assoc_config, thresholds)

    start = time.perf_counter()
    for step in scenario["steps"]:
        meas = extract_measurements(step["sensor_readings"][0])
        filt.step(meas, step["step"])
    return time.perf_counter() - start


def run_multi_sensor(scenario: dict, filter_cls, assoc_config) -> float:
    """Run multi-sensor filter, return elapsed time in seconds."""
    motion, sensor, birth = build_models(scenario)
    n_sens = scenario["num_sensors"]
    multi_sensor = SensorConfigMulti([sensor] * n_sens)
    thresholds = FilterThresholds()

    filt = filter_cls(motion, multi_sensor, birth, assoc_config, thresholds)

    start = time.perf_counter()
    for step in scenario["steps"]:
        meas_list = [extract_measurements(r) for r in step["sensor_readings"]]
        filt.step(meas_list, step["step"])
    return time.perf_counter() - start


def main():
    if not SCENARIOS_DIR.exists():
        print(f"Error: scenarios directory not found: {SCENARIOS_DIR}")
        print("Run `uv run python benchmarks/generate_scenarios.py` first.")
        return

    scenario_files = sorted(SCENARIOS_DIR.glob("*.json"))
    if not scenario_files:
        print(f"Error: no scenario files found in {SCENARIOS_DIR}")
        return

    print(f"{'Scenario':<25} | {'Filter':<20} | {'Time (ms)':>12}", flush=True)
    print("-" * 65, flush=True)

    for path in scenario_files:
        with open(path) as f:
            scenario = json.load(f)

        name = path.stem
        n_sens = scenario["num_sensors"]

        if n_sens == 1:
            # Single-sensor: test LMB with different associators
            for assoc_name, assoc_config in [
                ("LMB-LBP", AssociatorConfig.lbp()),
                ("LMB-Gibbs", AssociatorConfig.gibbs()),
                ("LMB-Murty", AssociatorConfig.murty()),
            ]:
                try:
                    t = run_single_sensor(scenario, FilterLmb, assoc_config)
                    print(f"{name:<25} | {assoc_name:<20} | {t*1000:>12.1f}", flush=True)
                except Exception:
                    print(f"{name:<25} | {assoc_name:<20} | {'ERROR':>12}", flush=True)

            # Also test LMBM (Gibbs only, as it doesn't support LBP)
            for assoc_name, assoc_config in [
                ("LMBM-Gibbs", AssociatorConfig.gibbs()),
                ("LMBM-Murty", AssociatorConfig.murty()),
            ]:
                try:
                    t = run_single_sensor(scenario, FilterLmbm, assoc_config)
                    print(f"{name:<25} | {assoc_name:<20} | {t*1000:>12.1f}", flush=True)
                except Exception:
                    print(f"{name:<25} | {assoc_name:<20} | {'ERROR':>12}", flush=True)

        else:
            # Multi-sensor: test different fusion strategies with LBP
            assoc_config = AssociatorConfig.lbp()
            for filter_cls in [FilterAaLmb, FilterGaLmb, FilterPuLmb, FilterIcLmb]:
                filter_name = filter_cls.__name__
                try:
                    t = run_multi_sensor(scenario, filter_cls, assoc_config)
                    print(f"{name:<25} | {filter_name:<20} | {t*1000:>12.1f}", flush=True)
                except Exception:
                    print(f"{name:<25} | {filter_name:<20} | {'ERROR':>12}", flush=True)

            # Also test multi-sensor LMBM
            try:
                t = run_multi_sensor(scenario, FilterMultisensorLmbm, AssociatorConfig.gibbs())
                print(f"{name:<25} | {'FilterMsLmbm':<20} | {t*1000:>12.1f}", flush=True)
            except Exception:
                print(f"{name:<25} | {'FilterMsLmbm':<20} | {'ERROR':>12}", flush=True)


if __name__ == "__main__":
    main()
