#!/usr/bin/env python3
"""Minimal benchmark runner for Python bindings.

Usage:
    run_python.py --scenario <path> --filter <name>

Output:
    Prints elapsed time in milliseconds as a single number.
    Exit 0 on success, non-zero on error.
"""

import argparse
import json
import sys
import time

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

# Filter configuration: (class, is_multi_sensor)
FILTER_CLASSES = {
    "LMB-LBP": (FilterLmb, False),
    "LMB-Gibbs": (FilterLmb, False),
    "LMB-Murty": (FilterLmb, False),
    "LMBM-Gibbs": (FilterLmbm, False),
    "LMBM-Murty": (FilterLmbm, False),
    "AA-LMB-LBP": (FilterAaLmb, True),
    "IC-LMB-LBP": (FilterIcLmb, True),
    "PU-LMB-LBP": (FilterPuLmb, True),
    "GA-LMB-LBP": (FilterGaLmb, True),
    "MS-LMBM-Gibbs": (FilterMultisensorLmbm, True),
}

THRESHOLDS = FilterThresholds(
    existence=1e-3, gm_weight=1e-4, max_components=100, gm_merge=float("inf")
)


def get_associator(filter_name: str) -> AssociatorConfig:
    """Get associator config based on filter name suffix."""
    if "LBP" in filter_name:
        return AssociatorConfig.lbp(100, 1e-6)
    elif "Gibbs" in filter_name:
        return AssociatorConfig.gibbs(1000)
    elif "Murty" in filter_name:
        return AssociatorConfig.murty(25)
    return AssociatorConfig.lbp(100, 1e-6)


def preprocess(scenario: dict):
    """Convert scenario JSON to filter inputs."""
    m = scenario["model"]
    bounds = scenario["bounds"]
    n_sensors = scenario["num_sensors"]

    # Motion model: constant velocity 2D
    motion = MotionModel.constant_velocity_2d(
        m["dt"], m["process_noise_std"], m["survival_probability"]
    )

    # Observation volume
    obs_vol = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])

    # Sensor model
    sensor = SensorModel.position_2d(
        m["measurement_noise_std"], m["detection_probability"], m["clutter_rate"], obs_vol
    )

    # Birth model
    birth_locs = [
        BirthLocation(i, np.array(loc), np.diag([2500.0, 2500.0, 100.0, 100.0]))
        for i, loc in enumerate(m["birth_locations"])
    ]
    birth = BirthModel(birth_locs, lmb_existence=0.01, lmbm_existence=0.001)

    # Multi-sensor config
    multi_sensor = SensorConfigMulti([sensor] * n_sensors)

    # Extract steps
    steps = []
    for step in scenario["steps"]:
        t = step["step"]
        readings = step["sensor_readings"]
        multi_meas = [np.array(s) if s else np.empty((0, 2)) for s in readings]
        single_meas = multi_meas[0] if multi_meas else np.empty((0, 2))
        steps.append((t, single_meas, multi_meas))

    return motion, sensor, multi_sensor, birth, steps


def run_filter(filt, steps, is_multi: bool) -> float:
    """Run filter and return elapsed time in milliseconds."""
    start = time.perf_counter()
    for t, single_meas, multi_meas in steps:
        _ = filt.step(multi_meas if is_multi else single_meas, t)
    return (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description="Minimal Python benchmark runner")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file")
    parser.add_argument("--filter", required=True, help="Filter name (e.g., LMB-LBP)")
    args = parser.parse_args()

    # Validate filter name
    if args.filter not in FILTER_CLASSES:
        print(f"Unknown filter: {args.filter}", file=sys.stderr)
        print(f"Available: {', '.join(FILTER_CLASSES.keys())}", file=sys.stderr)
        return 1

    # Load scenario
    with open(args.scenario) as f:
        scenario = json.load(f)

    # Preprocess
    motion, sensor, multi_sensor, birth, steps = preprocess(scenario)

    # Create filter
    filter_cls, is_multi = FILTER_CLASSES[args.filter]
    assoc = get_associator(args.filter)

    filt = filter_cls(motion, multi_sensor if is_multi else sensor, birth, assoc, THRESHOLDS)

    # Run benchmark
    elapsed_ms = run_filter(filt, steps, is_multi)

    # Output only the timing
    print(f"{elapsed_ms:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
