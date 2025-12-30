"""
Python benchmark runner for multi-object tracking filters.

Usage:
    uv run python benchmarks/run_python.py                    # Run all
    uv run python benchmarks/run_python.py --filter LMB       # Only LMB filters
    uv run python benchmarks/run_python.py --assoc Gibbs      # Only Gibbs associator
    uv run python benchmarks/run_python.py --scenario n5      # Only scenarios with 'n5'
"""

import argparse
import json
import signal
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
TIMEOUT_SEC = 10
THRESHOLDS = FilterThresholds(
    existence=1e-3, gm_weight=1e-4, max_components=100, gm_merge=float("inf")
)

# Associator configs (explicit values matching Rust)
LBP = lambda: AssociatorConfig.lbp(100, 1e-6)
GIBBS = lambda: AssociatorConfig.gibbs(1000)
MURTY = lambda: AssociatorConfig.murty(25)

# All Filter × Associator configurations
# Format: (name, filter_class, associator_factory, is_multi_sensor)
CONFIGS = [
    # Single-sensor
    ("LMB-LBP", FilterLmb, LBP, False),
    ("LMB-Gibbs", FilterLmb, GIBBS, False),
    ("LMB-Murty", FilterLmb, MURTY, False),
    ("LMBM-Gibbs", FilterLmbm, GIBBS, False),
    ("LMBM-Murty", FilterLmbm, MURTY, False),
    # Multi-sensor (can also run on single-sensor scenarios)
    ("AA-LMB-LBP", FilterAaLmb, LBP, True),
    ("AA-LMB-Gibbs", FilterAaLmb, GIBBS, True),
    ("AA-LMB-Murty", FilterAaLmb, MURTY, True),
    ("GA-LMB-LBP", FilterGaLmb, LBP, True),
    ("GA-LMB-Gibbs", FilterGaLmb, GIBBS, True),
    ("GA-LMB-Murty", FilterGaLmb, MURTY, True),
    ("PU-LMB-LBP", FilterPuLmb, LBP, True),
    ("PU-LMB-Gibbs", FilterPuLmb, GIBBS, True),
    ("PU-LMB-Murty", FilterPuLmb, MURTY, True),
    ("IC-LMB-LBP", FilterIcLmb, LBP, True),
    ("IC-LMB-Gibbs", FilterIcLmb, GIBBS, True),
    ("IC-LMB-Murty", FilterIcLmb, MURTY, True),
    ("MS-LMBM-Gibbs", FilterMultisensorLmbm, GIBBS, True),
    ("MS-LMBM-Murty", FilterMultisensorLmbm, MURTY, True),
]


class TimeoutError(Exception):
    pass


# Global state for timeout handler
_current_progress = {"steps_done": 0, "total_steps": 0}


def on_timeout(signum, frame):
    raise TimeoutError()


def preprocess(scenario):
    """Convert scenario to preprocessed measurements (done outside timing)."""
    m = scenario["model"]
    bounds = scenario["bounds"]
    n_sensors = scenario["num_sensors"]

    # Build models
    motion = MotionModel.constant_velocity_2d(
        m["dt"], m["process_noise_std"], m["survival_probability"]
    )
    obs_vol = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    sensor = SensorModel.position_2d(
        m["measurement_noise_std"], m["detection_probability"], m["clutter_rate"], obs_vol
    )
    birth_locs = [
        BirthLocation(i, np.array(loc), np.eye(4) * 100.0)
        for i, loc in enumerate(m["birth_locations"])
    ]
    birth = BirthModel(birth_locs, lmb_existence=0.01, lmbm_existence=0.001)
    multi_sensor = SensorConfigMulti([sensor] * n_sensors)

    # Preprocess measurements: list of (timestep, single_meas, multi_meas, n_true)
    steps = []
    for step in scenario["steps"]:
        t = step["step"]
        multi_meas = [
            np.array([[r[0], r[1]] for r in s]) if s else np.empty((0, 2))
            for s in step["sensor_readings"]
        ]
        single_meas = multi_meas[0] if multi_meas else np.empty((0, 2))
        # Ground truth: count unique object IDs (excluding clutter which has id=-1)
        n_true = len(set(int(r[2]) for s in step["sensor_readings"] for r in s if int(r[2]) >= 0))
        steps.append((t, single_meas, multi_meas, n_true))

    return motion, sensor, multi_sensor, birth, steps


def run_filter(filt, steps, is_multi):
    """Run filter on preprocessed steps, return (elapsed_ms, card_err, steps_done)."""
    global _current_progress
    n_est_list, n_true_list = [], []
    _current_progress = {"steps_done": 0, "total_steps": len(steps)}

    start = time.perf_counter()
    for i, (t, single_meas, multi_meas, n_true) in enumerate(steps):
        result = filt.step(multi_meas if is_multi else single_meas, t)
        n_est_list.append(result.num_tracks)
        n_true_list.append(n_true)
        _current_progress["steps_done"] = i + 1

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Simple accuracy: mean absolute cardinality error
    card_err = sum(abs(e - t) for e, t in zip(n_est_list, n_true_list, strict=False)) / len(
        n_est_list
    )

    return elapsed_ms, card_err, len(steps)


def main():
    parser = argparse.ArgumentParser(description="Run LMB filter benchmarks")
    parser.add_argument("--filter", "-f", help="Filter name substring (e.g., 'LMB', 'AA-LMB')")
    parser.add_argument("--assoc", "-a", help="Associator name substring (e.g., 'Gibbs', 'LBP')")
    parser.add_argument("--scenario", "-s", help="Scenario name substring (e.g., 'n5', 's2')")
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=TIMEOUT_SEC,
        help=f"Timeout in seconds (default: {TIMEOUT_SEC})",
    )
    args = parser.parse_args()

    scenarios = sorted(SCENARIOS_DIR.glob("*.json"))
    if not scenarios:
        print(f"No scenarios in {SCENARIOS_DIR}")
        return

    # Filter scenarios by name
    if args.scenario:
        scenarios = [p for p in scenarios if args.scenario.lower() in p.stem.lower()]
        if not scenarios:
            print(f"No scenarios matching '{args.scenario}'")
            return

    # Filter configs
    configs = CONFIGS
    if args.filter:
        configs = [(n, c, a, m) for n, c, a, m in configs if args.filter.lower() in n.lower()]
    if args.assoc:
        configs = [(n, c, a, m) for n, c, a, m in configs if args.assoc.lower() in n.lower()]
    if not configs:
        print(f"No configs matching filter='{args.filter}' assoc='{args.assoc}'")
        return

    # Setup timeout (Unix only)
    try:
        signal.signal(signal.SIGALRM, on_timeout)
        use_timeout = True
    except (AttributeError, ValueError):
        use_timeout = False

    print(f"{'Scenario':<22} | {'Filter':<18} | {'Time(ms)':>9} | {'CardErr':>7} | {'Progress':>8}")
    print("-" * 75)

    for path in scenarios:
        scenario = json.load(open(path))
        name = path.stem

        # Preprocess outside timing
        motion, sensor, multi_sensor, birth, steps = preprocess(scenario)

        for filter_name, filter_cls, assoc_fn, is_multi in configs:
            try:
                if use_timeout:
                    signal.alarm(args.timeout)

                # Create filter
                filt = filter_cls(
                    motion, multi_sensor if is_multi else sensor, birth, assoc_fn(), THRESHOLDS
                )

                # Run and time
                elapsed_ms, card_err, steps_done = run_filter(filt, steps, is_multi)

                if use_timeout:
                    signal.alarm(0)

                progress = f"{steps_done}/{len(steps)}"
                print(
                    f"{name:<22} | {filter_name:<18} | {elapsed_ms:>9.1f} | {card_err:>7.2f} | {progress:>8}"
                )

            except TimeoutError:
                if use_timeout:
                    signal.alarm(0)
                progress = f"{_current_progress['steps_done']}/{_current_progress['total_steps']}"
                print(f"{name:<22} | {filter_name:<18} | {'TIMEOUT':>9} | {'-':>7} | {progress:>8}")

            except Exception:
                if use_timeout:
                    signal.alarm(0)
                progress = f"{_current_progress['steps_done']}/{_current_progress['total_steps']}"
                print(f"{name:<22} | {filter_name:<18} | {'ERROR':>9} | {'-':>7} | {progress:>8}")


if __name__ == "__main__":
    main()
