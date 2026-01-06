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
def LBP():
    return AssociatorConfig.lbp(100, 1e-6)


def GIBBS():
    return AssociatorConfig.gibbs(1000)


def MURTY():
    return AssociatorConfig.murty(25)


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


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Simple O(n³) Hungarian algorithm for square/rectangular cost matrices."""
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    INF = float("inf")
    for i in range(1, n + 1):
        p[0], minv, used = i, np.full(m + 1, INF), np.zeros(m + 1, dtype=bool)
        j0 = 0
        while p[j0] != 0:
            used[j0], i0, delta, j1 = True, p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
        while j0:
            p[j0], j0 = p[way[j0]], way[j0]
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]


def compute_ospa(
    estimates: list[np.ndarray], truths: list[np.ndarray], c: float = 50.0, p: float = 2.0
) -> float:
    """Compute mean OSPA using Hungarian (optimal) assignment."""
    ospa_sum = 0.0
    for est, gt in zip(estimates, truths, strict=False):
        m, n = len(est), len(gt)
        if m == 0 and n == 0:
            continue
        if m == 0 or n == 0:
            ospa_sum += c
            continue
        # Build cost matrix and solve with Hungarian
        cost = np.array([[min(np.linalg.norm(e - g), c) ** p for g in gt] for e in est])
        if m <= n:
            matches = hungarian(cost)
            loc_sum = sum(cost[i, j] for i, j in matches)
        else:
            matches = hungarian(cost.T)
            loc_sum = sum(cost[j, i] for i, j in matches)
        ospa_sum += ((loc_sum + (c**p) * abs(m - n)) / max(m, n)) ** (1 / p)
    return ospa_sum / len(estimates) if estimates else 0.0


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
    # Birth covariance: [2500, 2500, 100, 100] for [x, y, vx, vy]
    # Matches fixture and MATLAB benchmark settings - create separate matrix for each location
    birth_locs = [
        BirthLocation(i, np.array(loc), np.diag([2500.0, 2500.0, 100.0, 100.0]))
        for i, loc in enumerate(m["birth_locations"])
    ]
    birth = BirthModel(birth_locs, lmb_existence=0.01, lmbm_existence=0.001)
    multi_sensor = SensorConfigMulti([sensor] * n_sensors)

    # Preprocess measurements: list of (timestep, single_meas, multi_meas, gt_positions)
    steps = []
    for step in scenario["steps"]:
        t = step["step"]
        multi_meas = [np.array(s) if s else np.empty((0, 2)) for s in step["sensor_readings"]]
        single_meas = multi_meas[0] if multi_meas else np.empty((0, 2))
        gt_positions = (
            np.array(step["ground_truth"]) if step.get("ground_truth") else np.empty((0, 2))
        )
        steps.append((t, single_meas, multi_meas, gt_positions))

    return motion, sensor, multi_sensor, birth, steps


def run_filter(filt, steps, is_multi):
    """Run filter on preprocessed steps, return (elapsed_ms, ospa, steps_done)."""
    global _current_progress
    estimated_positions = []
    ground_truth_positions = []
    _current_progress = {"steps_done": 0, "total_steps": len(steps)}

    # === TIMED SECTION ===
    start = time.perf_counter()
    for i, (t, single_meas, multi_meas, gt_positions) in enumerate(steps):
        result = filt.step(multi_meas if is_multi else single_meas, t)

        # Collect positions (minimal overhead: just extract x,y from state mean)
        # State is [x, vx, y, vy] for constant velocity 2D model
        est_pos = (
            np.array([[tr.mean[0], tr.mean[2]] for tr in result.tracks])
            if result.tracks
            else np.empty((0, 2))
        )
        estimated_positions.append(est_pos)
        ground_truth_positions.append(gt_positions)

        _current_progress["steps_done"] = i + 1

    elapsed_ms = (time.perf_counter() - start) * 1000
    # === END TIMED SECTION ===

    # Compute OSPA (outside timing - O(n³) per step due to Hungarian algorithm)
    ospa = compute_ospa(estimated_positions, ground_truth_positions, c=50.0, p=2.0)

    return elapsed_ms, ospa, len(steps)


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

    # Sort scenarios by complexity (n5 < n10 < n20 < n50, then s1 < s2 < s4 < s8)
    def complexity_key(path):
        import re

        match = re.search(r"n(\d+)_s(\d+)", path.stem)
        if match:
            n_objects, n_sensors = int(match.group(1)), int(match.group(2))
            return (n_objects, n_sensors)
        return (999, 999)  # Unknown format goes last

    scenarios = sorted(scenarios, key=complexity_key)

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

    print(f"{'Scenario':<22} | {'Filter':<18} | {'Time(ms)':>9} | {'OSPA':>7} | {'Progress':>8}")
    print("-" * 75)

    # Track which filters have timed out - once a filter times out, skip remaining scenarios
    timed_out_filters = set()

    for path in scenarios:
        scenario = json.load(open(path))
        name = path.stem

        # Preprocess outside timing
        motion, sensor, multi_sensor, birth, steps = preprocess(scenario)

        for filter_name, filter_cls, assoc_fn, is_multi in configs:
            # Skip if this filter already timed out on an easier scenario
            if filter_name in timed_out_filters:
                print(f"{name:<22} | {filter_name:<18} | {'SKIP':>9} | {'-':>7} | {'-':>8}")
                continue
            try:
                if use_timeout:
                    signal.alarm(args.timeout)

                # Create filter
                filt = filter_cls(
                    motion, multi_sensor if is_multi else sensor, birth, assoc_fn(), THRESHOLDS
                )

                # Run and time
                elapsed_ms, ospa, steps_done = run_filter(filt, steps, is_multi)

                if use_timeout:
                    signal.alarm(0)

                progress = f"{steps_done}/{len(steps)}"
                print(
                    f"{name:<22} | {filter_name:<18} | {elapsed_ms:>9.1f} | "
                    f"{ospa:>7.2f} | {progress:>8}"
                )

            except TimeoutError:
                if use_timeout:
                    signal.alarm(0)
                progress = f"{_current_progress['steps_done']}/{_current_progress['total_steps']}"
                print(f"{name:<22} | {filter_name:<18} | {'TIMEOUT':>9} | {'-':>7} | {progress:>8}")
                # Mark this filter as timed out - skip remaining scenarios
                timed_out_filters.add(filter_name)

            except Exception:
                if use_timeout:
                    signal.alarm(0)
                progress = f"{_current_progress['steps_done']}/{_current_progress['total_steps']}"
                print(f"{name:<22} | {filter_name:<18} | {'ERROR':>9} | {'-':>7} | {progress:>8}")


if __name__ == "__main__":
    main()
