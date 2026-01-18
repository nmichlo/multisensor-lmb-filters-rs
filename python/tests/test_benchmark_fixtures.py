"""
Data-driven fixture tests - ALL configuration comes from fixture files.

Usage:
    uv run pytest benchmarks/test_matlab_fixtures.py -v

Fixtures are fully self-describing: model params, filter config, expected outputs.
"""

import json
from pathlib import Path

import numpy as np
import pytest
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
    MotionModel,
    SensorConfigMulti,
    SensorModel,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# Tolerances
MEAN_TOLERANCE = 5.0
TRACK_COUNT_TOLERANCE = 1


def load_fixtures() -> list[Path]:
    """Find all fixture files."""
    if not FIXTURES_DIR.exists():
        return []
    return sorted(FIXTURES_DIR.glob("*.json"))


def build_filter_from_fixture(fixture: dict, scenario: dict):
    """Build filter entirely from fixture configuration."""
    model = fixture["model"]
    filt_cfg = fixture["filter"]
    thresh = fixture["thresholds"]
    n_sensors = fixture["num_sensors"]

    # Motion model from fixture
    motion = MotionModel.constant_velocity_2d(
        model["dt"], model["process_noise_std"], model["survival_probability"]
    )

    # Sensor model from fixture
    bounds = model["bounds"]
    obs_vol = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    sensor = SensorModel.position_2d(
        model["measurement_noise_std"],
        model["detection_probability"],
        model["clutter_rate"],
        obs_vol,
    )

    # Birth model from fixture
    # State ordering: [x, y, vx, vy] (matches MATLAB)
    birth_cov = np.array(model["birth_covariance"], dtype=np.float64)
    birth_locs = [
        BirthLocation(i, np.array(loc, dtype=np.float64), np.diag(birth_cov))
        for i, loc in enumerate(model["birth_locations"])
    ]
    birth = BirthModel(birth_locs, lmb_existence=model["birth_existence"], lmbm_existence=0.001)

    # Thresholds from fixture - stored locally for inline passing
    # gm_merge can be null (JSON's null for MATLAB's inf) or "Inf" string
    gm_merge_val = thresh["gm_merge"]
    if gm_merge_val is None or gm_merge_val == "Inf":
        gm_merge_val = float("inf")
    existence_threshold = thresh["existence"]
    gm_weight_threshold = thresh["gm_weight"]
    max_gm_components = thresh["max_components"]

    # Associator from fixture
    assoc_cfg = filt_cfg["associator"]
    assoc_type = assoc_cfg["type"]
    assoc_params = assoc_cfg["params"]

    if assoc_type == "LBP":
        assoc = AssociatorConfig.lbp(assoc_params["max_iterations"], assoc_params["tolerance"])
    elif assoc_type == "Gibbs":
        assoc = AssociatorConfig.gibbs(assoc_params["num_samples"])
    elif assoc_type == "Murty":
        assoc = AssociatorConfig.murty(assoc_params["num_assignments"])
    else:
        raise ValueError(f"Unknown associator: {assoc_type}")

    # Filter type from fixture
    filter_type = filt_cfg["type"]

    # Common threshold kwargs for LMB filters
    lmb_threshold_kwargs = {
        "existence_threshold": existence_threshold,
        "gm_weight_threshold": gm_weight_threshold,
        "max_gm_components": max_gm_components,
        "gm_merge_threshold": gm_merge_val,
    }

    # Common kwargs for LMBM filters
    lmbm_kwargs = {
        "existence_threshold": existence_threshold,
        "max_hypotheses": 25,
        "hypothesis_weight_threshold": 1e-3,
    }

    if filter_type == "LMB":
        return FilterLmb(motion, sensor, birth, assoc, **lmb_threshold_kwargs), False
    elif filter_type == "LMBM":
        return FilterLmbm(motion, sensor, birth, assoc, **lmbm_kwargs), False
    elif filter_type == "AA-LMB":
        sensors = SensorConfigMulti([sensor] * n_sensors)
        return FilterAaLmb(motion, sensors, birth, assoc, **lmb_threshold_kwargs), True
    elif filter_type == "GA-LMB":
        sensors = SensorConfigMulti([sensor] * n_sensors)
        return FilterGaLmb(motion, sensors, birth, assoc, **lmb_threshold_kwargs), True
    elif filter_type == "PU-LMB":
        sensors = SensorConfigMulti([sensor] * n_sensors)
        return FilterPuLmb(motion, sensors, birth, assoc, **lmb_threshold_kwargs), True
    elif filter_type == "IC-LMB":
        sensors = SensorConfigMulti([sensor] * n_sensors)
        return FilterIcLmb(motion, sensors, birth, assoc, **lmb_threshold_kwargs), True
    elif filter_type == "MS-LMBM":
        sensors = SensorConfigMulti([sensor] * n_sensors)
        return FilterMultisensorLmbm(motion, sensors, birth, assoc, **lmbm_kwargs), True
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def run_filter(filt, scenario: dict, is_multi: bool, num_steps: int) -> list[dict]:
    """Run filter on scenario measurements."""
    results = []
    for t in range(num_steps):
        step = scenario["steps"][t]
        if is_multi:
            meas = [np.array(s) if s else np.empty((0, 2)) for s in step["sensor_readings"]]
        else:
            meas = (
                np.array(step["sensor_readings"][0])
                if step["sensor_readings"][0]
                else np.empty((0, 2))
            )

        result = filt.step(meas, t)
        results.append(
            {
                "step": t,
                "num_tracks": len(result.tracks),
                "means": [list(tr.mean) for tr in result.tracks],
            }
        )
    return results


def compare_results(rust_results: list, fixture: dict) -> list[str]:
    """Compare Rust vs MATLAB fixture."""
    errors = []
    num_steps = min(len(rust_results), fixture["num_steps"])

    for t in range(num_steps):
        rust = rust_results[t]
        matlab = fixture["steps"][t]

        # Track count
        rust_count = rust["num_tracks"]
        matlab_count = matlab["num_tracks"]
        if abs(rust_count - matlab_count) > TRACK_COUNT_TOLERANCE:
            errors.append(f"Step {t}: count Rust={rust_count} MATLAB={matlab_count}")

        # Means comparison
        if rust_count > 0 and matlab_count > 0:
            rust_means = np.array(rust["means"])
            matlab_means = np.array([tr["mean"] for tr in matlab["tracks"]])

            for r_mean in rust_means:
                if len(matlab_means) == 0:
                    break
                dists = np.linalg.norm(matlab_means - r_mean, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] > MEAN_TOLERANCE:
                    errors.append(
                        f"Step {t}: unmatched Rust track at [{r_mean[0]:.1f}, {r_mean[1]:.1f}]"
                    )
                matlab_means = np.delete(matlab_means, min_idx, axis=0)

    return errors


# Discover fixtures dynamically
def pytest_generate_tests(metafunc):
    if "fixture_path" in metafunc.fixturenames:
        fixtures = load_fixtures()
        metafunc.parametrize("fixture_path", fixtures, ids=[f.stem for f in fixtures])


def test_matlab_equivalence(fixture_path: Path):
    """Test Rust filter matches MATLAB fixture."""
    fixture = json.load(open(fixture_path))

    # Load scenario referenced by fixture
    scenario_path = SCENARIOS_DIR / fixture["scenario_file"]
    if not scenario_path.exists():
        pytest.skip(f"Scenario not found: {fixture['scenario_file']}")
    scenario = json.load(open(scenario_path))

    # Build filter from fixture config
    filt, is_multi = build_filter_from_fixture(fixture, scenario)

    # Run filter
    rust_results = run_filter(filt, scenario, is_multi, fixture["num_steps"])

    # Compare
    errors = compare_results(rust_results, fixture)
    if errors:
        pytest.fail(f"{fixture['filter']['name']}:\n" + "\n".join(errors[:10]))


if __name__ == "__main__":
    # Manual test run
    for fixture_path in load_fixtures():
        print(f"\n{fixture_path.stem}...")
        fixture = json.load(open(fixture_path))

        scenario_path = SCENARIOS_DIR / fixture["scenario_file"]
        if not scenario_path.exists():
            print("  SKIP: scenario not found")
            continue

        scenario = json.load(open(scenario_path))
        filt, is_multi = build_filter_from_fixture(fixture, scenario)
        rust_results = run_filter(filt, scenario, is_multi, fixture["num_steps"])
        errors = compare_results(rust_results, fixture)

        if errors:
            print(f"  FAIL: {len(errors)} errors")
            for e in errors[:3]:
                print(f"    {e}")
        else:
            print("  PASS")
