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

FIXTURE_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"
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
    return load_fixture("step_ss_lmb_seed42.json")


@pytest.fixture
def lmbm_fixture():
    """Load single-sensor LMBM step-by-step fixture."""
    return load_fixture("step_ss_lmbm_seed42.json")


@pytest.fixture
def multisensor_lmb_fixture():
    """Load multi-sensor LMB step-by-step fixture (IC-LMB)."""
    return load_fixture("step_ms_lmb_seed42.json")


@pytest.fixture
def multisensor_lmbm_fixture():
    """Load multi-sensor LMBM step-by-step fixture."""
    return load_fixture("step_ms_lmbm_seed42.json")


@pytest.fixture
def aa_lmb_fixture():
    """Load AA-LMB step-by-step fixture."""
    return load_fixture("step_ms_aa_lmb_seed42.json")


@pytest.fixture
def ga_lmb_fixture():
    """Load GA-LMB step-by-step fixture."""
    return load_fixture("step_ms_ga_lmb_seed42.json")


@pytest.fixture
def pu_lmb_fixture():
    """Load PU-LMB step-by-step fixture."""
    return load_fixture("step_ms_pu_lmb_seed42.json")


@pytest.fixture
def ic_lmb_fixture():
    """Load IC-LMB step-by-step fixture."""
    return load_fixture("step_ms_ic_lmb_seed42.json")


@pytest.fixture(params=["aa", "ga", "pu", "ic"])
def ms_lmb_variant_fixture(request):
    """Parametrized fixture loading step-by-step data for each multisensor LMB variant.

    This fixture runs each test 4 times, once for each multi-sensor LMB variant:
    - AA-LMB (Arithmetic Average)
    - GA-LMB (Geometric Average)
    - PU-LMB (Parallel Update)
    - IC-LMB (Iterated Corrector)
    """
    variant = request.param
    fixture = load_fixture(f"step_ms_{variant}_lmb_seed42.json")
    fixture["variant"] = variant  # Add variant name to fixture for test identification
    return fixture


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

    # Handle shape compatibility: MATLAB may save (N,1) matrices as 1D arrays
    # When comparing, squeeze singleton dimensions to allow (N,1) vs (N,) comparisons
    if expected_arr.shape != actual.shape:
        # Try squeezing both arrays to handle (N,1) vs (N,) case
        expected_squeezed = np.squeeze(expected_arr)
        actual_squeezed = np.squeeze(actual)
        if expected_squeezed.shape == actual_squeezed.shape:
            expected_arr = expected_squeezed
            actual = actual_squeezed
        else:
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


# =============================================================================
# Track Data Loading and Comparison (for fixture validation)
# =============================================================================


def make_track_data(obj: dict):
    """Convert fixture object to _TrackData for setting filter state.

    Args:
        obj: Fixture object dict with keys: label, r, mu, Sigma, w

    Returns:
        _TrackData instance that can be passed to filter.set_tracks()
    """
    from multisensor_lmb_filters_rs import _TrackData

    # Parse label
    label = obj.get("label", [0, 0])
    if isinstance(label, list):
        label_tuple = (label[0], label[1])
    else:
        label_tuple = (0, label)

    # Parse weights (can be scalar or list)
    weights = obj["w"]
    if not isinstance(weights, list):
        weights = [weights]

    return _TrackData(
        label=label_tuple,
        existence=obj["r"],
        means=obj["mu"],
        covariances=obj["Sigma"],
        weights=weights,
    )


def load_prior_tracks(fixture: dict) -> list:
    """Load prior tracks from fixture for filter initialization.

    Returns list of _TrackData objects from fixture['step1_prediction']['input']['prior_objects'].
    """
    prior_objects = fixture["step1_prediction"]["input"]["prior_objects"]
    return [make_track_data(obj) for obj in prior_objects]


def make_birth_model_empty():
    """Create empty birth model (no birth locations)."""
    from multisensor_lmb_filters_rs import BirthModel

    return BirthModel(locations=[], lmb_existence=0.0, lmbm_existence=0.0)


def make_birth_model_from_fixture(fixture: dict):
    """Create BirthModel by extracting birth tracks from fixture predicted output.

    MATLAB fixtures include birth tracks in the predicted output. This function
    extracts those tracks and creates a matching birth model.

    Handles both LMB fixtures (predicted_objects list) and LMBM fixtures
    (predicted_hypothesis with parallel arrays).
    """
    from multisensor_lmb_filters_rs import BirthLocation, BirthModel

    timestep = fixture["timestep"]
    prediction_output = fixture["step1_prediction"]["output"]

    # Check if this is LMB (predicted_objects) or LMBM (predicted_hypothesis) format
    if "predicted_objects" in prediction_output:
        # LMB format: list of track objects
        predicted_objects = prediction_output["predicted_objects"]

        # Handle different label formats:
        # - Old format: obj["label"] = [birthTime, birthLocation]
        # - New format: obj["birthTime"], obj["birthLocation"]
        birth_tracks = []
        for obj in predicted_objects:
            if "label" in obj:
                birth_time = obj["label"][0]
            else:
                birth_time = obj.get("birthTime", 0)

            if birth_time == timestep:
                birth_tracks.append(obj)

        if not birth_tracks:
            return make_birth_model_empty()

        lmb_existence = birth_tracks[0]["r"]
        locations = []
        for obj in birth_tracks:
            # Handle different label formats
            if "label" in obj:
                birth_loc = obj["label"][1]
            else:
                birth_loc = obj.get("birthLocation", 0)

            locations.append(
                BirthLocation(
                    label=birth_loc,
                    mean=np.array(obj["mu"][0], dtype=np.float64),
                    covariance=np.array(obj["Sigma"][0], dtype=np.float64),
                )
            )
    else:
        # LMBM format: predicted_hypothesis with parallel arrays
        hyp = prediction_output["predicted_hypothesis"]
        birth_times = hyp["birthTime"]
        birth_locs = hyp["birthLocation"]
        r_values = hyp["r"]
        mu_values = hyp["mu"]
        sigma_values = hyp["Sigma"]

        # Find indices where birthTime == timestep (these are birth tracks)
        birth_indices = [i for i, bt in enumerate(birth_times) if bt == timestep]

        if not birth_indices:
            return make_birth_model_empty()

        lmb_existence = r_values[birth_indices[0]]
        locations = []
        for i in birth_indices:
            locations.append(
                BirthLocation(
                    label=birth_locs[i],
                    mean=np.array(mu_values[i], dtype=np.float64),
                    covariance=np.array(sigma_values[i], dtype=np.float64),
                )
            )

    # When extracting birth existence from fixture, use the same value for both
    # LMB and LMBM since the fixture already contains the correct filter-specific value.
    # MATLAB uses model.rB=0.03 for LMB and model.rBLmbm=0.045 for LMBM.
    return BirthModel(
        locations=locations,
        lmb_existence=lmb_existence,
        lmbm_existence=lmb_existence,  # Same value - fixture has filter-specific existence
    )


def compare_tracks(name: str, expected: list, actual: list, tol: float = TOLERANCE):
    """Compare full track data (GM components, existence, labels).

    Args:
        name: Name for error messages
        expected: List of fixture dicts with keys: label, r, mu, Sigma, w
        actual: List of _TrackData objects from filter
        tol: Numerical tolerance

    Raises:
        AssertionError: On first mismatch with detailed error message
    """
    if len(expected) != len(actual):
        raise AssertionError(
            f"{name}: track count mismatch - expected {len(expected)}, got {len(actual)}"
        )

    for i, (exp, act) in enumerate(zip(expected, actual)):
        prefix = f"{name}[{i}]"

        # Compare label
        exp_label = exp.get("label", [0, 0])
        if isinstance(exp_label, list):
            exp_label_tuple = (exp_label[0], exp_label[1])
        else:
            exp_label_tuple = (0, exp_label)

        if act.label != exp_label_tuple:
            raise AssertionError(f"{prefix}.label: expected {exp_label}, got {act.label}")

        # Compare existence (r)
        compare_scalar(f"{prefix}.existence", exp["r"], act.existence, tol)

        # Compare GM components
        exp_w = exp["w"] if isinstance(exp["w"], list) else [exp["w"]]
        compare_array(f"{prefix}.w", exp_w, act.w, tol)
        compare_array(f"{prefix}.mu", exp["mu"], act.mu, tol)
        compare_array(f"{prefix}.sigma", exp["Sigma"], act.sigma, tol)


def compare_association_matrices(name: str, expected: dict, actual, tol: float = TOLERANCE):
    """Compare association matrices from fixture against actual output.

    Args:
        name: Name for error messages
        expected: Fixture dict with keys: C, L, R, P, eta, posteriorParameters
        actual: _AssociationMatrices object from filter
        tol: Numerical tolerance
    """
    # Compare cost matrix
    if "C" in expected:
        compare_array(f"{name}.cost", expected["C"], actual.cost, tol)

    # Compare likelihood matrix
    if "L" in expected:
        compare_array(f"{name}.likelihood", expected["L"], actual.likelihood, tol)

    # Compare sampling probabilities
    if "P" in expected:
        compare_array(f"{name}.sampling_prob", expected["P"], actual.sampling_prob, tol)

    # Compare eta normalization factors
    if "eta" in expected:
        compare_array(f"{name}.eta", expected["eta"], actual.eta, tol)

    # Compare R matrix (miss probabilities)
    if "R" in expected:
        compare_array(f"{name}.miss_prob", expected["R"], actual.miss_prob, tol)

    # Compare posteriorParameters
    if "posteriorParameters" in expected:
        compare_posterior_parameters(
            f"{name}.posteriorParameters",
            expected["posteriorParameters"],
            actual.posterior_parameters,
            tol,
        )


def compare_posterior_parameters(name: str, expected: list, actual: list, tol: float = TOLERANCE):
    """Compare posteriorParameters from fixture against actual output.

    Args:
        name: Name for error messages
        expected: List of fixture dicts, each with keys: w, mu, Sigma
        actual: List of _PosteriorParameters objects from filter
        tol: Numerical tolerance

    MATLAB Fixture Format:
        posteriorParameters[i].w → shape (num_meas + 1, num_comp)
            Row 0 is miss hypothesis (equals prior weights)
            Rows 1+ are measurements
            Each row sums to 1.0 (likelihood-normalized component weights)

        posteriorParameters[i].mu → shape (num_meas * num_comp, state_dim)
            Flattened posterior means

        posteriorParameters[i].Sigma → shape (num_meas * num_comp, state_dim, state_dim)
            Flattened posterior covariances
    """
    if len(expected) != len(actual):
        raise AssertionError(
            f"{name}: count mismatch - expected {len(expected)}, got {len(actual)}"
        )

    for i, (exp, act) in enumerate(zip(expected, actual)):
        prefix = f"{name}[{i}]"

        # Compare component weights w
        if "w" in exp:
            compare_array(f"{prefix}.w", exp["w"], act.w, tol)

        # Compare posterior means mu
        if "mu" in exp:
            compare_array(f"{prefix}.mu", exp["mu"], act.mu, tol)

        # Compare posterior covariances Sigma
        if "Sigma" in exp:
            compare_array(f"{prefix}.Sigma", exp["Sigma"], act.sigma, tol)


def compare_lmbm_hypothesis(name: str, expected: dict, actual, tol: float = TOLERANCE):
    """Compare an LMBM hypothesis against expected fixture data.

    Args:
        name: Name for error messages
        expected: Fixture dict with keys: w, r, mu, Sigma, birthTime, birthLocation
        actual: _LmbmHypothesis object from filter
        tol: Numerical tolerance
    """
    # Compare weight (w is log-weight in fixture for step4, linear for step5)
    # Check if it's a log-weight (negative) or linear weight (positive < 1)
    exp_w = expected["w"]
    if exp_w < 0:
        # Log-weight comparison
        compare_scalar(f"{name}.log_weight", exp_w, actual.log_weight, tol)
    else:
        # Linear weight comparison
        compare_scalar(f"{name}.weight", exp_w, actual.weight, tol)

    # Compare existence probabilities (r) - convert list to numpy array
    compare_array(f"{name}.r", expected["r"], np.array(actual.r), tol)

    # Compare means (mu) - flatten to 2D array
    if "mu" in expected:
        compare_array(f"{name}.mu", expected["mu"], np.array(actual.mu), tol)

    # Compare covariances (Sigma)
    if "Sigma" in expected:
        compare_array(f"{name}.Sigma", expected["Sigma"], np.array(actual.sigma), tol)

    # Compare birth times
    if "birthTime" in expected:
        expected_bt = expected["birthTime"]
        actual_bt = list(actual.birth_time)
        if expected_bt != actual_bt:
            raise AssertionError(f"{name}.birthTime: expected {expected_bt}, got {actual_bt}")

    # Compare birth locations
    if "birthLocation" in expected:
        expected_bl = expected["birthLocation"]
        actual_bl = list(actual.birth_location)
        if expected_bl != actual_bl:
            raise AssertionError(f"{name}.birthLocation: expected {expected_bl}, got {actual_bl}")


def compare_lmbm_hypotheses(name: str, expected: list, actual: list, tol: float = TOLERANCE):
    """Compare a list of LMBM hypotheses against expected fixture data.

    Args:
        name: Name for error messages
        expected: List of fixture hypothesis dicts
        actual: List of _LmbmHypothesis objects from filter
        tol: Numerical tolerance
    """
    if len(expected) != len(actual):
        raise AssertionError(
            f"{name}: count mismatch - expected {len(expected)}, got {len(actual)}"
        )

    for i, (exp, act) in enumerate(zip(expected, actual)):
        compare_lmbm_hypothesis(f"{name}[{i}]", exp, act, tol)


# =============================================================================
# Multi-sensor LMB Variant Helpers
# =============================================================================


def get_multisensor_filter_class(variant: str):
    """Get the filter class for a multi-sensor LMB variant.

    Args:
        variant: One of "aa", "ga", "pu", "ic"

    Returns:
        Filter class (FilterAaLmb, FilterGaLmb, FilterPuLmb, FilterIcLmb)
    """
    import multisensor_lmb_filters_rs as lmb

    variant_map = {
        "aa": lmb.FilterAaLmb,
        "ga": lmb.FilterGaLmb,
        "pu": lmb.FilterPuLmb,
        "ic": lmb.FilterIcLmb,
    }
    if variant not in variant_map:
        raise ValueError(f"Unknown variant: {variant}. Expected one of {list(variant_map.keys())}")
    return variant_map[variant]


def load_prior_tracks_from_variant_fixture(fixture: dict) -> list:
    """Load prior tracks from a variant fixture.

    The variant fixtures have a slightly different structure than the
    original multisensor_lmb fixture:
    - step1_prediction.input.prior_objects contains the prior track data

    Returns:
        List of _TrackData objects for filter initialization
    """
    prior_objects = fixture["step1_prediction"]["input"]["prior_objects"]
    return [make_track_data_from_variant_fixture(obj) for obj in prior_objects]


def make_track_data_from_variant_fixture(obj: dict):
    """Convert a variant fixture object to _TrackData.

    Variant fixtures store label as (birthTime, birthLocation) tuple directly.
    """
    from multisensor_lmb_filters_rs import _TrackData

    # Parse label - variant fixtures use birthTime and birthLocation fields
    birth_time = obj.get("birthTime", obj.get("label", [0, 0])[0] if "label" in obj else 0)
    birth_location = obj.get("birthLocation", obj.get("label", [0, 0])[1] if "label" in obj else 0)
    label_tuple = (birth_time, birth_location)

    # Parse weights (can be scalar or list)
    weights = obj["w"]
    if not isinstance(weights, list):
        weights = [weights]

    return _TrackData(
        label=label_tuple,
        existence=obj["r"],
        means=obj["mu"],
        covariances=obj["Sigma"],
        weights=weights,
    )


def compare_fused_tracks(name: str, expected: list, actual: list, tol: float = TOLERANCE):
    """Compare fused tracks from fixture against actual filter output.

    Args:
        name: Name for error messages
        expected: List of fixture dicts with keys: r, w, mu, Sigma, birthTime, birthLocation
        actual: List of _TrackData objects from filter
        tol: Numerical tolerance
    """
    if len(expected) != len(actual):
        raise AssertionError(
            f"{name}: track count mismatch - expected {len(expected)}, got {len(actual)}"
        )

    for i, (exp, act) in enumerate(zip(expected, actual)):
        prefix = f"{name}[{i}]"

        # Compare label (birthTime, birthLocation)
        exp_birth_time = exp.get("birthTime", 0)
        exp_birth_loc = exp.get("birthLocation", 0)
        exp_label = (exp_birth_time, exp_birth_loc)

        if act.label != exp_label:
            raise AssertionError(f"{prefix}.label: expected {exp_label}, got {act.label}")

        # Compare existence (r)
        compare_scalar(f"{prefix}.existence", exp["r"], act.existence, tol)

        # Compare GM components
        exp_w = exp["w"] if isinstance(exp["w"], list) else [exp["w"]]
        compare_array(f"{prefix}.w", exp_w, act.w, tol)
        compare_array(f"{prefix}.mu", exp["mu"], act.mu, tol)
        compare_array(f"{prefix}.sigma", exp["Sigma"], act.sigma, tol)
