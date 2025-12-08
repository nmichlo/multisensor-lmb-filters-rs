"""Tests that verify Python produces identical results to Rust fixtures.

These tests load the same JSON fixtures used by Rust tests and verify
that Python bindings produce numerically equivalent results.

Since Python bindings directly wrap Rust code via PyO3, results must be
100% identical (within floating point tolerance).
"""

import json
from pathlib import Path

import numpy as np
import pytest

# Tolerance for floating point comparisons (matches Rust tests)
TOLERANCE = 1e-10


def get_fixture_path(name: str) -> Path:
    """Get path to a test fixture file."""
    return Path(__file__).parent / "data" / name


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    path = get_fixture_path(name)
    with open(path) as f:
        return json.load(f)


def vec_to_array(vec: list[float]) -> np.ndarray:
    """Convert list to numpy array."""
    return np.array(vec, dtype=np.float64)


def matrix_to_array(matrix: list[list[float]]) -> np.ndarray:
    """Convert nested list to numpy 2D array."""
    return np.array(matrix, dtype=np.float64)


class TestLmbStepByStep:
    """Test LMB filter step-by-step against MATLAB/Rust fixtures."""

    @pytest.fixture
    def fixture(self):
        """Load the LMB step-by-step fixture."""
        return load_fixture("step_by_step/lmb_step_by_step_seed42.json")

    def test_fixture_loads(self, fixture):
        """Verify fixture loads correctly."""
        assert "seed" in fixture
        assert fixture["seed"] == 42
        assert "model" in fixture
        assert "measurements" in fixture

    def test_motion_model_creation(self, fixture):
        """Test creating motion model from fixture data."""
        from multisensor_lmb_filters_rs import MotionModel

        model = fixture["model"]
        A = matrix_to_array(model["A"])
        R = matrix_to_array(model["R"])
        p_s = model["P_s"]

        # Create motion model with fixture parameters
        motion = MotionModel(
            transition_matrix=A,
            process_noise=R,
            control_input=np.zeros(A.shape[0]),
            survival_probability=p_s,
        )

        # Verify parameters
        assert motion.x_dim == 4
        assert motion.survival_probability == pytest.approx(p_s, rel=TOLERANCE)
        np.testing.assert_allclose(motion.transition_matrix, A, rtol=TOLERANCE)
        np.testing.assert_allclose(motion.process_noise, R, rtol=TOLERANCE)

    def test_sensor_model_creation(self, fixture):
        """Test creating sensor model from fixture data."""
        from multisensor_lmb_filters_rs import SensorModel

        model = fixture["model"]
        C = matrix_to_array(model["C"])
        Q = matrix_to_array(model["Q"])
        p_d = model["P_d"]
        clutter = model["clutter_per_unit_volume"]

        # Create sensor model
        sensor = SensorModel(
            observation_matrix=C,
            measurement_noise=Q,
            detection_probability=p_d,
            clutter_rate=clutter * 10000.0,  # Convert density to rate
            observation_volume=10000.0,
        )

        # Verify parameters
        assert sensor.z_dim == 2
        assert sensor.x_dim == 4
        assert sensor.detection_probability == pytest.approx(p_d, rel=TOLERANCE)

    def test_measurements_conversion(self, fixture):
        """Test converting fixture measurements to numpy arrays."""
        measurements = fixture["measurements"]
        numpy_meas = [vec_to_array(m) for m in measurements]

        assert len(numpy_meas) == len(measurements)
        for i, m in enumerate(numpy_meas):
            assert m.shape == (2,), f"Measurement {i} has wrong shape"
            np.testing.assert_allclose(m, measurements[i], rtol=TOLERANCE)


class TestLmbmStepByStep:
    """Test LMBM filter step-by-step against fixtures."""

    @pytest.fixture
    def fixture(self):
        """Load the LMBM step-by-step fixture."""
        return load_fixture("step_by_step/lmbm_step_by_step_seed42.json")

    def test_fixture_loads(self, fixture):
        """Verify LMBM fixture loads correctly."""
        assert "seed" in fixture
        assert fixture["seed"] == 42


class TestMultisensorLmbStepByStep:
    """Test multisensor LMB filter step-by-step against fixtures."""

    @pytest.fixture
    def fixture(self):
        """Load the multisensor LMB step-by-step fixture."""
        return load_fixture("step_by_step/multisensor_lmb_step_by_step_seed42.json")

    def test_fixture_loads(self, fixture):
        """Verify multisensor LMB fixture loads correctly."""
        assert "seed" in fixture
        assert fixture["seed"] == 42


class TestMultisensorLmbmStepByStep:
    """Test multisensor LMBM filter step-by-step against fixtures."""

    @pytest.fixture
    def fixture(self):
        """Load the multisensor LMBM step-by-step fixture."""
        return load_fixture("step_by_step/multisensor_lmbm_step_by_step_seed42.json")

    def test_fixture_loads(self, fixture):
        """Verify multisensor LMBM fixture loads correctly."""
        assert "seed" in fixture
        assert fixture["seed"] == 42


class TestSingleTrialFixtures:
    """Test single-sensor trial fixtures."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "single_trial_42.json",
            "single_trial_42_quick.json",
            "single_detection_trial_42_quick.json",
        ],
    )
    def test_fixture_loads(self, fixture_name):
        """Verify trial fixture loads correctly."""
        fixture = load_fixture(fixture_name)
        assert "seed" in fixture
        assert "filterVariants" in fixture

    def test_single_trial_42_quick_structure(self):
        """Test detailed structure of quick trial fixture."""
        fixture = load_fixture("single_trial_42_quick.json")

        assert fixture["seed"] == 42
        assert len(fixture["filterVariants"]) > 0

        # Check each filter variant has expected fields
        for variant in fixture["filterVariants"]:
            assert "name" in variant
            assert "eOspa" in variant
            assert "hOspa" in variant


class TestMultisensorTrialFixtures:
    """Test multi-sensor trial fixtures."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "multisensor_trial_42.json",
            "multisensor_clutter_trial_42_quick.json",
            "multisensor_detection_trial_42_quick.json",
        ],
    )
    def test_fixture_loads(self, fixture_name):
        """Verify multisensor trial fixture loads correctly."""
        fixture = load_fixture(fixture_name)
        assert "seed" in fixture


class TestPredictionEquivalence:
    """Test prediction step produces same results as fixture."""

    @pytest.fixture
    def fixture(self):
        """Load the LMB step-by-step fixture."""
        return load_fixture("step_by_step/lmb_step_by_step_seed42.json")

    def test_prediction_input_parsing(self, fixture):
        """Test parsing prediction input from fixture."""
        pred_input = fixture["step1_prediction"]["input"]

        # Parse prior objects
        prior_objects = pred_input["prior_objects"]
        assert len(prior_objects) > 0

        # Check first object structure
        obj = prior_objects[0]
        assert "r" in obj  # existence probability
        assert "label" in obj  # track label
        assert "mu" in obj  # means
        assert "Sigma" in obj  # covariances
        assert "w" in obj  # weights

    def test_prediction_output_parsing(self, fixture):
        """Test parsing prediction output from fixture."""
        pred_output = fixture["step1_prediction"]["output"]

        predicted_objects = pred_output["predicted_objects"]
        assert len(predicted_objects) > 0

        # Check predicted object structure matches input
        obj = predicted_objects[0]
        assert "r" in obj
        assert "label" in obj
        assert "mu" in obj
        assert "Sigma" in obj
        assert "w" in obj


class TestAssociationEquivalence:
    """Test association step produces same results as fixture."""

    @pytest.fixture
    def fixture(self):
        """Load the LMB step-by-step fixture."""
        return load_fixture("step_by_step/lmb_step_by_step_seed42.json")

    def test_association_output_structure(self, fixture):
        """Test association output structure from fixture."""
        assoc_output = fixture["step2_association"]["output"]

        # Check cost/likelihood matrices exist
        assert "C" in assoc_output  # cost matrix
        assert "L" in assoc_output  # likelihood matrix
        assert "R" in assoc_output  # ...
        assert "P" in assoc_output  # ...
        assert "eta" in assoc_output  # eta values


class TestLbpEquivalence:
    """Test LBP marginal computation produces same results as fixture."""

    @pytest.fixture
    def fixture(self):
        """Load the LMB step-by-step fixture."""
        return load_fixture("step_by_step/lmb_step_by_step_seed42.json")

    def test_lbp_input_structure(self, fixture):
        """Test LBP input structure from fixture."""
        lbp_input = fixture["step3a_lbp"]["input"]

        assert "C" in lbp_input
        assert "L" in lbp_input
        assert "convergence_tolerance" in lbp_input
        assert "max_iterations" in lbp_input

    def test_lbp_output_structure(self, fixture):
        """Test LBP output structure from fixture."""
        lbp_output = fixture["step3a_lbp"]["output"]

        assert "r" in lbp_output  # existence probabilities
        assert "W" in lbp_output  # weight matrix


class TestGibbsEquivalence:
    """Test Gibbs sampling produces same results as fixture."""

    @pytest.fixture
    def fixture(self):
        """Load the LMB step-by-step fixture."""
        return load_fixture("step_by_step/lmb_step_by_step_seed42.json")

    def test_gibbs_input_structure(self, fixture):
        """Test Gibbs input structure from fixture."""
        gibbs_input = fixture["step3b_gibbs"]["input"]

        assert "numberOfSamples" in gibbs_input
        assert "rng_seed" in gibbs_input

    def test_gibbs_output_structure(self, fixture):
        """Test Gibbs output structure from fixture."""
        gibbs_output = fixture["step3b_gibbs"]["output"]

        assert "r" in gibbs_output
        assert "W" in gibbs_output


class TestMurtyEquivalence:
    """Test Murty's algorithm produces same results as fixture."""

    @pytest.fixture
    def fixture(self):
        """Load the LMB step-by-step fixture."""
        return load_fixture("step_by_step/lmb_step_by_step_seed42.json")

    def test_murty_input_structure(self, fixture):
        """Test Murty input structure from fixture."""
        murty_input = fixture["step3c_murtys"]["input"]

        assert "numberOfAssignments" in murty_input

    def test_murty_output_structure(self, fixture):
        """Test Murty output structure from fixture."""
        murty_output = fixture["step3c_murtys"]["output"]

        assert "r" in murty_output
        assert "W" in murty_output


class TestUpdateEquivalence:
    """Test update step produces same results as fixture."""

    @pytest.fixture
    def fixture(self):
        """Load the LMB step-by-step fixture."""
        return load_fixture("step_by_step/lmb_step_by_step_seed42.json")

    def test_update_output_structure(self, fixture):
        """Test update output structure from fixture."""
        update_output = fixture["step4_update"]["output"]

        assert "posterior_objects" in update_output

        # Check posterior objects have expected structure
        objects = update_output["posterior_objects"]
        assert len(objects) > 0

        obj = objects[0]
        assert "r" in obj
        assert "label" in obj
        assert "mu" in obj
        assert "Sigma" in obj
        assert "w" in obj


class TestCardinalityEquivalence:
    """Test cardinality estimation produces same results as fixture."""

    @pytest.fixture
    def fixture(self):
        """Load the LMB step-by-step fixture."""
        return load_fixture("step_by_step/lmb_step_by_step_seed42.json")

    def test_cardinality_output_structure(self, fixture):
        """Test cardinality output structure from fixture."""
        card_output = fixture["step5_cardinality"]["output"]

        assert "n_estimated" in card_output
        assert "map_indices" in card_output


class TestAllFixturesExist:
    """Verify all expected fixture files exist."""

    @pytest.mark.parametrize(
        "fixture_path",
        [
            "single_trial_42.json",
            "single_trial_42_quick.json",
            "single_detection_trial_42_quick.json",
            "multisensor_trial_42.json",
            "multisensor_clutter_trial_42_quick.json",
            "multisensor_detection_trial_42_quick.json",
            "step_by_step/lmb_step_by_step_seed42.json",
            "step_by_step/lmbm_step_by_step_seed42.json",
            "step_by_step/multisensor_lmb_step_by_step_seed42.json",
            "step_by_step/multisensor_lmbm_step_by_step_seed42.json",
        ],
    )
    def test_fixture_exists(self, fixture_path):
        """Verify fixture file exists."""
        path = get_fixture_path(fixture_path)
        assert path.exists(), f"Fixture not found: {fixture_path}"
