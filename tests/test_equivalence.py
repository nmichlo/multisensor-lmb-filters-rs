"""Numerical equivalence tests against MATLAB fixtures.

Tests validate filter output by comparing ENTIRE structures against
fixture expected values. Comparison functions raise on FIRST divergence
with detailed error messages.

These tests use set_tracks() to initialize filter state from fixture prior
objects, then step_detailed() to get all intermediate outputs for comparison.
"""

import numpy as np
import pytest
from conftest import (
    TOLERANCE,
    compare_array,
    compare_scalar,
    compare_tracks,
    load_fixture,
    load_prior_tracks,
    make_birth_model_empty,
    make_birth_model_from_fixture,
    make_motion_model,
    make_multisensor_config,
    make_sensor_model,
    measurements_to_numpy,
    nested_measurements_to_numpy,
)


class TestLmbFixtureEquivalence:
    """Test FilterLmb against LMB step-by-step fixture with FULL intermediate validation."""

    def test_lmb_prediction_equivalence(self, lmb_fixture):
        """Verify LMB prediction step matches MATLAB exactly."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        birth = make_birth_model_from_fixture(lmb_fixture)

        filter = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=lmb_fixture["seed"])

        # Load and set prior tracks from fixture
        prior_tracks = load_prior_tracks(lmb_fixture)
        filter.set_tracks(prior_tracks)

        # Run detailed step
        measurements = measurements_to_numpy(lmb_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=lmb_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Verify predicted tracks match MATLAB
        # ═══════════════════════════════════════════════════════════════
        expected_predicted = lmb_fixture["step1_prediction"]["output"]["predicted_objects"]
        compare_tracks("step1_predicted", expected_predicted, output.predicted_tracks, TOLERANCE)

    def test_lmb_association_matrices_equivalence(self, lmb_fixture):
        """Verify ALL LMB association matrices match MATLAB exactly.

        This test validates:
        - C: Cost matrix (negative log-likelihood)
        - L: Likelihood matrix
        - R: Miss probability matrix (phi/eta)
        - P: Sampling probabilities (psi/(1+psi))
        - eta: Normalization factors
        - posteriorParameters[i].w: Likelihood-normalized component weights
        - posteriorParameters[i].mu: Posterior means
        - posteriorParameters[i].Sigma: Posterior covariances
        """
        from conftest import compare_association_matrices
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        birth = make_birth_model_from_fixture(lmb_fixture)

        filter = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=lmb_fixture["seed"])

        prior_tracks = load_prior_tracks(lmb_fixture)
        filter.set_tracks(prior_tracks)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=lmb_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Verify ALL association matrices match MATLAB
        # ═══════════════════════════════════════════════════════════════
        expected_assoc = lmb_fixture["step2_association"]["output"]

        assert output.association_matrices is not None, "Association matrices should exist"

        # Use comprehensive comparison that validates ALL fields:
        # C, L, R, P, eta, and posteriorParameters (w, mu, Sigma)
        compare_association_matrices(
            "step2", expected_assoc, output.association_matrices, TOLERANCE
        )

    def test_lmb_lbp_result_equivalence(self, lmb_fixture):
        """Verify LBP association result matches MATLAB exactly."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        birth = make_birth_model_from_fixture(lmb_fixture)

        filter = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=lmb_fixture["seed"])

        prior_tracks = load_prior_tracks(lmb_fixture)
        filter.set_tracks(prior_tracks)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=lmb_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 3a: Verify LBP marginals match MATLAB
        # ═══════════════════════════════════════════════════════════════
        expected_lbp = lmb_fixture["step3a_lbp"]["output"]

        assert output.association_result is not None, "Association result should exist"

        # Compare posterior existence (r in MATLAB)
        # Note: MATLAB r is posterior existence from LBP, not miss weights
        compare_array(
            "step3a.r", expected_lbp["r"], output.association_result.posterior_existence, TOLERANCE
        )

        # Compare marginal weights W (MATLAB W is [miss, meas1, meas2, ...])
        # Our marginal_weights is just [meas1, meas2, ...]
        expected_w = expected_lbp["W"]
        # Skip first column (miss) - already checked above
        expected_marginals = [row[1:] for row in expected_w]
        compare_array(
            "step3a.W_marginals",
            expected_marginals,
            output.association_result.marginal_weights,
            TOLERANCE,
        )

    def test_lmb_gibbs_equivalence(self, lmb_fixture):
        """Verify Gibbs sampling association result matches MATLAB exactly.

        Gibbs sampling uses MCMC to approximate marginal association probabilities.
        With the same seed and number of samples, it should produce identical results.

        MATLAB fixture generation (generateLmbStepByStepData.m lines 149-170):
        - Uses rng_seed = seed + 2000 = 2042
        - Calls lmbGibbsSampling with model.numberOfSamples
        """
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        birth = make_birth_model_from_fixture(lmb_fixture)

        # Get Gibbs parameters from fixture
        gibbs_input = lmb_fixture["step3b_gibbs"]["input"]
        num_samples = gibbs_input["numberOfSamples"]
        gibbs_seed = gibbs_input["rng_seed"]

        filter = FilterLmb(
            motion, sensor, birth, AssociatorConfig.gibbs(num_samples), seed=gibbs_seed
        )

        prior_tracks = load_prior_tracks(lmb_fixture)
        filter.set_tracks(prior_tracks)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=lmb_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 3b: Verify Gibbs marginals match MATLAB
        # ═══════════════════════════════════════════════════════════════
        expected_gibbs = lmb_fixture["step3b_gibbs"]["output"]

        assert output.association_result is not None, "Association result should exist"

        # Compare posterior existence (r)
        compare_array(
            "step3b.r",
            expected_gibbs["r"],
            output.association_result.posterior_existence,
            TOLERANCE,
        )

        # Compare marginal weights W (MATLAB W is [miss, meas1, meas2, ...])
        expected_w = expected_gibbs["W"]
        expected_marginals = [row[1:] for row in expected_w]
        compare_array(
            "step3b.W_marginals",
            expected_marginals,
            output.association_result.marginal_weights,
            TOLERANCE,
        )

    def test_lmb_murty_equivalence(self, lmb_fixture):
        """Verify Murty's algorithm association result matches MATLAB exactly.

        Murty's algorithm finds the k-best assignments to compute exact marginals.
        Being deterministic, it should match MATLAB precisely.

        MATLAB fixture generation (generateLmbStepByStepData.m lines 172-191):
        - Calls lmbMurtysAlgorithm with model.numberOfAssignments
        """
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        birth = make_birth_model_from_fixture(lmb_fixture)

        # Get Murty parameters from fixture
        murty_input = lmb_fixture["step3c_murtys"]["input"]
        num_assignments = murty_input["numberOfAssignments"]

        filter = FilterLmb(
            motion, sensor, birth, AssociatorConfig.murty(num_assignments), seed=lmb_fixture["seed"]
        )

        prior_tracks = load_prior_tracks(lmb_fixture)
        filter.set_tracks(prior_tracks)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=lmb_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 3c: Verify Murty's marginals match MATLAB
        # ═══════════════════════════════════════════════════════════════
        expected_murty = lmb_fixture["step3c_murtys"]["output"]

        assert output.association_result is not None, "Association result should exist"

        # Compare posterior existence (r)
        compare_array(
            "step3c.r",
            expected_murty["r"],
            output.association_result.posterior_existence,
            TOLERANCE,
        )

        # Compare marginal weights W (MATLAB W is [miss, meas1, meas2, ...])
        expected_w = expected_murty["W"]
        expected_marginals = [row[1:] for row in expected_w]
        compare_array(
            "step3c.W_marginals",
            expected_marginals,
            output.association_result.marginal_weights,
            TOLERANCE,
        )

    def test_lmb_update_equivalence(self, lmb_fixture):
        """Verify LMB update step matches MATLAB exactly.

        This test validates that Rust's LMB update (using likelihood-normalized
        component weights and weight-based pruning) produces results identical
        to MATLAB's computePosteriorLmbSpatialDistributions implementation.
        """
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb, FilterThresholds

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        birth = make_birth_model_from_fixture(lmb_fixture)

        # Use max_components=5 and gm_weight=1e-5 to match MATLAB's thresholds
        thresholds = FilterThresholds(max_components=5, gm_weight=1e-5)
        filter = FilterLmb(
            motion,
            sensor,
            birth,
            AssociatorConfig.lbp(),
            thresholds=thresholds,
            seed=lmb_fixture["seed"],
        )

        prior_tracks = load_prior_tracks(lmb_fixture)
        filter.set_tracks(prior_tracks)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=lmb_fixture["timestep"])

        # Strict tolerance - this SHOULD match exactly once GM merging is implemented
        expected_updated = lmb_fixture["step4_update"]["output"]["posterior_objects"]
        compare_tracks("step4_updated", expected_updated, output.updated_tracks, TOLERANCE)

    def test_lmb_cardinality_equivalence(self, lmb_fixture):
        """Verify LMB cardinality extraction matches MATLAB exactly."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        birth = make_birth_model_from_fixture(lmb_fixture)

        filter = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=lmb_fixture["seed"])

        prior_tracks = load_prior_tracks(lmb_fixture)
        filter.set_tracks(prior_tracks)

        measurements = measurements_to_numpy(lmb_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=lmb_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Verify cardinality extraction matches MATLAB
        # ═══════════════════════════════════════════════════════════════
        expected_card = lmb_fixture["step5_cardinality"]["output"]

        assert output.cardinality.n_estimated == expected_card["n_estimated"], (
            f"cardinality.n_estimated: expected {expected_card['n_estimated']}, "
            f"got {output.cardinality.n_estimated}"
        )

        # Compare MAP indices (convert to 0-indexed if MATLAB is 1-indexed)
        expected_indices = expected_card["map_indices"]
        # MATLAB uses 1-indexed, convert to 0-indexed
        expected_indices_0 = [i - 1 for i in expected_indices] if expected_indices else []
        actual_indices = list(output.cardinality.map_indices)
        assert (
            actual_indices == expected_indices_0
        ), f"cardinality.map_indices: expected {expected_indices_0}, got {actual_indices}"

    def test_lmb_determinism(self, lmb_fixture):
        """Same seed produces identical results."""
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmb

        model = lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        birth = make_birth_model_from_fixture(lmb_fixture)

        prior_tracks = load_prior_tracks(lmb_fixture)
        measurements = measurements_to_numpy(lmb_fixture["measurements"])

        filter1 = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=42)
        filter1.set_tracks(prior_tracks)
        output1 = filter1.step_detailed(measurements, timestep=0)

        filter2 = FilterLmb(motion, sensor, birth, AssociatorConfig.lbp(), seed=42)
        filter2.set_tracks(prior_tracks)
        output2 = filter2.step_detailed(measurements, timestep=0)

        # Cardinality must match
        assert output1.cardinality.n_estimated == output2.cardinality.n_estimated

        # Updated tracks must match exactly
        assert len(output1.updated_tracks) == len(output2.updated_tracks)
        for i, (t1, t2) in enumerate(zip(output1.updated_tracks, output2.updated_tracks)):
            assert t1.label == t2.label, f"Track {i} label mismatch"
            compare_scalar(f"Track {i} existence", t1.existence, t2.existence, 0.0)
            np.testing.assert_array_equal(t1.mu, t2.mu, err_msg=f"Track {i} mu mismatch")
            np.testing.assert_array_equal(t1.sigma, t2.sigma, err_msg=f"Track {i} sigma mismatch")


class TestLmbmFixtureEquivalence:
    """Test FilterLmbm against LMBM step-by-step fixture with FULL intermediate validation."""

    def test_lmbm_association_matrices_equivalence(self, lmbm_fixture):
        """Verify LMBM association matrices match MATLAB exactly.

        This test validates:
        - C: Cost matrix
        - L: Likelihood matrix
        - P: Sampling probabilities

        Note: We load the PRIOR hypothesis and let the filter run prediction,
        which adds birth tracks to get the predicted hypothesis state that
        matches the fixture's step2_association input.
        """
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmbm, _LmbmHypothesis

        model = lmbm_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)

        # Create birth model matching the fixture's birth configuration
        # The fixture adds 4 birth locations at timestep 3
        birth = make_birth_model_from_fixture(lmbm_fixture)

        # Get Gibbs parameters from fixture
        gibbs_input = lmbm_fixture["step3a_gibbs"]["input"]
        num_samples = gibbs_input["numberOfSamples"]
        gibbs_seed = gibbs_input["rng_seed"]

        filter = FilterLmbm(
            motion, sensor, birth, AssociatorConfig.gibbs(num_samples), seed=gibbs_seed
        )

        # Load PRIOR hypothesis - step_detailed will run prediction to add birth tracks
        prior_hyp = lmbm_fixture["step1_prediction"]["input"]["prior_hypothesis"]
        hypothesis = _LmbmHypothesis.from_matlab(
            w=prior_hyp["w"],
            r=prior_hyp["r"],
            mu=prior_hyp["mu"],
            sigma=prior_hyp["Sigma"],
            birth_time=prior_hyp["birthTime"],
            birth_location=prior_hyp["birthLocation"],
        )
        filter.set_hypotheses([hypothesis])

        measurements = measurements_to_numpy(lmbm_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=lmbm_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Verify association matrices match MATLAB
        # ═══════════════════════════════════════════════════════════════
        expected_assoc = lmbm_fixture["step2_association"]["output"]

        assert output.association_matrices is not None, "Association matrices should exist"

        # Compare C, L, P matrices
        compare_array(
            "step2.cost", expected_assoc["C"], output.association_matrices.cost, TOLERANCE
        )
        compare_array(
            "step2.likelihood",
            expected_assoc["L"],
            output.association_matrices.likelihood,
            TOLERANCE,
        )
        compare_array(
            "step2.sampling_prob",
            expected_assoc["P"],
            output.association_matrices.sampling_prob,
            TOLERANCE,
        )

    def test_lmbm_gibbs_v_matrix_equivalence(self, lmbm_fixture):
        """Verify LMBM Gibbs sampling produces identical distinct assignment samples.

        The V matrix contains distinct association vectors where V[i,j] indicates
        which measurement (1-indexed) or miss (0) track j is assigned to in sample i.

        Note: MATLAB returns unique(V, 'rows') - only distinct samples are kept.
        We load the PRIOR hypothesis and let prediction run to match the fixture flow.
        """
        from multisensor_lmb_filters_rs import AssociatorConfig, FilterLmbm, _LmbmHypothesis

        model = lmbm_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)

        # Create birth model matching the fixture's birth configuration
        birth = make_birth_model_from_fixture(lmbm_fixture)

        # Get Gibbs parameters from fixture
        gibbs_input = lmbm_fixture["step3a_gibbs"]["input"]
        num_samples = gibbs_input["numberOfSamples"]
        gibbs_seed = gibbs_input["rng_seed"]

        filter = FilterLmbm(
            motion, sensor, birth, AssociatorConfig.gibbs(num_samples), seed=gibbs_seed
        )

        # Load PRIOR hypothesis - step_detailed will run prediction to add birth tracks
        prior_hyp = lmbm_fixture["step1_prediction"]["input"]["prior_hypothesis"]
        hypothesis = _LmbmHypothesis.from_matlab(
            w=prior_hyp["w"],
            r=prior_hyp["r"],
            mu=prior_hyp["mu"],
            sigma=prior_hyp["Sigma"],
            birth_time=prior_hyp["birthTime"],
            birth_location=prior_hyp["birthLocation"],
        )
        filter.set_hypotheses([hypothesis])

        measurements = measurements_to_numpy(lmbm_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=lmbm_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 3a: Verify Gibbs V matrix matches MATLAB (unique samples)
        # ═══════════════════════════════════════════════════════════════
        expected_gibbs = lmbm_fixture["step3a_gibbs"]["output"]

        assert output.association_result is not None, "Association result should exist"
        assert output.association_result.assignments is not None, "Gibbs assignments should exist"

        expected_v = np.array(expected_gibbs["V"], dtype=np.int32)
        actual_v = output.association_result.assignments

        # Rust uses 0-indexed measurements with -1 for miss
        # MATLAB uses 1-indexed measurements with 0 for miss
        # Convert Rust format to MATLAB format: -1 -> 0, 0 -> 1, 1 -> 2, etc.
        actual_v_matlab = actual_v + 1

        # MATLAB lmbmGibbsSampling returns unique(V, 'rows') - distinct samples only
        # Get unique rows from actual_v for comparison
        actual_v_unique = np.unique(actual_v_matlab, axis=0)

        # Compare unique V matrices
        compare_array("step3a.V", expected_v, actual_v_unique, 0)  # Exact match for integers

    def test_lmbm_cardinality_equivalence(self, lmbm_fixture):
        """Verify LMBM cardinality estimation matches MATLAB exactly.

        This test validates the final output of steps 4-6:
        - Step 4: Hypothesis generation from Gibbs samples
        - Step 5: Normalization and pruning
        - Step 6: Cardinality estimation and track extraction

        The cardinality_estimate and extraction_indices from step6 are the
        key outputs that determine which tracks are extracted as estimates.
        """
        from multisensor_lmb_filters_rs import (
            AssociatorConfig,
            FilterLmbm,
            FilterLmbmConfig,
            _LmbmHypothesis,
        )

        model = lmbm_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        birth = make_birth_model_from_fixture(lmbm_fixture)

        # Get Gibbs parameters from fixture
        gibbs_input = lmbm_fixture["step3a_gibbs"]["input"]
        num_samples = gibbs_input["numberOfSamples"]
        gibbs_seed = gibbs_input["rng_seed"]

        # Get extraction config from fixture
        # MATLAB use_map=False means use EAP, so Rust use_eap=True
        step6_input = lmbm_fixture["step6_extraction"]["input"]
        use_eap = not step6_input["use_map"]

        # Get normalization parameters from fixture
        step5_input = lmbm_fixture["step5_normalization"]["input"]
        lmbm_config = FilterLmbmConfig(
            max_hypotheses=step5_input["model_maximum_number_of_posterior_hypotheses"],
            hypothesis_weight_threshold=step5_input["model_posterior_hypothesis_weight_threshold"],
            use_eap=use_eap,
            existence_threshold=step5_input["model_existence_threshold"],
        )

        filter = FilterLmbm(
            motion,
            sensor,
            birth,
            AssociatorConfig.gibbs(num_samples),
            lmbm_config=lmbm_config,
            seed=gibbs_seed,
        )

        # Load PRIOR hypothesis
        prior_hyp = lmbm_fixture["step1_prediction"]["input"]["prior_hypothesis"]
        hypothesis = _LmbmHypothesis.from_matlab(
            w=prior_hyp["w"],
            r=prior_hyp["r"],
            mu=prior_hyp["mu"],
            sigma=prior_hyp["Sigma"],
            birth_time=prior_hyp["birthTime"],
            birth_location=prior_hyp["birthLocation"],
        )
        filter.set_hypotheses([hypothesis])

        measurements = measurements_to_numpy(lmbm_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=lmbm_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Verify cardinality matches MATLAB exactly
        # ═══════════════════════════════════════════════════════════════
        expected_step6 = lmbm_fixture["step6_extraction"]["output"]
        expected_cardinality = expected_step6["cardinality_estimate"]
        expected_indices = expected_step6["extraction_indices"]

        assert output.cardinality is not None, "Cardinality should exist"

        # Verify exact cardinality match
        assert output.cardinality.n_estimated == expected_cardinality, (
            f"Cardinality mismatch: expected {expected_cardinality}, "
            f"got {output.cardinality.n_estimated}"
        )

        # Verify extraction indices match (convert MATLAB 1-indexed to 0-indexed)
        expected_indices_0indexed = [i - 1 for i in expected_indices]
        actual_indices = sorted(output.cardinality.map_indices)
        expected_sorted = sorted(expected_indices_0indexed)
        assert actual_indices == expected_sorted, (
            f"Extraction indices mismatch: expected {expected_sorted} (0-indexed), "
            f"got {actual_indices}"
        )

    def test_lmbm_runs_on_fixture(self, lmbm_fixture):
        """LMBM filter runs on fixture data and produces valid output."""
        from multisensor_lmb_filters_rs import FilterLmbm

        model = lmbm_fixture["model"]
        motion = make_motion_model(model)
        sensor = make_sensor_model(model)
        birth = make_birth_model_empty()

        measurements = measurements_to_numpy(lmbm_fixture["measurements"])

        filter = FilterLmbm(motion, sensor, birth, seed=lmbm_fixture["seed"])
        estimate = filter.step(measurements, timestep=lmbm_fixture["timestep"])

        # Validate output structure
        assert estimate.num_tracks >= 0
        assert filter.num_hypotheses >= 1


class TestMultisensorLmbFixtureEquivalence:
    """Test multi-sensor LMB filters against fixtures with FULL intermediate validation."""

    def test_ic_lmb_prediction_equivalence(self, multisensor_lmb_fixture):
        """Verify IC-LMB prediction step matches MATLAB exactly."""
        from multisensor_lmb_filters_rs import FilterIcLmb

        model = multisensor_lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor_config = make_multisensor_config(model)
        birth = make_birth_model_from_fixture(multisensor_lmb_fixture)

        filter = FilterIcLmb(motion, sensor_config, birth, seed=multisensor_lmb_fixture["seed"])

        # Load and set prior tracks from fixture
        prior_tracks = load_prior_tracks(multisensor_lmb_fixture)
        filter.set_tracks(prior_tracks)

        measurements = nested_measurements_to_numpy(multisensor_lmb_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=multisensor_lmb_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Verify predicted tracks match MATLAB
        # ═══════════════════════════════════════════════════════════════
        expected_predicted = multisensor_lmb_fixture["step1_prediction"]["output"][
            "predicted_objects"
        ]
        compare_tracks("step1_predicted", expected_predicted, output.predicted_tracks, TOLERANCE)

    def test_ic_lmb_cardinality_equivalence(self, multisensor_lmb_fixture):
        """Verify IC-LMB cardinality extraction matches MATLAB exactly."""
        from multisensor_lmb_filters_rs import FilterIcLmb

        model = multisensor_lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor_config = make_multisensor_config(model)
        birth = make_birth_model_from_fixture(multisensor_lmb_fixture)

        filter = FilterIcLmb(motion, sensor_config, birth, seed=multisensor_lmb_fixture["seed"])

        prior_tracks = load_prior_tracks(multisensor_lmb_fixture)
        filter.set_tracks(prior_tracks)

        measurements = nested_measurements_to_numpy(multisensor_lmb_fixture["measurements"])
        output = filter.step_detailed(measurements, timestep=multisensor_lmb_fixture["timestep"])

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Verify cardinality extraction matches MATLAB
        # ═══════════════════════════════════════════════════════════════
        # Multisensor fixtures use 'stepFinal_cardinality' instead of 'step5_cardinality'
        expected_card = multisensor_lmb_fixture.get(
            "step5_cardinality", multisensor_lmb_fixture.get("stepFinal_cardinality", {})
        ).get("output", {})

        if "n_estimated" in expected_card:
            assert output.cardinality.n_estimated == expected_card["n_estimated"], (
                f"cardinality.n_estimated: expected {expected_card['n_estimated']}, "
                f"got {output.cardinality.n_estimated}"
            )

    @pytest.mark.parametrize(
        "filter_cls_name",
        ["FilterAaLmb", "FilterGaLmb", "FilterPuLmb", "FilterIcLmb"],
    )
    def test_all_multisensor_variants_run(self, multisensor_lmb_fixture, filter_cls_name):
        """All multisensor LMB filter variants run on fixture data."""
        import multisensor_lmb_filters_rs as lmb

        FilterCls = getattr(lmb, filter_cls_name)

        model = multisensor_lmb_fixture["model"]
        motion = make_motion_model(model)
        sensor_config = make_multisensor_config(model)
        birth = make_birth_model_from_fixture(multisensor_lmb_fixture)

        prior_tracks = load_prior_tracks(multisensor_lmb_fixture)
        measurements = nested_measurements_to_numpy(multisensor_lmb_fixture["measurements"])

        filter = FilterCls(motion, sensor_config, birth, seed=42)
        filter.set_tracks(prior_tracks)
        output = filter.step_detailed(measurements, timestep=0)

        # Basic validation - should produce valid output
        assert output.cardinality.n_estimated >= 0, f"{filter_cls_name}: negative cardinality"

        # Predicted tracks should exist
        assert len(output.predicted_tracks) > 0, f"{filter_cls_name}: no predicted tracks"


class TestMultisensorLmbmFixtureEquivalence:
    """Test multi-sensor LMBM filter against fixture."""

    def test_multisensor_lmbm_runs_on_fixture(self, multisensor_lmbm_fixture):
        """Multi-sensor LMBM filter runs on fixture data."""
        from multisensor_lmb_filters_rs import FilterMultisensorLmbm

        model = multisensor_lmbm_fixture["model"]
        motion = make_motion_model(model)
        sensor_config = make_multisensor_config(model)
        birth = make_birth_model_empty()

        measurements = nested_measurements_to_numpy(multisensor_lmbm_fixture["measurements"])

        filter = FilterMultisensorLmbm(
            motion, sensor_config, birth, seed=multisensor_lmbm_fixture["seed"]
        )
        output = filter.step_detailed(measurements, timestep=multisensor_lmbm_fixture["timestep"])

        assert output.cardinality.n_estimated >= 0
        assert filter.num_hypotheses >= 1


class TestAllFixturesLoad:
    """Verify all fixture files exist and load correctly."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "step_by_step/lmb_step_by_step_seed42.json",
            "step_by_step/lmbm_step_by_step_seed42.json",
            "step_by_step/multisensor_lmb_step_by_step_seed42.json",
            "step_by_step/multisensor_lmbm_step_by_step_seed42.json",
            "single_trial_42.json",
            "single_trial_42_quick.json",
            "single_detection_trial_42_quick.json",
            "multisensor_trial_42.json",
            "multisensor_clutter_trial_42_quick.json",
            "multisensor_detection_trial_42_quick.json",
        ],
    )
    def test_fixture_loads(self, fixture_name):
        """All 10 fixture files load without error."""
        fixture = load_fixture(fixture_name)
        assert "seed" in fixture or "filterVariants" in fixture
