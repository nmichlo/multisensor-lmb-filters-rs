# Fixture Coverage Analysis - Actual Test Status

## Current Status

**Python tests**: 31 passed (100% pass rate)
**Rust tests**: 44 passed, 2 ignored (100% pass rate)

**Last Updated**: 2025-12-27 (Comprehensive audit)

## Coverage Matrix

### LMB FIXTURE (lmb_step_by_step_seed42.json)

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_objects | ✓ values | ✓ values | **COMPLETE** |
| step2.C (cost) | ✓ values | ✓ values | **COMPLETE** |
| step2.L (likelihood) | ✓ values | ✓ values | **COMPLETE** |
| step2.R (miss prob) | ✓ values | ✓ values | **COMPLETE** |
| step2.P (sampling) | ✓ values | ✓ values | **COMPLETE** |
| step2.eta | ✓ values | ✓ values | **COMPLETE** |
| step2.posteriorParameters.w | ✓ values | ✗ | Python ONLY (Rust: unit-level, Python: integration) |
| step2.posteriorParameters.mu | ✓ values | ✗ | Python ONLY (Rust: unit-level, Python: integration) |
| step2.posteriorParameters.Sigma | ✓ values | ✗ | Python ONLY (Rust: unit-level, Python: integration) |
| step3a_lbp.r | ✓ values | ✓ values | **COMPLETE** |
| step3a_lbp.W | ✓ values | ✓ values | **COMPLETE** |
| step3b_gibbs.r | ✓ values | ✗ | Python ONLY (RNG-dependent, tested via integration) |
| step3b_gibbs.W | ✓ values | ✗ | Python ONLY (RNG-dependent, tested via integration) |
| step3c_murtys.r | ✓ values | ✗ | Python ONLY (RNG-dependent, tested via integration) |
| step3c_murtys.W | ✓ values | ✗ | Python ONLY (RNG-dependent, tested via integration) |
| step4.posterior_objects | ✓ values | ✗ | Python ONLY (Requires full filter execution) |
| step5.n_estimated | ✓ values | ✗ | Python ONLY (Requires full filter execution) |
| step5.map_indices | ✓ values | ✗ | Python ONLY (Requires full filter execution) |

### LMBM FIXTURE (lmbm_step_by_step_seed42.json)

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_hypothesis.w | ✗ | ✓ values | Tested indirectly (step4 validates full hypothesis pipeline) |
| step1.predicted_hypothesis.r | ✗ | ✓ values | Tested indirectly (step4 validates full hypothesis pipeline) |
| step1.predicted_hypothesis.mu | ✗ | ✗ | Tested indirectly (step4 validates full hypothesis pipeline) |
| step1.predicted_hypothesis.Sigma | ✗ | ✗ | Tested indirectly (step4 validates full hypothesis pipeline) |
| step1.predicted_hypothesis.birthTime | ✗ | ✓ values | Tested indirectly (step4 validates full hypothesis pipeline) |
| step1.predicted_hypothesis.birthLocation | ✗ | ✗ | Tested indirectly (step4 validates full hypothesis pipeline) |
| step2.C | ✓ values | ✓ values | **COMPLETE** |
| step2.L | ✓ values | ✓ values | **COMPLETE** |
| step2.P | ✓ values | ✓ values | **COMPLETE** |
| step2.posteriorParameters.r | ✗ | ✓ values | **GAP: Add Python test** |
| step2.posteriorParameters.mu | ✗ | ✗ | **GAP: Add BOTH (not exposed in API)** |
| step2.posteriorParameters.Sigma | ✗ | ✗ | **GAP: Add BOTH (not exposed in API)** |
| step3a_gibbs.V | ✓ values | ✓ structure | Python ONLY (Rust: structure validated, RNG-dependent) |
| step3b_murtys.V | ✗ | ✓ structure | **GAP: Add Python test** |
| step4.new_hypotheses.w | ✓ values | ✓ structure | Python ONLY (Rust: structure validated, requires filter integration) |
| step4.new_hypotheses.r | ✓ values | ✓ structure | Python ONLY (Rust: structure validated, requires filter integration) |
| step4.new_hypotheses.mu | ✓ values | ✓ structure | Python ONLY (Rust: structure validated, requires filter integration) |
| step4.new_hypotheses.Sigma | ✓ values | ✓ structure | Python ONLY (Rust: structure validated, requires filter integration) |
| step4.new_hypotheses.birthTime | ✓ values | ✓ structure | Python ONLY (Rust: structure validated, requires filter integration) |
| step4.new_hypotheses.birthLocation | ✓ values | ✓ structure | Python ONLY (Rust: structure validated, requires filter integration) |
| step5.normalized_hypotheses.w | ✗ | ✓ values | **GAP: Add Python test (only sum validated currently)** |
| step5.normalized_hypotheses.r | ✗ | ✗ | **GAP: Add Python test (not exposed currently)** |
| step5.normalized_hypotheses.mu | ✗ | ✗ | **GAP: Add Python test (not exposed currently)** |
| step5.normalized_hypotheses.Sigma | ✗ | ✗ | **GAP: Add Python test (not exposed currently)** |
| step5.objects_likely_to_exist | ✓ values | ✓ values | **COMPLETE** |
| step6.cardinality_estimate | ✓ values | ✓ values | **COMPLETE** |
| step6.extraction_indices | ✓ values | ✓ values | **COMPLETE** |

### MULTISENSOR LMB FIXTURE (multisensor_lmb_step_by_step_seed42.json)

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_objects | ✓ values | ✓ values | **COMPLETE** |
| sensorUpdates[0].association.C | ✗ | ✓ values | **GAP: Add Python test** |
| sensorUpdates[0].association.L | ✗ | ✓ structure | **GAP: Add Python test** |
| sensorUpdates[0].association.R | ✗ | ✓ structure | **GAP: Add Python test** |
| sensorUpdates[0].association.P | ✗ | ✓ values | **GAP: Add Python test** |
| sensorUpdates[0].association.eta | ✗ | ✓ values | **GAP: Add Python test** |
| sensorUpdates[0].posteriorParameters.w | ✗ | ✗ | **GAP: Add Python test** |
| sensorUpdates[0].posteriorParameters.mu | ✗ | ✗ | **GAP: Add Python test** |
| sensorUpdates[0].posteriorParameters.Sigma | ✗ | ✗ | **GAP: Add Python test** |
| sensorUpdates[0].dataAssociation.r | ✗ | ✓ structure | **GAP: Add Python test** |
| sensorUpdates[0].dataAssociation.W | ✗ | ✓ structure | **GAP: Add Python test** |
| sensorUpdates[0].output.updated_objects | ✗ | ✓ structure | **GAP: Add Python value test** |
| sensorUpdates[1].* | same as [0] | same as [0] | Same status as sensor 0 |
| stepFinal.n_estimated | ✓ values | ✓ values | **COMPLETE** |
| stepFinal.map_indices | ✗ | ✓ structure | **GAP: Add Python test** |

### MULTISENSOR LMBM FIXTURE (multisensor_lmbm_step_by_step_seed42.json)

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_hypothesis.w | ✗ | ✓ values | **GAP: Add Python test** |
| step1.predicted_hypothesis.r | ✗ | ✓ values | **GAP: Add Python test** |
| step1.predicted_hypothesis.mu | ✗ | ✗ | **GAP: Add Python test** |
| step1.predicted_hypothesis.Sigma | ✗ | ✗ | **GAP: Add Python test** |
| step1.predicted_hypothesis.birthTime | ✗ | ✓ values | **GAP: Add Python test** |
| step1.predicted_hypothesis.birthLocation | ✗ | ✗ | **GAP: Add Python test** |
| step2.L | ✗ | ✓ structure | **GAP: Add Python test** |
| step2.posteriorParameters.r | ✗ | ✓ structure | **GAP: Add Python test** |
| step2.posteriorParameters.mu | ✗ | ✗ | **GAP: Add Python test** |
| step2.posteriorParameters.Sigma | ✗ | ✗ | **GAP: Add Python test** |
| step3_gibbs.A | ✗ | ✓ structure | **GAP: Add Python test** |
| step4.new_hypotheses.w | ✗ | ✓ structure | **GAP: Add Python test** |
| step4.new_hypotheses.r | ✗ | ✓ structure | **GAP: Add Python test** |
| step4.new_hypotheses.mu | ✗ | ✗ | **GAP: Add Python test** |
| step4.new_hypotheses.Sigma | ✗ | ✗ | **GAP: Add Python test** |
| step5.normalized_hypotheses.w | ✗ | ✓ structure | **GAP: Add Python test** |
| step5.normalized_hypotheses.r | ✗ | ✗ | **GAP: Add Python test** |
| step5.objects_likely_to_exist | ✗ | ✓ structure | **GAP: Add Python test** |
| step6.cardinality_estimate | ✗ | ✓ structure | **GAP: Add Python test** |
| step6.extraction_indices | ✗ | ✓ structure | **GAP: Add Python test** |

---

## Real Gaps Summary (Prioritized)

### CRITICAL GAPS (Require API Changes)

1. **LMBM posteriorParameters.mu/Sigma** - Not exposed in `StepDetailedOutput` API
   - Location: step2_association.output
   - Impact: Cannot validate Kalman filter update calculations
   - Decision: ACCEPT - Downstream hypothesis tests validate correctness

2. **LMBM normalized_hypotheses internals** - Not exposed in `StepDetailedOutput` API
   - Location: step5_normalization.output
   - Impact: Cannot validate individual hypothesis weights/states after normalization
   - Decision: ACCEPT - objects_likely_to_exist validates gating logic

### HIGH PRIORITY (Add Python Tests)

3. **Multisensor LMB per-sensor validation** (~11 fields per sensor × 2 sensors)
   - Missing: C, L, R, P, eta, posteriorParameters (all association output)
   - Missing: r, W (dataAssociation output)
   - Missing: updated_objects value comparison
   - Test: `test_multisensor_lmb_sensor_updates_equivalence()`

4. **Multisensor LMBM complete validation** (~20 fields, steps 1-6)
   - Currently: Only `test_multisensor_lmbm_runs_on_fixture()` exists
   - Missing: ALL step-by-step value comparisons
   - Tests needed:
     - `test_multisensor_lmbm_prediction_equivalence()`
     - `test_multisensor_lmbm_association_equivalence()`
     - `test_multisensor_lmbm_gibbs_a_matrix_equivalence()`
     - `test_multisensor_lmbm_hypotheses_equivalence()`
     - `test_multisensor_lmbm_normalization_equivalence()`
     - `test_multisensor_lmbm_cardinality_equivalence()`

5. **LMBM Murty's V matrix** - Python test missing
   - Test: `test_lmbm_murty_v_matrix_equivalence()`

### MEDIUM PRIORITY (Minor Completeness)

6. **Multisensor LMB map_indices** - Python test missing
   - Add to `test_ic_lmb_cardinality_equivalence()`

7. **LMBM posteriorParameters.r** - Python test missing
   - Add to `test_lmbm_association_matrices_equivalence()`

### ACCEPTED LIMITATIONS (Not Gaps)

- **Python-only tests**: Gibbs/Murty/full filter execution require RNG seeding
- **Rust structure-only tests**: Documented rationale in test comments
- **Indirect testing**: Prediction validated through downstream hypothesis tests

---

## Test Coverage Legend

| Symbol | Meaning |
|--------|---------|
| ✓ values | Numerical values compared with TOLERANCE=1e-10 |
| ✓ structure | Dimensions/validity checked (not full value comparison) |
| ✓ sum only | Only aggregate checked (e.g., weights sum to 1.0) |
| ✗ | Not tested |
| **COMPLETE** | Both Python and Rust have full value comparison |
| **GAP** | Missing test needs to be added |

## Implementation Strategy

1. **Multisensor tests first** - Biggest gaps, most value
2. **LMBM Murty test** - Quick win, completes single-sensor LMBM
3. **Minor additions** - map_indices, posteriorParameters.r
4. **API changes** - Only if time permits and justified by benefits
