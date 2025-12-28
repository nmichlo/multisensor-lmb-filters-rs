# Fixture Coverage Analysis - Actual Test Status

## Current Status

**Python tests**: 32 passed (100% pass rate)
**Rust tests**: 44 passed, 2 ignored (100% pass rate)

**Last Updated**: 2025-12-27 (Post-implementation update)

### Recent Changes
- ✅ **Fixed LMBM Murty support**: Updated Python binding to use `DynamicAssociator` (src/python/filters.rs:590)
- ✅ **Added test_lmbm_murty_v_matrix_equivalence**: Now validates Murty's K-best assignments with TOLERANCE=0 (exact integers)
- ✅ **Comprehensive audit**: Updated all coverage tables to reflect actual test state vs outdated TODOs

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
| step2.posteriorParameters.w | ✓ values | ✗ | **GAP: Add Rust test** |
| step2.posteriorParameters.mu | ✓ values | ✗ | **GAP: Add Rust test** |
| step2.posteriorParameters.Sigma | ✓ values | ✗ | **GAP: Add Rust test** |
| step3a_lbp.r | ✓ values | ✓ values | **COMPLETE** |
| step3a_lbp.W | ✓ values | ✓ values | **COMPLETE** |
| step3b_gibbs.r | ✓ values | ✗ | **GAP: Add Rust test (use SimpleRng with exact seed)** |
| step3b_gibbs.W | ✓ values | ✗ | **GAP: Add Rust test (use SimpleRng with exact seed)** |
| step3c_murtys.r | ✓ values | ✗ | **GAP: Add Rust test** |
| step3c_murtys.W | ✓ values | ✗ | **GAP: Add Rust test** |
| step4.posterior_objects | ✓ values | ✗ | **GAP: Add Rust test** |
| step5.n_estimated | ✓ values | ✗ | **GAP: Add Rust test** |
| step5.map_indices | ✓ values | ✗ | **GAP: Add Rust test** |

### LMBM FIXTURE (lmbm_step_by_step_seed42.json)

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_hypothesis.w | ✗ | ✓ values | **GAP: Add Python test** |
| step1.predicted_hypothesis.r | ✗ | ✓ values | **GAP: Add Python test** |
| step1.predicted_hypothesis.mu | ✗ | ✗ | **GAP: Add BOTH tests** |
| step1.predicted_hypothesis.Sigma | ✗ | ✗ | **GAP: Add BOTH tests** |
| step1.predicted_hypothesis.birthTime | ✗ | ✓ values | **GAP: Add Python test** |
| step1.predicted_hypothesis.birthLocation | ✗ | ✗ | **GAP: Add BOTH tests** |
| step2.C | ✓ values | ✓ values | **COMPLETE** |
| step2.L | ✓ values | ✓ values | **COMPLETE** |
| step2.P | ✓ values | ✓ values | **COMPLETE** |
| step2.posteriorParameters.r | ✗ | ✓ values | **GAP: Add Python test** |
| step2.posteriorParameters.mu | ✗ | ✗ | **GAP: Add BOTH (not exposed in API)** |
| step2.posteriorParameters.Sigma | ✗ | ✗ | **GAP: Add BOTH (not exposed in API)** |
| step3a_gibbs.V | ✓ values | ✗ | **GAP: Upgrade Rust to VALUE test** |
| step3b_murtys.V | ✓ values | ✗ | **GAP: Upgrade Rust to VALUE test** |
| step4.new_hypotheses.w | ✓ values | ✗ | **GAP: Upgrade Rust to VALUE test** |
| step4.new_hypotheses.r | ✓ values | ✗ | **GAP: Upgrade Rust to VALUE test** |
| step4.new_hypotheses.mu | ✓ values | ✗ | **GAP: Upgrade Rust to VALUE test** |
| step4.new_hypotheses.Sigma | ✓ values | ✗ | **GAP: Upgrade Rust to VALUE test** |
| step4.new_hypotheses.birthTime | ✓ values | ✗ | **GAP: Upgrade Rust to VALUE test** |
| step4.new_hypotheses.birthLocation | ✓ values | ✗ | **GAP: Upgrade Rust to VALUE test** |
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

### NO ACCEPTED LIMITATIONS

ALL tests must be VALUE-based with TOLERANCE=1e-10 (or 0 for integers) in BOTH Rust AND Python.
NO "structure-only" tests. NO "Python-only" or "Rust-only" tests. NO excuses.

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

---

## COMPREHENSIVE TODO LIST - 100% VALUE EQUIVALENCE

**CRITICAL RULES:**
- ALL tests must validate VALUES with TOLERANCE=1e-10 (or 0 for integers)
- NO "✓ structure" tests allowed - upgrade ALL to "✓ values"
- NO "✗" (untested) fields allowed
- NO skipping, NO laziness, NO exceptions
- Fix CODE when tests fail, NEVER weaken tests

### PYTHON TESTS - Add Missing Value Comparisons

#### LMB Single-Sensor (0 gaps - COMPLETE)
- ✅ All fields have value tests

#### LMBM Single-Sensor (7 gaps)
- [ ] **TODO-PY-LMBM-01**: Add `test_lmbm_prediction_equivalence()` - validate step1.predicted_hypothesis ALL fields:
  - w, r, mu, Sigma, birthTime, birthLocation vs fixture (TOLERANCE=1e-10)
  - Use `compare_lmbm_hypothesis()` helper
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-LMBM-02**: Add posteriorParameters.r to `test_lmbm_association_matrices_equivalence()`
  - Extend existing test at line ~357
  - Compare step2.posteriorParameters.r array (TOLERANCE=1e-10)
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-LMBM-03**: Expose posteriorParameters.mu/Sigma in `StepDetailedOutput` API
  - Modify: `src/filters/lmbm.rs` - add fields to association output struct
  - Modify: `src/python/filters.rs` - expose in Python binding
  - Add to fixture comparison in `test_lmbm_association_matrices_equivalence()`
  - Files: `src/filters/lmbm.rs`, `src/python/filters.rs`, `tests/test_equivalence.py`

- [ ] **TODO-PY-LMBM-04**: Add `test_lmbm_normalized_hypotheses_individual_weights()`
  - Replace weight sum check with INDIVIDUAL weight comparison
  - Compare step5.normalized_hypotheses.w for EACH hypothesis (TOLERANCE=1e-10)
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-LMBM-05**: Expose normalized_hypotheses.r/mu/Sigma in `StepDetailedOutput` API
  - Modify: `src/filters/lmbm.rs` - add fields to normalization output struct
  - Modify: `src/python/filters.rs` - expose in Python binding
  - Add `test_lmbm_normalized_hypotheses_complete()` for r/mu/Sigma
  - Files: `src/filters/lmbm.rs`, `src/python/filters.rs`, `tests/test_equivalence.py`

#### Multisensor LMB (26 gaps - 13 per sensor × 2 sensors)
- [ ] **TODO-PY-MSLMB-01**: Add `test_multisensor_lmb_sensor0_association_equivalence()`
  - Compare sensorUpdates[0].association: C, L, R, P, eta (TOLERANCE=1e-10)
  - Compare sensorUpdates[0].posteriorParameters: w, mu, Sigma (TOLERANCE=1e-10)
  - Use `compare_association_matrices()` helper
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-MSLMB-02**: Add `test_multisensor_lmb_sensor0_data_association_equivalence()`
  - Compare sensorUpdates[0].dataAssociation: r, W (TOLERANCE=1e-10)
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-MSLMB-03**: Add `test_multisensor_lmb_sensor0_updated_objects_equivalence()`
  - Compare sensorUpdates[0].output.updated_objects (TOLERANCE=1e-10)
  - Use `compare_tracks()` for full value validation
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-MSLMB-04**: Add `test_multisensor_lmb_sensor1_association_equivalence()`
  - Same as TODO-PY-MSLMB-01 but for sensorUpdates[1]
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-MSLMB-05**: Add `test_multisensor_lmb_sensor1_data_association_equivalence()`
  - Same as TODO-PY-MSLMB-02 but for sensorUpdates[1]
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-MSLMB-06**: Add `test_multisensor_lmb_sensor1_updated_objects_equivalence()`
  - Same as TODO-PY-MSLMB-03 but for sensorUpdates[1]
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-MSLMB-07**: Add map_indices to `test_ic_lmb_cardinality_equivalence()`
  - Extend existing test at line ~TBD
  - Compare stepFinal.map_indices array (TOLERANCE=0 for exact integer match)
  - File: `tests/test_equivalence.py`

#### Multisensor LMBM (20 gaps - almost completely untested)
- [ ] **TODO-PY-MSLMBM-01**: Add `test_multisensor_lmbm_prediction_equivalence()`
  - Compare step1.predicted_hypothesis: w, r, mu, Sigma, birthTime, birthLocation
  - Use `compare_lmbm_hypothesis()` helper (TOLERANCE=1e-10)
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-MSLMBM-02**: Add `test_multisensor_lmbm_association_equivalence()`
  - Compare step2.L (multi-dimensional array, TOLERANCE=1e-10)
  - Compare step2.posteriorParameters: r, mu, Sigma (TOLERANCE=1e-10)
  - Create helper `compare_multidimensional_array()` for nested lists
  - File: `tests/test_equivalence.py`, `tests/conftest.py`

- [ ] **TODO-PY-MSLMBM-03**: Add `test_multisensor_lmbm_gibbs_a_matrix_equivalence()`
  - Compare step3_gibbs.A (association matrix, TOLERANCE=0 for integers)
  - Validate [n_objects × n_sensors] per sample
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-MSLMBM-04**: Add `test_multisensor_lmbm_hypotheses_equivalence()`
  - Compare step4.new_hypotheses: w, r, mu, Sigma for ALL hypotheses
  - Full value comparison (TOLERANCE=1e-10)
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-MSLMBM-05**: Add `test_multisensor_lmbm_normalization_equivalence()`
  - Compare step5.normalized_hypotheses: w, r (TOLERANCE=1e-10)
  - Compare step5.objects_likely_to_exist (exact boolean match)
  - File: `tests/test_equivalence.py`

- [ ] **TODO-PY-MSLMBM-06**: Add `test_multisensor_lmbm_extraction_equivalence()`
  - Compare step6.cardinality_estimate (TOLERANCE=1e-10)
  - Compare step6.extraction_indices (TOLERANCE=0 for exact match)
  - File: `tests/test_equivalence.py`

### RUST TESTS - Upgrade Structure-Only to Value Equivalence

#### LMB Single-Sensor (7 structure gaps)
- [ ] **TODO-RS-LMB-01**: Add posteriorParameters value test to `test_lmb_association_equivalence()`
  - Currently missing Rust test for step2.posteriorParameters: w, mu, Sigma
  - Add comparison against fixture (TOLERANCE=1e-10)
  - File: `tests/lmb/matlab_equivalence.rs`

- [ ] **TODO-RS-LMB-02**: Add Gibbs r/W value test (if deterministic path exists)
  - Currently missing Rust test for step3b_gibbs: r, W
  - If RNG-dependent: Document rationale, accept Python-only coverage
  - If testable: Add `test_lmb_gibbs_result_equivalence()` (TOLERANCE=1e-10)
  - File: `tests/lmb/matlab_equivalence.rs`

- [ ] **TODO-RS-LMB-03**: Add Murty r/W value test (if deterministic path exists)
  - Currently missing Rust test for step3c_murtys: r, W
  - If RNG-dependent: Document rationale, accept Python-only coverage
  - If testable: Add `test_lmb_murty_result_equivalence()` (TOLERANCE=1e-10)
  - File: `tests/lmb/matlab_equivalence.rs`

- [ ] **TODO-RS-LMB-04**: Add posterior_objects value test (if unit-testable)
  - Currently missing Rust test for step4.posterior_objects
  - If requires full filter: Document rationale, accept Python-only coverage
  - If unit-testable: Add `test_lmb_posterior_objects_equivalence()` (TOLERANCE=1e-10)
  - File: `tests/lmb/matlab_equivalence.rs`

- [ ] **TODO-RS-LMB-05**: Add cardinality value test (if unit-testable)
  - Currently missing Rust test for step5: n_estimated, map_indices
  - If unit-testable: Add `test_lmb_cardinality_equivalence()` (TOLERANCE=1e-10)
  - File: `tests/lmb/matlab_equivalence.rs`

#### LMBM Single-Sensor (9 structure gaps)
- [ ] **TODO-RS-LMBM-01**: Upgrade `test_lmbm_gibbs_v_matrix_structure()` → VALUE test
  - Change from "✓ structure" to "✓ values"
  - Compare step3a_gibbs.V matrix VALUES against fixture (TOLERANCE=0 for integers)
  - File: `tests/lmb/lmbm_matlab_equivalence.rs`

- [ ] **TODO-RS-LMBM-02**: Upgrade `test_lmbm_murty_v_matrix_structure()` → VALUE test
  - Change from "✓ structure" to "✓ values"
  - Compare step3b_murtys.V matrix VALUES against fixture (TOLERANCE=0 for integers)
  - File: `tests/lmb/lmbm_matlab_equivalence.rs`

- [ ] **TODO-RS-LMBM-03**: Upgrade hypothesis generation tests → VALUE tests
  - Currently step4.new_hypotheses has "✓ structure" for w, r, mu, Sigma, birthTime, birthLocation
  - Change ALL to "✓ values" with TOLERANCE=1e-10
  - If requires full filter integration: Document rationale, ensure Python has full coverage
  - File: `tests/lmb/lmbm_matlab_equivalence.rs`

#### Multisensor LMB (8 structure gaps)
- [ ] **TODO-RS-MSLMB-01**: Upgrade sensor association L/R tests → VALUE tests
  - Currently sensorUpdates[0/1].association.L/R are "✓ structure"
  - Change to "✓ values" with TOLERANCE=1e-10
  - File: `tests/lmb/multisensor_matlab_equivalence.rs`

- [ ] **TODO-RS-MSLMB-02**: Upgrade sensor data association r/W tests → VALUE tests
  - Currently sensorUpdates[0/1].dataAssociation.r/W are "✓ structure"
  - Change to "✓ values" with TOLERANCE=1e-10
  - File: `tests/lmb/multisensor_matlab_equivalence.rs`

- [ ] **TODO-RS-MSLMB-03**: Upgrade sensor updated_objects tests → VALUE tests
  - Currently sensorUpdates[0/1].output.updated_objects are "✓ structure"
  - Change to "✓ values" using `compare_tracks()` equivalent (TOLERANCE=1e-10)
  - File: `tests/lmb/multisensor_matlab_equivalence.rs`

- [ ] **TODO-RS-MSLMB-04**: Upgrade stepFinal.map_indices test → VALUE test
  - Currently "✓ structure"
  - Change to "✓ values" with TOLERANCE=0 (exact integer match)
  - File: `tests/lmb/multisensor_matlab_equivalence.rs`

#### Multisensor LMBM (10 structure gaps)
- [ ] **TODO-RS-MSLMBM-01**: Upgrade step2.L test → VALUE test
  - Currently "✓ structure"
  - Change to "✓ values" for multi-dimensional array (TOLERANCE=1e-10)
  - File: `tests/lmb/multisensor_lmbm_matlab_equivalence.rs`

- [ ] **TODO-RS-MSLMBM-02**: Upgrade step2.posteriorParameters.r test → VALUE test
  - Currently "✓ structure"
  - Change to "✓ values" (TOLERANCE=1e-10)
  - File: `tests/lmb/multisensor_lmbm_matlab_equivalence.rs`

- [ ] **TODO-RS-MSLMBM-03**: Upgrade step3_gibbs.A test → VALUE test
  - Currently "✓ structure"
  - Change to "✓ values" for association matrix (TOLERANCE=0 for integers)
  - File: `tests/lmb/multisensor_lmbm_matlab_equivalence.rs`

- [ ] **TODO-RS-MSLMBM-04**: Upgrade step4.new_hypotheses w/r tests → VALUE tests
  - Currently "✓ structure"
  - Change to "✓ values" (TOLERANCE=1e-10)
  - File: `tests/lmb/multisensor_lmbm_matlab_equivalence.rs`

- [ ] **TODO-RS-MSLMBM-05**: Upgrade step5.normalized_hypotheses.w test → VALUE test
  - Currently "✓ structure"
  - Change to "✓ values" (TOLERANCE=1e-10)
  - File: `tests/lmb/multisensor_lmbm_matlab_equivalence.rs`

- [ ] **TODO-RS-MSLMBM-06**: Upgrade step5.objects_likely_to_exist test → VALUE test
  - Currently "✓ structure"
  - Change to "✓ values" (exact boolean match)
  - File: `tests/lmb/multisensor_lmbm_matlab_equivalence.rs`

- [ ] **TODO-RS-MSLMBM-07**: Upgrade step6 cardinality/extraction tests → VALUE tests
  - Currently "✓ structure" for cardinality_estimate and extraction_indices
  - Change to "✓ values" (TOLERANCE=1e-10 for estimate, TOLERANCE=0 for indices)
  - File: `tests/lmb/multisensor_lmbm_matlab_equivalence.rs`

---

## TODO SUMMARY COUNTS

**Python Tests to Add:** 20 new tests / test extensions
- LMBM: 5 tests
- Multisensor LMB: 7 tests
- Multisensor LMBM: 6 tests
- API changes: 2 (posteriorParameters exposure, normalized_hypotheses exposure)

**Rust Tests to Upgrade:** 34 structure→value upgrades
- LMB: 5 upgrades
- LMBM: 9 upgrades
- Multisensor LMB: 8 upgrades
- Multisensor LMBM: 10 upgrades

**Total Work Items:** 54 TODO items

**Completion Criteria:**
- [ ] ZERO "✗" in coverage matrix
- [ ] ZERO "✓ structure" in coverage matrix
- [ ] ALL tests use TOLERANCE=1e-10 (or 0 for integers)
- [ ] ALL Python tests pass
- [ ] ALL Rust tests pass
- [ ] Coverage matrix shows 100% "✓ values" in BOTH Python AND Rust columns
- [ ] NO "GAP" entries remaining
