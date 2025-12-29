# Fixture Coverage & Test Status

**⚠️ IMPORTANT**: This file MUST be updated when tests are added/modified. Coverage matrix is the source of truth.

## Current Status

**Test Counts**: 88 tests total (56 Rust + 32 Python)
- ✅ **Single-Sensor LMB**: 100% Rust VALUE coverage
- ✅ **Single-Sensor LMBM**: All tests pass with TOLERANCE=1e-10
- ✅ **Multisensor LMB**: 100% Rust VALUE coverage
- ✅ **Multisensor LMBM**: 4 VALUE tests (normalization tested in isolation)

**Completion**: 30% (15/51 TODO items ✅), 0% blocked (0/51 ⏸️), 71% remaining (36/51)

---

## 🚨 CRITICAL: Fixture Incompatibility - End-to-End vs Step-by-Step Testing

### Problem

**Cannot test end-to-end filter runs using step-by-step fixtures with independent RNG seeds per step.**

**Step-by-step fixture** (`multisensor_lmbm_step_by_step_seed42.json`):
- Each step uses INDEPENDENT RNG seed (e.g., `step3_gibbs.input.rng_seed = seed + 2000 = 2042`)
- Designed for testing INDIVIDUAL steps in isolation
- ✅ Works for: isolated Gibbs sampling, individual step validation

**End-to-end filter**:
- Uses ONE RNG that continuously advances through ALL steps
- Starts at `seed + 1000 = 1042`
- By Gibbs time, RNG is in DIFFERENT state than isolated step
- ❌ Result: Different sample counts (1 vs 15), incompatible with step-by-step fixture

### Solution Required

**Generate new end-to-end fixture**: `multisensor_lmbm_end_to_end_seed42.json`

1. Create `trials/generateMultisensorLmbmEndToEndFixture.m` (adapt from `generateMultisensorLmbmDebugFixture.m`)
2. Run full `runMultisensorLmbmFilter()` with ONE continuous RNG (`seed + 1000`)
3. Capture all outputs using SAME RNG state
4. Save to JSON with end-to-end structure

### Previously Blocked Tests (4 total) - NOW UNBLOCKED ✅

- ✅ `test_multisensor_lmbm_normalization_isolated_equivalence()` - Rust (tests normalization in isolation)
- ✅ TODO-RS-MSLMBM-05: step5.normalized_hypotheses.w (tested via isolated test)
- ✅ TODO-RS-MSLMBM-06: step5.objects_likely_to_exist (tested via isolated test)
- [ ] TODO-PY-MSLMBM-05: step5 normalization Python test (remaining work)

### Bugs Fixed During Investigation

✅ **Birth model means** corrected (test file:795-816):
- Was: `DVector::zeros(x_dim)` for all locations
- Now: `[-80, -20, 0, 40]` for x-positions (matches MATLAB)

---

## Coverage Matrix

**Legend**: ✓ values (TOLERANCE=1e-10), ✓ structure (dimensions only), ✗ (not tested), ⏸️ (blocked)

### LMB FIXTURE (lmb_step_by_step_seed42.json)

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_objects | ✓ values | ✓ values | **COMPLETE** |
| step2.C (cost) | ✓ values | ✓ values | **COMPLETE** |
| step2.L (likelihood) | ✓ values | ✓ values | **COMPLETE** |
| step2.R (miss prob) | ✓ values | ✓ values | **COMPLETE** |
| step2.P (sampling) | ✓ values | ✓ values | **COMPLETE** |
| step2.eta | ✓ values | ✓ values | **COMPLETE** |
| step2.posteriorParameters.w | ✓ values | ✓ values | **COMPLETE** |
| step2.posteriorParameters.mu | ✓ values | ✓ values | **COMPLETE** |
| step2.posteriorParameters.Sigma | ✓ values | ✓ values | **COMPLETE** |
| step3a_lbp.r | ✓ values | ✓ values | **COMPLETE** |
| step3a_lbp.W | ✓ values | ✓ values | **COMPLETE** |
| step3b_gibbs.r | ✓ values | ✓ values | **COMPLETE** |
| step3b_gibbs.W | ✓ values | ✓ values | **COMPLETE** |
| step3c_murtys.r | ✓ values | ✓ values | **COMPLETE** |
| step3c_murtys.W | ✓ values | ✓ values | **COMPLETE** |
| step4.posterior_objects | ✓ values | ✓ values | **COMPLETE** |
| step5.n_estimated | ✓ values | ✓ values | **COMPLETE** |
| step5.map_indices | ✓ values | ✓ values | **COMPLETE** |

### LMBM FIXTURE (lmbm_step_by_step_seed42.json)

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_hypothesis.w | ✗ | ✓ values | **GAP: Add Python** |
| step1.predicted_hypothesis.r | ✗ | ✓ values | **GAP: Add Python** |
| step1.predicted_hypothesis.mu | ✗ | ✗ | **GAP: Add BOTH** |
| step1.predicted_hypothesis.Sigma | ✗ | ✗ | **GAP: Add BOTH** |
| step1.predicted_hypothesis.birthTime | ✗ | ✓ values | **GAP: Add Python** |
| step1.predicted_hypothesis.birthLocation | ✗ | ✗ | **GAP: Add BOTH** |
| step2.C | ✓ values | ✓ values | **COMPLETE** |
| step2.L | ✓ values | ✓ values | **COMPLETE** |
| step2.P | ✓ values | ✓ values | **COMPLETE** |
| step2.posteriorParameters.r | ✗ | ✓ values | **GAP: Add Python** |
| step2.posteriorParameters.mu | ✗ | ✗ | **GAP: Add BOTH (not exposed)** |
| step2.posteriorParameters.Sigma | ✗ | ✗ | **GAP: Add BOTH (not exposed)** |
| step3a_gibbs.V | ✓ values | ✓ values | **COMPLETE** |
| step3b_murtys.V | ✓ values | ✓ values | **COMPLETE** |
| step4.new_hypotheses (all fields) | ✓ values | ✓ values | **COMPLETE** |
| step5.normalized_hypotheses.w | ✗ | ✓ values | **GAP: Add Python** |
| step5.normalized_hypotheses.r/mu/Sigma | ✗ | ✗ | **GAP: Add BOTH (not exposed)** |
| step5.objects_likely_to_exist | ✓ values | ✓ values | **COMPLETE** |
| step6.cardinality_estimate | ✓ values | ✓ values | **COMPLETE** |
| step6.extraction_indices | ✓ values | ✓ values | **COMPLETE** |

### MULTISENSOR LMB FIXTURE (multisensor_lmb_step_by_step_seed42.json)

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_objects | ✓ values | ✓ values | **COMPLETE** |
| sensorUpdates[0].association (C/L/R/P/eta) | ✗ | ✓ values | **GAP: Add Python** |
| sensorUpdates[0].posteriorParameters | ✗ | ✗ | **GAP: Add BOTH** |
| sensorUpdates[0].dataAssociation (r/W) | ✗ | ✓ values | **COMPLETE (Rust)** |
| sensorUpdates[0].updated_objects | ✗ | ✓ values | **COMPLETE (Rust)** |
| sensorUpdates[1].* (same fields) | ✗ | ✓ values | **COMPLETE (Rust)** |
| stepFinal.n_estimated | ✓ values | ✓ values | **COMPLETE** |
| stepFinal.map_indices | ✗ | ✓ values | **COMPLETE (Rust)** |

### MULTISENSOR LMBM FIXTURE (multisensor_lmbm_step_by_step_seed42.json)

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_hypothesis.w | ✗ | ✗ | **GAP: Add BOTH** |
| step1.predicted_hypothesis.r | ✗ | ✓ values | **GAP: Add Python** :466 |
| step1.predicted_hypothesis.mu/Sigma | ✗ | ✗ | **GAP: Add BOTH** |
| step1.predicted_hypothesis.birthTime | ✗ | ✓ values | **GAP: Add Python** :481 |
| step1.predicted_hypothesis.birthLocation | ✗ | ✓ values | **GAP: Add Python** :486 |
| step2.L | ✗ | ✓ values | **GAP: Add Python** :494 |
| step2.posteriorParameters.r | ✗ | ✓ values | **GAP: Add Python** :588 |
| step2.posteriorParameters.mu/Sigma | ✗ | ✗ | **GAP: Add BOTH** |
| step3_gibbs.A (sample count) | ✗ | ✓ values | **GAP: Add Python** :717 |
| step3_gibbs.A (sample content) | ✗ | ✗ | **GAP: Add BOTH** |
| step4.new_hypotheses (all fields) | ✗ | ✓ structure | **GAP: Upgrade Rust, Add Python** :728 |
| step5.normalized_hypotheses.w | ✗ | ✓ values | **GAP: Add Python** :784 (tested in isolation) |
| step5.normalized_hypotheses.r | ✗ | ✓ values | **GAP: Add Python** :784 (tested in isolation) |
| step5.objects_likely_to_exist | ✗ | ✓ values | **GAP: Add Python** :784 (tested in isolation) |
| step6.cardinality_estimate | ✗ | ✓ structure | **GAP: Upgrade Rust, Add Python** :920 |
| step6.extraction_indices | ✗ | ✓ structure | **GAP: Upgrade Rust, Add Python** :920 |

---

## TODO List - Path to 100% Coverage

### Python Tests (33 total)

**LMBM Single-Sensor (7 tests)**
- [ ] `test_lmbm_prediction_equivalence()` - w, r, mu, Sigma, birthTime, birthLocation
- [ ] Extend `test_lmbm_association_matrices_equivalence()` - add posteriorParameters.r
- [ ] Expose posteriorParameters.mu/Sigma in API
- [ ] `test_lmbm_normalized_hypotheses_individual_weights()` - replace sum check with individual weights
- [ ] Expose normalized_hypotheses.r/mu/Sigma in API

**Multisensor LMB (7 tests)**
- [ ] `test_multisensor_lmb_sensor0_association_equivalence()` - C, L, R, P, eta, posteriorParameters
- [ ] `test_multisensor_lmb_sensor0_data_association_equivalence()` - r, W
- [ ] `test_multisensor_lmb_sensor0_updated_objects_equivalence()` - full track comparison
- [ ] `test_multisensor_lmb_sensor1_*` (same 3 tests for sensor 1)
- [ ] Extend `test_ic_lmb_cardinality_equivalence()` - add map_indices

**Multisensor LMBM (19 tests)**
- [ ] `test_multisensor_lmbm_prediction_equivalence()` - all hypothesis fields
- [ ] `test_multisensor_lmbm_association_equivalence()` - L, posteriorParameters
- [ ] `test_multisensor_lmbm_gibbs_a_matrix_equivalence()` - sample count + content
- [ ] `test_multisensor_lmbm_hypotheses_equivalence()` - all new_hypotheses fields
- [ ] ⏸️ `test_multisensor_lmbm_normalization_equivalence()` - BLOCKED (need end-to-end fixture)
- [ ] `test_multisensor_lmbm_extraction_equivalence()` - cardinality + indices

### Rust Tests (16 remaining, 4 blocked)

**LMB Single-Sensor (6 tests)**
- [ ] `test_lmb_gibbs_result_equivalence()` - if deterministic path exists
- [ ] `test_lmb_murty_result_equivalence()` - if deterministic path exists
- [ ] `test_lmb_posterior_objects_equivalence()` - if unit-testable
- [ ] `test_lmb_cardinality_equivalence()` - n_estimated, map_indices

**LMBM Single-Sensor (3 tests)**
- Already have 3 VALUE tests ✅

**Multisensor LMB (4 tests)**
- [ ] Upgrade sensor0/1 association L/R tests to VALUE comparison
- Already have 4 VALUE tests ✅

**Multisensor LMBM (3 remaining, 0 blocked)**
- [x] ✅ step2.L VALUE test (line 494)
- [x] ✅ step2.posteriorParameters.r VALUE test (line 588)
- [x] ✅ step3_gibbs.A sample count VALUE test (line 641)
- [ ] TODO-RS-MSLMBM-04: Upgrade step4.new_hypotheses to VALUE tests (currently structure:728)
- [x] ✅ TODO-RS-MSLMBM-05: step5.normalized_hypotheses (isolated test line 784)
- [x] ✅ TODO-RS-MSLMBM-06: step5.objects_likely_to_exist (isolated test line 784)
- [ ] TODO-RS-MSLMBM-07: Upgrade step6 to VALUE tests (currently structure:920)

---

## Summary Counts

**Rust Test Upgrades**: 31 total → **15 ✅, 0 ⏸️, 16 remaining**
- LMB: 7 upgrades (1 ✅, 6 remaining)
- LMBM: 6 upgrades (3 ✅, 3 remaining)
- Multisensor LMB: 8 upgrades (4 ✅, 4 remaining)
- Multisensor LMBM: 10 upgrades (7 ✅, 0 ⏸️, 3 remaining)

**Python Tests to Add**: 20 new tests / extensions
- LMBM: 5 tests
- Multisensor LMB: 7 tests
- Multisensor LMBM: 6 tests
- API changes: 2 (posteriorParameters, normalized_hypotheses exposure)

**Total**: 51 TODO items → **15 ✅ (30%), 0 ⏸️ (0%), 36 remaining (71%)**

**Recently Unblocked Items (tested in isolation)**:
1. ✅ TODO-RS-MSLMBM-05: step5.normalized_hypotheses.w
2. ✅ TODO-RS-MSLMBM-06: step5.objects_likely_to_exist
3. [ ] TODO-PY-MSLMBM-05: step5 normalization Python test (remaining)

---

## Completion Criteria

- [ ] ZERO "✗" in coverage matrix
- [ ] ZERO "✓ structure" in coverage matrix
- [ ] ALL tests use TOLERANCE=1e-10 (or 0 for integers)
- [ ] ALL Python tests pass
- [ ] ALL Rust tests pass
- [ ] Coverage matrix shows 100% "✓ values" in BOTH columns
- [ ] NO "GAP" entries remaining
- [ ] NO "BLOCKED" entries (generate end-to-end fixture)

---

## Implementation Notes

**THE GOLDEN RULE**: When tests fail, fix CODE not tests. Never relax tolerance beyond 1e-10.

**Key Lessons**:
1. **Fixtures are source of truth** - MATLAB output defines correct behavior
2. **Check fixture type FIRST** - step-by-step vs end-to-end are incompatible
3. **RNG seeds matter** - each step may have independent `rng_seed` field
4. **Use hex for uint64** - Octave fprintf loses precision for large integers
5. **Column-major order** - MATLAB's `reshape` reads columns first
6. **SimpleRng required** - Must match MATLAB's xorshift64 exactly
7. **Uniform01 distribution** - Must use full 64-bit: `u / 2^64`

**Recent Major Fixes**:
- ✅ SimpleRng (xorshift64) implementation for exact MATLAB equivalence
- ✅ Uniform01 distribution matching MATLAB's u64→f64 conversion
- ✅ Birth model means corrected to match MATLAB positions
- ✅ Identified fixture incompatibility for end-to-end testing

**Files Modified**: `src/lmb/simple_rng.rs` (NEW), `src/lmb/multisensor/traits.rs`, `tests/lmb/multisensor_lmbm_matlab_equivalence.rs`
