# Fixture Coverage & Test Equivalence Plan

**IMPORTANT**: This file MUST be updated when tests are added/modified. Coverage matrix is the source of truth.

## Current Status

**Test Counts**: 295 tests total (210 Rust + 85 Python)

### Coverage Summary

| Filter Type | Step-by-Step Fixture | Python Tests | Rust Tests | Status |
|-------------|---------------------|--------------|------------|--------|
| LMB | Full (7 steps) | Comprehensive | Comprehensive | **COMPLETE** |
| LMBM | Full (7 steps) | Comprehensive | Comprehensive | **COMPLETE** |
| AA-LMB | Full (6 steps) | Comprehensive (7 tests) | Comprehensive (6 tests) | **COMPLETE** |
| GA-LMB | Full (6 steps) | Comprehensive (7 tests) | Comprehensive (6 tests) | **COMPLETE** |
| PU-LMB | Full (6 steps) | Comprehensive (7 tests) | Comprehensive (6 tests) | **COMPLETE** |
| IC-LMB | Full (6 steps) | Comprehensive (7 tests) | Comprehensive (6 tests) | **COMPLETE** |
| MS-LMBM | Full (6 steps) | Partial (missing w,mu,Sigma) | Partial | **OPTIONAL** |

**ALL TESTS USE TOLERANCE=1e-10 FOR NUMERICAL COMPARISONS**

---

## Implementation Plan: Complete Test Equivalence

### Phase 1: Generate Comprehensive MATLAB Fixtures

**File to Create:** `benchmarks/generate_step_by_step_fixtures.m`

Generate step-by-step fixtures with ALL intermediate values for each filter:

#### 1.1 Multi-Sensor LMB Step-by-Step Fixtures (AA, GA, PU, IC)

Each fixture needs these steps:
```
step1_prediction:
  - tracks: [{r, w, mu, Sigma, birthTime, birthLocation}, ...]

step2_per_sensor_association:
  - sensor0: {C, L, R, P, eta, posteriorParameters}
  - sensor1: {C, L, R, P, eta, posteriorParameters}

step3_per_sensor_data_association:
  - sensor0: {r, W}  # posterior existence, marginal weights
  - sensor1: {r, W}

step4_per_sensor_update:
  - sensor0: {tracks: [{r, w, mu, Sigma}, ...]}
  - sensor1: {tracks: [{r, w, mu, Sigma}, ...]}

step5_fusion:
  - fused_tracks: [{r, w, mu, Sigma}, ...]  # After merger

step6_cardinality:
  - n_estimated, map_indices
```

Generate 4 fixtures:
- `tests/data/step_by_step/aa_lmb_step_by_step_seed42.json`
- `tests/data/step_by_step/ga_lmb_step_by_step_seed42.json`
- `tests/data/step_by_step/pu_lmb_step_by_step_seed42.json`
- `tests/data/step_by_step/ic_lmb_step_by_step_seed42.json` (replace partial)

#### 1.2 LMBM Benchmark Fixture

Generate LMBM fixture for end-to-end validation:
- `benchmarks/fixtures/bouncing_n5_s1_LMBM_LBP.json`

#### 1.3 Fix Multi-Sensor LMBM Fixture

Add missing fields to existing fixture:
- `step1_prediction` needs: w, mu, Sigma (not just r, birthTime, birthLocation)

### Phase 2: Python Test Implementation

**File to Modify:** `tests/test_equivalence.py`

#### 2.1 Add Multi-Sensor LMB Comprehensive Tests

Create new test class: `TestMultisensorLmbStepByStepEquivalence`

```python
class TestMultisensorLmbStepByStepEquivalence:
    """Comprehensive multi-sensor LMB tests (like single-sensor LMB)."""

    @pytest.fixture(params=["aa", "ga", "pu", "ic"])
    def ms_lmb_fixture(self, request):
        """Load step-by-step fixture for each merger type."""
        ...

    def test_prediction_equivalence(self, ms_lmb_fixture): ...
    def test_per_sensor_association_matrices_equivalence(self, ms_lmb_fixture): ...
    def test_per_sensor_data_association_equivalence(self, ms_lmb_fixture): ...
    def test_per_sensor_update_equivalence(self, ms_lmb_fixture): ...
    def test_fusion_equivalence(self, ms_lmb_fixture): ...
    def test_cardinality_equivalence(self, ms_lmb_fixture): ...
```

#### 2.2 Fix Multi-Sensor LMBM Tests

Update `TestMultisensorLmbmFixtureEquivalence`:
- `test_multisensor_lmbm_prediction_full_equivalence`: Add w, mu, Sigma validation

### Phase 3: Rust Test Implementation

**File to Modify:** `tests/lmb/multisensor_matlab_equivalence.rs`

#### 3.1 Add Multi-Sensor LMB Comprehensive Tests

Mirror the Python tests in Rust:

```rust
// For each merger type (AA, GA, PU, IC)
#[test]
fn test_aa_lmb_prediction_equivalence() { ... }
#[test]
fn test_aa_lmb_per_sensor_association_equivalence() { ... }
#[test]
fn test_aa_lmb_per_sensor_update_equivalence() { ... }
#[test]
fn test_aa_lmb_fusion_equivalence() { ... }
#[test]
fn test_aa_lmb_cardinality_equivalence() { ... }
```

### Phase 4: LMBM Benchmark Tests

**Files to Modify:**
- `benchmarks/test_matlab_fixtures.py`: Add LMBM fixture test
- `tests/benchmark_fixture_equivalence.rs`: Add LMBM fixture test

### Phase 5: Validation

Run all tests and verify:
1. `cargo test --release` - All Rust tests pass
2. `uv run pytest -v` - All Python tests pass
3. Same number of tests for each filter type in both languages

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `benchmarks/generate_step_by_step_fixtures.m` | Create | Generate comprehensive fixtures |
| `tests/data/step_by_step/aa_lmb_step_by_step_seed42.json` | Create | AA-LMB fixture |
| `tests/data/step_by_step/ga_lmb_step_by_step_seed42.json` | Create | GA-LMB fixture |
| `tests/data/step_by_step/pu_lmb_step_by_step_seed42.json` | Create | PU-LMB fixture |
| `tests/data/step_by_step/ic_lmb_step_by_step_seed42.json` | Create | IC-LMB full fixture |
| `tests/data/step_by_step/multisensor_lmbm_step_by_step_seed42.json` | Modify | Add w,mu,Sigma |
| `benchmarks/fixtures/bouncing_n5_s1_LMBM_LBP.json` | Create | LMBM benchmark |
| `tests/test_equivalence.py` | Modify | Add comprehensive MS-LMB tests |
| `tests/lmb/multisensor_matlab_equivalence.rs` | Modify | Add comprehensive MS-LMB tests |
| `benchmarks/test_matlab_fixtures.py` | Modify | Add LMBM test |
| `tests/benchmark_fixture_equivalence.rs` | Modify | Add LMBM test |

---

## Test Count Target

| Filter | Python Tests | Rust Tests |
|--------|--------------|------------|
| LMB | 8 | 8 |
| LMBM | 7 | 7 |
| AA-LMB | 6 | 6 |
| GA-LMB | 6 | 6 |
| PU-LMB | 6 | 6 |
| IC-LMB | 6 | 6 |
| MS-LMBM | 6 | 6 |
| **Total** | **45** | **45** |

---

## Coverage Matrix

**Legend**: values (TOLERANCE=1e-10), structure (dimensions only), (not tested)

### LMB FIXTURE (lmb_step_by_step_seed42.json) - COMPLETE

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_objects | values | values | **COMPLETE** |
| step2.C (cost) | values | values | **COMPLETE** |
| step2.L (likelihood) | values | values | **COMPLETE** |
| step2.R (miss prob) | values | values | **COMPLETE** |
| step2.P (sampling) | values | values | **COMPLETE** |
| step2.eta | values | values | **COMPLETE** |
| step2.posteriorParameters.w | values | values | **COMPLETE** |
| step2.posteriorParameters.mu | values | values | **COMPLETE** |
| step2.posteriorParameters.Sigma | values | values | **COMPLETE** |
| step3a_lbp.r | values | values | **COMPLETE** |
| step3a_lbp.W | values | values | **COMPLETE** |
| step3b_gibbs.r | values | values | **COMPLETE** |
| step3b_gibbs.W | values | values | **COMPLETE** |
| step3c_murtys.r | values | values | **COMPLETE** |
| step3c_murtys.W | values | values | **COMPLETE** |
| step4.posterior_objects | values | values | **COMPLETE** |
| step5.n_estimated | values | values | **COMPLETE** |
| step5.map_indices | values | values | **COMPLETE** |

### LMBM FIXTURE (lmbm_step_by_step_seed42.json) - COMPLETE

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_hypothesis.w | values | values | **COMPLETE** |
| step1.predicted_hypothesis.r | values | values | **COMPLETE** |
| step1.predicted_hypothesis.mu | values | values | **COMPLETE** |
| step1.predicted_hypothesis.Sigma | values | values | **COMPLETE** |
| step1.predicted_hypothesis.birthTime | values | values | **COMPLETE** |
| step1.predicted_hypothesis.birthLocation | values | values | **COMPLETE** |
| step2.C | values | values | **COMPLETE** |
| step2.L | values | values | **COMPLETE** |
| step2.P | values | values | **COMPLETE** |
| step2.posteriorParameters (conditional) | values | values | **COMPLETE** |
| step3a_gibbs.V | values | values | **COMPLETE** |
| step3b_murtys.V | values | values | **COMPLETE** |
| step4.new_hypotheses (all fields) | values | values | **COMPLETE** |
| step5.normalized_hypotheses.w | values | values | **COMPLETE** |
| step5.normalized_hypotheses.r/mu/Sigma | values | values | **COMPLETE** |
| step5.objects_likely_to_exist | values | values | **COMPLETE** |
| step6.cardinality_estimate | values | values | **COMPLETE** |
| step6.extraction_indices | values | values | **COMPLETE** |

### MULTISENSOR LMB FIXTURE (multisensor_lmb_step_by_step_seed42.json) - PARTIAL

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_objects | values | values | **COMPLETE** |
| sensorUpdates[0].association (C/L/R/P/eta) | values | values | **COMPLETE** |
| sensorUpdates[0].posteriorParameters | values | values | **COMPLETE** |
| sensorUpdates[0].dataAssociation (r/W) | values | values | **COMPLETE** |
| sensorUpdates[0].updated_objects | values | values | **COMPLETE** |
| sensorUpdates[1].association (C/L/R/P/eta) | values | values | **COMPLETE** |
| sensorUpdates[1].posteriorParameters | values | values | **COMPLETE** |
| sensorUpdates[1].dataAssociation (r/W) | values | values | **COMPLETE** |
| sensorUpdates[1].updated_objects | values | values | **COMPLETE** |
| stepFinal.n_estimated | values | values | **COMPLETE** |
| stepFinal.map_indices | values | values | **COMPLETE** |
| **step5_fusion (AA/GA/PU)** | - | - | **NEEDS FIXTURE** |

### MULTISENSOR LMBM FIXTURE (multisensor_lmbm_step_by_step_seed42.json) - PARTIAL

| Field | Python | Rust | Status |
|-------|--------|------|--------|
| step1.predicted_hypothesis.r | values | values | **COMPLETE** |
| step1.predicted_hypothesis.birthTime | values | values | **COMPLETE** |
| step1.predicted_hypothesis.birthLocation | values | values | **COMPLETE** |
| **step1.predicted_hypothesis.w** | - | - | **NEEDS FIXTURE** |
| **step1.predicted_hypothesis.mu** | - | - | **NEEDS FIXTURE** |
| **step1.predicted_hypothesis.Sigma** | - | - | **NEEDS FIXTURE** |
| step2.L | structure | values | **COMPLETE** |
| step2.posteriorParameters | structure | values | **COMPLETE** |
| step3_gibbs.A (sample count) | values | values | **COMPLETE** |
| step4.new_hypotheses | structure | values | **COMPLETE** |
| step5.normalized_hypotheses | structure | values | **COMPLETE** |
| step5.objects_likely_to_exist | structure | values | **COMPLETE** |
| step6.cardinality_estimate | values | values | **COMPLETE** |
| step6.extraction_indices | values | values | **COMPLETE** |

---

## Implementation Notes

**THE GOLDEN RULE**: When tests fail, fix CODE not tests. Never relax tolerance beyond 1e-10.

**Key Lessons**:
1. **Fixtures are source of truth** - MATLAB output defines correct behavior
2. **IC-LMB is SEQUENTIAL** - Each sensor uses previous sensor's output (not parallel)
3. **Check fixture type FIRST** - step-by-step vs end-to-end are incompatible
4. **RNG seeds matter** - each step may have independent `rng_seed` field
5. **SimpleRng required** - Must match MATLAB's xorshift64 exactly
6. **Uniform01 distribution** - Must use full 64-bit: `u / 2^64`
7. **State ordering**: [x, y, vx, vy] (MATLAB convention, not [x, vx, y, vy])

---

## Historical Bug Fixes (Reference)

### RNG Precision Bug
**Problem**: `Rng::rand()` used `u64::MAX as f64 + 1.0` as divisor, loses precision.
**Fix**: Changed to `2_f64.powi(64)` matching MATLAB's `double(u) / (2^64)`.

### LBP sigma_mt_old Bug
**Problem**: MATLAB computes `B = Psi .* SigmaMT` at START of iteration, Rust recomputed after loop.
**Fix**: Pass `sigma_mt_old` (from before last iteration) to `compute_lbp_result()`.

### IC-LMB Sequential Processing Bug
**Problem**: IC-LMB used parallel processing (all sensors get same input).
**Fix**: Added `is_sequential()` to `Merger` trait; IC-LMB now passes each sensor's output to next.

### PU-LMB Cartesian Product Bug
**Problem**: Rust only used first GM component from each sensor.
**Fix**: Implemented full Cartesian product of all GM components across sensors (matching MATLAB).

### State Ordering Mismatch
**Problem**: Rust used [x, vx, y, vy], MATLAB uses [x, y, vx, vy].
**Fix**: Changed `MotionModel::constant_velocity_2d` and `SensorModel::position_sensor_2d`.

---

## Locations with 1e-15 Division Guards

These locations have guards that MATLAB doesn't have - may need review if numerical differences persist:

| File | Line | Context |
|------|------|---------|
| `src/common/utils.rs` | 48, 105, 221 | normalize functions |
| `src/common/association/gibbs.rs` | 198, 214, 288 | Gibbs sampling |
| `src/association/builder.rs` | 231, 411 | association matrix building |
| `src/components/update.rs` | 26, 62 | track update |
| `src/python/intermediate.rs` | 406 | Python conversion |

---

# EXHAUSTIVE TEST AUDIT (Senior Engineer Analysis)

## Test Counts
- **Rust**: 210 total (206 active + 4 ignored)
- **Python**: 85 total

## Legend
- **Need**: ✓=Critical | ~=Nice-to-have | ✗=Remove/Skip
- **Dup**: What it duplicates (if any)

---

## 1. LMB Single-Sensor MATLAB Equivalence

| Test | Python File:Function | Rust File:Function | Py Need | Rust Need | Dup? | Action |
|------|---------------------|-------------------|:-------:|:---------:|------|--------|
| Prediction | `test_equivalence.py:TestLmbStepByStepEquivalence.test_prediction_equivalence` | `matlab_equivalence.rs:test_new_api_all_tracks_equivalence` | ✓ | ✓ | No | Keep |
| Association C/L/R/P | `test_equivalence.py:TestLmbStepByStepEquivalence.test_association_matrices_equivalence` | `matlab_equivalence.rs:test_new_api_cost_matrix_equivalence` | ✓ | ✓ | No | Keep |
| Posterior params | (in assoc test) | `matlab_equivalence.rs:test_new_api_posterior_parameters_equivalence` | ~ | ~ | w/ Update | Keep |
| LBP result | `test_equivalence.py:TestLmbStepByStepEquivalence.test_lbp_data_association_equivalence` | `matlab_equivalence.rs:test_new_api_lbp_equivalence` | ✓ | ✓ | No | Keep |
| Gibbs result | `test_equivalence.py:TestLmbStepByStepEquivalence.test_gibbs_data_association_equivalence` | `matlab_equivalence.rs:test_new_api_gibbs_equivalence` | ✓ | ✓ | No | Keep |
| Murty result | `test_equivalence.py:TestLmbStepByStepEquivalence.test_murtys_data_association_equivalence` | `matlab_equivalence.rs:test_new_api_murtys_equivalence` | ✓ | ✓ | No | Keep |
| Update | `test_equivalence.py:TestLmbStepByStepEquivalence.test_update_equivalence` | `matlab_equivalence.rs:test_new_api_update_equivalence` | ✓ | ✓ | No | Keep |
| Cardinality | `test_equivalence.py:TestLmbStepByStepEquivalence.test_cardinality_equivalence` | `matlab_equivalence.rs:test_new_api_cardinality_n_equivalence` | ✓ | ✓ | No | Keep |
| Determinism | `test_equivalence.py:TestLmbStepByStepEquivalence.test_lmb_filter_step_is_deterministic` | ❌ | ~ | ✗ | No | Py-only OK |
| Psi/phi/eta diag | ❌ | `matlab_equivalence.rs:test_new_api_psi_phi_eta_vs_matlab` | ✗ | ~ | Debug | Consider remove |
| Filter step smoke | ❌ | `matlab_equivalence.rs:test_new_api_lmb_filter_step` | ✗ | ~ | Smoke | Consider remove |
| Prediction component | ❌ | `matlab_equivalence.rs:test_new_api_prediction_component_equivalence` | ✗ | ~ | Subsumes | Consider remove |
| Prediction track | ❌ | `matlab_equivalence.rs:test_new_api_prediction_track_equivalence` | ✗ | ~ | Subsumes | Consider remove |
| LBP runs smoke | ❌ | `matlab_equivalence.rs:test_new_api_lbp_runs_on_matlab_fixture` | ✗ | ~ | Smoke | Consider remove |

---

## 2. LMBM MATLAB Equivalence

| Test | Python File:Function | Rust File:Function | Py Need | Rust Need | Dup? | Action |
|------|---------------------|-------------------|:-------:|:---------:|------|--------|
| Prediction full | `test_equivalence.py:TestLmbmStepByStepEquivalence.test_prediction_equivalence` | `lmbm_matlab_equivalence.rs:test_lmbm_prediction_equivalence` | ✓ | ✓ | No | Keep |
| Association matrices | `test_equivalence.py:TestLmbmStepByStepEquivalence.test_association_matrices_equivalence` | `lmbm_matlab_equivalence.rs:test_lmbm_association_matrices_equivalence` | ✓ | ✓ | No | Keep |
| Gibbs V-matrix | `test_equivalence.py:TestLmbmStepByStepEquivalence.test_gibbs_v_matrix_equivalence` | `lmbm_matlab_equivalence.rs:test_lmbm_gibbs_v_matrix_equivalence` | ✓ | ✓ | No | Keep |
| Murty V-matrix | `test_equivalence.py:TestLmbmStepByStepEquivalence.test_murtys_v_matrix_equivalence` | `lmbm_matlab_equivalence.rs:test_lmbm_murtys_v_matrix_equivalence` | ✓ | ✓ | No | Keep |
| Step4 hypothesis | `test_equivalence.py:TestLmbmStepByStepEquivalence.test_step4_hypothesis_generation` | `lmbm_matlab_equivalence.rs:test_lmbm_hypothesis_generation_equivalence` | ✓ | ✓ | No | Keep |
| Step5 normalization | `test_equivalence.py:TestLmbmStepByStepEquivalence.test_step5_normalization` | `lmbm_matlab_equivalence.rs:test_lmbm_normalization_equivalence` | ✓ | ✓ | No | Keep |
| Cardinality | `test_equivalence.py:TestLmbmStepByStepEquivalence.test_cardinality_equivalence` | ❌ | ✓ | ✓ | No | **Add to Rust** |
| Extraction | ❌ | `lmbm_matlab_equivalence.rs:test_lmbm_extraction_equivalence` | ~ | ✓ | ~ Cardinality | Keep |
| Runs on fixture | `test_equivalence.py:TestLmbmStepByStepEquivalence.test_lmbm_filter_step_runs_on_fixture` | ❌ | ~ | ✗ | Smoke | Py-only OK |

---

## 3. Multi-Sensor LMB (IC-LMB baseline fixture)

| Test | Python File:Function | Rust File:Function | Py Need | Rust Need | Dup? | Action |
|------|---------------------|-------------------|:-------:|:---------:|------|--------|
| Prediction | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_prediction_equivalence` | `multisensor_matlab_equivalence.rs:test_multisensor_lmb_prediction_equivalence` | ✓ | ✓ | No | Keep |
| S0 assoc matrices | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_sensor0_association_matrices_equivalence` | `multisensor_matlab_equivalence.rs:test_multisensor_lmb_sensor0_association_equivalence` | ✓ | ✓ | No | Keep |
| S0 posterior params | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_sensor0_posterior_parameters_equivalence` | ❌ | ~ | ✗ | w/ Update | Consider remove |
| S0 data assoc | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_sensor0_data_association_equivalence` | `multisensor_matlab_equivalence.rs:test_multisensor_lmb_sensor0_data_association_equivalence` | ✓ | ✓ | No | Keep |
| S0 update | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_sensor0_update_equivalence` | `multisensor_matlab_equivalence.rs:test_multisensor_lmb_sensor0_update_equivalence` | ✓ | ✓ | No | Keep |
| S1 assoc matrices | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_sensor1_association_matrices_equivalence` | `multisensor_matlab_equivalence.rs:test_multisensor_lmb_sensor1_association_equivalence` | ~ | ~ | Same as S0 | Consider remove |
| S1 data assoc | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_sensor1_data_association_equivalence` | `multisensor_matlab_equivalence.rs:test_multisensor_lmb_sensor1_data_association_equivalence` | ~ | ~ | Same as S0 | Consider remove |
| S1 update | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_sensor1_update_equivalence` | `multisensor_matlab_equivalence.rs:test_multisensor_lmb_sensor1_update_equivalence` | ~ | ~ | Same as S0 | Consider remove |
| Cardinality | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_cardinality_equivalence` | `multisensor_matlab_equivalence.rs:test_multisensor_lmb_cardinality_equivalence` | ✓ | ✓ | No | Keep |
| All variants run | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_all_multisensor_variants_run` | ❌ | ~ | ✗ | Smoke | Py-only OK |

---

## 4. Multi-Sensor LMB Variants (AA/GA/PU/IC) - CRITICAL ANALYSIS

### Why Most Variant Tests Are Redundant

The prediction, association, data_association, update, and cardinality code paths are **IDENTICAL** across all 4 variants. The ONLY difference is the **fusion step**. Running the same code 4× proves nothing.

| Step | Code Path | Varies by Variant? |
|------|-----------|:------------------:|
| Prediction | `lmbPredictionStep()` | ❌ Same |
| Association | `generateLmbSensorAssociationMatrices()` | ❌ Same |
| Data Association | `loopyBeliefPropagation()` | ❌ Same |
| Per-sensor Update | `computePosteriorLmbSpatialDistributions()` | ❌ Same |
| **Fusion** | `aaLmbTrackMerging/gaLmbTrackMerging/puLmbTrackMerging` | **✓ DIFFERENT** |
| Cardinality | `lmbMapCardinalityEstimate()` | ❌ Same |

### Variant Tests Table

| Test | Python File:Function | Rust File:Function | Py Need | Rust Need | Action |
|------|---------------------|-------------------|:-------:|:---------:|--------|
| AA prediction | `test_equivalence.py:TestMultisensorLmbVariantsStepByStepEquivalence.test_prediction_equivalence[aa]` | `multisensor_variants_matlab_equivalence.rs:test_aa_lmb_prediction_equivalence` | ✗ | ✗ | **Keep 1, remove 3** |
| GA prediction | `test_equivalence.py:...[ga]` | `...:test_ga_lmb_prediction_equivalence` | ✗ | ✗ | Redundant |
| PU prediction | `test_equivalence.py:...[pu]` | `...:test_pu_lmb_prediction_equivalence` | ✗ | ✗ | Redundant |
| IC prediction | `test_equivalence.py:...[ic]` | `...:test_ic_lmb_prediction_equivalence` | ✗ | ✗ | Redundant |
| AA assoc matrices | `test_equivalence.py:...test_per_sensor_association_matrices_equivalence[aa]` | `...:test_aa_lmb_association_matrices_equivalence` | ✗ | ✗ | **Keep 1, remove 3** |
| GA assoc matrices | `...test_per_sensor_association_matrices_equivalence[ga]` | `...:test_ga_lmb_association_matrices_equivalence` | ✗ | ✗ | Redundant |
| PU assoc matrices | `...test_per_sensor_association_matrices_equivalence[pu]` | `...:test_pu_lmb_association_matrices_equivalence` | ✗ | ✗ | Redundant |
| IC assoc matrices | `...test_per_sensor_association_matrices_equivalence[ic]` | `...:test_ic_lmb_association_matrices_equivalence` | ✗ | ✗ | Redundant |
| AA posterior params | `...test_per_sensor_posterior_parameters_equivalence[aa]` | ❌ | ✗ | ✗ | **Remove all** |
| GA posterior params | `...[ga]` | ❌ | ✗ | ✗ | Redundant |
| PU posterior params | `...[pu]` | ❌ | ✗ | ✗ | Redundant |
| IC posterior params | `...[ic]` | ❌ | ✗ | ✗ | Redundant |
| AA data assoc | `...test_per_sensor_data_association_equivalence[aa]` | `...:test_aa_lmb_data_association_equivalence` | ✗ | ✗ | **Keep 1, remove 3** |
| GA data assoc | `...[ga]` | `...:test_ga_lmb_data_association_equivalence` | ✗ | ✗ | Redundant |
| PU data assoc | `...[pu]` | `...:test_pu_lmb_data_association_equivalence` | ✗ | ✗ | Redundant |
| IC data assoc | `...[ic]` | `...:test_ic_lmb_data_association_equivalence` | ✗ | ✗ | Redundant |
| AA update | `...test_per_sensor_update_equivalence[aa]` | `...:test_aa_lmb_update_output_equivalence` | ✗ | ✗ | **Keep 1, remove 3** |
| GA update | `...[ga]` | `...:test_ga_lmb_update_output_equivalence` | ✗ | ✗ | Redundant |
| PU update | `...[pu]` | `...:test_pu_lmb_update_output_equivalence` | ✗ | ✗ | Redundant |
| IC update | `...[ic]` | `...:test_ic_lmb_update_output_equivalence` | ✗ | ✗ | Redundant |
| **AA fusion** | `...test_fusion_equivalence[aa]` | `...:test_aa_lmb_fusion_equivalence` | **✓** | **✓** | **CRITICAL - Keep** |
| **GA fusion** | `...test_fusion_equivalence[ga]` | `...:test_ga_lmb_fusion_equivalence` | **✓** | **✓** | **CRITICAL - Keep** |
| **PU fusion** | `...test_fusion_equivalence[pu]` | `...:test_pu_lmb_fusion_equivalence` | **✓** | **✓** | **CRITICAL - Keep** |
| IC fusion | `...test_fusion_equivalence[ic]` (skipped) | ❌ | ✗ | ✗ | N/A (no fusion step) |
| AA cardinality | `...test_cardinality_equivalence[aa]` | `...:test_aa_lmb_cardinality_equivalence` | ✗ | ✗ | **Keep 1, remove 3** |
| GA cardinality | `...[ga]` | `...:test_ga_lmb_cardinality_equivalence` | ✗ | ✗ | Redundant |
| PU cardinality | `...[pu]` | `...:test_pu_lmb_cardinality_equivalence` | ✗ | ✗ | Redundant |
| IC cardinality | `...[ic]` | `...:test_ic_lmb_cardinality_equivalence` | ✗ | ✗ | Redundant |

---

## 5. Multi-Sensor LMBM

| Test | Python File:Function | Rust File:Function | Py Need | Rust Need | Action |
|------|---------------------|-------------------|:-------:|:---------:|--------|
| Prediction | `test_equivalence.py:TestMultisensorLmbmFixtureEquivalence.test_prediction_equivalence` | `multisensor_lmbm_matlab_equivalence.rs:test_ms_lmbm_prediction_equivalence` | ✓ | ✓ | Keep |
| Association L | `test_equivalence.py:...test_per_sensor_association_l_matrices_equivalence` | `...:test_ms_lmbm_association_l_matrices_equivalence` | ✓ | ✓ | Keep |
| Association post r | ❌ | `...:test_ms_lmbm_association_posterior_r_equivalence` | ~ | ✓ | Rust has more |
| Gibbs | `test_equivalence.py:...test_gibbs_sampling_count_equivalence` | `...:test_ms_lmbm_gibbs_sampling_equivalence` | ✓ | ✓ | Keep |
| Hypothesis gen | ❌ | `...:test_ms_lmbm_hypothesis_generation_equivalence` | ~ | ✓ | Rust has more |
| Normalization | ❌ | `...:test_ms_lmbm_normalization_equivalence` | ~ | ✓ | Rust has more |
| Extraction | `test_equivalence.py:...test_cardinality_and_extraction_equivalence` | `...:test_ms_lmbm_extraction_equivalence` | ✓ | ✓ | Keep |
| Runs on fixture | `test_equivalence.py:...test_multisensor_lmbm_filter_step_runs` | ❌ | ~ | ✗ | Py-only OK |

---

## 6. Infrastructure/Utility Tests

| Test | Python File:Function | Rust File:Function | Py Need | Rust Need | Action |
|------|---------------------|-------------------|:-------:|:---------:|--------|
| Fixture loads (×14) | `test_equivalence.py:TestFixtureLoading.test_*` | ❌ | ~ | ✗ | Py-only OK |
| Sensor output fields | `test_equivalence.py:TestMultisensorLmbFixtureEquivalence.test_sensor_output_fields_present` | ❌ | ~ | ✗ | API test |
| RNG equivalence | ❌ | `rng_equivalence.rs:*` | ✗ | ✓ | Keep |
| Utils (factorial etc) | ❌ | `utils.rs:467-513` | ✗ | ~ | Keep |
| Marginal eval n1/n2 | ❌ | `marginal_evaluations.rs:test_marginal_*_n1/n2` | ✗ | ✓ | Keep |
| Marginal eval n3 | ❌ | `marginal_evaluations.rs:test_marginal_*_n3` | ✗ | ~ | Ignored OK |
| Marginal comprehensive | ❌ | `marginal_evaluations.rs:test_comprehensive_*` | ✗ | ~ | Ignored OK |

---

## ACTION PLAN

### CRITICAL: Add to Rust (3 tests) ✅ DONE

```
File: tests/lmb/multisensor_variants_matlab_equivalence.rs

Added:
- test_aa_lmb_fusion_equivalence()
- test_ga_lmb_fusion_equivalence()
- test_pu_lmb_fusion_equivalence()

These test the ONLY code paths that differ between variants.
```

### RECOMMENDED: Add to Rust (1 test) - Future

```
File: tests/lmb/lmbm_matlab_equivalence.rs

Add:
- test_lmbm_cardinality_equivalence()

Python has this, Rust doesn't. Minor gap.
```

### CLEANUP: Remove from Python (Future)

These are redundant - same code tested multiple times:

| Remove | Reason |
|--------|--------|
| `test_per_sensor_posterior_parameters_equivalence` | Redundant with update |
| `test_sensor1_*` tests (4 tests) | Same code as sensor0 |
| 12 variant duplicates | Same code path tested 4× |

**Estimated savings**: ~20 tests worth of redundancy

### CLEANUP: Consider Removing from Rust (Future)

| Remove | Reason |
|--------|--------|
| `test_new_api_prediction_component_equivalence` | Subsumed by track test |
| `test_new_api_prediction_track_equivalence` | Subsumed by all tracks test |
| `test_new_api_psi_phi_eta_vs_matlab` | Debug only |
| `test_new_api_lbp_runs_on_matlab_fixture` | Smoke test |
| `test_new_api_lmb_filter_step` | Smoke test |
| 16 variant duplicates | Same code path tested 4× |

**Estimated savings**: ~20 tests worth of redundancy

---

## SENIOR ENGINEER VERDICT

### On "Variant tests are important":

**Disagree with blanket statement.** Here's the nuance:

1. **Variant FUSION tests are important** - These test different algorithms (AA/GA/PU use different merging)
2. **Variant non-fusion tests are NOT important** - Same code, different data

The current setup runs ~25 redundant tests across Python and Rust. This:
- Wastes CI time
- Creates maintenance burden
- Provides false sense of coverage
- Obscures real gaps (the 3 fusion tests that were missing)

### Recommended Final State (After Cleanup)

| Category | Python | Rust | Total |
|----------|--------|------|-------|
| LMB single | 8 | 12 | 20 |
| LMBM | 9 | 9 | 18 |
| MS-LMB baseline | 6 | 7 | 13 |
| MS-LMB variants | **7** (1 each + 3 fusion) | **8** (1 each + 3 fusion + summary) | 15 |
| MS-LMBM | 5 | 8 | 13 |
| Infra/Utils | 17 | 20 | 37 |
| **Total** | **52** | **64** | **116** |

Current: 85 + 210 = 295 tests
Recommended: ~116 tests (60% reduction)

### Immediate Action (Minimum Viable) ✅ DONE

Added 3 fusion tests to Rust. Everything else is optional optimization.
