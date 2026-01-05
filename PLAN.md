# Fixture Coverage & Test Equivalence Plan

**IMPORTANT**: This file MUST be updated when tests are added/modified. Coverage matrix is the source of truth.

## Current Status

**Test Counts**: 120+ tests total (56 Rust + 65+ Python)

### Coverage Summary

| Filter Type | Step-by-Step Fixture | Python Tests | Rust Tests | Status |
|-------------|---------------------|--------------|------------|--------|
| LMB | Full (7 steps) | Comprehensive | Comprehensive | **COMPLETE** |
| LMBM | Full (7 steps) | Comprehensive | Comprehensive | **COMPLETE** |
| AA-LMB | None | Basic only | Basic only | **NEEDS WORK** |
| GA-LMB | None | Basic only | Basic only | **NEEDS WORK** |
| PU-LMB | None | Basic only | Basic only | **NEEDS WORK** |
| IC-LMB | Partial (per-sensor) | Per-sensor only | Per-sensor only | **NEEDS WORK** |
| MS-LMBM | Full (6 steps) | Partial (missing w,mu,Sigma) | Partial | **NEEDS WORK** |

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
