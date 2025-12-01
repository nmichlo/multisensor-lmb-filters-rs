# Code Quality Refactoring TODOs

This document tracks the progress of code deduplication and quality improvements.

**Goal**: Deduplicate code, introduce traits and common functions across all algorithms while maintaining 100% MATLAB equivalence.

---

## Current Status

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| Phase 1 | LBP Refactoring | ✅ Complete | Extracted shared message-passing logic |
| Phase 2 | Common Utilities | ✅ Complete | robust_inverse, log_sum_exp, normalize_log_weights |
| Phase 3 | Likelihood Helpers | ✅ Complete | Refactored lmb + lmbm association |
| Phase 4 | Prediction Helpers | ✅ Complete | Shared prediction functions |
| Phase 5 | Multisensor LMB Deduplication | ✅ Complete | Extracted shared utils for gating, MAP, trajectories |
| Phase 6 | Multisensor LMBM Deduplication | ✅ Complete | determine_linear_index() moved to mod.rs |
| Phase 7 | Multisensor Association Helpers | ✅ Complete | Using robust_inverse(), log_gaussian_normalizing_constant() |
| Phase 8 | Common Utility Functions | ✅ Complete | update_existence_missed_detection() helper |
| Phase 9 | Track Merging Refactoring | ✅ Complete | Using robust_inverse() in GA/PU functions |
| Phase 10 | Model Accessor Methods | ✅ Complete | Centralizing Option handling for multisensor params |
| Phase 11 | Function Extraction | ✅ Complete | Added canonical form helpers to linalg.rs |
| Phase 12 | Remaining Deduplication | ✅ Complete | Analysis: minimal remaining opportunities |
| Phase 13 | API Standardization | ✅ Complete | Reviewed: naming matches MATLAB, docs comprehensive |
| Phase 14 | Documentation & Constants | ✅ Complete | Added constants.rs module |
| Phase 15 | Helper Function Adoption | ✅ Complete | ground_truth.rs, merging.rs updated |

---

## Phase 1: LBP Refactoring ✅ COMPLETE

**File**: `src/common/association/lbp.rs`

**Problem**: `loopy_belief_propagation` and `fixed_loopy_belief_propagation` share ~80% identical code.

**Tasks**:
- [x] Extract `lbp_message_passing_iteration()` inner function
- [x] Extract `compute_lbp_result()` function
- [x] Update `loopy_belief_propagation` to use shared code
- [x] Update `fixed_loopy_belief_propagation` to use shared code
- [x] Run all tests to verify MATLAB equivalence

**Outcome**:
- Original: 231 lines (two functions with 80% duplication)
- Refactored: 212 lines (two helper functions + two thin wrappers)
- Lines saved: ~19 lines
- Code duplication: 80% → 0%
- All 79 unit tests + 150+ integration tests pass

---

## Phase 2: Common Utilities ✅ COMPLETE

**File**: `src/common/linalg.rs`

**Problem**: Multiple files have similar Cholesky-with-fallback and log-sum-exp patterns.

**Tasks**:
- [x] Add `robust_inverse()` function
- [x] Add `robust_solve()` function
- [x] Add `robust_solve_vec()` function
- [x] `log_sum_exp()` already existed - added tests
- [x] `normalize_log_weights()` already existed - added tests
- [x] Run all tests

**Outcome**:
- Added 3 new utility functions for robust matrix operations
- Added 10 unit tests for linalg utilities
- All 89+ unit tests + 150+ integration tests pass
- Functions available for future refactoring of association files

---

## Phase 3: Likelihood Helpers ✅ COMPLETE

**Files**:
- `src/common/linalg.rs` (add helpers)
- `src/lmb/association.rs` (refactor)
- `src/lmbm/association.rs` (refactor)

**Problem**: Association files compute log-likelihood ratios with nearly identical code.

**Tasks**:
- [x] Add `compute_innovation_params()` function
- [x] Add `log_gaussian_normalizing_constant()` function
- [x] Add `compute_kalman_gain()` function
- [x] Add `compute_measurement_log_likelihood()` function
- [x] Add `compute_kalman_updated_mean()` function
- [x] Refactor `lmb/association.rs`
- [x] Refactor `lmbm/association.rs`
- [x] Run all tests

**Outcome**:
- Added 5 new likelihood helper functions to `linalg.rs`
- Added 5 unit tests for the helpers
- Refactored `lmb/association.rs`: ~49 → ~44 lines in core loop
- Refactored `lmbm/association.rs`: ~69 → ~38 lines in core loop (significant simplification)
- Eliminated ~50 lines of duplicated Cholesky/SVD fallback code
- All 170+ tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Note**: Multi-sensor association files (`multisensor_lmb/association.rs`, `multisensor_lmbm/association.rs`) have more complex per-sensor logic and would require different helper signatures. Left for future work if needed.

---

## Phase 4: Prediction Helpers ✅ COMPLETE

**Files**:
- `src/common/linalg.rs` (add helpers)
- `src/lmb/prediction.rs` (refactor)
- `src/lmbm/prediction.rs` (refactor)

**Problem**: Nearly identical Chapman-Kolmogorov prediction logic in LMB and LMBM.

**Tasks**:
- [x] Add `predict_mean()` function
- [x] Add `predict_covariance()` function
- [x] Add `predict_existence()` function
- [x] Refactor `lmb/prediction.rs`
- [x] Refactor `lmbm/prediction.rs`
- [x] Run all tests

**Outcome**:
- Added 3 new prediction helper functions to `linalg.rs`
- Added 3 unit tests for the helpers
- Refactored both prediction files to use shared helpers
- More readable code: `predict_mean(&mu, &a, &u)` vs `&a * &mu + &u`
- All 170+ tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Note**: Originally planned as a trait-based solution, but helper functions provide the same deduplication benefit with less API disruption. The data structures (`Object` vs `Hypothesis`) remain unchanged.

---

## Phase 5: Multisensor LMB Deduplication ✅ COMPLETE

**Files**:
- `src/multisensor_lmb/parallel_update.rs`
- `src/multisensor_lmb/iterated_corrector.rs`
- `src/multisensor_lmb/utils.rs` (NEW)

**Problem**: `parallel_update.rs` and `iterated_corrector.rs` shared ~100 lines of identical code.

**Tasks**:
- [x] Create `src/multisensor_lmb/utils.rs`
- [x] Extract `gate_and_export_tracks()` - track gating and trajectory export
- [x] Extract `extract_map_state_estimates()` - MAP cardinality extraction
- [x] Extract `update_existence_no_measurements_sensor()` - no-measurement update
- [x] Extract `update_object_trajectories()` - trajectory length/timestamp updates
- [x] Extract `export_remaining_trajectories()` - final trajectory export
- [x] Refactor `parallel_update.rs` to use utils
- [x] Refactor `iterated_corrector.rs` to use utils
- [x] Run all tests (175 pass)

**Outcome**:
- Created `utils.rs` with 5 shared utility functions
- Added 4 unit tests for the utilities
- Both filter files now use shared code for gating, MAP extraction, and trajectories
- Data association code left separate (implementations differ intentionally)
- All 175 tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Note**: Data association switch was NOT extracted because the implementations differ significantly between parallel_update.rs (complex Murty) and iterated_corrector.rs (simple Murty).

---

## Phase 6: Multisensor LMBM Deduplication ✅ COMPLETE

**Files**:
- `src/multisensor_lmbm/gibbs.rs`
- `src/multisensor_lmbm/hypothesis.rs`
- `src/multisensor_lmbm/mod.rs`

**Problem**: `determine_linear_index()` function was exactly duplicated (27 lines each including docs and test).

**Tasks**:
- [x] Move `determine_linear_index()` to `mod.rs`
- [x] Update `gibbs.rs` to import from mod
- [x] Update `hypothesis.rs` to import from mod
- [x] Run all tests (175 pass)

**Outcome**:
- Moved function to `mod.rs` with single test
- Both files now import from parent module
- Eliminated ~50 lines of duplicated code (function + docs + test in each file)
- All 175 tests pass with unchanged tolerances
- MATLAB equivalence maintained

---

## Phase 7: Multisensor Association → Use linalg.rs Helpers ✅ COMPLETE

**Files**:
- `src/multisensor_lmb/association.rs`
- `src/multisensor_lmbm/association.rs`

**Problem**: Both files manually implemented Cholesky/SVD fallback instead of using centralized helpers.

**Tasks**:
- [x] Refactor `multisensor_lmb/association.rs`:
  - [x] Use `robust_inverse()` instead of manual fallback chain
  - [x] Use `log_gaussian_normalizing_constant()` for normalizing constant
- [x] Refactor `multisensor_lmbm/association.rs`:
  - [x] Use `robust_inverse()` instead of manual fallback chain
  - [x] Use `log_gaussian_normalizing_constant()` for normalizing constant
- [x] Run all tests (175 pass)

**Outcome**:
- Replaced ~15 lines of manual Cholesky fallback per file
- Both files now use centralized helpers from `linalg.rs`
- All 175 tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Note**: `compute_measurement_log_likelihood()` was not used because the multisensor likelihood computation is more complex (stacked measurements, block diagonal Q matrices).

---

## Phase 8: Common Utility Functions ✅ COMPLETE

**Files**:
- `src/common/utils.rs` (added helper)
- `src/lmb/update.rs` (refactored)
- `src/multisensor_lmb/utils.rs` (refactored)

**Problem**: Existence update formula duplicated across filters.

**Tasks**:
- [x] Add `update_existence_missed_detection()` to `common/utils.rs`
- [x] Refactor `lmb/update.rs::update_no_measurements` to use helper
- [x] Refactor `multisensor_lmb/utils.rs::update_existence_no_measurements_sensor` to use helper
- [x] Run all tests (176 pass - +1 for new helper test)

**Outcome**:
- Added `update_existence_missed_detection()` scalar helper function
- Formula `r' = r*(1-p_d) / (1 - r*p_d)` now centralized
- Added 1 unit test for the helper
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Note**: Trajectory update helper was not added as it would provide minimal deduplication benefit - the trajectory logic differs between single-sensor (Object struct) and multi-sensor (Trajectory struct) contexts.

---

## Phase 9: Track Merging Refactoring ✅ COMPLETE

**Files**:
- `src/multisensor_lmb/merging.rs`

**Problem**: GA and PU merging functions had manual matrix inversion fallback chains.

**Tasks**:
- [x] Analyze merging functions for shared patterns
- [x] Replace manual inversions in `ga_lmb_track_merging()` with `robust_inverse()`
- [x] Replace manual inversions in `pu_lmb_track_merging()` with `robust_inverse()`
- [x] Run all tests

**Outcome**:
- Replaced 5 manual matrix inversion fallback chains with `robust_inverse()`
- Lines saved: ~25 (5 fallback chains × 5 lines each)
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Note**: The original plan to extract shared loop structure was not implemented because AA, GA, and PU merging algorithms are fundamentally different:
- AA: Concatenates weighted GM components
- GA: Fuses in canonical form with moment matching
- PU: Information form fusion with decorrelation
The only shared code was the matrix inversion fallback logic, which is now centralized via `robust_inverse()`.

---

## Phase 10: Model Accessor Methods ✅ COMPLETE

**File**: `src/common/types.rs` (Model impl block)

**Problem**: Scattered `Option` handling for multi-sensor parameters throughout codebase:
```rust
let p_d = model.detection_probability_multisensor.as_ref()
    .map(|v| v[sensor_idx])
    .unwrap_or(model.detection_probability);
```

**Tasks**:
- [x] Add `get_detection_probability(sensor_idx)` accessor
- [x] Add `get_observation_matrix(sensor_idx)` accessor
- [x] Add `get_measurement_noise(sensor_idx)` accessor
- [x] Add `get_clutter_rate(sensor_idx)` accessor
- [x] Add `get_clutter_per_unit_volume(sensor_idx)` accessor
- [x] Add `is_multisensor()` helper
- [x] Update `multisensor_lmb/association.rs` to use accessors
- [x] Update `multisensor_lmbm/association.rs` to use accessors
- [x] Update `multisensor_lmb/utils.rs` to use accessors
- [x] Run all tests

**Outcome**:
- Added 6 accessor methods to Model impl block
- Updated 3 files to use accessor methods instead of manual Option handling
- All 175+ tests pass with unchanged tolerances
- MATLAB equivalence maintained

---

## Phase 11: Function Extraction ✅ COMPLETE

**Goal**: Extract well-named helper functions from large functions/loops to improve readability.

**Files**:
- `src/common/linalg.rs` (added canonical form helpers)

**Tasks**:
- [x] Add `CanonicalGaussian` struct for information form representation
- [x] Add `to_canonical_form()` - converts (mu, sigma, weight) to (K, h, g)
- [x] Add `from_canonical_form()` - converts (K, h, g) back to (mu, sigma, g_out)
- [x] Add `to_weighted_canonical_form()` - for weighted fusion (GA merging)
- [x] Add 4 unit tests for canonical form helpers
- [x] Run all tests (175+ pass)

**Outcome**:
- Added `CanonicalGaussian` struct and 3 conversion functions to `linalg.rs`
- Added 4 unit tests for the helpers
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Analysis of other planned extractions**:
- `process_sensor_measurement()`: Would only be called from one place, provides minimal benefit
- `compute_marginal_association_probabilities()`: Murty implementations differ significantly between PU (complex with W_indicator/J matrices) and IC (simple marginalization), so deduplication not feasible
- `update_object_trajectory()`: Already exists as `update_object_trajectories()` in `multisensor_lmb/utils.rs`; LMB and LMBM filters have different struct types (Object vs Trajectory) requiring different logic

**Note**: The canonical form helpers are available for future refactoring of merging.rs GA/PU functions, but directly refactoring those functions risks breaking MATLAB numerical equivalence due to subtle differences in the existing implementations.

---

## Phase 12: Remaining Deduplication ✅ COMPLETE

**Problem**: Data association dispatch partially duplicated between PU (~120 lines) and IC (~70 lines).

**Analysis**:
- LBP and Gibbs branches are identical between PU and IC (~15 lines each)
- Murty branch differs significantly (PU has complex W_indicator/J matrix handling vs IC's simple marginalization)
- Extraction would only partially deduplicate and may reduce readability

**Decision**: After analysis, determined that remaining deduplication opportunities are minimal:
- LBP/Gibbs extraction would save ~30 lines but reduce code locality and readability
- Murty implementations are fundamentally different and cannot be unified
- The canonical form helpers (added in Phase 11) represent the most valuable extraction

**Outcome**:
- Phase analyzed and documented
- No further extraction performed (cost/benefit analysis shows minimal value)
- Codebase already well-deduplicated from Phases 1-11

---

## Phase 13: API Standardization ✅ COMPLETE

**Goal**: Consistent naming, parameter order, and return types.

**Analysis**:
- Function names already follow a consistent pattern: `{filter}_{operation}_{step}` (e.g., `lmb_prediction_step`)
- Names match MATLAB function names for cross-reference (e.g., `lmb_prediction_step` → `lmbPredictionStep.m`)
- MATLAB reference comments already present in 26+ source files
- Parameter order is already consistent within each filter type

**Decision**: After review, determined that function renaming would be counterproductive:
- Current names match MATLAB conventions, aiding cross-reference
- Renaming would break API compatibility for any users of the library
- Deprecation aliases would add complexity without clear benefit
- The risk of breaking MATLAB numerical equivalence through refactoring outweighs benefits

**Documentation coverage verified**:
- All algorithm files have "Matches MATLAB xxx.m exactly" comments
- Files without MATLAB references are module re-exports and utility functions

**Outcome**:
- API naming reviewed and documented as intentionally matching MATLAB
- MATLAB reference documentation already comprehensive
- No API changes made (risk/benefit analysis shows renaming has higher cost than value)

---

## Phase 14: Documentation & Constants ✅ COMPLETE

**Tasks**:
- [x] Create `src/common/constants.rs` with magic numbers:
  - `EPSILON_EXISTENCE: f64 = 1e-15`
  - `ESF_ADJUSTMENT: f64 = 1e-6`
  - `SVD_TOLERANCE: f64 = 1e-10`
  - `DEFAULT_LBP_TOLERANCE: f64 = 1e-6`
  - `DEFAULT_GM_WEIGHT_THRESHOLD: f64 = 1e-6`
- [x] Updated `src/lmb/cardinality.rs` to use `EPSILON_EXISTENCE` and `ESF_ADJUSTMENT`
- [x] Updated `src/common/linalg.rs` to use `SVD_TOLERANCE`
- [x] MATLAB reference comments already comprehensive (verified in Phase 13)
- [x] Run all tests (175+ pass)

**Outcome**:
- Created `constants.rs` module with 5 named constants
- Partially adopted constants in key files (`cardinality.rs`, `linalg.rs`)
- Remaining files use hardcoded values but are documented
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Note**: Many source files still use hardcoded `1e-15` and `1e-6` values. Full adoption is lower priority as the current values match MATLAB exactly. Constants module provides a reference for future updates.

---

## Phase 15: Helper Function Adoption ✅ COMPLETE

**Goal**: Ensure all helper functions from Phase 10-11 are actually being used throughout the codebase.

**Files Updated**:
- `src/common/ground_truth.rs` - Updated to use Model accessors (`get_clutter_rate`, `get_detection_probability`, `get_measurement_noise`, `get_observation_matrix`)
- `src/multisensor_lmb/merging.rs` - Updated GA and PU merging to use canonical form helpers

**Tasks**:
- [x] Audit codebase for places still using old patterns
- [x] Update `ground_truth.rs` to use Model accessors (Phase 10)
- [x] Update GA merging to use `to_weighted_canonical_form()` and `from_canonical_form()` (Phase 11)
- [x] Update PU merging to use `to_canonical_form()` and `from_canonical_form()` (Phase 11)
- [x] Run all tests (175+ pass)

**Outcome**:
- `ground_truth.rs`: Removed 4 instances of `.as_ref().unwrap()` pattern
- `merging.rs`: GA function now uses canonical form helpers (~15 lines cleaner)
- `merging.rs`: PU function now uses canonical form helpers (~20 lines cleaner)
- Removed unused `robust_inverse` import from `merging.rs`
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

---

## Completed Work

_(Items move here as they are completed)_

---

---

## Performance Optimization (Phase 16+)

**Goal**: Optimize LMBM algorithm performance while maintaining MATLAB numerical equivalence.

### Benchmark Results (LMBM Filter)

| Optimization | Time | Change | Notes |
|--------------|------|--------|-------|
| **Baseline** | 23.14s | - | 10.7M likelihood calls |
| + LTO/Codegen | 23.17s | ~0% | Expected - bottleneck is algorithmic |
| + Q Matrix Cache + #[inline] | 22.15s | -4.3% | Avoids 10.7M Q/C matrix clones |
| + robust_inverse_with_log_det | 20.91s | -9.6% | Single Cholesky for inverse + log-det |
| + Stack-allocated indices | 20.20s | -12.7% | Avoids 21.4M Vec allocations |
| + mimalloc allocator | 11.88s → 9.32s | -21.5% | Feature-gated custom allocator |
| **+ rayon parallelization** | 9.32s → **2.81s** | **3.3x faster** | Feature-gated parallel loop |

**Total improvement: 23.14s → 2.81s (8.2x faster)**

### Access Pattern Analysis (gibbs-trace feature)

Access pattern instrumentation confirms lazy likelihood viability:
- **Typical access ratio**: 5-17% of total entries
- **Larger matrices**: as low as 3-6% access
- **Potential savings**: 83-95% of likelihood computations

This validates that the 10.7M upfront likelihood computations could be reduced to ~500K-1.8M on-demand computations.

### Completed Optimizations

- [x] **LTO/Codegen** (Cargo.toml): `lto = "thin"`, `codegen-units = 1`, `opt-level = 3`
- [x] **Q Matrix Cache**: Pre-cache Q and C matrices per sensor at top of `generate_multisensor_lmbm_association_matrices()` (avoids 10.7M clones)
- [x] **#[inline] annotations**: Added to `convert_from_linear_to_cartesian_inplace()` and `determine_log_likelihood_ratio()`
- [x] **robust_inverse_with_log_det()**: Combined inverse + log-det computation in single Cholesky decomposition
- [x] **Stack-allocated indices**: Replaced Vec allocations with `[usize; MAX_SENSORS]` in index conversion loop (avoids 21.4M heap allocations)
- [x] **mimalloc allocator**: Feature-gated custom allocator (21.5% speedup)
- [x] **gibbs-trace instrumentation**: Access pattern tracing to validate lazy likelihood (5-17% access ratio confirmed)
- [x] **rayon parallelization**: Feature-gated parallel loop (3.3x speedup)

### Pending Optimizations (Quick Wins)

_(All quick wins completed)_

### Future Optimizations (Higher Effort)

- [ ] **Lazy Likelihood** (Expected: 5-20x based on access analysis - **HIGHEST PRIORITY**)
  - Compute likelihoods on-demand during Gibbs sampling instead of upfront
  - Access pattern analysis shows only 5-17% of entries accessed
  - Would reduce 10.7M computations to ~500K-1.8M
  - Requires architectural change: closure-based or HashMap-cached approach

- [ ] **Rayon Parallelization** (Expected: 3-8x on multi-core)
  - Parallelize likelihood computation across entries
  - Can combine with lazy likelihood for maximum benefit

- [ ] **Workspace buffer reuse** (Expected: modest improvement)
  - Reuse allocated buffers across likelihood computations
  - Less impactful now that mimalloc handles allocation overhead

- [ ] **In-place matrix operations** (Expected: 1.5-2x)
  - Replace temporaries with in-place GEMV operations

### Files Modified

- `Cargo.toml`: Added release profile optimizations, `mimalloc`, `gibbs-trace`, and `rayon` features
- `src/lib.rs`: Added feature-gated mimalloc global allocator
- `src/common/linalg.rs`: Added `robust_inverse_with_log_det()`
- `src/multisensor_lmbm/association.rs`:
  - Q/C matrix caching
  - Combined inverse+log-det via `robust_inverse_with_log_det()`
  - Stack-allocated index arrays (`MAX_SENSORS` constant, `convert_from_linear_to_cartesian_inplace()`)
  - `#[inline]` annotations on hot functions
  - **Rayon parallel loop** (`into_par_iter()` with cfg feature gate)
- `src/multisensor_lmbm/gibbs.rs`: Added access pattern tracing (gibbs-trace feature)
- `src/multisensor_lmbm/filter.rs`: Integrated tracing hooks around Gibbs sampling
- `src/multisensor_lmbm/mod.rs`: Exported tracing functions

---

## Validation Checklist

After each phase:
- [ ] `cargo test --release` - all 150+ tests pass
- [ ] `cargo check` - no new warnings
- [ ] Update this file with completion status
- [ ] Update `changelog.md` with summary
