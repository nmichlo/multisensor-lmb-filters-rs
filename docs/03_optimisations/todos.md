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
| Phase 9 | Track Merging Refactoring | ⏳ Pending | AA/GA/PU share ~60% loop structure |

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

## Phase 9: Track Merging Refactoring ⏳ Pending

**Files**:
- `src/multisensor_lmb/merging.rs`

**Problem**: Three merging functions (AA, GA, PU) share ~60% identical loop structure for:
- Weighted sum of existence probabilities
- GM component concatenation
- Weight sorting and capping
- Renormalization

**Tasks**:
- [ ] Identify common merge loop pattern
- [ ] Create `merge_sensor_distributions_core()` helper or similar
- [ ] Refactor `aa_lmb_track_merging()` to use shared logic
- [ ] Refactor `ga_lmb_track_merging()` to use shared logic
- [ ] Refactor `pu_lmb_track_merging()` to use shared logic
- [ ] Run all tests

**Expected savings**: ~100 lines

---

## Completed Work

_(Items move here as they are completed)_

---

## Validation Checklist

After each phase:
- [ ] `cargo test --release` - all 150+ tests pass
- [ ] `cargo check` - no new warnings
- [ ] Update this file with completion status
- [ ] Update `changelog.md` with summary
