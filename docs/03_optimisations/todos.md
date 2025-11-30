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

## Completed Work

_(Items move here as they are completed)_

---

## Validation Checklist

After each phase:
- [ ] `cargo test --release` - all 150+ tests pass
- [ ] `cargo check` - no new warnings
- [ ] Update this file with completion status
- [ ] Update `changelog.md` with summary
