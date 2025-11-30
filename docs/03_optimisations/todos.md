# Code Quality Refactoring TODOs

This document tracks the progress of code deduplication and quality improvements.

**Goal**: Deduplicate code, introduce traits and common functions across all algorithms while maintaining 100% MATLAB equivalence.

---

## Current Status

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| Phase 1 | LBP Refactoring | ✅ Complete | Extracted shared message-passing logic |
| Phase 2 | Common Utilities | ✅ Complete | robust_inverse, log_sum_exp, normalize_log_weights |
| Phase 3 | Likelihood Helpers | ⏳ Pending | Kalman gain, innovation params |
| Phase 4 | Prediction Trait | ⏳ Pending | BernoulliPrediction trait |

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

## Phase 3: Likelihood Helpers

**Files**:
- `src/common/linalg.rs` (add helpers)
- `src/lmb/association.rs` (refactor)
- `src/lmbm/association.rs` (refactor)
- `src/multisensor_lmb/association.rs` (refactor)
- `src/multisensor_lmbm/association.rs` (refactor)

**Problem**: All four association files compute log-likelihood ratios with nearly identical code.

**Tasks**:
- [ ] Add `compute_innovation_params()` function
- [ ] Add `log_gaussian_normalizing_constant()` function
- [ ] Add `compute_kalman_gain()` function
- [ ] Add `compute_log_likelihood()` function
- [ ] Refactor `lmb/association.rs`
- [ ] Refactor `lmbm/association.rs`
- [ ] Refactor `multisensor_lmb/association.rs`
- [ ] Refactor `multisensor_lmbm/association.rs`
- [ ] Run all tests

**Expected outcome**: ~50-70 lines saved per file

---

## Phase 4: Prediction Trait

**Files**:
- `src/common/prediction.rs` (new file)
- `src/lmb/prediction.rs` (refactor)
- `src/lmbm/prediction.rs` (refactor)

**Problem**: Nearly identical Chapman-Kolmogorov prediction logic in LMB and LMBM.

**Tasks**:
- [ ] Create `BernoulliPrediction` trait
- [ ] Implement trait for `Object`
- [ ] Implement trait for `Hypothesis`
- [ ] Refactor `lmb/prediction.rs`
- [ ] Refactor `lmbm/prediction.rs`
- [ ] Run all tests

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
