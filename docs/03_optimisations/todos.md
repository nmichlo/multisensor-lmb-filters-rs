# Code Quality Refactoring TODOs

This document tracks the progress of code deduplication and quality improvements.

**Goal**: Deduplicate code, introduce traits and common functions across all algorithms while maintaining 100% MATLAB equivalence.

---

## Current Status

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| Phase 1 | LBP Refactoring | 🔄 In Progress | Extract shared message-passing logic |
| Phase 2 | Common Utilities | ⏳ Pending | robust_inverse, log_sum_exp |
| Phase 3 | Likelihood Helpers | ⏳ Pending | Kalman gain, innovation params |
| Phase 4 | Prediction Trait | ⏳ Pending | BernoulliPrediction trait |

---

## Phase 1: LBP Refactoring

**File**: `src/common/association/lbp.rs`

**Problem**: `loopy_belief_propagation` and `fixed_loopy_belief_propagation` share ~80% identical code.

**Tasks**:
- [ ] Extract `lbp_message_passing_iteration()` inner function
- [ ] Extract `compute_lbp_result()` function
- [ ] Update `loopy_belief_propagation` to use shared code
- [ ] Update `fixed_loopy_belief_propagation` to use shared code
- [ ] Run all tests to verify MATLAB equivalence

**Expected outcome**: ~40 lines saved, 0% duplication in LBP module

---

## Phase 2: Common Utilities

**File**: `src/common/linalg.rs`

**Problem**: Multiple files have similar Cholesky-with-fallback and log-sum-exp patterns.

**Tasks**:
- [ ] Add `robust_inverse()` function
- [ ] Add `robust_solve()` function
- [ ] Add `log_sum_exp()` function
- [ ] Add `normalize_log_weights()` function
- [ ] Run all tests

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
