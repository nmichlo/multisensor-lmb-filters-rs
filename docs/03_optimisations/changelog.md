# Optimisation Changelog

This document tracks code quality and performance improvements to the codebase.

---

## 2025-11-30: Phase 1 - LBP Refactoring

**File**: `src/common/association/lbp.rs`

**Changes**:
- Extracted `lbp_message_passing_iteration()` helper function (41 lines)
- Extracted `compute_lbp_result()` helper function (44 lines)
- Simplified `loopy_belief_propagation()` to use shared helpers
- Simplified `fixed_loopy_belief_propagation()` to use shared helpers

**Impact**:
- Code duplication: 80% → 0%
- Lines: 231 → 212 (19 lines saved)
- All 150+ tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: The two public LBP functions were nearly identical, differing only in their iteration strategy (convergence-based vs fixed count). By extracting the shared message-passing and result computation logic, we eliminated duplication while preserving the exact numerical behavior required for MATLAB equivalence.

---

## 2025-11-30: Phase 2 - Common Utilities

**File**: `src/common/linalg.rs`

**Changes**:
- Added `robust_inverse()` - Cholesky → LU → SVD fallback chain
- Added `robust_solve()` - Matrix RHS version
- Added `robust_solve_vec()` - Vector RHS version
- Added 10 unit tests for linalg utilities (including existing `log_sum_exp` and `normalize_log_weights`)

**Impact**:
- Unit tests: 79 → 89 (+10)
- New utilities available for future association file refactoring
- All 150+ integration tests pass unchanged

**Rationale**: Multiple association files have similar Cholesky-with-fallback patterns. These utilities provide a standardized approach that can be incrementally adopted. The `log_sum_exp` and `normalize_log_weights` functions already existed but lacked tests.

---

## 2025-11-30: Phase 3 - Likelihood Helpers

**Files**:
- `src/common/linalg.rs` (new helpers)
- `src/lmb/association.rs` (refactored)
- `src/lmbm/association.rs` (refactored)

**Changes**:
- Added `compute_innovation_params()` - computes predicted measurement and innovation covariance
- Added `log_gaussian_normalizing_constant()` - computes log normalizing constant using Cholesky when possible
- Added `compute_kalman_gain()` - computes Kalman gain, updated covariance, and Z inverse with robust fallbacks
- Added `compute_measurement_log_likelihood()` - computes Gaussian log-likelihood for measurement
- Added `compute_kalman_updated_mean()` - computes Kalman-updated state mean
- Refactored `lmb/association.rs` to use new helpers
- Refactored `lmbm/association.rs` to use new helpers (removed ~30 lines of Cholesky/SVD fallback code)
- Added 5 unit tests for likelihood helpers

**Impact**:
- Unit tests: 89 → 94 (+5)
- `lmbm/association.rs` core loop: 69 → 38 lines (45% reduction)
- Eliminated duplicated Cholesky → LU → SVD fallback logic
- All 170+ tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: Both LMB and LMBM association files computed marginal likelihood ratios with nearly identical patterns: innovation covariance, log normalizing constant, Kalman gain, and measurement likelihood. By extracting these into reusable helpers, we reduced code duplication and centralized the robust matrix inversion fallback logic.
