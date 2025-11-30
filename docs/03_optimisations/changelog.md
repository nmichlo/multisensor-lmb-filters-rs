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

---

## 2025-11-30: Phase 4 - Prediction Helpers

**Files**:
- `src/common/linalg.rs` (new helpers)
- `src/lmb/prediction.rs` (refactored)
- `src/lmbm/prediction.rs` (refactored)

**Changes**:
- Added `predict_mean()` - computes mu' = A * mu + u
- Added `predict_covariance()` - computes Sigma' = A * Sigma * A' + R
- Added `predict_existence()` - computes r' = p_s * r
- Refactored `lmb/prediction.rs` to use helpers
- Refactored `lmbm/prediction.rs` to use helpers
- Added 3 unit tests for prediction helpers

**Impact**:
- Unit tests: 94 → 97 (+3)
- Both prediction files now use identical helper functions
- More readable and self-documenting code
- All 170+ tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: Both LMB and LMBM prediction steps implement the same Chapman-Kolmogorov equations for linear Gaussian motion models. While the data structures differ (`Vec<Object>` vs `Hypothesis`), the core math is identical. Helper functions provide semantic clarity and ensure consistency.

---

## 2025-11-30: Phase 5 - Multisensor LMB Deduplication

**Files**:
- `src/multisensor_lmb/utils.rs` (NEW)
- `src/multisensor_lmb/parallel_update.rs` (refactored)
- `src/multisensor_lmb/iterated_corrector.rs` (refactored)

**Changes**:
- Created `utils.rs` with 5 shared utility functions:
  - `update_existence_no_measurements_sensor()` - missed detection update
  - `gate_and_export_tracks()` - existence gating and trajectory export
  - `extract_map_state_estimates()` - MAP cardinality extraction
  - `update_object_trajectories()` - trajectory length/timestamp updates
  - `export_remaining_trajectories()` - final trajectory export
- Refactored `parallel_update.rs` to use shared utilities
- Refactored `iterated_corrector.rs` to use shared utilities
- Added 4 unit tests for the utility functions

**Impact**:
- Unit tests: 97 → 101 (+4)
- Both multi-sensor filter files now share gating, MAP, and trajectory code
- `parallel_update.rs`: ~470 → ~410 lines (~60 lines saved)
- `iterated_corrector.rs`: ~270 → ~180 lines (~90 lines saved)
- All 175 tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: The parallel_update and iterated_corrector LMB filters shared significant duplicated code for track gating, MAP state extraction, and trajectory management. By extracting these into a shared utils module, we reduced maintenance burden and ensured consistent behavior. Data association code was intentionally NOT shared because the implementations differ (parallel_update has more complex Murty handling).

---

## 2025-11-30: Phase 6 - Multisensor LMBM Deduplication

**Files**:
- `src/multisensor_lmbm/mod.rs` (added shared function)
- `src/multisensor_lmbm/gibbs.rs` (refactored)
- `src/multisensor_lmbm/hypothesis.rs` (refactored)

**Changes**:
- Moved `determine_linear_index()` function to `mod.rs`
- Updated `gibbs.rs` to import from parent module
- Updated `hypothesis.rs` to import from parent module
- Consolidated duplicate tests into single test in `mod.rs`

**Impact**:
- Lines saved: ~50 (27 per file: function + docs + test)
- Both LMBM modules now share the index computation logic
- All 175 tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: The `determine_linear_index()` function was exactly duplicated in both gibbs.rs and hypothesis.rs, converting multi-dimensional MATLAB-style indices to linear array indices. Moving it to the parent module eliminates the duplication while keeping it accessible to both files.

---

## 2025-11-30: Phase 7 - Multisensor Association Helpers

**Files**:
- `src/multisensor_lmb/association.rs` (refactored)
- `src/multisensor_lmbm/association.rs` (refactored)

**Changes**:
- Refactored `multisensor_lmb/association.rs`:
  - Use `robust_inverse()` instead of manual Cholesky → try_inverse → SVD chain
  - Use `log_gaussian_normalizing_constant()` for computing log normalizing constant
- Refactored `multisensor_lmbm/association.rs`:
  - Use `robust_inverse()` instead of manual try_inverse → SVD chain
  - Use `log_gaussian_normalizing_constant()` for computing eta

**Impact**:
- Lines saved: ~30 (15 per file)
- Both multisensor association files now use centralized linalg helpers
- All 175 tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: The multisensor association files had manually implemented matrix inversion fallbacks and Gaussian normalizing constant computation. By using the centralized helpers from `linalg.rs`, we ensure consistent behavior and reduce maintenance burden. The full `compute_measurement_log_likelihood()` helper was not applicable because multisensor likelihood computation involves stacked measurement vectors and block diagonal noise covariances.

---

## 2025-11-30: Phase 8 - Common Utility Functions

**Files**:
- `src/common/utils.rs` (added helper)
- `src/lmb/update.rs` (refactored)
- `src/multisensor_lmb/utils.rs` (refactored)

**Changes**:
- Added `update_existence_missed_detection()` helper function to `common/utils.rs`
- Refactored `lmb/update.rs::update_no_measurements()` to use the helper
- Refactored `multisensor_lmb/utils.rs::update_existence_no_measurements_sensor()` to use the helper
- Added 1 unit test for the helper function

**Impact**:
- Unit tests: 101 → 102 (+1)
- Centralized the missed detection formula: r' = r*(1-p_d) / (1 - r*p_d)
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: The missed detection existence update formula was duplicated across multiple files. By centralizing it in `common/utils.rs`, we ensure consistent behavior and reduce maintenance burden. Trajectory update helpers were not added as the logic differs significantly between single-sensor and multi-sensor contexts.

---

## 2025-11-30: Phase 9 - Track Merging Refactoring

**File**: `src/multisensor_lmb/merging.rs`

**Changes**:
- Replaced 5 manual matrix inversion fallback chains with `robust_inverse()`:
  - `ga_lmb_track_merging()`: 2 locations (t_inv, sigma_ga)
  - `pu_lmb_track_merging()`: 3 locations (k_prior, k_c, sigma)
- Removed ~25 lines of duplicated fallback logic

**Impact**:
- Lines saved: ~25
- Both GA and PU merging now use centralized `robust_inverse()` from `linalg.rs`
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: The GA and PU track merging functions had manual matrix inversion fallback chains (try_inverse → cholesky → SVD pseudo-inverse). By using the centralized `robust_inverse()` helper, we ensure consistent fallback behavior and reduce maintenance burden. The AA merging function was not modified as it doesn't require matrix inversion.
