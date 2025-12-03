# Optimisation Changelog

This document tracks code quality and performance improvements to the codebase.

---

## 2025-12-03: Filter Standardization & Code Quality

**Goal**: Standardize flow, deduplicate code, and improve API consistency across all 4 filter implementations.

### Phase 1: High Priority - Completed

| Change | Files Affected |
|--------|----------------|
| Standardized STEP 1-7 flow comments | `lmb.rs`, `lmbm.rs`, `multisensor_lmb.rs`, `multisensor_lmbm.rs` |
| Extract `update_existence_from_marginals()` | `common_ops.rs` (new), `lmb.rs`, `multisensor_lmb.rs` |
| Extract `predict_all_hypotheses()` | `common_ops.rs` (new), `lmbm.rs`, `multisensor_lmbm.rs` |
| Add `MultisensorConfig::z_dim()` | `types/config.rs`, `multisensor_lmb.rs`, `multisensor_lmbm.rs` |
| Fix MultisensorLMBM associator storage | `multisensor_lmbm.rs` (stores instance, not PhantomData) |

### Phase 1: Medium Priority - Completed

| Change | Files Affected |
|--------|----------------|
| Add input validation | `multisensor_lmb.rs` (matches `multisensor_lmbm.rs`) |
| Extract inline to private method | `lmb.rs` (`update_existence_no_measurements()`) |

### Phase 1: Low Priority - Completed

| Change | Files Affected |
|--------|----------------|
| Rename `with_associator()` → `with_associator_type()` | `multisensor_lmbm.rs` (API consistency) |
| Extract magic number to constant | `lmbm.rs` (`LOG_UNDERFLOW = -700.0`) |

### Phase 2: Magic Number Consolidation - Completed

| Change | Files Affected |
|--------|----------------|
| Add `NUMERICAL_ZERO` constant (1e-15) | `mod.rs`, replaced 9 inline uses in `common_ops.rs`, `traits.rs` |
| Add `UNDERFLOW_THRESHOLD` constant (1e-300) | `mod.rs`, replaced 2 inline uses in `lmbm.rs` |
| Document default associator choices | `lmb.rs` (why LBP), `lmbm.rs` (why Gibbs) |
| Document missing `with_gm_pruning()` | `lmbm.rs` (single-component tracks don't need GM pruning) |

### Impact

- All filters now have identical STEP 1-7 flow comments for easy cross-reference
- ~40 lines of duplicate code extracted to `common_ops.rs`
- Consistent constructor naming across all filters
- Input validation added to `multisensor_lmb.rs` (prevents silent failures)
- `z_dim()` standardized with documentation explaining original differences
- All magic numbers in filter module now use named constants
- Default associator choices documented (LMB→LBP, LMBM→Gibbs)
- All 114 tests pass, MATLAB equivalence maintained

### Documentation

- Updated plan file: `.claude/plans/linear-pondering-yeti.md`
- z_dim() documentation explains original implementations
- Default associator section added to LmbFilter and LmbmFilter docs
- Note about single-component tracks explains why LMBM lacks GM pruning

---

## 2025-12-02: New API & Legacy Cleanup

**Major Refactoring**: Replaced legacy function-based API with new trait-based API.

### New API Structure

```
src/
├── types/              # Core data types (Track, MotionModel, SensorModel)
├── components/         # Shared algorithms (prediction, update)
├── association/        # Data association (likelihood, builder)
├── filter/             # Filter implementations (LmbFilter, LmbmFilter, etc.)
├── common/             # Low-level utilities (kept: association/, linalg, rng)
└── lmb/cardinality.rs  # MAP cardinality estimation
```

### Deleted Legacy Modules

- `src/lmbm/` - entire module (replaced by `filter/lmbm.rs`)
- `src/multisensor_lmb/` - entire module (replaced by `filter/multisensor_lmb.rs`)
- `src/multisensor_lmbm/` - entire module (replaced by `filter/multisensor_lmbm.rs`)
- `src/lmb/filter.rs`, `prediction.rs`, `update.rs`, `association.rs`, `data_association.rs`
- `src/common/model.rs`, `types.rs`, `ground_truth.rs`, `metrics.rs`

### Deleted Tests & Examples

- 24 test files that depended on legacy modules
- 8 example files that used legacy API

### Impact

- Test count: 265+ → 145 tests (removed legacy-specific tests)
- All remaining tests pass
- MATLAB equivalence maintained at 1e-12 tolerance
- Clean trait-based API: `Filter`, `Associator`, `Merger`

### Notes

The optimization phases documented below (Phases 1-15) refer to the OLD legacy code that has now been deleted. They are kept here for historical reference only.

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

---

## 2025-12-01: Phase 10 - Model Accessor Methods

**File**: `src/common/types.rs`

**Changes**:
- Added `impl Model` block with 6 accessor methods:
  - `get_detection_probability(sensor_idx: Option<usize>)` - returns per-sensor or default
  - `get_observation_matrix(sensor_idx: Option<usize>)` - returns per-sensor C matrix
  - `get_measurement_noise(sensor_idx: Option<usize>)` - returns per-sensor Q matrix
  - `get_clutter_rate(sensor_idx: Option<usize>)` - returns per-sensor clutter rate
  - `get_clutter_per_unit_volume(sensor_idx: Option<usize>)` - returns per-sensor clutter density
  - `is_multisensor()` - checks if model is multi-sensor
- Updated `multisensor_lmb/association.rs` to use accessors (lines 83-88)
- Updated `multisensor_lmbm/association.rs` to use accessors throughout
- Updated `multisensor_lmb/utils.rs` to use accessor (line 26)

**Impact**:
- Centralized Option handling for multi-sensor parameters
- Reduced ~15 lines of repetitive Option unwrapping code
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: The codebase had repetitive patterns like `model.detection_probability_multisensor.as_ref().map(|v| v[s]).unwrap_or(model.detection_probability)` in multiple files. By centralizing this logic in accessor methods, we ensure consistent behavior and reduce maintenance burden. The accessors take `Option<usize>` to handle both single-sensor (None) and multi-sensor (Some(idx)) cases uniformly.

---

## 2025-12-01: Phase 11 - Canonical Form Helpers

**File**: `src/common/linalg.rs`

**Changes**:
- Added `CanonicalGaussian` struct for information/canonical form representation
- Added `to_canonical_form()` - converts (mu, sigma, weight) to (K, h, g)
- Added `from_canonical_form()` - converts (K, h, g) back to (mu, sigma, g_out)
- Added `to_weighted_canonical_form()` - for weighted fusion (GA merging)
- Added 4 unit tests for the canonical form helpers

**Impact**:
- Unit tests: 101 → 105 (+4)
- New utilities available for future track merging refactoring
- All 175+ tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: The GA and PU track merging algorithms in `merging.rs` convert between moment form (mu, sigma) and canonical/information form (K, h, g) for Gaussian fusion. By extracting these conversions to reusable helpers, we provide a standardized approach that can be adopted in future refactoring. The helpers use `robust_inverse()` for numerical stability.

**Note**: Other planned Phase 11 extractions were analyzed and determined to provide minimal benefit:
- `process_sensor_measurement()`: Would only be called from one place
- `compute_marginal_association_probabilities()`: Murty implementations differ significantly between PU and IC
- `update_object_trajectory()`: Already exists in `multisensor_lmb/utils.rs`

---

## 2025-12-01: Phase 14 - Constants Module

**File**: `src/common/constants.rs` (NEW)

**Changes**:
- Created `constants.rs` module with named constants:
  - `EPSILON_EXISTENCE: f64 = 1e-15` - threshold for near-zero existence probabilities
  - `ESF_ADJUSTMENT: f64 = 1e-6` - adjustment to avoid unit existence probabilities
  - `SVD_TOLERANCE: f64 = 1e-10` - tolerance for SVD pseudo-inverse
  - `DEFAULT_LBP_TOLERANCE: f64 = 1e-6` - default LBP convergence tolerance
  - `DEFAULT_GM_WEIGHT_THRESHOLD: f64 = 1e-6` - default GM pruning threshold
- Updated `src/lmb/cardinality.rs` to use `EPSILON_EXISTENCE` and `ESF_ADJUSTMENT`
- Updated `src/common/linalg.rs` to use `SVD_TOLERANCE`

**Impact**:
- Centralized magic numbers with documentation
- Key files (`cardinality.rs`, `linalg.rs`) now use named constants
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: Magic numbers like `1e-15` and `1e-6` appeared throughout the codebase without clear documentation. By defining named constants with documentation, we improve code readability and maintainability. Full adoption is deferred as the priority was to create the reference module.

---

## 2025-12-01: Phase 15 - Helper Function Adoption

**Files**:
- `src/common/ground_truth.rs` (refactored)
- `src/multisensor_lmb/merging.rs` (refactored)

**Changes**:
- `ground_truth.rs`: Updated to use Model accessor methods:
  - `model.get_clutter_rate(Some(s))` instead of `model.clutter_rate_multisensor.as_ref().unwrap()[s]`
  - `model.get_detection_probability(Some(s))` instead of `model.detection_probability_multisensor.as_ref().unwrap()[s]`
  - `model.get_measurement_noise(Some(s))` instead of `model.q_multisensor.as_ref().unwrap()[s].clone()`
  - `model.get_observation_matrix(Some(s))` instead of `model.c_multisensor.as_ref().unwrap()[s].clone()`
- `merging.rs`: Updated GA merging to use canonical form helpers:
  - `to_weighted_canonical_form(&nu, &t, sensor_weights[s])` replaces manual K, h, g computation
  - `from_canonical_form(&fused)` replaces manual back-conversion
- `merging.rs`: Updated PU merging to use canonical form helpers:
  - `to_canonical_form(&mu, &sigma, weight)` for prior and sensor components
  - `from_canonical_form(&canonical)` for back-conversion to moment form
- Removed unused `robust_inverse` import from `merging.rs`

**Impact**:
- `ground_truth.rs`: 4 instances of `.as_ref().unwrap()` pattern removed
- `merging.rs` GA function: ~15 lines cleaner
- `merging.rs` PU function: ~20 lines cleaner
- Canonical form helpers now actively used (not just in tests)
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: An audit revealed that helper functions from Phase 10 (Model accessors) and Phase 11 (canonical form helpers) were defined but not fully adopted. By updating `ground_truth.rs` and `merging.rs` to use these helpers, we ensure consistent behavior and demonstrate the value of the extracted utilities.
