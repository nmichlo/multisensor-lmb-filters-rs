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

---

## 2025-12-01: Phase 16 - Performance Optimization (LMBM)

**Files**:
- `Cargo.toml` (release profile)
- `src/common/linalg.rs` (new function)
- `src/multisensor_lmbm/association.rs` (optimizations)

**Changes**:
- Added release profile to `Cargo.toml`: `lto = "thin"`, `codegen-units = 1`, `opt-level = 3`
- Added `robust_inverse_with_log_det()` to `linalg.rs` - computes inverse and log-det in single Cholesky
- Pre-cached Q and C matrices per sensor in `generate_multisensor_lmbm_association_matrices()`
- Added `#[inline]` annotations to `convert_from_linear_to_cartesian()` and `determine_log_likelihood_ratio()`
- Updated `determine_log_likelihood_ratio()` to use cached matrices and combined inverse+log-det

**Impact**:
- Benchmark: **23.14s → 20.20s** (12.7% improvement)
- Allocations: **34.8 GB → 29.7 GB** (15% reduction in cumulative allocations)
- Eliminated 10.7M Q/C matrix clones per run
- Eliminated redundant Cholesky decomposition in likelihood computation
- Eliminated 21.4M Vec heap allocations via stack-allocated index arrays
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: Profiling showed `determine_log_likelihood_ratio()` was called 10.7 million times with 34.8 GB cumulative allocations. The main inefficiencies were:
1. Cloning Q and C matrices on every call instead of caching them once per timestep
2. Computing Cholesky decomposition twice: once for `robust_inverse()` and once for `log_gaussian_normalizing_constant()`
3. Allocating Vec<usize> for index conversion on every iteration (21.4M allocations)
By caching matrices, combining the inverse+log-det computation, and using stack-allocated arrays, we reduce both time and memory pressure.

---

## 2025-12-01: Phase 17 - Custom Allocator and Access Pattern Analysis

**Files**:
- `Cargo.toml` (new features)
- `src/lib.rs` (global allocator)
- `src/multisensor_lmbm/gibbs.rs` (access tracing)
- `src/multisensor_lmbm/filter.rs` (tracing integration)
- `src/multisensor_lmbm/mod.rs` (exports)

**Changes**:
- Added `mimalloc` feature with feature-gated global allocator
- Added `gibbs-trace` feature for access pattern instrumentation
- Instrumented Gibbs sampling to track likelihood matrix accesses
- Added `reset_access_trace()`, `get_access_stats()`, `print_access_report()` functions

**Impact**:
- **mimalloc benchmark**: 11.88s → 9.32s (**21.5% faster**)
- **Access pattern analysis** confirms lazy likelihood viability:
  - Typical access ratio: **5-17%** of total entries
  - Larger matrices: as low as **3-6%** access
  - Potential savings: **83-95%** of likelihood computations
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**:
1. **mimalloc**: The custom allocator provides significant speedup for allocation-heavy workloads like LMBM with minimal code changes. Feature-gated to remain optional.
2. **Access pattern tracing**: Before implementing lazy likelihood (a significant architectural change), instrumentation confirms the hypothesis that Gibbs sampling only accesses a small fraction of the precomputed likelihood matrix, validating the potential 10-100x improvement from on-demand computation.

**Key Finding**: The 10.7M likelihood computations done upfront could potentially be reduced to ~500K-1.8M on-demand computations (5-17% of total), making lazy likelihood the highest-impact optimization remaining.

---

## 2025-12-01: Phase 18 - Rayon Parallelization

**Files**:
- `Cargo.toml` (new feature)
- `src/multisensor_lmbm/association.rs` (parallel loop)

**Changes**:
- Added `rayon` feature with feature-gated parallel implementation
- Parallelized the 10.7M iteration likelihood computation loop using `into_par_iter()`
- Each iteration is independent (writes to unique index) - embarrassingly parallel

**Impact**:
- **Benchmark with rayon + mimalloc**: 9.32s → 2.81s (**3.3x faster**)
- **Total improvement from baseline**: 23.14s → 2.81s (**8.2x faster**)
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Usage**:
```bash
# Use all optimizations for best performance
cargo run --release --features rayon,mimalloc --example multi_sensor -- --filter-type LMBM
```

**Rationale**: The likelihood computation loop iterates 10.7M times with no shared mutable state - each iteration computes values for a unique index. This is trivially parallelizable with rayon's `into_par_iter()`. The parallel version collects results and then unpacks them, avoiding any contention.

---

## 2025-12-01: Phase 19 - Workspace Buffer Reuse (REVERTED)

**Status**: ❌ FAILED & REVERTED

**Attempted Changes**:
- Added `LmbmLikelihoodWorkspace` struct with pre-allocated buffers
- Reused buffers across likelihood computations

**Result**: 2.76s (only 1% improvement from 2.79s)

**Why it failed**:
- mimalloc already handles allocation efficiently
- Most allocation overhead is in nalgebra operator temporaries, not explicit buffers
- Added code complexity for negligible gain

**Decision**: Reverted. Not worth the added complexity.

---

## 2025-12-01: Phase 20 - Lazy Likelihood Computation (REVERTED)

**Status**: ❌ FAILED & REVERTED

**Attempted Changes**:
- Created `LazyLikelihood` struct with on-demand computation and HashMap memoization
- Changed Gibbs sampling to compute likelihoods lazily instead of upfront
- Reduced likelihood computations from 10.7M to ~445K (96% reduction)

**Result**: 4.94s (slower than rayon's 2.79s)

**Why it failed**:
- While lazy reduced computations by 96%, serial execution couldn't compete with parallel eager computation
- HashMap lookups and RefCell borrow checking added per-access overhead
- The 10.7M computations are embarrassingly parallel - rayon handles them efficiently

**Key Lesson**: Parallelization (rayon) provides better speedup than avoiding computation (lazy) when the work is embarrassingly parallel. The access pattern analysis was correct (only 5-17% accessed), but parallel eager still wins.

**Decision**: Reverted. Rayon parallelization is the superior approach.

---

## 2025-12-01: Phase 21 - Clone Elimination

**Files**:
- `src/multisensor_lmbm/filter.rs` (clone elimination)
- `src/multisensor_lmbm/association.rs` (signature change)
- `tests/step_by_step_validation.rs` (updated tests)

**Changes**:
- Removed measurement vector cloning in filter.rs:
  - Changed from `measurements[s][t].clone()` to slice references
  - API signature changed: `&[Vec<DVector<f64>>]` → `&[&[DVector<f64>]]`
- Removed hypothesis cloning in filter.rs:
  - Changed from `hypotheses[i].clone()` to `std::mem::replace(..., Hypothesis::empty())`
  - Takes ownership instead of cloning since hypotheses are overwritten at end of each timestep
- Updated association.rs and tests to use new signature

**Impact**:
- **Benchmark**: 2.79s → 2.77s (**~0.7% improvement**)
- Cleaner code with less unnecessary work
- All tests pass with unchanged tolerances
- MATLAB equivalence maintained

**Rationale**: The filter was cloning measurements and hypotheses unnecessarily. Measurements are read-only and hypotheses are overwritten at the end of each timestep. By using references and ownership transfer, we eliminate unnecessary memory operations. The small improvement is expected since these clones were not in the hot 10.7M iteration loop.

---

## 2025-12-01: Phase G1 - Static Types for Hypothesis

**Files**:
- `src/common/types.rs` (new type aliases, updated Hypothesis struct)
- `src/multisensor_lmbm/association.rs` (static → dynamic conversions)
- `src/multisensor_lmbm/hypothesis.rs` (dynamic → static conversions)
- `src/multisensor_lmbm/filter.rs` (static → dynamic for output)
- `src/lmbm/association.rs` (conversions)
- `src/lmbm/hypothesis.rs` (conversions)
- `src/lmbm/prediction.rs` (conversions)
- Test files updated for new types

**Changes**:
- Added `State4 = SVector<f64, 4>` and `Cov4x4 = SMatrix<f64, 4, 4>` type aliases
- Changed `Hypothesis.mu: Vec<DVector<f64>>` → `Vec<State4>` (stack-allocated 4D vectors)
- Changed `Hypothesis.sigma: Vec<DMatrix<f64>>` → `Vec<Cov4x4>` (stack-allocated 4x4 matrices)
- Added conversions at boundaries where dynamic types are needed (posterior params, output structs)
- Kept `MultisensorLmbmPosteriorParameters` using dynamic types (less frequently accessed)

**Impact**:
- **Benchmark**: 0.65s → 0.59s (**~9% improvement**)
- Eliminates inner heap allocations for Hypothesis mu/sigma
- Better cache locality (contiguous [f64; 4] and [f64; 16] instead of heap pointers)
- All tests pass (without rayon; with rayon, parallel Gibbs produces more samples as expected)

**Rationale**: The Hypothesis struct is accessed 10.7M times in the hot path. Using static types for the 4D state vectors and 4x4 covariance matrices eliminates heap allocations and improves cache locality. The x_dimension is always 4 in this codebase, making static types safe.

---

## Summary: Final Optimization State

**Current best configuration**: `--features rayon,mimalloc`

| Phase | Optimization | Time | vs Previous | Total Speedup |
|-------|-------------|------|-------------|---------------|
| Baseline | None | 13.90s | - | 1.0x |
| Early | +lto+cache+logdet+stack | 12.20s | -12.2% | 1.1x |
| Early | +mimalloc | 9.58s | -21.4% | 1.5x |
| Early | +rayon (parallel likelihood) | 2.79s | -70.9% | 5.0x |
| A | Clone elimination | 2.77s | -0.7% | 5.0x |
| B | Measurement gating | 1.48s | -46.6% | 9.4x |
| C | Deferred posterior params | 1.38s | -6.8% | 10.1x |
| F | Parallel Gibbs chains | 0.65s | -52.9% | 21.4x |
| **G1** | **Static types (State4/Cov4x4)** | **0.59s** | **-9.2%** | **23.6x** |

**Total speedup: 23.6x** (13.90s → 0.59s)

**Target**: <0.5s (~1.2x more improvement needed)

See `docs/03_optimisations/benchmarks.md` for full methodology and raw data.
