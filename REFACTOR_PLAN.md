# Comprehensive Cleanup & Unification Plan for multisensor-lmb-filters-rs

## Executive Summary
Addresses ~60% code duplication between single/multi-sensor filters, API inconsistencies, and architectural gaps. Prioritizes extensibility and production-grade qualities.

## Core Principles (NON-NEGOTIABLE)
1. **Delete old code** - Complete rewrite, not migration. Old implementations MUST be deleted.
2. **Fix impl, not tests** - Numeric equivalence at 1e-10. MATLAB fixtures are ground truth.
3. **One implementation per filter type** - `LmbAlgorithm` for LMB, `LmbmAlgorithm` for LMBM.
4. **Extensibility via traits** - `FilterAlgorithm`, `Associator`, `Merger`, `UpdateScheduler`, `StepReporter`, `MotionModelBehavior`, `SensorModelBehavior`
5. **Always use rust-analyzer-lsp** MCP/plugin instead of manual grep
6. **Prefer bulk renaming** with unique names instead of manual edits

## Unified Filter Architecture

### Why Not Full Generic `FilterCore<S, A, U, M>`?
LMB, LMBM, NORFAIR/SORT have fundamentally different semantics (state types, existence models, association methods, cardinality). Full generics would create type explosion and force unnatural abstractions.

### Chosen Approach: `FilterAlgorithm` Trait
```rust
pub trait FilterAlgorithm: Send + Sync {
    type State: Clone;
    type Measurements;
    type DetailedOutput;
    fn predict(&mut self, motion: &dyn MotionModelBehavior, timestep: usize);
    fn inject_birth(&mut self, birth: &BirthModel, timestep: usize);
    fn associate_and_update<R: Rng>(&mut self, rng: &mut R, measurements: &Self::Measurements, sensors: &SensorConfig) -> Result<(), FilterError>;
    fn normalize_and_gate(&mut self, config: &GatingConfig);
    fn extract_estimate(&self, timestamp: usize) -> StateEstimate;
    fn extract_detailed(&self) -> Self::DetailedOutput;
}

pub struct Filter<A: FilterAlgorithm> { algorithm: A, motion: MotionModel, sensors: SensorConfig, birth: BirthModel, gating: GatingConfig, trajectories: Vec<Trajectory> }
```

### Type Aliases
```rust
pub type LmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, SingleSensorScheduler>>;
pub type IcLmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, SequentialScheduler>>;
pub type AaLmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, ParallelScheduler<ArithmeticAverageMerger>>>;
pub type GaLmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, ParallelScheduler<GeometricAverageMerger>>>;
pub type PuLmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, ParallelScheduler<ParallelUpdateMerger>>>;
pub type LmbmFilter = UnifiedFilter<LmbmStrategy<SingleSensorLmbmAssociator<GibbsAssociator>>>;
pub type MultisensorLmbmFilter = UnifiedFilter<LmbmStrategy<MultisensorLmbmAssociator<MultisensorGibbsAssociator>>>;
```

## Current State Analysis
- **Rust**: 41 files, ~15,851 LOC | **Python Tests**: 1,875 LOC | **Filters**: 7 types | **Associators**: 3 | **Mergers**: 4

| Issue | Severity |
|-------|----------|
| Single/multi-sensor step() ~60% overlap | HIGH |
| LMBM predict/update duplicated | HIGH |
| No `MotionModel` trait (blocks IMM) | HIGH |
| 4 trajectory update functions duplicated | MEDIUM |
| Type aliases unusable with `::new()` | MEDIUM |

## Phase Structure Rules
1. Tests may ONLY change to match new APIs (signatures, imports, config types)
2. Tests must NOT change expected numeric values, tolerances, or behavior
3. If test fails due to numeric differences, IMPLEMENTATION is wrong
4. Each phase MUST end with plan update before next phase

---

## Phase 0: Plan Setup ✅
- [x] Write complete plan to `./REFACTOR_PLAN.md`
- [x] Verify tests pass (`cargo test --release` - 39 Rust, `uv run pytest` - 86 Python)
- [x] Baseline established on 2026-01-16

## Phase 1: Zero-Copy Measurement Input ✅
Created `src/lmb/measurements.rs` with `MeasurementSource` trait (~700 LOC).
- [x] Create `MeasurementSource` trait with GATs for zero-copy iteration
- [x] Implement for `&[DVector]`, `&[Vec<DVector>]`, `&[&[DVector]]`
- [x] Add 9 unit tests verifying zero-copy guarantee
- **NOTE**: Integration into filter cores deferred to Phase 8

## Phase 2: Type-Safe Configuration ✅
Created `CommonConfig`, `LmbFilterConfig`, `LmbmFilterConfig` with builders (~400 LOC).
- [x] Create config structs with builder patterns
- [x] Add 10 unit tests
- **NOTE**: Filter constructor updates deferred to Phase 7A/7B

## Phase 3: Extensible Traits for Models ✅
- [x] Create `MotionModelBehavior` and `SensorModelBehavior` traits
- [x] Implement for existing structs
- **NOTE**: Filter integration deferred to Phase 7A/7B

## Phase 4: Update Strategy Pattern ✅
Created `src/lmb/scheduler.rs` (~600 LOC).
- [x] Create `UpdateScheduler` trait
- [x] Implement `ParallelScheduler<M: Merger>`, `SequentialScheduler`, `SingleSensorScheduler`, `DynamicScheduler`
- **NOTE**: Integration deferred to Phase 7A/7B

## Phase 5: Observability via StepReporter ✅
Created `src/lmb/reporter.rs` (~600 LOC).
- [x] Create `StepReporter` trait with 9 callback methods
- [x] Implement `NoOpReporter`, `DebugReporter`, `LoggingReporter`, `CompositeReporter<A, B>`
- **NOTE**: Hook integration deferred to Phase 8

## Phase 6: Numerical Robustness ✅
Added to `src/common/linalg.rs` (~200 LOC).
- [x] Add `robust_cholesky()` with exponential regularization
- [x] Create `LinalgWarning` enum, `CholeskyResult` enum
- **NOTE**: Integration into Kalman updates deferred to Phase 8

## Phase 7A: Delete Old LMB Code ✅
**Deleted 1285 LOC** - using unified `LmbFilterCore` from `core.rs`.
- [x] **DELETE** `src/lmb/singlesensor/lmb.rs` (-549 LOC)
- [x] **DELETE** `src/lmb/multisensor/lmb.rs` (-736 LOC)
- [x] Update `mod.rs` exports, Python bindings, bench files
- [x] Verification: All tests pass

## Phase 7B: Delete Old LMBM Code ✅
**Deleted 1634 LOC** - using unified `LmbmFilterCore` from `core_lmbm.rs`.
- [x] **DELETE** `src/lmb/singlesensor/lmbm.rs` (-733 LOC)
- [x] **DELETE** `src/lmb/multisensor/lmbm.rs` (-901 LOC)
- [x] **DELETE** `src/lmb/singlesensor/` directory
- [x] Verification: All tests pass

## Phase 7C: API Simplification ✅
Created `src/lmb/factory.rs` with 7 factory functions.
- [x] Create factory functions: `lmb_filter()`, `ic_lmb_filter()`, `aa_lmb_filter()`, `ga_lmb_filter()`, `pu_lmb_filter()`, `lmbm_filter()`, `multisensor_lmbm_filter()`
- [x] Merge `SensorConfig` into `config.rs`, delete `LmbmSensorConfig`
- [x] Remove generic params from type aliases
- [ ] Remove constructor impl blocks (deferred to 7D)

## Phase 7D: Dead Code Cleanup ✅
**Deleted ~500 LOC** - ONE way to create filters.
- [x] Delete 6 redundant constructor impl blocks from `core.rs`
- [x] Delete `SensorVariant`, `CommonConfigBuilder`, `LmbFilterConfigBuilder`, `LmbmFilterConfigBuilder`, `FilterParamsBuilder`
- [x] Delete `SingleSensorLmbmStrategy::new()`, `MultisensorLmbmStrategy::new()`
- [x] Verification: All tests pass

**API After**: Factory functions for common cases, `with_scheduler()`/`with_strategy()` as escape hatch.

## Phase 7E: (Merged into Phase 8.7)

## Phase 8: Full LMB/LMBM Unification ✅
**Deleted core.rs (~1166 LOC) + core_lmbm.rs (~1450 LOC). Created strategy.rs (~1700 LOC) + unified.rs (~600 LOC). Net: ~1300 LOC reduction.**

### 8.1: Rename Hypothesis Struct ✅
- [x] Rename `LmbmHypothesis` → `Hypothesis` in `types.rs`
- [x] Add `lmb()` constructor for single-hypothesis state

### 8.2: Create UpdateStrategy Trait ✅
- [x] Create `src/lmb/strategy.rs` with `UpdateStrategy` trait
- [x] Define `UpdateContext`, `CommonPruneConfig`, `LmbPruneConfig`, `LmbmPruneConfig`, `UpdateIntermediate`

### 8.3: Implement LmbStrategy ✅
- [x] Implement for `SingleSensorScheduler`, `SequentialScheduler`, `ParallelScheduler<M>`

### 8.4: Implement LmbmStrategy ✅
- [x] Implement for `SingleSensorLmbmStrategy`, `MultisensorLmbmStrategy`

### 8.5: Create UnifiedFilter Struct ✅
- [x] Create `src/lmb/unified.rs` with `UnifiedFilter<S: UpdateStrategy>`
- [x] Implement `Filter` trait for all scheduler variants
- [x] Add `step_detailed()` with sensor_updates and correct cardinality

### 8.6: Update Factory Functions ✅
- [x] All 7 factory functions return `UnifiedFilter<...>`

### 8.7: Update Python Bindings ✅
Updated 7 filter classes to wrap `UnifiedFilter` internally (backward-compatible API preserved):
- [x] `PyFilterLmb`, `PyFilterLmbm`, `PyFilterIcLmb`, `PyFilterAaLmb`, `PyFilterGaLmb`, `PyFilterPuLmb`, `PyFilterMultisensorLmbm`
- [x] Add `set_tracks()`, `get_config()` to `UnifiedFilter`
- [x] Add config getters to `UpdateStrategy` trait
- [x] Fix cardinality computation (from `updated_tracks` before pruning)
- [x] Verification: 86 Python tests pass at 1e-10

### 8.8: Delete Old Cores ✅
- [x] **DELETE** `src/lmb/core.rs` (-1166 LOC)
- [x] **DELETE** `src/lmb/core_lmbm.rs` (-1450 LOC)

### 8.9: Clean Up common_ops.rs ✅
- [x] Kept: `predict_all_hypotheses()`, `update_hypothesis_trajectories()`, `init_hypothesis_birth_trajectories()` (still used by strategy.rs)

**Completion**: All 13 Rust tests, 86 Python tests pass. Single unified architecture.

## Phase 9: (Merged into Phase 8)

---

## Phase 10: Complete API Migration (Delete All Backward Compat) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**Goal**: Delete ALL backward compatibility code. Update ALL tests to use new APIs. Per the refactor principles: "Delete old code, don't keep for backward compat."

**CRITICAL**: This phase explicitly updates tests to new APIs - tests are NOT exempt from API migration.

**NB** FOLLOW THE "## Core Principles" section at the top.

---

### 10.1: Delete Deprecated Rust Types (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Files**: `src/lmb/types.rs`, `src/lmb/config.rs`, `src/lib.rs`

| Type to DELETE | Location | Replacement |
|----------------|----------|-------------|
| `LmbmHypothesis` (deprecated alias) | `types.rs:L~650` | Use `Hypothesis` directly |
| `FilterParams` | `config.rs` | Split into specific configs |
| `FilterThresholds` | `config.rs` | `CommonPruneConfig` + `LmbPruneConfig` |
| `LmbmConfig` | `config.rs` | `LmbmPruneConfig` |
| `to_legacy_lmbm_config()` | `config.rs` | DELETE |
| `#[allow(deprecated)] pub use lmb::LmbmHypothesis` | `lib.rs` | DELETE |

- [x] Delete `LmbmHypothesis` type alias from `types.rs`
- [x] Delete `FilterParams` struct from `config.rs`
- [ ] Delete `FilterThresholds` struct from `config.rs`
- [ ] Delete `LmbmConfig` struct from `config.rs`
- [ ] Delete `to_legacy_lmbm_config()` method from `LmbmFilterConfig`
- [x] Delete deprecated re-export from `lib.rs`
- [x] Update `src/lmb/mod.rs` exports

---

### 10.2: Delete Old Python Config Classes

**NB** FOLLOW THE "## Core Principles" section at the top.

**File**: `src/python/filters.rs`

| Class to DELETE | Replacement |
|-----------------|-------------|
| `PyFilterThresholds` | Inline params on filter constructors |
| `PyFilterLmbmConfig` | Inline params on filter constructors |
| `PyAssociatorConfig` | Inline params on filter constructors |
| `_LmbmHypothesis` | Rename to `_Hypothesis` |

**New Python API** (filter constructors with inline params):

```python
# OLD API (DELETE)
thresholds = FilterThresholds(max_components=5, gm_weight=1e-5)
lmbm_config = FilterLmbmConfig(max_hypotheses=100, hypothesis_weight_threshold=1e-6)
assoc = AssociatorConfig.lbp(max_iterations=100, tolerance=1e-3)
filter = FilterLmb(motion, sensor, birth, assoc, thresholds=thresholds)

# NEW API (all params inline)
filter = FilterLmb(
    motion, sensor, birth,
    # Association params
    associator="lbp",  # or "gibbs", "murty"
    lbp_max_iterations=100,
    lbp_tolerance=1e-3,
    gibbs_num_samples=1000,
    murty_num_assignments=100,
    # Prune params (CommonPruneConfig)
    existence_threshold=0.001,
    min_trajectory_length=3,
    # LMB-specific prune params (LmbPruneConfig)
    gm_weight_threshold=1e-5,
    max_gm_components=5,
    gm_merge_threshold=float('inf'),
    # Seed
    seed=42,
)

# LMBM filters get LMBM-specific params
filter = FilterLmbm(
    motion, sensor, birth,
    associator="gibbs",
    gibbs_num_samples=1000,
    # Prune params (CommonPruneConfig)
    existence_threshold=0.001,
    min_trajectory_length=3,
    # LMBM-specific prune params (LmbmPruneConfig)
    hypothesis_weight_threshold=1e-6,
    max_hypotheses=100,
    use_eap=False,
    seed=42,
)
```

- [x] Update `PyFilterLmb::new()` to accept inline params
- [x] Update `PyFilterLmbm::new()` to accept inline params
- [x] Update `PyFilterIcLmb::new()` to accept inline params
- [x] Update `PyFilterAaLmb::new()` to accept inline params
- [x] Update `PyFilterGaLmb::new()` to accept inline params
- [x] Update `PyFilterPuLmb::new()` to accept inline params
- [x] Update `PyFilterMultisensorLmbm::new()` to accept inline params
- [ ] DELETE `PyFilterThresholds` class
- [ ] DELETE `PyFilterLmbmConfig` class
- [ ] DELETE `PyAssociatorConfig` class
- [x] Rename `_LmbmHypothesis` to `_Hypothesis`

---

### 10.3: Update Python `__init__.py` Exports

**File**: `python/multisensor_lmb_filters_rs/__init__.py`

```python
# DELETE these exports:
# - FilterThresholds
# - FilterLmbmConfig
# - AssociatorConfig
# - _LmbmHypothesis (rename to _Hypothesis)

# KEEP these exports:
__all__ = [
    # Filters
    "FilterLmb",
    "FilterLmbm",
    "FilterIcLmb",
    "FilterAaLmb",
    "FilterGaLmb",
    "FilterPuLmb",
    "FilterMultisensorLmbm",
    # Models
    "MotionModel",
    "SensorModel",
    "MultisensorConfig",
    "BirthModel",
    "BirthLocation",
    # Internal types (underscore prefix)
    "_Track",
    "_GaussianComponent",
    "_Hypothesis",  # Renamed from _LmbmHypothesis
    "_StepOutput",
    "_SensorUpdateOutput",
]
```

- [ ] Remove `FilterThresholds` from exports
- [ ] Remove `FilterLmbmConfig` from exports
- [ ] Remove `AssociatorConfig` from exports
- [x] Rename `_LmbmHypothesis` to `_Hypothesis` in exports

---

### 10.4: Migrate ALL Python Tests to New API

**NB** FOLLOW THE "## Core Principles" section at the top.

**CRITICAL**: Tests must use new API. No backward compat exemptions.

**Files to update**:
- `python/tests/test_equivalence.py` (~50 usages)
- `python/tests/test_benchmark_fixtures.py` (~10 usages)
- `python/tests/conftest.py` (~5 usages)

#### Migration patterns:

```python
# OLD: FilterThresholds
thresholds = FilterThresholds(max_components=5, gm_weight=1e-5)
filter = FilterLmb(..., thresholds=thresholds)

# NEW: Inline params
filter = FilterLmb(..., max_gm_components=5, gm_weight_threshold=1e-5)
```

```python
# OLD: FilterLmbmConfig
lmbm_config = FilterLmbmConfig(max_hypotheses=100, hypothesis_weight_threshold=1e-6)
filter = FilterLmbm(..., lmbm_config=lmbm_config)

# NEW: Inline params
filter = FilterLmbm(..., max_hypotheses=100, hypothesis_weight_threshold=1e-6)
```

```python
# OLD: AssociatorConfig
filter = FilterLmb(..., AssociatorConfig.lbp(max_iterations=100, tolerance=1e-3))

# NEW: Inline params
filter = FilterLmb(..., associator="lbp", lbp_max_iterations=100, lbp_tolerance=1e-3)
```

```python
# OLD: _LmbmHypothesis
from multisensor_lmb_filters_rs import _LmbmHypothesis
hypothesis = _LmbmHypothesis.from_matlab(...)

# NEW: _Hypothesis
from multisensor_lmb_filters_rs import _Hypothesis
hypothesis = _Hypothesis.from_matlab(...)
```

- [x] Update `test_equivalence.py`: Replace all `FilterThresholds` usages
- [ ] Update `test_equivalence.py`: Replace all `FilterLmbmConfig` usages
- [ ] Update `test_equivalence.py`: Replace all `AssociatorConfig` usages
- [x] Update `test_equivalence.py`: Replace all `_LmbmHypothesis` with `_Hypothesis`
- [x] Update `test_benchmark_fixtures.py`: Same replacements
- [x] Update `conftest.py`: Replace `_LmbmHypothesis` references

---

### 10.5: Update Rust Test Files (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Files**: `tests/ss_lmb.rs`, `tests/ss_lmbm.rs`, `tests/ms_lmb.rs`, `tests/ms_lmbm.rs`, `tests/ms_variants.rs`

- [x] Remove any uses of deprecated `LmbmHypothesis` alias
- [x] Update to use `Hypothesis` directly
- [x] Remove any uses of `FilterParams`, `FilterThresholds`

---

### 10.6: Clean Up Legacy Adapter Comments (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**File**: `src/lmb/traits.rs`

The legacy adapters (`legacy_lbp`, `legacy_gibbs`, `legacy_murtys`, `RngAdapter`) are kept as internal implementation detail. Update comments to clarify they're intentional internal wrappers, not backward compat:

```rust
// Before: "use crate::common::association::gibbs as legacy_gibbs;"
// After: "use crate::common::association::gibbs as internal_gibbs;"
```

- [ ] Rename `legacy_*` imports to `internal_*` (cosmetic, not functional)
- [ ] Update comments to clarify these are internal implementations

---

### 10.7: Clean Up mod.rs Exports (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**File**: `src/lmb/mod.rs`

Final exports (NO backward compat types):

```rust
// Unified filter
pub use unified::{UnifiedFilter, Hypothesis};
pub use strategy::{UpdateStrategy, LmbStrategy, LmbmStrategy};
pub use strategy::{CommonPruneConfig, LmbPruneConfig, LmbmPruneConfig};

// Type aliases
pub use unified::{LmbFilter, IcLmbFilter, AaLmbFilter, GaLmbFilter, PuLmbFilter};
pub use unified::{LmbmFilter, MultisensorLmbmFilter};

// Traits
pub use traits::{Filter, Associator, Updater, Merger};
pub use scheduler::UpdateScheduler;

// Configuration
pub use config::{MotionModel, SensorModel, SensorConfig, BirthModel, AssociationConfig};

// Schedulers
pub use scheduler::{SingleSensorScheduler, SequentialScheduler, ParallelScheduler};

// Fusion mergers
pub use multisensor::fusion::{ArithmeticAverageMerger, GeometricAverageMerger, ParallelUpdateMerger};

// Errors
pub use errors::{FilterError, AssociationError};

// DELETED: FilterParams, FilterThresholds, LmbmConfig, LmbmHypothesis
```

- [ ] Remove `FilterParams` export
- [ ] Remove `FilterThresholds` export
- [ ] Remove `LmbmConfig` export
- [ ] Remove `LmbmHypothesis` export
- [ ] Add `CommonPruneConfig`, `LmbPruneConfig`, `LmbmPruneConfig` exports

---

### 10.8: Update lib.rs Exports (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**File**: `src/lib.rs`

```rust
// DELETED exports:
// - FilterParams
// - FilterThresholds
// - LmbmConfig
// - #[allow(deprecated)] LmbmHypothesis

// NEW exports:
pub use lmb::{CommonPruneConfig, LmbPruneConfig, LmbmPruneConfig};
```

- [ ] Remove deprecated exports
- [ ] Add new config type exports

---

### 10.9: Update Documentation (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

- [ ] Update `lib.rs` doc example to use new API (already uses factory functions, verify no old types)
- [ ] Update any doc comments referencing old types

---

## Additional Code Smells from Architecture Review

The following issues were identified during architecture review and should be addressed as part of Phase 10.

---

### 10.10: Eliminate Configuration Leakage (HIGH PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: 7+ `is_hypothesis_based()` checks scattered throughout `unified.rs` (lines ~350, ~378, ~405, ~450, ~502, ~518, ~551). Each check is a configuration leakage where filter behavior diverges based on strategy type rather than polymorphism.

**File**: `src/lmb/unified.rs`

**Pattern to eliminate**:
```rust
// BAD: Configuration leakage
if self.strategy.is_hypothesis_based() {
    // LMBM-specific code path
} else {
    // LMB-specific code path
}
```

**Solution**: Each divergence point should be a trait method that strategies implement differently:

```rust
// GOOD: Polymorphism
trait UpdateStrategy {
    fn finalize_step(&self, hypotheses: &[Hypothesis], intermediate: &UpdateIntermediate) -> StepOutput;
    fn build_detailed_output(&self, ...) -> StepDetailedOutput;
    // ... other divergence points
}
```

**Tasks**:
- [ ] Audit all 7+ `is_hypothesis_based()` usage sites
- [ ] For each site, create trait method that captures the divergent behavior
- [ ] Implement method differently in `LmbStrategy` vs `LmbmStrategy`
- [ ] Delete `is_hypothesis_based()` from trait

---

### 10.11: Integrate Zero-Copy MeasurementSource (HIGH PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: `MeasurementSource` trait was created in Phase 1 but never integrated. All filter `step()` methods still accept raw slices, forcing allocations.

**Files**: `src/lmb/unified.rs`, `src/lmb/strategy.rs`, `src/lmb/measurements.rs`

**Current** (Phase 1 created but unused):
```rust
// measurements.rs - EXISTS but unused
pub trait MeasurementSource {
    type Iter<'a>: Iterator<Item = DVector<f64>> where Self: 'a;
    fn measurements(&self) -> Self::Iter<'_>;
    fn len(&self) -> usize;
}

// unified.rs - Still uses raw slices
pub fn step(&mut self, rng: &mut R, measurements: &[DVector<f64>], ts: usize) -> Result<...>
```

**After**:
```rust
// unified.rs - Uses trait for zero-copy
pub fn step<M: MeasurementSource>(&mut self, rng: &mut R, measurements: &M, ts: usize) -> Result<...>
```

**Tasks**:
- [ ] Update `UnifiedFilter::step()` to accept `impl MeasurementSource`
- [ ] Update `UpdateStrategy::update()` to accept `impl MeasurementSource`
- [ ] Update Python bindings to use zero-copy wrappers
- [ ] Verify no allocations in hot path with benchmarks

---

### 10.12: Consolidate Python Filter Classes (HIGH PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: 7 nearly identical Python filter classes (`PyFilterLmb`, `PyFilterLmbm`, `PyFilterIcLmb`, etc.) with 489 lines of duplicated code. Phase 8.7 updated internals but kept class proliferation.

**File**: `src/python/filters.rs`

**Current** (7 classes × ~70 lines each):
```python
# Python API - 7 separate classes
filter = FilterLmb(motion, sensor, birth, ...)
filter = FilterLmbm(motion, sensor, birth, ...)
filter = FilterIcLmb(motion, sensors, birth, ...)
# ... 4 more
```

**After** (Strategy Object Pattern from original Phase 8.7 plan):
```python
# Python API - 1 Filter + 2 Strategy classes
filter = Filter(motion, sensor, birth, strategy=LmbStrategy(...))
filter = Filter(motion, sensor, birth, strategy=LmbmStrategy(...))
filter = Filter(motion, sensors, birth, strategy=LmbStrategy(..., scheduler="ic"))
```

**Tasks**:
- [ ] Create `PyLmbStrategy` class with LMB-specific params
- [ ] Create `PyLmbmStrategy` class with LMBM-specific params
- [ ] Create unified `PyFilter` class accepting any strategy
- [ ] DELETE all 7 old filter classes
- [ ] Update Python `__init__.py` exports
- [ ] Update ALL Python tests to new API

---

### 10.13: Fix Strategy-Specific Defaults Returning 0 (HIGH PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: Trait methods like `gm_weight_threshold()`, `max_gm_components()`, `lmbm_config()` return 0 or empty when called on wrong strategy type. This violates LSP - calling these on LMBM returns meaningless values.

**File**: `src/lmb/strategy.rs`

**Current** (LSP violation):
```rust
impl UpdateStrategy for LmbmStrategy {
    fn gm_weight_threshold(&self) -> f64 { 0.0 }  // Meaningless for LMBM
    fn max_gm_components(&self) -> usize { 0 }    // Meaningless for LMBM
}
```

**Options**:
1. Return `Option<f64>` / `Option<usize>` (explicit absence)
2. Remove from trait, make strategy-specific via downcast
3. Use associated types to make config type-safe

**Recommended**: Option 1 - Return `Option`:
```rust
trait UpdateStrategy {
    fn gm_weight_threshold(&self) -> Option<f64> { None }
    fn max_gm_components(&self) -> Option<usize> { None }
    fn lmbm_config(&self) -> Option<&LmbmPruneConfig> { None }
}

impl UpdateStrategy for LmbStrategy {
    fn gm_weight_threshold(&self) -> Option<f64> { Some(self.prune_config.gm_weight_threshold) }
}
```

**Tasks**:
- [ ] Change trait method signatures to return `Option<...>`
- [ ] Update all call sites to handle `Option`
- [ ] Update Python bindings to expose only relevant config

---

### 10.14: Consolidate impl Filter Blocks (MEDIUM PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: 5 nearly identical `impl Filter for UnifiedFilter<LmbStrategy<A, S>>` blocks in `unified.rs`. Each block has the same structure with minor differences in measurement types.

**File**: `src/lmb/unified.rs`

**Current** (5 nearly identical blocks):
```rust
impl<A: Associator> Filter for UnifiedFilter<LmbStrategy<A, SingleSensorScheduler>> { ... }
impl<A: Associator> Filter for UnifiedFilter<LmbStrategy<A, SequentialScheduler>> { ... }
impl<A: Associator, M: Merger> Filter for UnifiedFilter<LmbStrategy<A, ParallelScheduler<M>>> { ... }
impl<A: LmbmAssociator> Filter for UnifiedFilter<LmbmStrategy<SingleSensorLmbmAssociator<A>>> { ... }
impl Filter for UnifiedFilter<LmbmStrategy<MultisensorLmbmAssociator<MultisensorGibbsAssociator>>> { ... }
```

**Options**:
1. Macro to generate impl blocks
2. Blanket impl with associated type for measurements
3. Keep as-is (explicit but verbose)

**Tasks**:
- [ ] Evaluate whether blanket impl is feasible
- [ ] If not, create macro to reduce boilerplate
- [ ] Document reason for keeping separate impls if necessary

---

### 10.15: Refactor step_detailed() (MEDIUM PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: `step_detailed()` in `unified.rs` is 113 lines. Complex method doing too many things.

**File**: `src/lmb/unified.rs` (lines ~500-613)

**Tasks**:
- [ ] Extract `build_predicted_hypotheses()` helper
- [ ] Extract `build_sensor_updates()` helper
- [ ] Extract `build_cardinality_estimate()` helper
- [ ] Extract `build_final_estimate()` helper
- [ ] Reduce `step_detailed()` to orchestration of helpers
- [ ] Target: < 50 lines for main method

---

### 10.16: Refactor UpdateContext (MEDIUM PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: `UpdateContext` bundles 5 references (motion, sensors, birth, association_config, common_prune). It's a "God Object" for passing state around.

**File**: `src/lmb/strategy.rs`

**Options**:
1. Keep as-is (it's a parameter object, not a god object - debatable)
2. Split into `PredictContext` and `UpdateContext`
3. Pass individual params where needed

**Tasks**:
- [ ] Audit which methods actually need which fields
- [ ] If clear split exists, create separate context types
- [ ] Document rationale if keeping as-is

---

### 10.17: Make gm_merge_threshold Configurable (MEDIUM PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: `gm_merge_threshold` is hardcoded to `f64::INFINITY` in `LmbPruneConfig::default()`. This effectively disables GM merging.

**File**: `src/lmb/strategy.rs`

**Current**:
```rust
impl Default for LmbPruneConfig {
    fn default() -> Self {
        Self {
            gm_weight_threshold: 1e-5,
            max_gm_components: 100,
            gm_merge_threshold: f64::INFINITY,  // Hardcoded - disables merging
        }
    }
}
```

**Tasks**:
- [ ] Add `gm_merge_threshold` param to factory functions
- [ ] Add to Python filter constructors
- [ ] Document what reasonable values are (e.g., Mahalanobis distance)
- [ ] Consider whether default should be `INFINITY` or a reasonable value

---

### 10.18: Fix Python Accessor Mismatch (MEDIUM PRIORITY)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: `set_tracks()` exists on LMB filters but not LMBM filters. Inconsistent API.

**File**: `src/python/filters.rs`

**Tasks**:
- [ ] Add `set_tracks()` to all Python filter classes (or `set_hypotheses()`)
- [ ] Or document why LMBM doesn't support track injection
- [ ] Ensure consistent API across all filter types

---

### 10.19: Replace unwrap() with expect() (LOW PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: Bare `unwrap()` calls provide no context on panic. Should use `expect("reason")`.

**Files**: Various in `src/lmb/`

**Tasks**:
- [ ] Grep for `.unwrap()` in lmb module
- [ ] Replace with `.expect("context")` or proper error handling
- [ ] Prioritize hot paths and public API

---

### 10.20: Review Macro Usage (LOW PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: Macro boilerplate may obscure logic. Evaluate if macros add or reduce clarity.

**Files**: `src/lmb/unified.rs` (impl_filter! macro?)

**Tasks**:
- [ ] Audit macro usage in lmb module
- [ ] If macros obscure logic, consider replacing with explicit code
- [ ] If macros help, document why

---

### 10.21: Extract Track Extraction Helper (LOW PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: Track extraction code (getting means/covariances from tracks) repeated 6x across codebase.

**Files**: `src/lmb/unified.rs`, `src/lmb/strategy.rs`, `src/lmb/output.rs`

**Tasks**:
- [ ] Identify all track extraction patterns
- [ ] Create shared `extract_track_states()` helper in `common_ops.rs`
- [ ] Update all call sites to use helper

---

### 10.22: Make Merger/Scheduler Configurable (LOW PRIORITY) (NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

**NB** FOLLOW THE "## Core Principles" section at the top.

**Problem**: Merger and scheduler types are hardcoded in type aliases. Can't change at runtime.

**Current**:
```rust
pub type AaLmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, ParallelScheduler<ArithmeticAverageMerger>>>;
```

**Options**:
1. Keep as-is (compile-time selection is fine for most use cases)
2. Add `DynamicScheduler` / `DynamicMerger` for runtime selection
3. Use trait objects for runtime polymorphism

**Tasks**:
- [ ] Evaluate if runtime selection is actually needed
- [ ] If needed, implement dynamic variants
- [ ] Document when to use static vs dynamic selection

---

### Verification

```bash
# Must all pass with NO backward compat types:
cargo test --release
cargo clippy --all-targets
uv run maturin develop --release
uv run pytest python/tests/ -v
```

---

### Phase 10 Priority Summary

(NB: ALWAYS USE THE rust-analyzer-lsp Plugin)

| Priority | Sections | Description |
|----------|----------|-------------|
| **HIGH** | 10.1-10.9 | Delete deprecated types, migrate tests |
| **HIGH** | 10.10 | Eliminate `is_hypothesis_based()` configuration leakage |
| **HIGH** | 10.11 | Integrate `MeasurementSource` zero-copy trait |
| **HIGH** | 10.12 | Consolidate 7 Python filter classes → Strategy Object Pattern |
| **HIGH** | 10.13 | Fix strategy-specific defaults returning 0 (LSP violation) |
| **MEDIUM** | 10.14 | Consolidate 5 `impl Filter` blocks |
| **MEDIUM** | 10.15 | Refactor `step_detailed()` (113 lines → <50) |
| **MEDIUM** | 10.16 | Refactor `UpdateContext` god object |
| **MEDIUM** | 10.17 | Make `gm_merge_threshold` configurable |
| **MEDIUM** | 10.18 | Fix Python accessor mismatch (`set_tracks`) |
| **LOW** | 10.19 | Replace `unwrap()` with `expect()` |
| **LOW** | 10.20 | Review macro usage |
| **LOW** | 10.21 | Extract track extraction helper |
| **LOW** | 10.22 | Make merger/scheduler configurable |

---

### Summary of Deletions

| Category | Items Deleted |
|----------|---------------|
| Rust types | `LmbmHypothesis`, `FilterParams`, `FilterThresholds`, `LmbmConfig` |
| Rust methods | `to_legacy_lmbm_config()`, `is_hypothesis_based()` |
| Python classes | `PyFilterThresholds`, `PyFilterLmbmConfig`, `PyAssociatorConfig`, 7 filter classes (Phase 10.12) |
| Python exports | `FilterThresholds`, `FilterLmbmConfig`, `AssociatorConfig`, `_LmbmHypothesis`, `FilterLmb`, `FilterLmbm`, etc. (Phase 10.12) |
| Test usages | ~65 instances across 3 test files |

### API Changes Summary

| Old API | New API |
|---------|---------|
| `FilterThresholds(...)` | Inline params on filter constructor |
| `FilterLmbmConfig(...)` | Inline params on filter constructor |
| `AssociatorConfig.lbp(...)` | `associator="lbp"` + inline params |
| `_LmbmHypothesis` | `_Hypothesis` |
| `LmbmHypothesis` (Rust) | `Hypothesis` |
| 7 filter classes (`FilterLmb`, etc.) | `Filter` + strategy classes (Phase 10.12) |
| `is_hypothesis_based()` | Polymorphic trait methods (Phase 10.10) |
| Raw slice measurements | `impl MeasurementSource` (Phase 10.11) |
| `gm_weight_threshold() -> f64` | `gm_weight_threshold() -> Option<f64>` (Phase 10.13) |

---

## Phase 11: Python Test Cleanup

**Goal**: Eliminate 50%+ of test code via parameterization.

### 1. Implementation Tasks
- [ ] Create `FilterTestCase` dataclass for parameterized testing
- [ ] Refactor `test_equivalence.py` to use parameterized fixtures
- [ ] Remove duplicated sensor 0/1 tests (use `@pytest.mark.parametrize`)
- [ ] Consolidate fixture loading functions in `conftest.py`

### 2. Implementation Design

```python
@dataclass
class FilterTestCase:
    filter_class: type
    fixture_prefix: str
    is_multisensor: bool

FILTER_TEST_CASES = [
    FilterTestCase(FilterLmb, "lmb", False),
    FilterTestCase(FilterLmbm, "lmbm", False),
    FilterTestCase(FilterIcLmb, "ic_lmb", True),
    FilterTestCase(FilterAaLmb, "aa_lmb", True),
    FilterTestCase(FilterGaLmb, "ga_lmb", True),
    FilterTestCase(FilterPuLmb, "pu_lmb", True),
    FilterTestCase(FilterMultisensorLmbm, "multisensor_lmbm", True),
]

@pytest.fixture(params=FILTER_TEST_CASES, ids=lambda c: c.fixture_prefix)
def filter_case(request):
    return request.param
```

### 3. Verification
- [ ] Verify all tests still cover same code paths
- [ ] Verify 1e-10 tolerance unchanged
- [ ] Confirm NO numeric outputs changed

---

## Phase 12: NORFAIR Integration (Future)

**Goal**: Integrate norfair-rs as `NorfairStrategy` implementing `UpdateStrategy` trait.

### 1. Files to CREATE
- [ ] `src/lmb/norfair.rs` - `NorfairStrategy<D: Distance>` implementation
- [ ] `src/lmb/norfair/distance.rs` - Distance trait and implementations

### 2. Type Aliases
```rust
pub type NorfairFilter<D = EuclideanDistance> = UnifiedFilter<NorfairStrategy<D>>;
pub type SortFilter = UnifiedFilter<SortStrategy>;  // Future
pub type ByteTrackFilter = UnifiedFilter<ByteTrackStrategy>;  // Future
```

### 3. Key Differences from LMB/LMBM
| Aspect | NORFAIR | LMB/LMBM |
|--------|---------|----------|
| Birth | From unmatched detections | From BirthModel locations |
| Existence | Hit counter (integer) | Probability (0.0-1.0) |
| Association | Greedy/Hungarian | LBP/Gibbs/Murty |
| Gating | Age threshold | Existence threshold |

---

## Verification Strategy

### Per-Phase Testing
1. **Before**: `cargo test --release` must pass
2. **API changes only**: Tests may need API updates but NOT behavior changes
3. **Tolerance unchanged**: All equivalence tests at 1e-10
4. **After**: Update `./REFACTOR_PLAN.md` with completion status

```bash
cargo test --release
uv run pytest python/tests/ -v
```

---

## Summary

**Net LOC deleted**: ~3419 (Phase 7A: 1285 + Phase 7B: 1634 + Phase 7D: ~500)
**After Phase 8**: ~5919 LOC deleted total, Python API: 21→14 types

### Success Metrics
1. Single `UnifiedFilter<S: UpdateStrategy>` for ALL filter types
2. All tests at 1e-10 tolerance
3. No old code (core.rs, core_lmbm.rs, singlesensor/*, multisensor/lmb.rs, multisensor/lmbm.rs ALL DELETED)
4. ONE way to do each thing (factory functions + escape hatch)
5. Clear path for NORFAIR/SORT/ByteTrack via `UpdateStrategy` trait

### Future Algorithm Support
| Algorithm | Current | After | Notes |
|-----------|---------|-------|-------|
| NORFAIR/SORT/ByteTrack | Blocked | Easy | Implement `UpdateStrategy` |
| GNN/JPDA Associator | Easy | Easy | Implement `Associator` |
| IMM/Custom Motion | Blocked | Easy | Implement `MotionModelBehavior` |
