# Comprehensive Cleanup & Unification Plan for multisensor-lmb-filters-rs

## Executive Summary

This plan addresses major code duplication (~60% overlap between single/multi-sensor filters), API inconsistencies, and architectural gaps. The design prioritizes **extensibility** (downstream can add custom implementations) and **production-grade** qualities: type safety, numerical robustness, and real-time capability.

---

## Core Principles

**These principles are NON-NEGOTIABLE for this refactor:**

1. **Delete old code, don't keep for backward compat**
   - This is a complete rewrite, not a migration
   - Old implementations (singlesensor/*.rs, multisensor/lmb.rs, multisensor/lmbm.rs) MUST be deleted
   - Type aliases maintain API compatibility where sensible, but old code is gone

2. **If tests fail, fix the implementation, not the test**
   - Numeric equivalence at 1e-10 tolerance
   - MATLAB fixtures are ground truth
   - Never relax tolerance to make tests pass

3. **One implementation per filter type**
   - `LmbFilterCore` is THE LMB implementation (single AND multi-sensor)
   - `LmbmFilterCore` is THE LMBM implementation (single AND multi-sensor)
   - No parallel old/new implementations coexisting

4. **Extensibility through traits, not inheritance**
   - `Associator` trait for custom association algorithms
   - `Merger` trait for custom fusion strategies
   - `UpdateScheduler` trait for custom sensor processing
   - `StepReporter` trait for custom observability
   - `MotionModelBehavior` / `SensorModelBehavior` for custom models

---

## Current State Analysis

### Codebase Statistics
- **Rust**: 41 files, ~15,851 lines in `src/`
- **Python Tests**: 1,875 lines in `test_equivalence.py` alone
- **Filters**: 7 types (2 single-sensor, 5 multi-sensor)
- **Associators**: 3 (LBP, Gibbs, Murty)
- **Mergers**: 4 (AA, GA, PU, IC)

### Critical Issues Identified

| Category | Issue | Files | Severity |
|----------|-------|-------|----------|
| **Duplication** | Single/multi-sensor step() ~60% overlap | `singlesensor/lmb.rs`, `multisensor/lmb.rs` | HIGH |
| **Duplication** | LMBM predict/update logic duplicated | `singlesensor/lmbm.rs`, `multisensor/lmbm.rs` | HIGH |
| **Duplication** | 4 trajectory update functions | `common_ops.rs:418,664,689,731` | MEDIUM |
| **Duplication** | 3 pruning functions | `common/utils.rs:39`, `common_ops.rs:33,254` | MEDIUM |
| **Inconsistency** | Type aliases unusable with `::new()` | `multisensor/lmb.rs:24-28` | MEDIUM |
| **Inconsistency** | Multiple config patterns coexist | `config.rs`, filter `with_*()` methods | MEDIUM |
| **Architecture** | No `MotionModel` trait (only struct) | `config.rs:9-77` | HIGH (blocks IMM) |
| **Architecture** | No `Predictor` trait | `components/prediction.rs` | HIGH (blocks extensibility) |
| **Python** | Massive test duplication | `test_equivalence.py` | MEDIUM |

---

## Phase Structure Requirements

**RULES** for EVERY phase:
1. Tests may ONLY change to match new APIs (function signatures, import paths, config types)
2. Tests must NOT change expected numeric values, tolerances, or behavior
3. If a test fails due to numeric differences, the IMPLEMENTATION is wrong - fix code, not tests
4. Each phase MUST end with plan update before starting next phase

---

## Phase 0: Plan Setup ✅

**Goal**: Write the refactor plan to the repo and establish tracking.

### TODO
- [x] Write complete plan to `./REFACTOR_PLAN.md`
- [x] Verify tests pass before starting (`cargo test --release`) - ✅ All 39 Rust tests pass
- [x] Verify Python tests pass (`uv run pytest python/tests/ -v`) - ✅ 86 passed, 1 skipped
- [x] Baseline established on 2026-01-16

---

## Phase 1: Zero-Copy Measurement Input ✅

**Goal**: Accept measurements without forcing allocations.

### 1. Implementation Tasks
- [x] Create `src/lmb/measurements.rs` with `MeasurementSource` trait
- [x] Implement for common types: `&[DVector]`, `&[Vec<DVector>]`, `&[&[DVector]]`
- [x] Export from `src/lmb/mod.rs`

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [x] Add unit tests for `MeasurementSource` trait implementations (9 tests)
- [x] Verify ALL tests pass at 1e-10 tolerance
- [x] Confirm NO numeric outputs changed

### 3. Update Plan & TODOs
- [x] Mark completed tasks in `./REFACTOR_PLAN.md`
- [x] Document any deviations or learnings
- [x] Verify phase is complete before proceeding

### Completion Notes
- Added `SingleSensorMeasurements`, `VecOfVecsMeasurements`, `SliceOfSlicesMeasurements` wrappers
- All wrappers use GATs for zero-copy iteration
- Includes `From` conversions for ergonomic use
- 9 unit tests verifying zero-copy guarantee and correct behavior
- **NOTE**: Integration into filter cores deferred to Phase 8

### Files Modified
- `src/lmb/measurements.rs` - **NEW** (~700 LOC)
- `src/lmb/mod.rs` - Export new types

---

## Phase 2: Type-Safe Configuration (No "God Config") ✅

**Goal**: Make illegal states unrepresentable via composition.

### 1. Implementation Tasks
- [x] Create `CommonConfig` struct for shared settings
- [x] Create `LmbFilterConfig` struct with `common: CommonConfig` + GM-specific fields
- [x] Create `LmbmFilterConfig` struct with `common: CommonConfig` + hypothesis-specific fields
- [x] Create builders for each config type

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [x] Add 10 unit tests for new config types
- [x] Verify all tests pass at 1e-10 tolerance with `cargo test --release`
- [x] Confirm NO numeric outputs changed

### Completion Notes
- Created `CommonConfig`, `LmbFilterConfig`, `LmbmFilterConfig` with builder patterns
- **NOTE**: Filter constructor updates deferred to Phase 7A/7B cleanup

### Files Modified
- `src/lmb/config.rs` - Added new config types (~400 LOC)

---

## Phase 3: Extensible Traits for Models (Open for Extension) ✅

**Goal**: Enable downstream custom implementations WITHOUT modifying upstream.

### 1. Implementation Tasks
- [x] Create `MotionModelBehavior` trait
- [x] Create `SensorModelBehavior` trait
- [x] Implement traits for existing `MotionModel` and `SensorModel` structs
- [x] Add `Box<dyn MotionModelBehavior>` support for custom models

### Completion Notes
- Created both traits with full method signatures
- Implemented for existing structs
- **NOTE**: Filter integration deferred to Phase 7A/7B cleanup

### Files Modified
- `src/lmb/config.rs` - Added trait definitions

---

## Phase 4: Update Strategy Pattern (No Boolean Flags) ✅

**Goal**: Invert control flow so strategies own the loop, not the filter core.

### 1. Implementation Tasks
- [x] Create `src/lmb/scheduler.rs` with `UpdateScheduler` trait
- [x] Implement `ParallelScheduler<M: Merger>` for AA/GA/PU fusion
- [x] Implement `SequentialScheduler` for IC fusion
- [x] Implement `SingleSensorScheduler` for single-sensor filters
- [x] Implement `DynamicScheduler` enum for Python bindings

### Completion Notes
- Created all scheduler types
- **NOTE**: Integration into filter cores deferred to Phase 7A/7B cleanup

### Files Modified
- `src/lmb/scheduler.rs` - **NEW** (~600 LOC)

---

## Phase 5: Observability via StepReporter ✅

**Goal**: Enable debugging without polluting core logic.

### 1. Implementation Tasks
- [x] Create `src/lmb/reporter.rs` with `StepReporter` trait
- [x] Implement `NoOpReporter` (zero-cost default)
- [x] Implement `DebugReporter` (collects all events)
- [x] Implement `LoggingReporter` (log crate integration)
- [x] Implement `CompositeReporter<A, B>` (combine reporters)

### Completion Notes
- Created all reporter types with 9 callback methods
- **NOTE**: Hook integration into filters deferred to Phase 8

### Files Modified
- `src/lmb/reporter.rs` - **NEW** (~600 LOC)

---

## Phase 6: Numerical Robustness (Self-Healing Math) ✅

**Goal**: Filter survives numerical edge cases without crashing.

### 1. Implementation Tasks
- [x] Add `robust_cholesky()` to `src/common/linalg.rs`
- [x] Create `LinalgWarning` enum for recoverable failures
- [x] Create `CholeskyResult` enum distinguishing standard vs regularized decomposition

### Completion Notes
- `robust_cholesky()` auto-regularizes with exponentially increasing epsilon on failure
- **NOTE**: Integration into Kalman updates deferred to Phase 8

### Files Modified
- `src/common/linalg.rs` - Added robust functions (~200 LOC)

---

## Phase 7A: LMB Integration (Delete Old, Use Core)

**Goal**: Make `LmbFilterCore` the ONLY LMB implementation. Delete all old code.

### 1. Files to DELETE (MANDATORY)
- [ ] `src/lmb/singlesensor/lmb.rs` (~549 LOC) - **DELETE ENTIRELY**
- [ ] `src/lmb/multisensor/lmb.rs` (~736 LOC) - **DELETE ENTIRELY**

### 2. Files to MODIFY
- [ ] `src/lmb/singlesensor/mod.rs` - Remove `pub mod lmb;` and `pub use lmb::LmbFilter;`
- [ ] `src/lmb/multisensor/mod.rs` - Remove LMB exports, keep fusion/traits for now
- [ ] `src/lmb/mod.rs` - Change exports to use core module:
  ```rust
  // REMOVE these lines:
  pub use singlesensor::LmbFilter;
  pub use multisensor::{AaLmbFilter, GaLmbFilter, IcLmbFilter, MultisensorLmbFilter, PuLmbFilter};

  // ADD these lines (type aliases from core.rs):
  pub use core::{LmbFilter, IcLmbFilter, AaLmbFilter, GaLmbFilter, PuLmbFilter};
  ```
- [ ] `src/python/filters.rs` - Update imports to use `LmbFilterCore` variants

### 3. Type Alias Strategy
The following type aliases in `core.rs` maintain API compatibility:
```rust
pub type LmbFilter<A = LbpAssociator> = LmbFilterCore<A, SingleSensorScheduler>;
pub type IcLmbFilter<A = LbpAssociator> = LmbFilterCore<A, SequentialScheduler>;
pub type AaLmbFilter<A = LbpAssociator> = LmbFilterCore<A, ParallelScheduler<ArithmeticAverageMerger>>;
pub type GaLmbFilter<A = LbpAssociator> = LmbFilterCore<A, ParallelScheduler<GeometricAverageMerger>>;
pub type PuLmbFilter<A = LbpAssociator> = LmbFilterCore<A, ParallelScheduler<ParallelUpdateMerger>>;
```

### 4. Breaking Changes (Accepted)
- `MultisensorLmbFilter<A, M>` type is REMOVED - use specific type alias instead
- Constructor signatures may differ - update call sites

### 5. Verification
```bash
cargo test --release        # Must pass
uv run pytest python/tests/ -v  # Must pass with same numeric results
```

### 6. Update Plan & TODOs
- [ ] Mark completed tasks in `./REFACTOR_PLAN.md`
- [ ] Verify phase is complete before proceeding

---

## Phase 7B: LMBM Integration (Delete Old, Use Core)

**Goal**: Make `LmbmFilterCore` the ONLY LMBM implementation. Delete all old code.

### 1. Files to DELETE (MANDATORY)
- [ ] `src/lmb/singlesensor/lmbm.rs` (~733 LOC) - **DELETE ENTIRELY**
- [ ] `src/lmb/multisensor/lmbm.rs` (~901 LOC) - **DELETE ENTIRELY**
- [ ] `src/lmb/singlesensor/` - **DELETE ENTIRE DIRECTORY** (should be empty after this)

### 2. Files to MODIFY
- [ ] `src/lmb/mod.rs` - Remove singlesensor module, update LMBM exports:
  ```rust
  // REMOVE:
  pub mod singlesensor;
  pub use singlesensor::LmbmFilter;
  pub use multisensor::MultisensorLmbmFilter;

  // ADD (type aliases from core_lmbm.rs):
  pub use core_lmbm::{LmbmFilter, MultisensorLmbmFilter};
  ```
- [ ] `src/lmb/multisensor/mod.rs` - Remove `pub mod lmbm;` and LMBM exports
- [ ] `src/python/filters.rs` - Update imports to use `LmbmFilterCore` variants

### 3. Type Alias Strategy
```rust
pub type LmbmFilter<A = GibbsAssociator> = LmbmFilterCore<SingleSensorLmbmStrategy<A>>;
pub type MultisensorLmbmFilter<A = MultisensorGibbsAssociator> = LmbmFilterCore<MultisensorLmbmStrategy<A>>;
```

### 4. Verification
```bash
cargo test --release
uv run pytest python/tests/ -v
```

### 5. Update Plan & TODOs
- [ ] Mark completed tasks in `./REFACTOR_PLAN.md`
- [ ] Verify phase is complete before proceeding

---

## Phase 8: Infrastructure Integration

**Goal**: Wire up orphaned infrastructure (MeasurementSource, StepReporter, etc.) into filter cores.

### 1. MeasurementSource Integration
- [ ] Update `LmbFilterCore::step()` to accept `impl MeasurementSource` instead of raw slices
- [ ] Update `LmbmFilterCore::step()` similarly
- [ ] Add conversion impls so existing call sites work

### 2. StepReporter Integration
- [ ] Add `reporter: Option<&mut dyn StepReporter>` to `step()` / `step_detailed()` methods
- [ ] Call reporter hooks at appropriate points:
  - `on_prediction()` after prediction step
  - `on_birth()` after birth injection
  - `on_association()` after data association
  - `on_sensor_update()` after each sensor update
  - `on_fusion()` after track fusion (multi-sensor)
  - `on_pruning()` after track pruning

### 3. Config Type Integration
- [ ] Update filter constructors to accept `LmbFilterConfig` / `LmbmFilterConfig`
- [ ] Remove or deprecate old `FilterParams` usage

### 4. Verification
```bash
cargo test --release
uv run pytest python/tests/ -v
```

---

## Phase 9: Python Bindings Update

**Goal**: Update Python bindings to use unified cores.

### 1. Update Filter Imports
- [ ] `src/python/filters.rs` - Change:
  ```rust
  // FROM (old):
  use crate::lmb::singlesensor::lmb::LmbFilter;
  use crate::lmb::multisensor::lmb::MultisensorLmbFilter;
  use crate::lmb::singlesensor::lmbm::LmbmFilter;
  use crate::lmb::multisensor::lmbm::MultisensorLmbmFilter;

  // TO (new):
  use crate::lmb::core::{LmbFilterCore, LmbFilter, IcLmbFilter, AaLmbFilter, GaLmbFilter, PuLmbFilter};
  use crate::lmb::core_lmbm::{LmbmFilterCore, LmbmFilter, MultisensorLmbmFilter};
  ```

### 2. Update PyFilter Structs
- [ ] `PyFilterLmb` wraps `LmbFilterCore<DynamicAssociator, SingleSensorScheduler>`
- [ ] `PyFilterAaLmb` wraps `LmbFilterCore<DynamicAssociator, ParallelScheduler<ArithmeticAverageMerger>>`
- [ ] Similar for GA, PU, IC variants
- [ ] `PyFilterLmbm` wraps `LmbmFilterCore<SingleSensorLmbmStrategy<DynamicAssociator>>`
- [ ] `PyFilterMultisensorLmbm` wraps `LmbmFilterCore<MultisensorLmbmStrategy<...>>`

### 3. Verification
```bash
uv run pytest python/tests/ -v  # All 87 tests must pass
```

---

## Phase 10: API Cleanup

**Goal**: Remove deprecated types, consolidate exports, clean up public API.

### 1. Remove Deprecated Types
- [ ] Audit `FilterParams` usage - remove if unused externally
- [ ] Remove old `SensorVariant` if replaced by `SensorSet`
- [ ] Remove backward-compat shims added in earlier phases

### 2. Consolidate Config Types
- [ ] Decide: Keep both `FilterThresholds` and `LmbFilterConfig`, or merge?
- [ ] Decide: Keep both `LmbmConfig` and `LmbmFilterConfig`, or merge?

### 3. Clean Up mod.rs Exports
Final `src/lmb/mod.rs` should export:
```rust
// Core filter implementations
pub use core::{LmbFilterCore, LmbFilter, IcLmbFilter, AaLmbFilter, GaLmbFilter, PuLmbFilter, SensorSet};
pub use core_lmbm::{LmbmFilterCore, LmbmFilter, MultisensorLmbmFilter, LmbmSensorSet};

// Traits
pub use traits::{Filter, Associator, Updater, Merger};
pub use scheduler::UpdateScheduler;
pub use reporter::StepReporter;

// Configuration
pub use config::{MotionModel, SensorModel, BirthModel, AssociationConfig, ...};

// Infrastructure
pub use measurements::MeasurementSource;
pub use reporter::{NoOpReporter, DebugReporter, LoggingReporter, CompositeReporter};
pub use scheduler::{SingleSensorScheduler, SequentialScheduler, ParallelScheduler, DynamicScheduler};

// Fusion mergers
pub use multisensor::fusion::{ArithmeticAverageMerger, GeometricAverageMerger, ParallelUpdateMerger, IteratedCorrectorMerger};

// Errors
pub use errors::{FilterError, AssociationError};
```

### 4. Update Documentation
- [ ] Update module-level docs in `mod.rs`
- [ ] Update `lib.rs` examples to use new API

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

## Verification Strategy

### Per-Phase Testing Requirements
1. **Before starting**: `cargo test --release` must pass
2. **API changes only**: Tests may need API updates but NOT behavior changes
3. **Tolerance unchanged**: All equivalence tests at 1e-10
4. **After completing**: Update `./REFACTOR_PLAN.md` with completion status

### Commands
```bash
# Before each phase:
cargo test --release

# After each phase:
cargo test --release
uv run pytest python/tests/ -v
```

---

## Critical Files Summary

| File | Action | Impact |
|------|--------|--------|
| `src/lmb/singlesensor/lmb.rs` | **DELETE** | -549 LOC |
| `src/lmb/singlesensor/lmbm.rs` | **DELETE** | -733 LOC |
| `src/lmb/singlesensor/mod.rs` | **DELETE** | Directory removed |
| `src/lmb/multisensor/lmb.rs` | **DELETE** | -736 LOC |
| `src/lmb/multisensor/lmbm.rs` | **DELETE** | -901 LOC |
| `src/lmb/core.rs` | **KEEP** | Primary LMB impl (~1400 LOC) |
| `src/lmb/core_lmbm.rs` | **KEEP** | Primary LMBM impl (~1600 LOC) |
| `src/lmb/mod.rs` | **MODIFY** | Update exports |
| `src/python/filters.rs` | **MODIFY** | Use new cores |
| `python/tests/test_equivalence.py` | **REFACTOR** | -1200 LOC via parameterization |

**Net LOC change**: Delete ~2919 LOC of old filters, keep ~3000 LOC unified cores = **~40% reduction** in filter code

---

## Implementation Order

```
Phase 0-6 (Foundation)      ─► ✅ COMPLETE (infrastructure created)
         │
         ▼
Phase 7A (LMB Cleanup)      ─┐
Phase 7B (LMBM Cleanup)     ─┼─► DELETE OLD CODE
         │                  ─┘
         ▼
Phase 8 (Infrastructure)    ─► Wire up MeasurementSource, StepReporter
         │
         ▼
Phase 9 (Python Bindings)   ─► Update to use unified cores
         │
         ▼
Phase 10 (API Cleanup)      ─► Remove deprecated types
         │
         ▼
Phase 11 (Python Tests)     ─► Parameterize tests
```

---

## Success Metrics

1. **Code Reduction**: ~40% reduction in `src/lmb/` filter code
2. **No Tolerance Changes**: All tests at 1e-10
3. **Single Implementation**: One `LmbFilterCore`, one `LmbmFilterCore`
4. **No Old Code**: singlesensor/lmb.rs, multisensor/lmb.rs, etc. DELETED
5. **Test Reduction**: `test_equivalence.py` < 800 LOC
6. **Extensibility**: Downstream can add custom motion/sensor models
7. **Infrastructure Integrated**: MeasurementSource, StepReporter actually used

---

## Future Algorithm Support Analysis

| Algorithm | Current | After Refactor | Notes |
|-----------|---------|----------------|-------|
| **GNN Associator** | Easy | Easy | Just add new `Associator` impl |
| **JPDA Associator** | Easy | Easy | Just add new `Associator` impl |
| **PMBM** | Medium | Easy | New hypothesis management |
| **Full GLMB** | Medium | Easy | Extend hypothesis tracking |
| **IMM** | Blocked | Easy | Downstream implements `MotionModelBehavior` |
| **Custom Motion** | Blocked | Easy | Downstream implements `MotionModelBehavior` |
| **Custom Sensor** | Blocked | Easy | Downstream implements `SensorModelBehavior` |
