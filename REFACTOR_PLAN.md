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

3. **One algorithm implementation per filter type**
   - `LmbAlgorithm` is THE LMB implementation (single AND multi-sensor)
   - `LmbmAlgorithm` is THE LMBM implementation (single AND multi-sensor)
   - Future: `NorfairAlgorithm`, `SortAlgorithm` for simpler trackers
   - No parallel old/new implementations coexisting

4. **Extensibility through traits, not inheritance**
   - `FilterAlgorithm` trait for different tracking paradigms (LMB, LMBM, NORFAIR, SORT)
   - `Associator` trait for custom association algorithms
   - `Merger` trait for custom fusion strategies
   - `UpdateScheduler` trait for custom sensor processing
   - `StepReporter` trait for custom observability
   - `MotionModelBehavior` / `SensorModelBehavior` for custom models

---

## Unified Filter Architecture

### Why Not Full Generic `FilterCore<S, A, U, M>`?

LMB, LMBM, and simpler trackers (NORFAIR, SORT, ByteTrack) have **fundamentally different semantics**:

| Aspect | LMB | LMBM | NORFAIR/SORT |
|--------|-----|------|--------------|
| **State** | `Vec<Track>` (GM mixture) | `Vec<LmbmHypothesis>` | `Vec<TrackedObject>` |
| **Existence** | Probabilistic (0.0-1.0) | Hypothesis weights | Hit counters (age) |
| **Association** | Marginal (soft) | Joint hypothesis (sampled) | Greedy/Hungarian (hard) |
| **Cardinality** | From existence marginals | From hypothesis mixture | Fixed per frame |

A full generic `FilterCore<State, Assoc, Update, Motion>` would:
- Create type parameter explosion
- Force unnatural abstractions (can't meaningfully swap LMB association into NORFAIR)
- Obscure fundamental semantic differences

### Chosen Approach: `FilterAlgorithm` Trait

```rust
/// Core abstraction - each filter type implements this
pub trait FilterAlgorithm: Send + Sync {
    type State: Clone;           // Vec<Track> | Vec<LmbmHypothesis> | Vec<TrackedObject>
    type Measurements;           // &[DVector] | impl MeasurementSource | Vec<Detection>
    type DetailedOutput;         // Algorithm-specific debug output

    fn predict(&mut self, motion: &dyn MotionModelBehavior, timestep: usize);
    fn inject_birth(&mut self, birth: &BirthModel, timestep: usize);
    fn associate_and_update<R: Rng>(
        &mut self,
        rng: &mut R,
        measurements: &Self::Measurements,
        sensors: &SensorSet,
    ) -> Result<(), FilterError>;
    fn normalize_and_gate(&mut self, config: &GatingConfig);
    fn extract_estimate(&self, timestamp: usize) -> StateEstimate;
    fn extract_detailed(&self) -> Self::DetailedOutput;
}

/// Unified filter wrapper - shared infrastructure
pub struct Filter<A: FilterAlgorithm> {
    algorithm: A,
    motion: MotionModel,  // or Box<dyn MotionModelBehavior>
    sensors: SensorSet,
    birth: BirthModel,
    gating: GatingConfig,
    trajectories: Vec<Trajectory>,
}

impl<A: FilterAlgorithm> Filter<A> {
    pub fn step<R: Rng>(
        &mut self,
        rng: &mut R,
        measurements: &A::Measurements,
        ts: usize,
    ) -> Result<StateEstimate, FilterError> {
        self.algorithm.predict(&self.motion, ts);
        self.algorithm.inject_birth(&self.birth, ts);
        self.algorithm.associate_and_update(rng, measurements, &self.sensors)?;
        self.algorithm.normalize_and_gate(&self.gating);
        Ok(self.algorithm.extract_estimate(ts))
    }
}
```

### Algorithm Implementations

| Algorithm | Type Parameters | Description |
|-----------|-----------------|-------------|
| `LmbAlgorithm<A: Associator, S: UpdateScheduler>` | Associator + Scheduler | Marginal GM posteriors |
| `LmbmAlgorithm<S: LmbmAssociator>` | LMBM strategy | Hypothesis mixtures |
| `NorfairAlgorithm<D: Distance>` | Distance function | Hit counter lifecycle (future) |
| `SortAlgorithm` | None | Hungarian + Kalman (future) |

### Type Aliases (API Compatibility)

```rust
// LMB variants
pub type LmbFilter<A = LbpAssociator> = Filter<LmbAlgorithm<A, SingleSensorScheduler>>;
pub type IcLmbFilter<A = LbpAssociator> = Filter<LmbAlgorithm<A, SequentialScheduler>>;
pub type AaLmbFilter<A = LbpAssociator> = Filter<LmbAlgorithm<A, ParallelScheduler<ArithmeticAverageMerger>>>;
pub type GaLmbFilter<A = LbpAssociator> = Filter<LmbAlgorithm<A, ParallelScheduler<GeometricAverageMerger>>>;
pub type PuLmbFilter<A = LbpAssociator> = Filter<LmbAlgorithm<A, ParallelScheduler<ParallelUpdateMerger>>>;

// LMBM variants
pub type LmbmFilter<A = GibbsAssociator> = Filter<LmbmAlgorithm<SingleSensorLmbmStrategy<A>>>;
pub type MultisensorLmbmFilter<A = MultisensorGibbsAssociator> = Filter<LmbmAlgorithm<MultisensorLmbmStrategy<A>>>;

// Future: Simpler trackers
pub type NorfairFilter<D = EuclideanDistance> = Filter<NorfairAlgorithm<D>>;
```

### Benefits

1. **Clear API**: `Filter<LmbAlgorithm>` vs `Filter<NorfairAlgorithm>`
2. **Shared infrastructure**: Motion models, birth, trajectories, reporters live in `Filter<A>`
3. **Each algorithm owns its complexity**: No forced abstractions
4. **Easy to add new trackers**: Just implement `FilterAlgorithm` trait
5. **NORFAIR integration path**: Rewrite norfair-rs core as `NorfairAlgorithm`

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

## Phase 7A: Delete Old LMB Code + Use Unified Core ✅

**Goal**: Delete old duplicate LMB implementations. Use unified `LmbFilterCore` from `core.rs`.

**NOTE**: The full `FilterAlgorithm` trait creation is deferred. The immediate goal of eliminating 1285 LOC of duplicate code is achieved by using the existing `LmbFilterCore<A, S>` from `core.rs`.

### 1. Files DELETED ✅
- [x] `src/lmb/singlesensor/lmb.rs` (~549 LOC) - **DELETED**
- [x] `src/lmb/multisensor/lmb.rs` (~736 LOC) - **DELETED**

### 2. Files MODIFIED ✅
- [x] `src/lmb/singlesensor/mod.rs` - Removed `pub mod lmb;` and LMB exports
- [x] `src/lmb/multisensor/mod.rs` - Removed LMB exports, added `MultisensorMeasurements` type
- [x] `src/lmb/mod.rs` - Updated to export LMB filters from `core.rs`:
  ```rust
  // Single-sensor filters (from unified cores)
  pub use core::{LmbFilter, LmbFilterCore};

  // Multi-sensor LMB filters (from unified core)
  pub use core::{AaLmbFilter, GaLmbFilter, IcLmbFilter, PuLmbFilter};
  ```
- [x] `src/lib.rs` - Removed `MultisensorLmbFilter` export
- [x] `src/bench_utils.rs` - Updated to use new type aliases and constructors
- [x] `src/python/filters.rs` - Updated imports and constructors
- [x] `tests/bench_fixtures.rs` - Updated imports and constructors

### 3. Type Aliases (Now in core.rs)
```rust
pub type LmbFilter<A = LbpAssociator> = LmbFilterCore<A, SingleSensorScheduler>;
pub type IcLmbFilter<A = LbpAssociator> = LmbFilterCore<A, SequentialScheduler>;
pub type AaLmbFilter<A = LbpAssociator> = LmbFilterCore<A, ParallelScheduler<ArithmeticAverageMerger>>;
pub type GaLmbFilter<A = LbpAssociator> = LmbFilterCore<A, ParallelScheduler<GeometricAverageMerger>>;
pub type PuLmbFilter<A = LbpAssociator> = LmbFilterCore<A, ParallelScheduler<ParallelUpdateMerger>>;
```

### 4. Breaking Changes (Accepted)
- `MultisensorLmbFilter<A, M>` type is REMOVED - use specific type aliases

### 5. Verification ✅
```bash
cargo test --release        # ✅ All tests pass
uv run pytest python/tests/ -v  # ✅ 86 passed, 1 skipped
```

### Completion Notes (2026-01-17)
- Deleted 1285 LOC of duplicate code
- All tests pass with unchanged numeric results
- Python bindings updated to use unified core

---

## Phase 7B: Delete Old LMBM Code + Use Unified Core ✅

**Goal**: Delete old duplicate LMBM implementations. Use unified `LmbmFilterCore` from `core_lmbm.rs`.

**NOTE**: Like Phase 7A, the full `FilterAlgorithm` trait creation is deferred. The immediate goal of eliminating 1634 LOC of duplicate code is achieved by using the existing `LmbmFilterCore` from `core_lmbm.rs`.

### 1. Files DELETED ✅
- [x] `src/lmb/singlesensor/lmbm.rs` (~733 LOC) - **DELETED**
- [x] `src/lmb/multisensor/lmbm.rs` (~901 LOC) - **DELETED**
- [x] `src/lmb/singlesensor/` - **ENTIRE DIRECTORY DELETED**

### 2. Files MODIFIED ✅
- [x] `src/lmb/mod.rs` - Removed `pub mod singlesensor;`
- [x] `src/lmb/multisensor/mod.rs` - Removed `pub mod lmbm;`

### 3. Type Aliases (Already in core_lmbm.rs)
```rust
pub type LmbmFilter<A = GibbsAssociator> = LmbmFilterCore<SingleSensorLmbmStrategy<A>>;
pub type MultisensorLmbmFilter = LmbmFilterCore<MultisensorLmbmStrategy<MultisensorGibbsAssociator>>;
```

### 4. Verification ✅
```bash
cargo test --release        # ✅ All tests pass
uv run pytest python/tests/ -v  # ✅ 86 passed, 1 skipped
```

### Completion Notes (2026-01-17)
- Deleted 1634 LOC of duplicate code (733 + 901)
- All tests pass with unchanged numeric results
- Total code deleted in Phase 7A+7B: 2919 LOC

---

## Phase 7C: API Simplification ✅

**Goal**: Flatten constructor chains, remove unnecessary generics from public API, merge duplicated types.

### Problem Analysis

The explore agent identified these issues:

| Issue | Location | Impact |
|-------|----------|--------|
| Generic type aliases | `core.rs:1117-1143` | `LmbFilter<A = LbpAssociator>` exposes implementation details |
| Constructor proliferation | `core.rs` (15 impl blocks) | 6 constructors all funnel to `with_scheduler()` |
| Duplicated enums | `core.rs`, `core_lmbm.rs` | `SensorSet` duplicated as `LmbmSensorSet` |
| Deep call chains | `core_lmbm.rs` | 4-level indirection: `new()` → `with_associator()` → `with_scheduler()` → internal |

### 1. Create `factory.rs` for All Filter Constructors

**New file**: `src/lmb/factory.rs`

```rust
//! Factory functions for creating LMB and LMBM filters.
//!
//! These functions provide simple, one-call construction for common filter configurations.
//! For custom associators or schedulers, use `LmbFilterCore::with_scheduler()` directly.

use crate::lmb::core::{LmbFilterCore, LmbFilter, IcLmbFilter, AaLmbFilter, GaLmbFilter, PuLmbFilter};
use crate::lmb::core_lmbm::{LmbmFilterCore, LmbmFilter, MultisensorLmbmFilter};
use crate::lmb::config::{MotionModel, SensorModel, MultisensorConfig, BirthModel, AssociationConfig};
// ... imports

/// Create a single-sensor LMB filter with default LBP associator.
pub fn lmb_filter(
    motion: MotionModel,
    sensor: SensorModel,
    birth: BirthModel,
    association: AssociationConfig,
) -> LmbFilter {
    LmbFilterCore::with_scheduler(
        motion,
        sensor.into(),
        birth,
        association,
        LbpAssociator,
        SingleSensorScheduler::new(),
    )
}

/// Create an IC-LMB (Iterated Corrector) multi-sensor filter.
pub fn ic_lmb_filter(
    motion: MotionModel,
    sensors: MultisensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
) -> IcLmbFilter {
    LmbFilterCore::with_scheduler(
        motion,
        sensors.into(),
        birth,
        association,
        LbpAssociator,
        SequentialScheduler::new(),
    )
}

/// Create an AA-LMB (Arithmetic Average) multi-sensor filter.
pub fn aa_lmb_filter(
    motion: MotionModel,
    sensors: MultisensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
    max_hypotheses: usize,
) -> AaLmbFilter {
    let merger = ArithmeticAverageMerger::uniform(sensors.num_sensors(), max_hypotheses);
    LmbFilterCore::with_scheduler(
        motion,
        sensors.into(),
        birth,
        association,
        LbpAssociator,
        ParallelScheduler::new(merger),
    )
}

/// Create a GA-LMB (Geometric Average) multi-sensor filter.
pub fn ga_lmb_filter(
    motion: MotionModel,
    sensors: MultisensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
) -> GaLmbFilter {
    let merger = GeometricAverageMerger::uniform(sensors.num_sensors());
    LmbFilterCore::with_scheduler(
        motion,
        sensors.into(),
        birth,
        association,
        LbpAssociator,
        ParallelScheduler::new(merger),
    )
}

/// Create a PU-LMB (Parallel Update) multi-sensor filter.
pub fn pu_lmb_filter(
    motion: MotionModel,
    sensors: MultisensorConfig,
    birth: BirthModel,
    association: AssociationConfig,
) -> PuLmbFilter {
    let merger = ParallelUpdateMerger::new(Vec::new());
    LmbFilterCore::with_scheduler(
        motion,
        sensors.into(),
        birth,
        association,
        LbpAssociator,
        ParallelScheduler::new(merger),
    )
}

/// Create a single-sensor LMBM filter with default Gibbs associator.
pub fn lmbm_filter(
    motion: MotionModel,
    sensor: SensorModel,
    birth: BirthModel,
    lmbm_config: LmbmConfig,
) -> LmbmFilter { ... }

/// Create a multi-sensor LMBM filter.
pub fn multisensor_lmbm_filter(
    motion: MotionModel,
    sensors: MultisensorConfig,
    birth: BirthModel,
    lmbm_config: LmbmConfig,
) -> MultisensorLmbmFilter { ... }
```

- [x] Create `src/lmb/factory.rs` with all 7 factory functions
- [x] Export from `src/lmb/mod.rs`

### 2. Merge Duplicated SensorSet Enums

**Current**: Two nearly identical enums
```rust
// core.rs
pub enum SensorSet { Single(SensorModel), Multi(MultisensorConfig) }

// core_lmbm.rs
pub enum LmbmSensorSet { Single(SensorModel), Multi(MultisensorConfig) }
```

**After**: Single shared enum in `config.rs`
```rust
// config.rs
pub enum SensorSet {
    Single(SensorModel),
    Multi(MultisensorConfig),
}
```

- [x] Move `SensorSet` to `config.rs`
- [x] Delete `LmbmSensorSet` from `core_lmbm.rs`
- [x] Update both cores to use shared `SensorSet`

### 3. Simplify Type Aliases (Remove Generics from Public API)

**Current**: Generics leak into public API
```rust
pub type LmbFilter<A = LbpAssociator> = LmbFilterCore<A, SingleSensorScheduler>;
```

**After**: Concrete types for common cases
```rust
// No generics visible to users
pub type LmbFilter = LmbFilterCore<LbpAssociator, SingleSensorScheduler>;
pub type IcLmbFilter = LmbFilterCore<LbpAssociator, SequentialScheduler>;
pub type AaLmbFilter = LmbFilterCore<LbpAssociator, ParallelScheduler<ArithmeticAverageMerger>>;
pub type GaLmbFilter = LmbFilterCore<LbpAssociator, ParallelScheduler<GeometricAverageMerger>>;
pub type PuLmbFilter = LmbFilterCore<LbpAssociator, ParallelScheduler<ParallelUpdateMerger>>;

// For custom associators - use LmbFilterCore<A, S> directly
```

- [x] Remove `<A = LbpAssociator>` from all type aliases in `core.rs`
- [x] Remove generic parameters from type aliases in `core_lmbm.rs`

### 4. Remove Constructor Impl Blocks from Core Files

**Current** (`core.rs`): 15 impl blocks, 6 specialized constructors
```rust
impl LmbFilterCore<LbpAssociator, SingleSensorScheduler> {
    pub fn new(...) -> Self { Self::with_scheduler(...) }
}
impl<A: Associator> LmbFilterCore<A, SingleSensorScheduler> {
    pub fn with_associator(...) -> Self { Self::with_scheduler(...) }
}
// ... 4 more constructor impl blocks
```

**After**: Keep only essential impl blocks
```rust
// Generic implementation - the one escape hatch
impl<A: Associator, S: UpdateScheduler> LmbFilterCore<A, S> {
    pub fn with_scheduler(...) -> Self { ... }  // Keep this for custom configs
    // ... shared methods
}

// Per-scheduler step() implementations (required for different Measurements types)
impl<A: Associator> LmbFilterCore<A, SingleSensorScheduler> { ... }
impl<A: Associator> LmbFilterCore<A, SequentialScheduler> { ... }
impl<A: Associator, M: Merger> LmbFilterCore<A, ParallelScheduler<M>> { ... }
```

- [ ] Remove `new()`, `new_ic()`, `new_parallel()` from `core.rs`
- [ ] Remove `with_associator()`, `with_associator_ic()`, `with_associator_parallel()` from `core.rs`
- [ ] Keep only `with_scheduler()` as escape hatch
- [ ] Apply same cleanup to `core_lmbm.rs`

### 5. Files Summary

| File | Action |
|------|--------|
| `src/lmb/factory.rs` | **NEW** - All 7 factory functions |
| `src/lmb/config.rs` | Add shared `SensorSet` enum |
| `src/lmb/core.rs` | Remove constructor impl blocks, simplify type aliases |
| `src/lmb/core_lmbm.rs` | Remove `LmbmSensorSet`, remove constructor impl blocks |
| `src/lmb/mod.rs` | Export `factory::*` functions |
| `src/python/filters.rs` | Update to use factory functions |
| `tests/bench_fixtures.rs` | Update to use factory functions |

### 6. API Before/After

```rust
// BEFORE (verbose, exposes internals)
let filter: LmbFilter<LbpAssociator> = LmbFilter::new(motion, sensor, birth, config);
let filter: AaLmbFilter<LbpAssociator> = AaLmbFilter::new_parallel(
    motion, sensors, birth, config,
    ParallelScheduler::new(ArithmeticAverageMerger::uniform(2, 100))
);

// AFTER (simple factory functions)
use multisensor_lmb_filters_rs::lmb::{lmb_filter, aa_lmb_filter};

let filter = lmb_filter(motion, sensor, birth, config);
let filter = aa_lmb_filter(motion, sensors, birth, config, 100);

// Custom associator (escape hatch - explicit about complexity)
let filter: LmbFilterCore<MyAssociator, SingleSensorScheduler> =
    LmbFilterCore::with_scheduler(motion, sensor.into(), birth, config, MyAssociator, SingleSensorScheduler::new());
```

### 7. Verification
```bash
cargo test --release
uv run pytest python/tests/ -v
```

### 8. Success Criteria
- [x] New `factory.rs` with 7 factory functions
- [x] Zero generic parameters visible in standard type aliases
- [x] Single `SensorSet` enum (no duplication)
- [ ] ≤4 impl blocks per core file (down from 15/10) - deferred, not blocking
- [x] All tests pass unchanged

### Completion Notes (2026-01-17)
- Created `factory.rs` with 7 factory functions: `lmb_filter()`, `ic_lmb_filter()`, `aa_lmb_filter()`, `ga_lmb_filter()`, `pu_lmb_filter()`, `lmbm_filter()`, `multisensor_lmbm_filter()`
- Merged `SensorSet` into `config.rs`, deleted `LmbmSensorSet`
- Simplified type aliases to remove generic parameters
- Updated `bench_utils.rs` and test files to use new API
- Constructor impl block removal moved to Phase 7D

---

## Phase 7D: Dead Code Cleanup ✅

**Goal**: Remove ALL redundant code paths. ONE way to do each thing.

**Principle**:
- Factory functions = public API for common cases
- `with_scheduler()` / `with_strategy()` = escape hatch for custom configs
- DELETE everything else

### 1. Deleted Redundant Constructors ✅

The following redundant constructors were deleted:

| Item | File | Status |
|------|------|--------|
| 6 redundant constructor impl blocks | `core.rs` | ✅ Deleted (commit 76c7679) |
| `SensorVariant` enum + impl | `config.rs` | ✅ Deleted (commit 76c7679) |
| `CommonConfigBuilder` | `config.rs` | ✅ Deleted (commit 76c7679) |
| `LmbFilterConfigBuilder` | `config.rs` | ✅ Deleted (commit 76c7679) |
| `LmbmFilterConfigBuilder` | `config.rs` | ✅ Deleted (commit 76c7679) |
| `FilterParamsBuilder` | `config.rs` | ✅ Deleted (commit 76c7679) |
| `SingleSensorLmbmStrategy::new()` | `core_lmbm.rs` | ✅ Deleted (commit 11b9281) |
| `MultisensorLmbmStrategy::new()` | `core_lmbm.rs` | ✅ Deleted (commit 11b9281) |

### 2. API After Cleanup

```rust
// === PUBLIC API (Factory Functions) ===
use multisensor_lmb_filters_rs::lmb::{lmb_filter, aa_lmb_filter, lmbm_filter, ...};

let filter = lmb_filter(motion, sensor, birth, assoc_config);
let filter = aa_lmb_filter(motion, sensors, birth, assoc_config, max_hyp);
let filter = lmbm_filter(motion, sensor, birth, assoc_config, lmbm_config);

// === ESCAPE HATCH (Custom Associators) ===
use multisensor_lmb_filters_rs::lmb::{LmbFilterCore, LmbmFilterCore};

let filter: LmbFilterCore<MyAssociator, SingleSensorScheduler> =
    LmbFilterCore::with_scheduler(motion, sensor.into(), birth, assoc, MyAssociator, scheduler);

let filter: LmbmFilterCore<SingleSensorLmbmStrategy<MyAssociator>> =
    LmbmFilterCore::with_strategy(motion, sensor.into(), birth, assoc, lmbm, strategy);
```

### 3. Verification ✅

```bash
cargo test --release        # ✅ All tests pass
cargo clippy --all-targets  # ✅ No errors
uv run pytest python/tests/ -v  # ✅ 86 passed, 1 skipped
```

### Completion Notes (2026-01-17)

- Deleted all redundant constructors from `core.rs` and `core_lmbm.rs`
- Deleted `SensorVariant` (duplicate of `SensorSet`)
- Deleted unused config builders
- Deleted `SingleSensorLmbmStrategy::new()` and `MultisensorLmbmStrategy::new()`
- Made strategy `associator` fields `pub(crate)` for internal struct construction
- Factory functions and `with_scheduler()`/`with_strategy()` are the only ways to create filters

---

## Phase 7E: (Merged into Phase 8.7)

**Note**: Python API simplification is now part of Phase 8.7. This avoids rewriting
Python bindings twice (once for API simplification, once for UnifiedFilter migration).

See Phase 8.7 for the combined scope.

---

## Phase 8: Full LMB/LMBM Unification ✅

**Goal**: Delete both `core.rs` and `core_lmbm.rs`. Replace with a single `UnifiedFilter<S: UpdateStrategy>` that handles LMB, LMBM, and future algorithms (NORFAIR, SORT) through strategy parameterization.

**Key Insight**: The "single component per track in LMBM" is an **updater constraint**, not a structural requirement. `Track` already supports `SmallVec<[GaussianComponent; 4]>`. We can represent:
- **LMB**: Single hypothesis with multi-component tracks
- **LMBM**: Multiple hypotheses with single-component tracks

### 8.1: Rename Hypothesis Struct ✅

**File**: `src/lmb/types.rs`

- [x] Rename `LmbmHypothesis` → `Hypothesis`
- [x] Add `lmb()` constructor for single-hypothesis LMB state
- [x] Update all usages (grep for `LmbmHypothesis`)

### 8.2: Create UpdateStrategy Trait ✅

**File**: `src/lmb/strategy.rs` (NEW)

```rust
pub trait UpdateStrategy: Send + Sync {
    type Measurements;
    fn predict(&self, hypotheses: &mut Vec<Hypothesis>, motion: &MotionModel, birth: &BirthModel, ts: usize);
    fn update<R: Rng>(&self, rng: &mut R, hypotheses: &mut Vec<Hypothesis>, meas: &Self::Measurements, ctx: &UpdateContext) -> Result<UpdateIntermediate, FilterError>;
    fn prune(&self, hypotheses: &mut Vec<Hypothesis>, trajectories: &mut Vec<Trajectory>, config: &PruneConfig);
    fn extract(&self, hypotheses: &[Hypothesis], ts: usize) -> StateEstimate;
    fn name(&self) -> &'static str;
    fn is_hypothesis_based(&self) -> bool;
}
```

- [x] Create `src/lmb/strategy.rs`
- [x] Define `UpdateStrategy` trait
- [x] Define `UpdateContext`, `CommonPruneConfig`, `LmbPruneConfig`, `LmbmPruneConfig`, `UpdateIntermediate`
- [x] Export from `mod.rs`

### 8.3: Implement LmbStrategy ✅

**File**: `src/lmb/strategy.rs`

```rust
pub struct LmbStrategy<A: Associator, S: UpdateScheduler> {
    associator: A,
    scheduler: S,
    updater: MarginalUpdater,
    prune_config: LmbPruneConfig,
}
```

**Key behaviors**:
- `predict()`: Calls `predict_tracks()` on `hypotheses[0].tracks`
- `update()`: Delegates to scheduler, uses `MarginalUpdater`, captures per-sensor data
- `prune()`: GM component pruning + track gating (NOT hypothesis pruning)
- `extract()`: MAP cardinality from single hypothesis
- Invariant: `hypotheses.len() == 1` always

- [x] Implement for `SingleSensorScheduler`
- [x] Implement for `SequentialScheduler` (with per-sensor update capture)
- [x] Implement for `ParallelScheduler<M>` (with per-sensor update capture)

### 8.4: Implement LmbmStrategy ✅

**File**: `src/lmb/strategy.rs`

```rust
pub struct LmbmStrategy<S: LmbmAssociator> {
    inner: S,
    prune_config: LmbmPruneConfig,
}
```

**Key behaviors**:
- `predict()`: Calls `predict_tracks()` on each hypothesis
- `update()`: Delegates to `LmbmAssociator`, which branches hypotheses
- `prune()`: Hypothesis weight normalization + gating + track pruning
- `extract()`: Weighted cardinality across hypotheses (MAP or EAP)

- [x] Implement for single-sensor (`SingleSensorLmbmStrategy`)
- [x] Implement for multi-sensor (`MultisensorLmbmStrategy`)

### 8.5: Create UnifiedFilter Struct ✅

**File**: `src/lmb/unified.rs` (NEW)

```rust
pub struct UnifiedFilter<S: UpdateStrategy> {
    motion: MotionModel,
    sensors: SensorSet,
    birth: BirthModel,
    association_config: AssociationConfig,
    common_prune: CommonPruneConfig,
    hypotheses: Vec<Hypothesis>,
    trajectories: Vec<Trajectory>,
    strategy: S,
}
```

- [x] Create `src/lmb/unified.rs`
- [x] Implement `UnifiedFilter` struct
- [x] Implement `Filter` trait for all scheduler variants
- [x] Add `step_detailed()` for fixture validation (with sensor_updates and correct cardinality)
- [x] Export from `mod.rs`

### 8.6: Update Factory Functions ✅

**File**: `src/lmb/factory.rs`

- [x] Update all 7 factory functions to return `UnifiedFilter<...>`
- [x] Update type aliases in `mod.rs`

### 8.7: Update Python Bindings ✅

**Goal**: Update Python bindings to use `UnifiedFilter` internally while preserving backward-compatible API.

**Implementation Note**: Instead of the Strategy Object Pattern originally planned, we kept the existing 7 filter classes (`FilterLmb`, `FilterLmbm`, `FilterIcLmb`, `FilterAaLmb`, `FilterGaLmb`, `FilterPuLmb`, `FilterMultisensorLmbm`) but updated their internals to wrap `UnifiedFilter<...>` with appropriate strategies. This approach:
- Preserves backward compatibility for existing users
- Reduces migration effort
- Still achieves the goal of unified internal architecture

#### Changes Made

**Updated** (7 filter classes - now wrap `UnifiedFilter`):
- `PyFilterLmb` → `inner: UnifiedFilter<LmbStrategy<DynamicAssociator, SingleSensorScheduler>>`
- `PyFilterLmbm` → `inner: UnifiedFilter<LmbmStrategy<SingleSensorLmbmStrategy<GibbsAssociator>>>`
- `PyFilterIcLmb` → `inner: UnifiedFilter<LmbStrategy<LbpAssociator, SequentialScheduler>>`
- `PyFilterAaLmb` → `inner: UnifiedFilter<LmbStrategy<LbpAssociator, ParallelScheduler<ArithmeticAverageMerger>>>`
- `PyFilterGaLmb` → `inner: UnifiedFilter<LmbStrategy<LbpAssociator, ParallelScheduler<GeometricAverageMerger>>>`
- `PyFilterPuLmb` → `inner: UnifiedFilter<LmbStrategy<LbpAssociator, ParallelScheduler<ParallelUpdateMerger>>>`
- `PyFilterMultisensorLmbm` → `inner: UnifiedFilter<LmbmStrategy<MultisensorLmbmStrategy<MultisensorGibbsAssociator>>>`

**Added**:
- `set_tracks()` method to `UnifiedFilter` for setting prior tracks
- `get_config()` method to `UnifiedFilter` returning `FilterConfigSnapshot`
- Config getter methods to `UpdateStrategy` trait: `gm_weight_threshold()`, `max_gm_components()`, `lmbm_config()`
- `sensor_updates` field to `UpdateIntermediate` for per-sensor update capture
- Per-sensor data capture in `ParallelScheduler` and `SequentialScheduler` update methods

**Fixed**:
- Cardinality computed from `updated_tracks` (before pruning) to match MATLAB behavior
- `objects_likely_to_exist` computed from `prune()` return value (after normalization for LMBM)

#### Verification ✅

```bash
cargo test --release        # ✅ All 13 tests pass
uv run pytest python/tests/ -v  # ✅ 86 passed, 1 skipped
```

### 8.8: Delete Old Cores ✅

- [x] Delete `src/lmb/core.rs` (~1166 LOC)
- [x] Delete `src/lmb/core_lmbm.rs` (~1450 LOC)
- [x] Update `mod.rs` imports

### 8.9: Clean Up common_ops.rs ✅

**KEPT** (still used by strategy.rs):
- `predict_all_hypotheses()` - used by strategy implementations
- `update_hypothesis_trajectories()` - used by `UpdateStrategy::update_trajectories()`
- `init_hypothesis_birth_trajectories()` - used by `UpdateStrategy::init_birth_trajectories()`

**NOTE**: These functions are still needed because `UnifiedFilter::step_detailed()` delegates to the strategy which calls these helpers. They are NOT redundant.

**KEPT**: All component-level operations, gating functions, extraction functions

- [x] Verified functions are still in use
- [x] No changes needed - helpers are correctly delegated to by strategy trait default methods

### Completion Notes (2026-01-18)

Phase 8 is now complete. Key achievements:
- Deleted `core.rs` (~1166 LOC) and `core_lmbm.rs` (~1450 LOC)
- Created `strategy.rs` (~1700 LOC) with `UpdateStrategy` trait and implementations
- Created `unified.rs` (~600 LOC) with `UnifiedFilter<S>` struct
- Updated all 7 Python filter classes to use `UnifiedFilter` internally
- All 86 Python tests pass at 1e-10 tolerance
- All 13 Rust tests pass
- Net LOC reduction: ~1300 LOC (deleted ~2600, added ~1300)
- Single unified architecture for all filter types

### Type Aliases (Final)

```rust
// LMB variants
pub type LmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, SingleSensorScheduler>>;
pub type IcLmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, SequentialScheduler>>;
pub type AaLmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, ParallelScheduler<ArithmeticAverageMerger>>>;
pub type GaLmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, ParallelScheduler<GeometricAverageMerger>>>;
pub type PuLmbFilter = UnifiedFilter<LmbStrategy<LbpAssociator, ParallelScheduler<ParallelUpdateMerger>>>;

// LMBM variants
pub type LmbmFilter = UnifiedFilter<LmbmStrategy<SingleSensorLmbmAssociator<GibbsAssociator>>>;
pub type MultisensorLmbmFilter = UnifiedFilter<LmbmStrategy<MultisensorLmbmAssociator<MultisensorGibbsAssociator>>>;
```

### Files Summary

| File | Action | Impact |
|------|--------|--------|
| `src/lmb/types.rs` | Modify | Rename `LmbmHypothesis` → `Hypothesis` |
| `src/lmb/strategy.rs` | **NEW** | `UpdateStrategy` trait + implementations |
| `src/lmb/unified.rs` | **NEW** | `UnifiedFilter<S>` struct |
| `src/lmb/core.rs` | **DELETE** | ~900 LOC removed |
| `src/lmb/core_lmbm.rs` | **DELETE** | ~1200 LOC removed |
| `src/lmb/factory.rs` | Modify | Use `UnifiedFilter` |
| `src/lmb/common_ops.rs` | Modify | Delete 3 redundant functions |
| `src/lmb/mod.rs` | Modify | Update exports |
| `src/python/filters.rs` | Modify | Update wrapper types |

**Net LOC change**: ~-1500 LOC (delete 2100, add ~600 for unified code)

### Verification

After each sub-phase:
```bash
cargo test --release
cargo clippy --all-targets
```

After Phase 8.8:
```bash
uv run pytest python/tests/ -v  # All 87 Python tests pass at 1e-10 tolerance
```

---

## Phase 9: (Merged into Phase 8)

**Note**: Python bindings update is now Phase 8.7. This phase is no longer needed as a separate step.

---

## Phase 10: API Cleanup

**Goal**: Remove deprecated types, consolidate exports, clean up public API.

### 1. Remove Deprecated Types
- [ ] Remove `FilterParams` if unused
- [ ] Remove any backward-compat type aliases (e.g., `LmbmHypothesis` alias if added)
- [ ] Remove unused config types

### 2. Consolidate Config Types
- [ ] Merge `FilterThresholds` into `PruneConfig` or delete
- [ ] Merge `LmbmConfig` fields into `PruneConfig`
- [ ] Single source of truth for all filter parameters

### 3. Clean Up mod.rs Exports
Final `src/lmb/mod.rs` should export:
```rust
// Unified filter
pub use unified::{UnifiedFilter, Hypothesis};
pub use strategy::{UpdateStrategy, LmbStrategy, LmbmStrategy, PruneConfig};

// Type aliases
pub use unified::{LmbFilter, IcLmbFilter, AaLmbFilter, GaLmbFilter, PuLmbFilter};
pub use unified::{LmbmFilter, MultisensorLmbmFilter};

// Traits
pub use traits::{Filter, Associator, Updater, Merger};
pub use scheduler::UpdateScheduler;

// Configuration
pub use config::{MotionModel, SensorModel, SensorSet, BirthModel, AssociationConfig};

// Schedulers
pub use scheduler::{SingleSensorScheduler, SequentialScheduler, ParallelScheduler};

// Fusion mergers
pub use multisensor::fusion::{ArithmeticAverageMerger, GeometricAverageMerger, ParallelUpdateMerger};

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

## Phase 12: NORFAIR Integration (Future)

**Goal**: Integrate norfair-rs as `NorfairStrategy` implementing `UpdateStrategy` trait.

### 1. Files to CREATE
- [ ] `src/lmb/norfair.rs` - `NorfairStrategy<D: Distance>` implementation
- [ ] `src/lmb/norfair/distance.rs` - Distance trait and implementations

### 2. Implementation Details
```rust
pub struct NorfairStrategy<D: Distance = EuclideanDistance> {
    distance: D,
    hit_counter_config: HitCounterConfig,
}

impl<D: Distance> UpdateStrategy for NorfairStrategy<D> {
    type Measurements = Vec<Detection>;

    fn predict(&self, hypotheses: &mut Vec<Hypothesis>, motion: &MotionModel, _birth: &BirthModel, ts: usize) {
        // NORFAIR: single hypothesis, Kalman predict for each track
        // Birth is handled in update (from unmatched detections), not here
        for track in &mut hypotheses[0].tracks {
            // Kalman predict
        }
    }

    fn update<R: Rng>(&self, _rng: &mut R, hypotheses: &mut Vec<Hypothesis>, meas: &Self::Measurements, ctx: &UpdateContext) -> Result<UpdateIntermediate, FilterError> {
        let tracks = &mut hypotheses[0].tracks;
        // Greedy/Hungarian matching by distance
        // Update matched tracks (Kalman update)
        // Increment/decrement hit counters
        // Create new tracks from unmatched detections (birth)
        Ok(UpdateIntermediate::default())
    }

    fn prune(&self, hypotheses: &mut Vec<Hypothesis>, trajectories: &mut Vec<Trajectory>, config: &PruneConfig) {
        // Remove dead tracks (hit_counter < 0)
    }

    fn extract(&self, hypotheses: &[Hypothesis], ts: usize) -> StateEstimate {
        // Return confirmed tracks (hit_counter > init_threshold)
    }

    fn name(&self) -> &'static str { "NORFAIR" }
    fn is_hypothesis_based(&self) -> bool { false }
}
```

### 3. Type Aliases
```rust
pub type NorfairFilter<D = EuclideanDistance> = UnifiedFilter<NorfairStrategy<D>>;
pub type SortFilter = UnifiedFilter<SortStrategy>;  // Future
pub type ByteTrackFilter = UnifiedFilter<ByteTrackStrategy>;  // Future
```

### 4. NORFAIR-RS Migration Path
- Extract core tracking logic from norfair-rs into `NorfairStrategy`
- Adapt to use shared `MotionModel` (optional - can use internal Kalman)
- Keep existing norfair-rs API as compatibility layer wrapping `UnifiedFilter<NorfairStrategy>`
- Share distance functions with LMB/LMBM where applicable

### 5. Key Differences from LMB/LMBM
| Aspect | NORFAIR | LMB/LMBM |
|--------|---------|----------|
| Birth | From unmatched detections | From BirthModel locations |
| Existence | Hit counter (integer) | Probability (0.0-1.0) |
| Association | Greedy/Hungarian | LBP/Gibbs/Murty |
| Gating | Age threshold | Existence threshold |

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

### Phase 7A Completed ✅
| File | Action | Impact |
|------|--------|--------|
| `src/lmb/singlesensor/lmb.rs` | **DELETED** ✅ | -549 LOC |
| `src/lmb/multisensor/lmb.rs` | **DELETED** ✅ | -736 LOC |
| `src/lmb/mod.rs` | **MODIFIED** ✅ | Re-export from core.rs |
| `src/lmb/singlesensor/mod.rs` | **MODIFIED** ✅ | Removed lmb exports |
| `src/lmb/multisensor/mod.rs` | **MODIFIED** ✅ | Added MultisensorMeasurements type |
| `src/lib.rs` | **MODIFIED** ✅ | Removed MultisensorLmbFilter |
| `src/python/filters.rs` | **MODIFIED** ✅ | Use LmbFilterCore from core.rs |
| `src/bench_utils.rs` | **MODIFIED** ✅ | Use new type aliases |
| `tests/bench_fixtures.rs` | **MODIFIED** ✅ | Use new type aliases |

### Phase 7B Completed ✅
| File | Action | Impact |
|------|--------|--------|
| `src/lmb/singlesensor/lmbm.rs` | **DELETED** ✅ | -733 LOC |
| `src/lmb/multisensor/lmbm.rs` | **DELETED** ✅ | -901 LOC |
| `src/lmb/singlesensor/` | **DELETED** ✅ | Directory removed |
| `src/lmb/mod.rs` | **MODIFIED** ✅ | Removed singlesensor module |
| `src/lmb/multisensor/mod.rs` | **MODIFIED** ✅ | Removed lmbm module |

### Phase 7D Completed ✅
| File | Action | Impact |
|------|--------|--------|
| `src/lmb/core.rs` | Deleted 6 constructor impl blocks | ~-150 LOC |
| `src/lmb/core_lmbm.rs` | Deleted 4 constructor impl blocks + strategy `new()` | ~-110 LOC |
| `src/lmb/config.rs` | Deleted `SensorVariant`, unused builders | ~-240 LOC |
| `src/lmb/factory.rs` | Updated to use direct struct construction | minimal |
| `src/bench_utils.rs` | Updated to use direct struct construction | minimal |
| `src/python/filters.rs` | Updated to use direct struct construction | minimal |

### Phase 8 (Pending) - Full Unification + Python API Simplification
| File | Action | Impact |
|------|--------|--------|
| `src/lmb/types.rs` | Rename `LmbmHypothesis` → `Hypothesis` | minimal |
| `src/lmb/strategy.rs` | **NEW** | `UpdateStrategy` trait + implementations |
| `src/lmb/unified.rs` | **NEW** | `UnifiedFilter<S>` struct |
| `src/lmb/core.rs` | **DELETE** | ~-900 LOC |
| `src/lmb/core_lmbm.rs` | **DELETE** | ~-1200 LOC |
| `src/lmb/factory.rs` | Modify | Use `UnifiedFilter` |
| `src/lmb/common_ops.rs` | Modify | Delete 3 redundant functions |
| `src/python/filters.rs` | DELETE 7 filter classes + 2 config classes, ADD 3 (Filter, LmbStrategy, LmbmStrategy) | ~-400 LOC net (Phase 8.7) |
| `src/python/models.rs` | DELETE `FilterThresholds`, `FilterLmbmConfig` | ~-100 LOC (Phase 8.7) |
| `python/multisensor_lmb_filters_rs/__init__.py` | Clean exports (14 types) | minimal (Phase 8.7) |
| `python/tests/test_equivalence.py` | Update to Strategy Object Pattern API | minimal (Phase 8.7) |

### Future Phases
| File | Action | Impact |
|------|--------|--------|
| `src/lmb/norfair.rs` | **NEW** (Phase 12) | NorfairStrategy (future) |
| `python/tests/test_equivalence.py` | **REFACTOR** (Phase 11) | -1200 LOC via parameterization |

**Net LOC change so far**: Deleted ~3419 LOC (Phase 7A: 1285 + Phase 7B: 1634 + Phase 7D: ~500)
**After Phase 8**: ~5919 LOC deleted total (core.rs + core_lmbm.rs + Python simplification), Python API: 21 types → 14 types (Strategy Object Pattern)

---

## Implementation Order

```
Phase 0-6 (Foundation)      ─► ✅ COMPLETE (infrastructure created)
         │
         ▼
Phase 7A (LMB Cleanup)      ─► ✅ COMPLETE (deleted 1285 LOC, using core.rs)
         │
         ▼
Phase 7B (LMBM Cleanup)     ─► ✅ COMPLETE (deleted 1634 LOC, using core_lmbm.rs)
         │
         ▼
Phase 7C (API Simplify)     ─► ✅ COMPLETE (factory.rs, merged SensorSet, simplified aliases)
         │
         ▼
Phase 7D (Dead Code)        ─► ✅ COMPLETE (deleted ~500 LOC, ONE way to create filters)
         │
         ▼
Phase 8 (Full Unification)  ─► Delete core.rs + core_lmbm.rs, create UnifiedFilter<S: UpdateStrategy>
         │
         ├─► 8.7 (Python Bindings + API Simplification):
         │       - Wrap UnifiedFilter<...>
         │       - 7 filter classes → 1 Filter + 2 Strategies (LmbStrategy, LmbmStrategy)
         │       - 14 public types (Strategy Object Pattern)
         ▼
Phase 10 (API Cleanup)      ─► Remove deprecated types, clean exports
         │
         ▼
Phase 11 (Python Tests)     ─► Parameterize tests
         │
         ▼
Phase 12 (NORFAIR)          ─► (Future) Implement NorfairStrategy : UpdateStrategy
```

---

## Success Metrics

1. **Unified Architecture**: Single `UnifiedFilter<S: UpdateStrategy>` for ALL filter types
2. **No Tolerance Changes**: All tests at 1e-10
3. **Single Filter Implementation**: One `UnifiedFilter` struct, strategy determines behavior
4. **No Old Code**: core.rs, core_lmbm.rs, singlesensor/*.rs, multisensor/lmb.rs, multisensor/lmbm.rs ALL DELETED
5. **ONE Way to Do Each Thing**: Factory functions for common cases, `UnifiedFilter::with_strategy()` for custom
6. **No Duplicate Types**: `SensorVariant` deleted, unused builders deleted, `LmbmHypothesis` renamed to `Hypothesis`
7. **Test Reduction**: `test_equivalence.py` < 800 LOC
8. **Extensibility**:
   - Custom trackers: Implement `UpdateStrategy` trait (NORFAIR, SORT, ByteTrack)
   - Custom motion/sensor: Implement behavior traits
   - Custom association: Implement `Associator` trait
9. **Future-Ready**: Clear path for NORFAIR/SORT/ByteTrack via `UpdateStrategy` trait
10. **Python API Simplified** (Phase 8.7 - Strategy Object Pattern):
    - 1 unified `Filter` class + 2 strategy classes (`LmbStrategy`, `LmbmStrategy`)
    - Old 7 filter classes DELETED, old 2 config classes DELETED
    - 14 public types total (down from 21)
    - Internal `_` types not exported in `__init__.py`
    - Type-safe: `LmbStrategy` only accepts LMB params, `LmbmStrategy` only accepts LMBM params
    - Future-proof: Adding `NorfairStrategy` doesn't require changing `Filter` class

---

## Future Algorithm Support Analysis

| Algorithm | Current | After Refactor | Notes |
|-----------|---------|----------------|-------|
| **NORFAIR Tracker** | Blocked | Easy | Implement `FilterAlgorithm` trait |
| **SORT Tracker** | Blocked | Easy | Implement `FilterAlgorithm` trait |
| **ByteTrack** | Blocked | Easy | Implement `FilterAlgorithm` trait |
| **GNN Associator** | Easy | Easy | Just add new `Associator` impl |
| **JPDA Associator** | Easy | Easy | Just add new `Associator` impl |
| **PMBM** | Medium | Easy | New `FilterAlgorithm` impl |
| **Full GLMB** | Medium | Easy | Extend hypothesis tracking |
| **IMM** | Blocked | Easy | Downstream implements `MotionModelBehavior` |
| **Custom Motion** | Blocked | Easy | Downstream implements `MotionModelBehavior` |
| **Custom Sensor** | Blocked | Easy | Downstream implements `SensorModelBehavior` |
