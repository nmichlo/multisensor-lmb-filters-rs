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

## Phase 7D: Dead Code Cleanup

**Goal**: Remove ALL redundant code paths. ONE way to do each thing.

**Principle**:
- Factory functions = public API for common cases
- `with_scheduler()` / `with_strategy()` = escape hatch for custom configs
- DELETE everything else

### 1. Delete Redundant Constructors from core.rs

Delete these impl blocks (~150 LOC):

| Impl Block | Method | Replaced By |
|------------|--------|-------------|
| `impl LmbFilterCore<LbpAssociator, SingleSensorScheduler>` | `new()` | `lmb_filter()` |
| `impl<A: Associator> LmbFilterCore<A, SingleSensorScheduler>` | `with_associator()` | `with_scheduler()` directly |
| `impl LmbFilterCore<LbpAssociator, SequentialScheduler>` | `new_ic()` | `ic_lmb_filter()` |
| `impl<A: Associator> LmbFilterCore<A, SequentialScheduler>` | `with_associator_ic()` | `with_scheduler()` directly |
| `impl<M: Merger> LmbFilterCore<LbpAssociator, ParallelScheduler<M>>` | `new_parallel()` | `aa/ga/pu_lmb_filter()` |
| `impl<A: Associator, M: Merger> LmbFilterCore<A, ParallelScheduler<M>>` | `with_associator_parallel()` | `with_scheduler()` directly |

**KEEP**:
- [ ] `impl<A: Associator, S: UpdateScheduler> LmbFilterCore<A, S>` with `with_scheduler()`
- [ ] `impl<A: Associator, S: UpdateScheduler> LmbFilterCore<A, S>` with `with_gm_pruning()`, `with_gm_merge_threshold()`

### 2. Delete Redundant Constructors from core_lmbm.rs

Delete these impl blocks (~100 LOC):

| Impl Block | Method | Replaced By |
|------------|--------|-------------|
| `impl LmbmFilterCore<SingleSensorLmbmStrategy<GibbsAssociator>>` | `new()` | `lmbm_filter()` |
| `impl<A: LmbmAssociator> LmbmFilterCore<SingleSensorLmbmStrategy<A>>` | `with_associator()` | `with_strategy()` directly |
| `impl LmbmFilterCore<MultisensorLmbmStrategy<MultisensorGibbsAssociator>>` | `new_multisensor()` | `multisensor_lmbm_filter()` |
| `impl<A: MultisensorAssociator> LmbmFilterCore<MultisensorLmbmStrategy<A>>` | `with_multisensor_associator()` | `with_strategy()` directly |

**KEEP**:
- [ ] `impl<S: LmbmStrategy> LmbmFilterCore<S>` with `with_strategy()`

### 3. Delete SensorVariant (Duplicate of SensorSet)

In `config.rs`:
- [ ] Delete `SensorVariant` enum and its impl block (lines ~1236-1272)
- [ ] Update `FilterParams` to use `SensorSet` instead of `SensorVariant`
- [ ] Update any code that references `SensorVariant`

**Analysis**: `SensorVariant` and `SensorSet` are identical:
```rust
// Both have exactly the same structure:
pub enum SensorSet { Single(SensorModel), Multi(MultisensorConfig) }
pub enum SensorVariant { Single(SensorModel), Multi(MultisensorConfig) }
```
- `SensorSet`: 53 usages (keep)
- `SensorVariant`: 15 usages (delete)

### 4. Delete Unused Config Builders

In `config.rs`, delete these structs and impl blocks (~200 LOC):
- [ ] `CommonConfigBuilder` - ZERO usages in tests/benches/production
- [ ] `LmbFilterConfigBuilder` - ZERO usages
- [ ] `LmbmFilterConfigBuilder` - ZERO usages
- [ ] `FilterParamsBuilder` - ZERO usages

### 5. Update Callers

Check and update any code that uses deleted constructors:
- [ ] `src/bench_utils.rs` - uses `LmbFilterCore::with_scheduler()` already ✓
- [ ] `src/python/filters.rs` - verify uses factory functions or `with_scheduler()`
- [ ] `tests/bench_fixtures.rs` - verify uses factory functions
- [ ] `tests/ss_lmb.rs`, `tests/ss_lmbm.rs` - update if needed

### 6. Clean Up Exports

In `src/lmb/mod.rs`:
- [ ] Remove `SensorVariant` from re-exports
- [ ] Remove builder types from re-exports if they were exported

### 7. Verification

```bash
cargo test --release
cargo clippy --all-targets
uv run pytest python/tests/ -v
```

### 8. Success Criteria

- [ ] ≤2 constructor impl blocks in `core.rs` (down from 8+)
- [ ] ≤1 constructor impl block in `core_lmbm.rs` (down from 5+)
- [ ] `SensorVariant` deleted (~40 LOC)
- [ ] Unused builders deleted (~200 LOC)
- [ ] All tests pass unchanged
- [ ] ONE obvious way to create each filter type

### 9. API After Cleanup

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

### 10. Files Summary

| File | Changes | LOC Impact |
|------|---------|------------|
| `src/lmb/core.rs` | Delete 6 constructor impl blocks | ~-150 LOC |
| `src/lmb/core_lmbm.rs` | Delete 4 constructor impl blocks | ~-100 LOC |
| `src/lmb/config.rs` | Delete `SensorVariant`, delete unused builders | ~-240 LOC |
| `src/lmb/mod.rs` | Update exports | minimal |
| `tests/*.rs` | Update any callers of deleted constructors | minimal |

**Total estimated deletion**: ~490 LOC

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

## Phase 12: NORFAIR Integration (Future)

**Goal**: Integrate norfair-rs as `NorfairAlgorithm` implementing `FilterAlgorithm` trait.

### 1. Files to CREATE
- [ ] `src/filter/norfair.rs` - `NorfairAlgorithm<D: Distance>` implementation
- [ ] `src/filter/norfair/distance.rs` - Distance trait and implementations
- [ ] `src/filter/norfair/tracked_object.rs` - TrackedObject type

### 2. Implementation Details
```rust
pub struct NorfairAlgorithm<D: Distance = EuclideanDistance> {
    tracks: Vec<TrackedObject>,
    distance: D,
    hit_counter_config: HitCounterConfig,
}

impl<D: Distance> FilterAlgorithm for NorfairAlgorithm<D> {
    type State = TrackedObject;
    type Measurements = Vec<Detection>;
    type DetailedOutput = NorfairDetailedOutput;

    fn predict(&mut self, motion: &dyn MotionModelBehavior, timestep: usize) {
        // Kalman predict for each track
    }

    fn inject_birth(&mut self, _birth: &BirthModel, _timestep: usize) {
        // NORFAIR creates new tracks from unmatched detections, not from birth model
    }

    fn associate_and_update<R: Rng>(
        &mut self,
        _rng: &mut R,
        measurements: &Self::Measurements,
        _sensors: &SensorSet,
    ) -> Result<(), FilterError> {
        // Greedy/Hungarian matching by distance
        // Update matched tracks (Kalman update)
        // Increment/decrement hit counters
        // Create new tracks from unmatched detections
    }

    fn normalize_and_gate(&mut self, _config: &GatingConfig) {
        // Remove dead tracks (hit_counter < 0)
    }

    fn extract_estimate(&self, timestamp: usize) -> StateEstimate {
        // Return confirmed tracks (hit_counter > init_threshold)
    }

    fn extract_detailed(&self) -> Self::DetailedOutput { ... }
}
```

### 3. Type Aliases
```rust
pub type NorfairFilter<D = EuclideanDistance> = Filter<NorfairAlgorithm<D>>;
pub type SortFilter = Filter<SortAlgorithm>;  // Future
pub type ByteTrackFilter = Filter<ByteTrackAlgorithm>;  // Future
```

### 4. NORFAIR-RS Migration Path
- Extract core tracking logic from norfair-rs into `NorfairAlgorithm`
- Adapt to use shared `MotionModelBehavior` trait (optional - can use internal Kalman)
- Keep existing norfair-rs API as compatibility layer wrapping `Filter<NorfairAlgorithm>`
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

### Phase 7D (Pending)
| File | Action | Impact |
|------|--------|--------|
| `src/lmb/core.rs` | Delete 6 constructor impl blocks | ~-150 LOC |
| `src/lmb/core_lmbm.rs` | Delete 4 constructor impl blocks | ~-100 LOC |
| `src/lmb/config.rs` | Delete `SensorVariant`, delete unused builders | ~-240 LOC |
| `src/lmb/mod.rs` | Update exports | minimal |

### Future Phases
| File | Action | Impact |
|------|--------|--------|
| `src/filter/norfair.rs` | **NEW** (Phase 12) | NorfairAlgorithm (future) |
| `python/tests/test_equivalence.py` | **REFACTOR** | -1200 LOC via parameterization |

**Net LOC change so far**: Deleted 2919 LOC of old filters (Phase 7A: 1285 + Phase 7B: 1634)
**After Phase 7D**: ~3409 LOC deleted total

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
Phase 7D (Dead Code)        ─► Delete redundant constructors, SensorVariant, unused builders (~490 LOC)
         │
         ▼
Phase 8 (Infrastructure)    ─► Wire up MeasurementSource, StepReporter into Filter
         │
         ▼
Phase 9 (Python Bindings)   ─► Already updated in Phase 7A ✅
         │
         ▼
Phase 10 (API Cleanup)      ─► Remove deprecated types, clean exports
         │
         ▼
Phase 11 (Python Tests)     ─► Parameterize tests
         │
         ▼
Phase 12 (NORFAIR)          ─► (Future) Add NorfairAlgorithm, integrate norfair-rs
```

---

## Success Metrics

1. **Unified Architecture**: `LmbFilterCore<A, S>` and `LmbmFilterCore<S>` with factory functions
2. **No Tolerance Changes**: All tests at 1e-10
3. **Single Algorithm per Type**: One core implementation per filter family
4. **No Old Code**: singlesensor/*.rs, multisensor/lmb.rs, multisensor/lmbm.rs DELETED ✅
5. **ONE Way to Do Each Thing**: Factory functions for common cases, `with_scheduler()`/`with_strategy()` for custom
6. **No Duplicate Types**: `SensorVariant` deleted, unused builders deleted
7. **Test Reduction**: `test_equivalence.py` < 800 LOC
8. **Extensibility**:
   - Custom trackers: Implement `FilterAlgorithm` trait (future)
   - Custom motion/sensor: Implement behavior traits
   - Custom association: Implement `Associator` trait
9. **Infrastructure Integrated**: MeasurementSource, StepReporter wired into filters
10. **Future-Ready**: Clear path for NORFAIR/SORT/ByteTrack integration

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
