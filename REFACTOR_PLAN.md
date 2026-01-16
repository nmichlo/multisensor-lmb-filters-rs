# Comprehensive Cleanup & Unification Plan for multisensor-lmb-filters-rs

## Executive Summary

This plan addresses major code duplication (~60% overlap between single/multi-sensor filters), API inconsistencies, and architectural gaps. The design prioritizes **extensibility** (downstream can add custom implementations) and **production-grade** qualities: type safety, numerical robustness, and real-time capability.

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

## Phase 0: Plan Setup

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

### Implementation Design

```rust
/// Zero-copy measurement abstraction using iterators
pub trait MeasurementSource {
    type SensorIter<'a>: Iterator<Item = Self::MeasIter<'a>> where Self: 'a;
    type MeasIter<'a>: Iterator<Item = &'a DVector<f64>> where Self: 'a;

    fn num_sensors(&self) -> usize;
    fn sensors(&self) -> Self::SensorIter<'_>;
}

// Concrete implementations for common cases
impl<'a> MeasurementSource for &'a [DVector<f64>] { /* single sensor */ }
impl<'a> MeasurementSource for &'a [Vec<DVector<f64>>] { /* multi-sensor */ }
impl<'a> MeasurementSource for &'a [&'a [DVector<f64>]] { /* slice of slices */ }
```

**Rationale**: Avoids forcing `Vec<Vec<...>>` allocation on users with zero-copy data sources.

### Files to Modify
- `src/lmb/measurements.rs` - **NEW**
- `src/lmb/mod.rs` - Export new types

---

## Phase 2: Type-Safe Configuration (No "God Config") ✅

**Goal**: Make illegal states unrepresentable via composition.

### 1. Implementation Tasks
- [x] Create `CommonConfig` struct for shared settings
- [x] Create `LmbFilterConfig` struct with `common: CommonConfig` + GM-specific fields
- [x] Create `LmbmFilterConfig` struct with `common: CommonConfig` + hypothesis-specific fields
- [x] Create builders for each config type (`CommonConfigBuilder`, `LmbFilterConfigBuilder`, `LmbmFilterConfigBuilder`)
- [ ] Update filter constructors to use appropriate config type (deferred to Phase 8-9 unification)

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [x] Add 10 unit tests for new config types
- [x] Verify all tests pass at 1e-10 tolerance with `cargo test --release`
- [x] Confirm NO numeric outputs changed

### 3. Update Plan & TODOs
- [x] Mark completed tasks in `./REFACTOR_PLAN.md`
- [x] Document any deviations or learnings
- [x] Verify phase is complete before proceeding

### Completion Notes
- Created `CommonConfig`, `LmbFilterConfig`, `LmbmFilterConfig` with builder patterns
- Existing `FilterThresholds` and `LmbmConfig` kept for backward compatibility
- Filter constructor updates deferred to Phase 8-9 when filters are unified
- Type safety demonstrated: LmbFilterConfig has GM fields, LmbmFilterConfig has hypothesis fields
- Added `to_legacy_lmbm_config()` for backward compatibility conversion

### Implementation Design

```rust
/// Common configuration shared by all filters
#[derive(Clone, Debug)]
pub struct CommonConfig {
    pub existence_threshold: f64,
    pub min_trajectory_length: usize,
    pub max_trajectory_length: usize,
}

/// LMB-specific configuration (Gaussian mixture)
#[derive(Clone, Debug)]
pub struct LmbConfig {
    pub common: CommonConfig,
    pub gm_weight_threshold: f64,
    pub max_gm_components: usize,
    pub gm_merge_threshold: f64,
}

/// LMBM-specific configuration (hypothesis mixture)
#[derive(Clone, Debug)]
pub struct LmbmConfig {
    pub common: CommonConfig,
    pub max_hypotheses: usize,
    pub hypothesis_weight_threshold: f64,
    pub use_eap: bool,
}
```

**Rationale**: You physically cannot set `max_hypotheses` on an LMB filter - the field doesn't exist.

### Files to Modify
- `src/lmb/config.rs` - Split into `CommonConfig`, `LmbConfig`, `LmbmConfig`
- `src/lmb/builder.rs` - Separate builders per config type

---

## Phase 3: Extensible Traits for Models (Open for Extension) ✅

**Goal**: Enable downstream custom implementations WITHOUT modifying upstream.

### 1. Implementation Tasks
- [x] Create `MotionModelBehavior` trait
- [x] Create `SensorModelBehavior` trait
- [x] Implement traits for existing `MotionModel` and `SensorModel` structs
- [x] Keep existing struct constructors working (backward compat)
- [x] Add `Box<dyn MotionModelBehavior>` support for custom models

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [x] Add 8 unit tests for trait behavior (predict_state, predict_covariance, accessors, etc.)
- [x] Add 2 trait object tests verifying Box<dyn ...> support
- [x] Verify all tests pass with `cargo test --release`
- [x] Confirm NO numeric outputs changed

### 3. Update Plan & TODOs
- [x] Mark completed tasks in `./REFACTOR_PLAN.md`
- [x] Document any deviations or learnings
- [x] Verify phase is complete before proceeding

### Completion Notes
- Created `MotionModelBehavior` trait with methods: `predict_state`, `predict_covariance`, `survival_probability`, `x_dim`, `transition_matrix` (optional), `process_noise` (optional)
- Created `SensorModelBehavior` trait with methods: `predict_measurement`, `measurement_jacobian`, `measurement_noise`, `detection_probability`, `clutter_rate`, `observation_volume`, `clutter_density` (default impl), `z_dim`, `x_dim`, `observation_matrix` (optional)
- Implemented traits for existing `MotionModel` and `SensorModel` structs
- Optional methods return `Some(&DMatrix)` for linear models, `None` for nonlinear (future extension)
- Both traits require `Send + Sync` for thread safety with trait objects
- Excluded `as_serializable()` method from plan - can be added later if needed
- All 21 config tests pass (including 10 new Phase 3 tests)

### Implementation Design

```rust
/// Trait for motion models - OPEN for downstream extension
pub trait MotionModelBehavior: Send + Sync {
    fn predict_state(&self, state: &DVector<f64>) -> DVector<f64>;
    fn predict_covariance(&self, cov: &DMatrix<f64>) -> DMatrix<f64>;
    fn survival_probability(&self) -> f64;
    fn state_dim(&self) -> usize;

    /// Optional: Enable serialization for concrete types
    fn as_serializable(&self) -> Option<&dyn erased_serde::Serialize> { None }
}

/// Existing linear motion model implements the trait
#[derive(Clone, Debug)]
pub struct LinearMotionModel {
    pub transition_matrix: DMatrix<f64>,
    pub process_noise: DMatrix<f64>,
    pub survival_probability: f64,
}

impl MotionModelBehavior for LinearMotionModel { /* ... */ }
```

**Rationale**: Trait objects allow downstream extension without modifying upstream crate. Serialization is opt-in via `as_serializable()` for types that support it.

### Files to Modify
- `src/lmb/config.rs` - Add traits, implement for existing structs
- `src/lmb/mod.rs` - Export traits

---

## Phase 4: Update Strategy Pattern (No Boolean Flags)

**Goal**: Invert control flow so strategies own the loop, not the filter core.

### 1. Implementation Tasks
- [ ] Create `src/lmb/scheduler.rs` with `UpdateScheduler` trait
- [ ] Implement `ParallelScheduler<M: Merger>` for AA/GA/PU fusion
- [ ] Implement `SequentialScheduler` for IC fusion and single-sensor
- [ ] Export from `src/lmb/mod.rs`

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [ ] Verify all tests pass with `cargo test --release`
- [ ] Confirm NO numeric outputs changed

### 3. Update Plan & TODOs
- [ ] Mark completed tasks in `./REFACTOR_PLAN.md`
- [ ] Document any deviations or learnings
- [ ] Verify phase is complete before proceeding

### Implementation Design

```rust
/// Controls how sensors are processed during update - OPEN for extension
pub trait UpdateScheduler<A: Associator>: Send + Sync {
    fn execute_update<M: MeasurementSource, R: rand::Rng>(
        &self,
        tracks: &mut Vec<Track>,
        measurements: &M,
        sensors: &SensorSet,
        associator: &A,
        updater: &MarginalUpdater,
        config: &LmbConfig,
        rng: &mut R,
        deadline: Option<Instant>,
        reporter: Option<&mut dyn StepReporter>,
    ) -> Result<(), FilterError>;
}

/// Parallel: Update all sensors independently, then fuse
pub struct ParallelScheduler<M: Merger> {
    merger: M,
}

/// Sequential: Output of sensor A is input to sensor B (Iterated Corrector)
pub struct SequentialScheduler;
```

**Rationale**: No `is_sequential()` boolean flag. The scheduler owns the iteration logic.

### Files to Modify
- `src/lmb/scheduler.rs` - **NEW**
- `src/lmb/fusion.rs` - Simplify to just `Merger` trait

---

## Phase 5: Observability via StepReporter

**Goal**: Enable debugging without polluting core logic.

### 1. Implementation Tasks
- [ ] Create `src/lmb/reporter.rs` with `StepReporter` trait
- [ ] Implement `NoOpReporter` (zero-cost default)
- [ ] Implement `DebugReporter` (collects all events)
- [ ] Add reporter hooks to internal filter methods
- [ ] Export from `src/lmb/mod.rs`

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [ ] Add test using `DebugReporter` to verify hooks fire
- [ ] Verify all existing tests pass with `cargo test --release`
- [ ] Confirm NO numeric outputs changed

### 3. Update Plan & TODOs
- [ ] Mark completed tasks in `./REFACTOR_PLAN.md`
- [ ] Document any deviations or learnings
- [ ] Verify phase is complete before proceeding

### Implementation Design

```rust
/// Zero-cost observability for debugging and research - OPEN for extension
pub trait StepReporter {
    fn on_prediction(&mut self, tracks: &[Track]) {}
    fn on_birth(&mut self, new_tracks: &[Track]) {}
    fn on_association(&mut self, sensor_idx: usize, matrices: &AssociationMatrices) {}
    fn on_sensor_update(&mut self, sensor_idx: usize, tracks: &[Track]) {}
    fn on_fusion(&mut self, tracks: &[Track]) {}
    fn on_pruning(&mut self, removed: &[Track], kept: &[Track]) {}
}

pub struct NoOpReporter;
impl StepReporter for NoOpReporter {}
```

**Rationale**: Researchers need intermediate data without return type explosion.

### Files to Modify
- `src/lmb/reporter.rs` - **NEW**

---

## Phase 6: Numerical Robustness (Self-Healing Math)

**Goal**: Filter survives numerical edge cases without crashing.

### 1. Implementation Tasks
- [ ] Add `robust_cholesky()` to `src/common/linalg.rs`
- [ ] Add `robust_inverse()` to `src/common/linalg.rs`
- [ ] Create `LinalgWarning` enum for recoverable failures
- [ ] Update `src/components/update.rs` to use robust functions
- [ ] Add logging for regularization events

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [ ] Add tests for edge cases (near-singular matrices)
- [ ] Verify all existing tests pass with `cargo test --release`
- [ ] Confirm NO numeric outputs changed (robust path should not activate on normal inputs)

### 3. Update Plan & TODOs
- [ ] Mark completed tasks in `./REFACTOR_PLAN.md`
- [ ] Document any deviations or learnings
- [ ] Verify phase is complete before proceeding

### Implementation Design

```rust
/// Robust Cholesky that auto-regularizes on failure
pub fn robust_cholesky(matrix: &DMatrix<f64>) -> Result<Cholesky<f64, Dyn>, LinalgWarning> {
    // First attempt: standard Cholesky
    if let Some(chol) = matrix.clone().cholesky() {
        return Ok(chol);
    }

    // Second attempt: add small regularization to diagonal
    let mut regularized = matrix.clone();
    let eps = 1e-6 * matrix.diagonal().abs().max();
    for i in 0..regularized.nrows() {
        regularized[(i, i)] += eps;
    }

    if let Some(chol) = regularized.cholesky() {
        return Err(LinalgWarning::Regularized { chol, epsilon: eps });
    }

    Err(LinalgWarning::Failed)
}
```

**Rationale**: In production, covariance matrices become non-positive-definite. Filter must survive.

### Files to Modify
- `src/common/linalg.rs` - Add robust functions
- `src/components/update.rs` - Use robust functions

---

## Phase 7: Time-Budgeted Association

**Goal**: Real-time safety for latency-sensitive applications.

### 1. Implementation Tasks
- [ ] Add `deadline: Option<Instant>` parameter to `Associator::associate()`
- [ ] Update `LbpAssociator` to respect deadline
- [ ] Update `GibbsAssociator` to respect deadline (early exit)
- [ ] Update `MurtyAssociator` to respect deadline (early exit)
- [ ] Add `step_with_deadline()` method to filters

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [ ] Add test verifying early exit on deadline
- [ ] Verify all existing tests pass with `cargo test --release` (None deadline = no change)
- [ ] Confirm NO numeric outputs changed

### 3. Update Plan & TODOs
- [ ] Mark completed tasks in `./REFACTOR_PLAN.md`
- [ ] Document any deviations or learnings
- [ ] Verify phase is complete before proceeding

### Implementation Design

```rust
pub trait Associator: Send + Sync {
    fn associate<R: rand::Rng>(
        &self,
        matrices: &AssociationMatrices,
        config: &AssociationConfig,
        rng: &mut R,
        deadline: Option<Instant>,  // NEW: Optional time budget
    ) -> Result<AssociationResult, AssociationError>;

    fn name(&self) -> &'static str;
}
```

**Rationale**: When complexity explodes, missing a deadline is worse than a degraded estimate.

### Files to Modify
- `src/lmb/traits.rs` - Add `deadline` parameter
- All associator implementations

---

## Phase 8: Core Filter Unification

**Goal**: Extract common algorithm into generic `LmbFilterCore`.

### 1. Implementation Tasks
- [ ] Create `src/lmb/core.rs` with `LmbFilterCore<A, S>` struct
- [ ] Implement `step()` and `step_with_options()` methods
- [ ] Create type aliases: `LmbFilter`, `IcLmbFilter`, `AaLmbFilter`, etc.
- [ ] Refactor `src/lmb/singlesensor/lmb.rs` to use type alias
- [ ] Refactor `src/lmb/multisensor/lmb.rs` to use type alias

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [ ] Update test imports if paths changed
- [ ] Verify all LMB tests pass at 1e-10 tolerance with `cargo test --release`
- [ ] Confirm NO numeric outputs changed

### 3. Update Plan & TODOs
- [ ] Mark completed tasks in `./REFACTOR_PLAN.md`
- [ ] Document any deviations or learnings
- [ ] Verify phase is complete before proceeding

### Implementation Design

```rust
/// Generic LMB filter core - parameterized by Associator and UpdateScheduler
pub struct LmbFilterCore<A: Associator, S: UpdateScheduler<A>> {
    motion: Box<dyn MotionModelBehavior>,
    sensors: SensorSet,
    birth: BirthModel,
    associator: A,
    scheduler: S,
    updater: MarginalUpdater,
    config: LmbConfig,
    tracks: Vec<Track>,
    trajectories: Vec<Trajectory>,
}

// Type aliases for backward compatibility
pub type LmbFilter<A = LbpAssociator> = LmbFilterCore<A, SequentialScheduler>;
pub type IcLmbFilter<A = LbpAssociator> = LmbFilterCore<A, SequentialScheduler>;
pub type AaLmbFilter<A = LbpAssociator> = LmbFilterCore<A, ParallelScheduler<ArithmeticAverageMerger>>;
```

### Files to Modify
- `src/lmb/core.rs` - **NEW**
- `src/lmb/singlesensor/lmb.rs` - Refactor to type alias
- `src/lmb/multisensor/lmb.rs` - Refactor to type alias

---

## Phase 9: LMBM Filter Unification

**Goal**: Same pattern for LMBM filters.

### 1. Implementation Tasks
- [ ] Create `src/lmb/core_lmbm.rs` with `LmbmFilterCore<A>` struct
- [ ] Create type aliases: `LmbmFilter`, `MultisensorLmbmFilter`
- [ ] Refactor `src/lmb/singlesensor/lmbm.rs` to use type alias
- [ ] Refactor `src/lmb/multisensor/lmbm.rs` to use type alias

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [ ] Update test imports if paths changed
- [ ] Verify all LMBM tests pass at 1e-10 tolerance with `cargo test --release`
- [ ] Confirm NO numeric outputs changed

### 3. Update Plan & TODOs
- [ ] Mark completed tasks in `./REFACTOR_PLAN.md`
- [ ] Document any deviations or learnings
- [ ] Verify phase is complete before proceeding

### Implementation Design

```rust
pub struct LmbmFilterCore<A: Associator> {
    motion: Box<dyn MotionModelBehavior>,
    sensors: SensorSet,
    birth: BirthModel,
    associator: A,
    config: LmbmConfig,
    hypotheses: Vec<LmbmHypothesis>,
    trajectories: Vec<Trajectory>,
}
```

### Files to Modify
- `src/lmb/core_lmbm.rs` - **NEW**
- `src/lmb/singlesensor/lmbm.rs` - Type alias
- `src/lmb/multisensor/lmbm.rs` - Type alias

---

## Phase 10: PyO3 Wrapper Macro

**Goal**: Generate Python wrappers without manual boilerplate.

### 1. Implementation Tasks
- [ ] Create `src/python/macros.rs` with `py_filter_wrapper!` macro
- [ ] Replace manual wrappers in `src/python/filters.rs` with macro invocations
- [ ] Update Python config types for split `LmbConfig`/`LmbmConfig`

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [ ] Verify all Python tests pass with `uv run pytest tests/ -v`
- [ ] Confirm NO numeric outputs changed

### 3. Update Plan & TODOs
- [ ] Mark completed tasks in `./REFACTOR_PLAN.md`
- [ ] Document any deviations or learnings
- [ ] Verify phase is complete before proceeding

### Implementation Design

```rust
macro_rules! py_filter_wrapper {
    ($py_name:ident, $rust_type:ty, $config_type:ty) => {
        #[pyclass(name = stringify!($py_name))]
        pub struct $py_name {
            inner: $rust_type,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new(...) -> PyResult<Self> { ... }
            fn step(&mut self, ...) -> PyResult<PyStateEstimate> { ... }
            fn step_detailed(&mut self, ...) -> PyResult<PyStepOutput> { ... }
        }
    };
}

// Generate all wrappers
py_filter_wrapper!(FilterLmb, LmbFilter<DynamicAssociator>, PyLmbConfig);
py_filter_wrapper!(FilterLmbm, LmbmFilter<DynamicAssociator>, PyLmbmConfig);
// ... etc
```

### Files to Modify
- `src/python/macros.rs` - **NEW**
- `src/python/filters.rs` - Replace with macro invocations

---

## Phase 11: Python Test Cleanup

**Goal**: Eliminate 50%+ of test code via parameterization.

### 1. Implementation Tasks
- [ ] Create `FilterTestCase` dataclass for parameterized testing
- [ ] Refactor `test_equivalence.py` to use parameterized fixtures
- [ ] Remove duplicated sensor 0/1 tests (use `@pytest.mark.parametrize`)
- [ ] Consolidate fixture loading functions in `conftest.py`

### 2. Update Tests (API ONLY - NO BEHAVIOR/NUMERIC CHANGES)
- [ ] Verify all tests still cover same code paths
- [ ] Verify 1e-10 tolerance unchanged with `uv run pytest tests/ -v`
- [ ] Confirm NO numeric outputs changed

### 3. Update Plan & TODOs
- [ ] Mark completed tasks in `./REFACTOR_PLAN.md`
- [ ] Mark entire refactor complete

### Implementation Design

```python
@dataclass
class FilterTestCase:
    filter_class: type
    config_class: type
    fixture_prefix: str
    is_multisensor: bool
    sensors_to_test: List[int]

FILTER_TEST_CASES = [
    FilterTestCase(FilterLmb, LmbConfig, "lmb", False, [0]),
    FilterTestCase(FilterIcLmb, LmbConfig, "ic_lmb", True, [0, 1]),
    # ... all 7 filters
]

@pytest.fixture(params=FILTER_TEST_CASES, ids=lambda c: c.fixture_prefix)
def filter_case(request):
    return request.param
```

### Files to Modify
- `python/tests/test_equivalence.py` - Parameterize (1875 -> ~600 LOC)
- `python/tests/conftest.py` - Add unified fixtures

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
cd python && uv run pytest tests/ -v
```

---

## Critical Files Summary

| File | Action | Impact |
|------|--------|--------|
| `src/lmb/core.rs` | **NEW** | Unified LMB filter (~400 LOC) |
| `src/lmb/core_lmbm.rs` | **NEW** | Unified LMBM filter (~350 LOC) |
| `src/lmb/measurements.rs` | **NEW** | Zero-copy measurement input (~80 LOC) |
| `src/lmb/scheduler.rs` | **NEW** | Update scheduler trait (~200 LOC) |
| `src/lmb/reporter.rs` | **NEW** | Observability hooks (~50 LOC) |
| `src/lmb/config.rs` | Refactor | Split into `LmbConfig`, `LmbmConfig` |
| `src/common/linalg.rs` | Extend | Add robust math functions |
| `src/python/macros.rs` | **NEW** | PyO3 wrapper generation |
| `src/lmb/singlesensor/lmb.rs` | Refactor | ~80% reduction (type alias) |
| `src/lmb/multisensor/lmb.rs` | Refactor | ~80% reduction (type alias) |
| `python/tests/test_equivalence.py` | Refactor | ~60% reduction |

---

## Future Algorithm Support Analysis

| Algorithm | Current | After Refactor | Notes |
|-----------|---------|----------------|-------|
| **GNN Associator** | Easy | Easy | Just add new `Associator` impl |
| **JPDA Associator** | Easy | Easy | Just add new `Associator` impl |
| **PMBM** | Medium | Easy | New hypothesis management |
| **Full GLMB** | Medium | Easy | Extend hypothesis tracking |
| **Appearance** | Hard | Medium | Track extension + distance metrics |
| **IMM** | Blocked | Easy | Downstream implements `MotionModelBehavior` |
| **Extended Target** | Hard | Medium | New association structure |
| **TBD** | Very Hard | Hard | Different likelihood model |
| **Custom Motion** | Blocked | Easy | Downstream implements `MotionModelBehavior` |
| **Custom Sensor** | Blocked | Easy | Downstream implements `SensorModelBehavior` |

---

## Success Metrics

1. **Code Reduction**: ~40% reduction in `src/lmb/` LOC
2. **No Tolerance Changes**: All tests at 1e-10
3. **API Stability**: Python bindings unchanged (or only additive)
4. **Performance**: No benchmark regression
5. **Test Reduction**: `test_equivalence.py` < 800 LOC
6. **Extensibility**: Downstream can add custom motion/sensor models
7. **Real-time Ready**: Deadline-aware association

---

## Implementation Order

```
Phase 0 (Plan Setup)        ─► Write ./REFACTOR_PLAN.md
         │
         ▼
Phase 1 (Measurements)      ─┐
Phase 2 (Config Split)      ─┼─► Foundation
Phase 3 (Model Traits)      ─┘
         │
         ▼
Phase 4 (UpdateScheduler)   ─┐
Phase 5 (StepReporter)      ─┼─► Architecture
Phase 6 (Robust Math)       ─┤
Phase 7 (Time Budgets)      ─┘
         │
         ▼
Phase 8 (LMB Core)          ─┐
Phase 9 (LMBM Core)         ─┼─► Filter Unification
         │                  ─┘
         ▼
Phase 10 (PyO3 Macros)      ─► Python Integration
         │
         ▼
Phase 11 (Python Tests)     ─► Cleanup
```

---

## Critique Response Log

### Addressed Critiques

| Critique | Resolution | Phase |
|----------|------------|-------|
| **God Config** | Split into `LmbConfig`, `LmbmConfig` with composition | Phase 2 |
| **`is_sequential()` leak** | Replaced with `UpdateScheduler` trait that owns the loop | Phase 4 |
| **Observability** | Added `StepReporter` trait with zero-cost no-op default | Phase 5 |
| **PyO3 boilerplate** | Macro-based wrapper generation | Phase 10 |
| **Numerical fragility** | Robust self-healing math functions | Phase 6 |
| **Vec<Vec> allocations** | Iterator-based zero-copy input | Phase 1 |
| **Time budgets** | Deadline parameter on `Associator::associate` | Phase 7 |

### Considered But Accepted Tradeoffs

| Critique | Decision | Reasoning |
|----------|----------|-----------|
| **Identity runtime check** | Keep `assert_eq!(tracks.len(), 1)` | For single-sensor, this check runs once per step (~μs cost). Code clarity benefit outweighs negligible performance cost. |
| **Serialization** | Use trait objects, NOT enum dispatch | **Extensibility trumps serializability.** Downstream users MUST be able to add custom motion/sensor models without modifying upstream. Serialization can be opt-in via `as_serializable()` method for types that support it. |
