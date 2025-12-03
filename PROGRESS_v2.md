# Prak v2 Refactoring Progress

## Goal

Transform Prak into a high-quality, extensible open-source multi-object tracking library supporting multiple algorithm families (LMB, SORT, ByteTrack, Norfair).

## Design Decisions

- **Clean API break** - no backwards compatibility shims
- **Type-safe weights** - `LogWeight`/`LinearWeight` wrappers
- **Algorithm-family directories** - `src/algorithms/{lmb,sort,bytetrack}/`
- **Keep "Filter" names** - `LmbFilter` implements `Tracker` trait but keeps name

---

## Current Status

**Phase:** Not started
**Last Updated:** 2025-12-03

---

## TODO List

### Batch 1: Core Abstractions
- [ ] Create `src/core/mod.rs`
- [ ] Create `src/core/tracker.rs` - Core `Tracker` trait
- [ ] Create `src/core/track.rs` - Abstract `Track` trait + `TrackId`
- [ ] Create `src/core/detection.rs` - `Detection` trait
- [ ] Create `src/core/output.rs` - Move output types from `types/output.rs`
- [ ] Create `src/core/weights.rs` - `LogWeight`, `LinearWeight` newtypes
- [ ] Create `src/core/config.rs` - Configuration traits
- [ ] Create `src/motion/mod.rs`
- [ ] Create `src/motion/traits.rs` - `MotionModel` trait
- [ ] Create `src/motion/constant_velocity.rs` - Move existing `MotionModel`
- [ ] Centralize magic numbers in `src/common/constants.rs`

### Batch 2: Component Consolidation
- [ ] Create `src/components/kalman.rs` - Consolidated Kalman operations
- [ ] Create `src/components/existence.rs` - Consolidated existence updates
- [ ] Create `src/components/pruning.rs` - GM/hypothesis/track pruning
- [ ] Create `src/components/gating.rs` - Mahalanobis gating
- [ ] Create `src/common/robust.rs` - Robust matrix operations

### Batch 3: Association Refactor
- [ ] Create `src/association/traits.rs` - Unified `Associator` trait
- [ ] Refactor `src/common/association/lbp.rs` to implement unified trait
- [ ] Refactor `src/common/association/gibbs.rs` to implement unified trait
- [ ] Refactor `src/common/association/murtys.rs` to implement unified trait
- [ ] Add `src/association/greedy.rs` stub (for future SORT)
- [ ] Add `src/association/cascade.rs` stub (for future DeepSORT)

### Batch 4: Algorithm Migration
- [ ] Create `src/algorithms/mod.rs`
- [ ] Create `src/algorithms/lmb/mod.rs`
- [ ] Create `src/algorithms/lmb/types.rs` - `GaussianComponent`, `LmbmHypothesis`
- [ ] Create `src/algorithms/lmb/config.rs` - LMB-specific config
- [ ] Migrate `LmbFilter` to `src/algorithms/lmb/lmb.rs`
- [ ] Migrate `LmbmFilter` to `src/algorithms/lmb/lmbm.rs`
- [ ] Create `src/algorithms/lmb/multisensor/mod.rs`
- [ ] Migrate `MultisensorLmbFilter` to `src/algorithms/lmb/multisensor/lmb.rs`
- [ ] Extract fusion strategies to `src/algorithms/lmb/multisensor/fusion.rs`
- [ ] Migrate `MultisensorLmbmFilter` to `src/algorithms/lmb/multisensor/lmbm.rs`
- [ ] Move cardinality to `src/algorithms/lmb/cardinality.rs`

### Batch 5: Cleanup
- [ ] Remove old `src/filter/` directory
- [ ] Remove old `src/types/` directory
- [ ] Remove old `src/lmb/` directory
- [ ] Update `src/lib.rs` with new module structure
- [ ] Create `src/prelude.rs` with convenient re-exports
- [ ] Documentation pass on all public items

### Batch 6: Test Verification
- [ ] Run all MATLAB equivalence tests - must pass
- [ ] Update test imports to new paths
- [ ] Add integration tests for `Tracker` trait

---

## Target Module Structure

```
src/
├── lib.rs
├── prelude.rs
│
├── core/                    # Algorithm-agnostic abstractions
│   ├── mod.rs
│   ├── tracker.rs           # pub trait Tracker
│   ├── track.rs             # Core Track trait
│   ├── detection.rs         # Detection types
│   ├── output.rs            # StateEstimate, Trajectory
│   ├── weights.rs           # LogWeight, LinearWeight
│   └── config.rs            # Config traits
│
├── components/              # Reusable building blocks
│   ├── mod.rs
│   ├── kalman.rs
│   ├── existence.rs
│   ├── trajectory.rs
│   ├── pruning.rs
│   └── gating.rs
│
├── association/             # Data association algorithms
│   ├── mod.rs
│   ├── traits.rs
│   ├── lbp.rs
│   ├── gibbs.rs
│   ├── murty.rs
│   ├── hungarian.rs
│   ├── greedy.rs            # Future: SORT
│   └── cascade.rs           # Future: DeepSORT
│
├── motion/                  # Motion models
│   ├── mod.rs
│   ├── traits.rs
│   ├── constant_velocity.rs
│   └── constant_acceleration.rs
│
├── algorithms/              # Tracking algorithm families
│   ├── mod.rs
│   ├── lmb/                 # LMB family
│   │   ├── mod.rs
│   │   ├── types.rs
│   │   ├── config.rs
│   │   ├── lmb.rs
│   │   ├── lmbm.rs
│   │   ├── cardinality.rs
│   │   └── multisensor/
│   │       ├── mod.rs
│   │       ├── lmb.rs
│   │       ├── lmbm.rs
│   │       └── fusion.rs
│   ├── sort/                # Future
│   └── bytetrack/           # Future
│
└── common/
    ├── mod.rs
    ├── constants.rs
    ├── linalg.rs
    ├── robust.rs
    └── rng.rs
```

---

## Key Traits

### Tracker (core)
```rust
pub trait Tracker {
    type State;
    type Detection;

    fn step<R: Rng>(
        &mut self,
        detections: &[Self::Detection],
        timestamp: usize,
        rng: &mut R,
    ) -> Result<StateEstimate, TrackerError>;

    fn state(&self) -> &Self::State;
    fn reset(&mut self);
    fn state_dim(&self) -> usize;
    fn detection_dim(&self) -> usize;
}
```

### LogWeight / LinearWeight (type-safe)
```rust
#[derive(Copy, Clone)]
pub struct LogWeight(f64);

#[derive(Copy, Clone)]
pub struct LinearWeight(f64);

impl LogWeight {
    pub fn to_linear(self) -> LinearWeight;
    pub fn log_sum_exp(weights: &[Self]) -> Self;
}
```

---

## Success Criteria

1. All MATLAB equivalence tests pass
2. Clean module structure matching plan
3. No code duplication
4. Type-safe weights throughout
5. Unified `Tracker` trait implemented by all filters
6. Clear extension points for SORT/ByteTrack
7. Documentation on all public items

---

## Notes

- Full plan details: `.claude/plans/drifting-greeting-acorn.md`
- Previous progress: `PROGRESS.md` (archived)
