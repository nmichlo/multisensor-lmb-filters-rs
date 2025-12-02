# PRAK Library Refactoring Progress

## Status: Phase 1 Complete - Module Structure Created

**Last Updated:** 2025-12-02

---

## Completed Work

### Step 1: Module Skeleton Creation ✅

All new module structures have been created and compile successfully.

#### types/ module
- [x] `src/types/mod.rs` - Module exports
- [x] `src/types/track.rs` - Track, GaussianComponent, TrackLabel, TrajectoryHistory, LmbmHypothesis
- [x] `src/types/config.rs` - MotionModel, SensorModel, FilterParams, etc.
- [x] `src/types/output.rs` - StateEstimate, EstimatedTrack, FilterOutput, Trajectory

#### components/ module
- [x] `src/components/mod.rs` - Module exports
- [x] `src/components/prediction.rs` - `predict_tracks()`, `predict_track()`, `predict_component()`
- [x] `src/components/update.rs` - `update_existence_no_detection()`, `update_existence_with_measurement()`

#### association/ module
- [x] `src/association/mod.rs` - Module exports
- [x] `src/association/likelihood.rs` - `LikelihoodWorkspace`, `compute_likelihood()`, `compute_log_likelihood()`
- [x] `src/association/builder.rs` - `AssociationBuilder`, `AssociationMatrices`, `PosteriorGrid`

#### filter/ module
- [x] `src/filter/mod.rs` - Module exports
- [x] `src/filter/traits.rs` - Filter, Associator, Merger, Updater traits + placeholder implementations
- [x] `src/filter/errors.rs` - FilterError, AssociationError

#### lib.rs
- [x] Updated to export new modules alongside legacy modules
- [x] New API (v2) and Legacy API documented

### Dependencies Added
- [x] `smallvec = "1.11"` added to Cargo.toml

### Tests
- [x] All existing tests pass (`cargo test --release`)
- [x] New unit tests added for types, components, association

---

## Next Steps

### Step 2: Wire Up Associator Implementations

Connect the placeholder `LbpAssociator`, `GibbsAssociator`, `MurtyAssociator` to the actual implementations in `common/association/`:

1. `filter/traits.rs` - LbpAssociator.associate() should call `lbp::loopy_belief_propagation()`
2. `filter/traits.rs` - GibbsAssociator.associate() should call `gibbs::gibbs_sampling()`
3. `filter/traits.rs` - MurtyAssociator.associate() should call `murtys::murtys_kbest()`

### Step 3: Implement MarginalUpdater and HardAssignmentUpdater

Fill in the actual update logic for:
- `MarginalUpdater` - LMB marginal reweighting
- `HardAssignmentUpdater` - LMBM hard selection

### Step 4: Implement LmbFilter

Create `src/filter/lmb.rs` with full `LmbFilter` implementation:
- Uses new types (`Track`, `FilterParams`)
- Uses `predict_tracks()` from components
- Uses `AssociationBuilder` from association
- Uses `LbpAssociator` (or selectable via config)
- Uses `MarginalUpdater`

### Step 5: Implement LmbmFilter

Create `src/filter/lmbm.rs` with `LmbmFilter` implementation.

### Step 6: Implement MultisensorLmbFilter<M: Merger>

Create `src/filter/multisensor_lmb.rs` with generic merger support:
- `ArithmeticAverageMerger`
- `GeometricAverageMerger`
- `ParallelUpdateMerger`
- `IteratedCorrectorMerger`

### Step 7: Implement MultisensorLmbmFilter

Create `src/filter/multisensor_lmbm.rs`.

### Step 8: Migration Testing

1. Create comparative tests that run both old and new implementations
2. Verify numerical equivalence with MATLAB
3. Run benchmarks to check for performance regression

### Step 9: Cleanup

1. Remove legacy modules once new implementations are validated
2. Update examples to use new API
3. Update documentation

---

## Files Overview

### New Files Created
```
src/
├── types/
│   ├── mod.rs          ✅
│   ├── track.rs        ✅ (pre-existing, reviewed)
│   ├── config.rs       ✅ (pre-existing, reviewed)
│   └── output.rs       ✅
├── components/
│   ├── mod.rs          ✅
│   ├── prediction.rs   ✅
│   └── update.rs       ✅
├── association/
│   ├── mod.rs          ✅
│   ├── likelihood.rs   ✅
│   └── builder.rs      ✅
├── filter/
│   ├── mod.rs          ✅
│   ├── traits.rs       ✅
│   └── errors.rs       ✅
└── lib.rs              ✅ (modified)
```

### Legacy Files (to be replaced)
```
src/
├── common/
│   ├── types.rs        → replaced by src/types/
│   ├── model.rs        → replaced by src/types/config.rs
│   └── association/    → algorithms moved to src/association/
├── lmb/                → replaced by src/filter/lmb.rs
├── lmbm/               → replaced by src/filter/lmbm.rs
├── multisensor_lmb/    → replaced by src/filter/multisensor_lmb.rs
└── multisensor_lmbm/   → replaced by src/filter/multisensor_lmbm.rs
```

---

## Design Decisions

1. **Runtime dimensions** - Using DVector/DMatrix for Python binding compatibility
2. **SmallVec** - Stack allocation for GM components (typically 1-4)
3. **Trait-based** - Filter, Associator, Merger, Updater traits for flexibility
4. **Error types** - FilterError and AssociationError instead of panics
5. **Parallel structure** - All modules created together, implementation follows

---

## Notes

- Old code remains functional during migration
- Both APIs coexist in lib.rs
- Tests verify no regression in existing functionality
