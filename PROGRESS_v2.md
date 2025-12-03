# Prak v2 Refactoring Progress

## Goal

Transform Prak into a high-quality, extensible open-source multi-object tracking library supporting multiple algorithm families (LMB, SORT, ByteTrack, Norfair).

## Design Decisions

- **Clean API break** - no backwards compatibility shims
- **Type-safe weights** - `LogWeight`/`LinearWeight` wrappers (DEFERRED)
- **Algorithm-family directories** - `src/algorithms/{lmb,sort,bytetrack}/` (DEFERRED)
- **Keep "Filter" names** - `LmbFilter` implements `Tracker` trait but keeps name

---

## Current Status

**Phase:** Library cleanup complete - LMB code is well-organized
**Last Updated:** 2025-12-03

**Current structure:**
```
src/
├── lib.rs
├── association/           # Association matrix building
│   ├── mod.rs
│   ├── builder.rs         # AssociationBuilder, AssociationMatrices
│   └── likelihood.rs      # Likelihood computation
├── common/                # Low-level utilities
│   ├── mod.rs
│   ├── association/       # Algorithm implementations
│   │   ├── lbp.rs         # Loopy Belief Propagation
│   │   ├── gibbs.rs       # Gibbs sampling
│   │   ├── murtys.rs      # Murty's k-best
│   │   └── hungarian.rs   # Hungarian algorithm
│   ├── constants.rs       # Numerical constants
│   ├── linalg.rs          # Kalman, Mahalanobis, robust ops
│   └── rng.rs             # RNG utilities
├── components/            # Shared algorithms
│   ├── mod.rs
│   ├── prediction.rs      # Track prediction
│   └── update.rs          # Existence updates
└── lmb/                   # LMB filter family
    ├── mod.rs             # Re-exports
    ├── builder.rs         # FilterBuilder traits
    ├── cardinality.rs     # MAP cardinality estimation
    ├── common_ops.rs      # Pruning, gating, normalization
    ├── config.rs          # MotionModel, SensorModel, etc.
    ├── errors.rs          # FilterError
    ├── output.rs          # StateEstimate, Trajectory
    ├── traits.rs          # Associator, Merger, Updater traits + impls
    ├── types.rs           # Track, GaussianComponent, etc.
    ├── singlesensor/
    │   ├── mod.rs
    │   ├── lmb.rs         # LmbFilter
    │   └── lmbm.rs        # LmbmFilter
    └── multisensor/
        ├── mod.rs
        ├── lmb.rs         # MultisensorLmbFilter + AA/GA/PU/IC aliases
        ├── lmbm.rs        # MultisensorLmbmFilter
        ├── fusion.rs      # Merger implementations
        └── traits.rs      # MultisensorAssociator trait
```

---

## Changelog

### 2025-12-03: Phase 2 - Cleanup & Deduplication

**Completed:**
- Removed deprecated `types` and `filter` re-export modules from `src/lib.rs`
- Extracted fusion strategies (~370 lines) to `src/lmb/multisensor/fusion.rs`
- Created `src/lmb/builder.rs` with `FilterBuilder` and `LmbFilterBuilder` traits
- Implemented builder traits for all 4 filter types, removing duplicate methods
- Fixed `crate::types::` references in dependent files
- All tests passing

---

## TODO Summary

Most planned work is **already complete** or **deferred** until needed:

| Batch | Status | Notes |
|-------|--------|-------|
| 1: Core Abstractions | SKIPPED | Defer until SORT/ByteTrack needed |
| 2: Component Consolidation | DONE | Already consolidated in linalg.rs, update.rs, common_ops.rs |
| 3: Association Refactor | DONE | Associator trait exists with LBP/Gibbs/Murty impls |
| 4: Algorithm Migration | DEFERRED | Current src/lmb/ structure is clean |
| 5: Cleanup | MOSTLY DONE | Deprecated modules removed |
| 6: Test Verification | DONE | All tests pass |

---

## Detailed TODO List

### Batch 1: Core Abstractions (SKIPPED - defer until SORT/ByteTrack needed)
- [ ] ~~Create `src/core/tracker.rs`~~ - Tracker trait for multiple algorithms
- [ ] ~~Create `src/core/weights.rs`~~ - LogWeight/LinearWeight newtypes
- [ ] ~~Create `src/motion/traits.rs`~~ - Generic MotionModel trait

### Batch 2: Component Consolidation ✅ COMPLETE
Already consolidated in existing files:
- [x] Kalman operations → `src/common/linalg.rs`
- [x] Existence updates → `src/components/update.rs`
- [x] Pruning/gating → `src/lmb/common_ops.rs`
- [x] Mahalanobis distance → `src/common/linalg.rs`
- [x] Robust matrix ops → `src/common/linalg.rs`

### Batch 3: Association Refactor ✅ COMPLETE
Already implemented:
- [x] `Associator` trait → `src/lmb/traits.rs`
- [x] `LbpAssociator` → `src/lmb/traits.rs`
- [x] `GibbsAssociator` → `src/lmb/traits.rs`
- [x] `MurtyAssociator` → `src/lmb/traits.rs`
- [ ] Greedy associator (SORT) - DEFER
- [ ] Cascade associator (DeepSORT) - DEFER

### Batch 4: Algorithm Migration (DEFERRED)
Current `src/lmb/` structure is clean. Migration to `src/algorithms/lmb/` deferred until SORT/ByteTrack are actually added.

### Batch 5: Cleanup ✅ MOSTLY COMPLETE
- [x] Remove deprecated `src/filter/` re-exports
- [x] Remove deprecated `src/types/` re-exports
- [x] Extracted fusion strategies to separate file
- [x] Created builder traits for deduplication
- [ ] Create `src/prelude.rs` - OPTIONAL
- [ ] Documentation pass - OPTIONAL

### Batch 6: Test Verification ✅ COMPLETE
- [x] All MATLAB equivalence tests pass
- [x] All unit tests pass

---

## Future Work (When Adding SORT/ByteTrack)

When adding new tracking algorithms:

1. Create `src/algorithms/` directory
2. Move `src/lmb/` to `src/algorithms/lmb/`
3. Create `src/core/tracker.rs` with unified `Tracker` trait
4. Add `src/algorithms/sort/` with SORT implementation
5. Add greedy/cascade associators

---

## Success Criteria ✅

1. ✅ All MATLAB equivalence tests pass
2. ✅ Clean module structure
3. ✅ Minimal code duplication (builder traits, common_ops)
4. ⏸️ Type-safe weights (deferred)
5. ⏸️ Unified `Tracker` trait (deferred)
6. ⏸️ Extension points for SORT/ByteTrack (deferred)
7. ⏸️ Documentation pass (optional)

---

## Notes

- Plan file: `.claude/plans/drifting-greeting-acorn.md`
- The library is in a clean, usable state
- Future extensibility work deferred until actually needed
