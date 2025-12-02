# PRAK Library Refactoring Progress

## Status: Step 5 Complete - LmbmFilter Implemented

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

### Step 2: Wire Up Associator Implementations ✅

Connected the placeholder associators to the actual implementations in `common/association/`:

1. [x] `LbpAssociator.associate()` → calls `legacy_lbp::loopy_belief_propagation()`
   - Converts between new and legacy `AssociationMatrices` types
   - Extracts miss weights from first column, marginal weights from remaining columns

2. [x] `GibbsAssociator.associate()` → calls `legacy_gibbs::lmb_gibbs_sampling()`
   - Builds `GibbsAssociationMatrices` (p, l, r, c) from new matrices
   - Uses `RngAdapter` to bridge `rand::Rng` to legacy `common::rng::Rng` trait
   - Converts sampled associations to new format (-1 for miss, 0-indexed measurements)

3. [x] `MurtyAssociator.associate()` → calls `legacy_murtys::murtys_algorithm_wrapper()`
   - Uses cost matrix directly from new matrices
   - Computes marginal probabilities from weighted k-best assignments
   - Handles dummy assignments (clutter) correctly

### Step 3: Implement MarginalUpdater and HardAssignmentUpdater ✅

Implemented the track update strategies:

1. [x] `MarginalUpdater` - LMB marginal reweighting
   - Reweights GM components by marginal association probabilities
   - Creates new components for each (prior component, measurement) pair
   - Miss case: prior components weighted by miss probability
   - Detection case: posterior components weighted by marginal probabilities
   - Prunes components below weight threshold, caps to max components
   - Renormalizes kept components

2. [x] `HardAssignmentUpdater` - LMBM hard selection
   - Uses sampled association events for deterministic assignment
   - Selects single posterior for each track (single-component result)
   - Detection: replaces all components with single posterior, sets existence=1.0
   - Miss: keeps prior state unchanged
   - Falls back to best association if no samples available

### Step 4: Implement LmbFilter ✅

Created `src/filter/lmb.rs` with full `LmbFilter<A: Associator>` implementation:

1. [x] `LmbFilter<A: Associator>` struct with generic associator
   - Default type parameter `LbpAssociator` for convenience
   - Stores motion, sensor, birth models and association config
   - Maintains tracks and trajectories
   - Configurable thresholds (existence, GM pruning, trajectory length)

2. [x] Constructor methods
   - `LmbFilter::new()` - Default constructor with LBP associator
   - `LmbFilter::from_params()` - Create from FilterParams (extracts single sensor)
   - `LmbFilter::with_associator_type()` - Custom associator

3. [x] Builder-style configuration
   - `with_existence_threshold()` - Set gating threshold
   - `with_min_trajectory_length()` - Set minimum trajectory to save
   - `with_gm_pruning()` - Set GM component pruning parameters

4. [x] Core methods
   - `gate_tracks()` - Remove low-existence tracks, save long trajectories
   - `extract_estimates()` - MAP cardinality estimation for state extraction
   - `update_trajectories()` - Record track states at each timestep
   - `init_birth_trajectories()` - Initialize trajectory recording for births
   - `update_existence_from_association()` - Apply association result to existence

5. [x] `Filter` trait implementation
   - `step()` - Full predict-update cycle with measurements
   - `state()` - Get current tracks
   - `reset()` - Clear all tracks and trajectories
   - `x_dim()` / `z_dim()` - State/measurement dimensions

6. [x] Unit tests pass
   - test_filter_creation
   - test_filter_step_no_measurements
   - test_filter_step_with_measurements
   - test_filter_multiple_steps
   - test_filter_reset

### Step 5: Implement LmbmFilter ✅

Created `src/filter/lmbm.rs` with full `LmbmFilter<A: Associator>` implementation:

1. [x] `LmbmFilter<A: Associator>` struct with generic associator
   - Default type parameter `GibbsAssociator` for sampling-based association
   - Maintains multiple weighted hypotheses (`Vec<LmbmHypothesis>`)
   - Each hypothesis has single-component tracks (hard assignments)
   - Stores motion, sensor, birth models and LMBM-specific config

2. [x] Constructor methods
   - `LmbmFilter::new()` - Default constructor with Gibbs associator
   - `LmbmFilter::from_params()` - Create from FilterParams
   - `LmbmFilter::with_associator_type()` - Custom associator (e.g., Murty)

3. [x] Hypothesis management
   - `predict_hypotheses()` - Apply motion model to all hypotheses
   - `generate_posterior_hypotheses()` - Create new hypotheses from association samples
   - `normalize_and_gate_hypotheses()` - Log-sum-exp normalization, pruning, sorting
   - `gate_tracks()` - Remove low-existence tracks across hypotheses

4. [x] Core methods
   - `build_log_likelihood_matrix()` - Compute log-likelihoods for hypothesis weights
   - `update_existence_no_measurements()` - Missed detection update
   - `extract_estimates()` - MAP/EAP cardinality estimation from hypothesis mixture
   - `update_trajectories()` / `init_birth_trajectories()` - Trajectory tracking

5. [x] `Filter` trait implementation
   - `step()` - Full predict-update-normalize cycle
   - `state()` - Returns current hypotheses
   - `reset()` - Clears all hypotheses, reinitializes with empty hypothesis

6. [x] Unit tests pass (6 tests)
   - test_filter_creation
   - test_filter_step_no_measurements
   - test_filter_step_with_measurements
   - test_filter_multiple_steps
   - test_filter_reset
   - test_hypothesis_normalization

---

## Next Steps

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
│   ├── errors.rs       ✅
│   ├── lmb.rs          ✅ (Step 4)
│   └── lmbm.rs         ✅ (Step 5)
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
