# Python Fixture Equivalence Tests - Implementation Plan

## Status: COMPLETE (with known numerical divergences)

## Summary

**22 tests pass, 3 tests fail with numerical divergences**

The test infrastructure is complete and working. Tests now:
- Load prior tracks from fixtures using `set_tracks()`
- Get full intermediate outputs using `step_detailed()`
- Compare ALL intermediate values against MATLAB expected values

### Known Numerical Divergences (to investigate)

1. **Association matrices shape mismatch** - P matrix `(9, 17)` vs `(9, 18)`
   - Likely difference in miss-probability column handling
2. **LBP result numerical mismatch** - miss_weights value `1.0` vs expected `0.002`
   - Suggests issue with LBP algorithm or existence updates
3. **Update step existence mismatch** - `0.94` vs expected `0.77`
   - Related to LBP issue above

## TODO List

### Phase 1: Rust Core Changes - COMPLETE

- [x] **1.1** Add `StepDetailedOutput` struct to `src/lmb/types.rs`
- [x] **1.2** Add `set_tracks()` method to `LmbFilter` in `src/lmb/singlesensor/lmb.rs`
- [x] **1.3** Add `step_detailed()` method to `LmbFilter` in `src/lmb/singlesensor/lmb.rs`
- [x] **1.4** Add `set_tracks()` and `step_detailed()` to `MultisensorLmbFilter` in `src/lmb/multisensor/lmb.rs`
- [x] **1.5** Add `set_hypotheses()` and `step_detailed()` to `LmbmFilter` in `src/lmb/singlesensor/lmbm.rs`
- [x] **1.6** Add `set_hypotheses()` and `step_detailed()` to `MultisensorLmbmFilter` in `src/lmb/multisensor/lmbm.rs`
- [x] **1.7** Run `cargo test` to verify Rust changes compile and pass (22 pass, 2 ignored)

### Phase 2: Python Bindings - COMPLETE

- [x] **2.1** Add `_TrackData.__new__()` constructor in `src/python/intermediate.rs`
- [x] **2.2** Add `to_track()` conversion method in `src/python/intermediate.rs`
- [x] **2.3** Update `PyStepOutput` to handle optional fields in `src/python/intermediate.rs`
- [x] **2.4** Add `set_tracks()`, `get_tracks()`, `step_detailed()` to `PyFilterLmb` in `src/python/filters.rs`
- [x] **2.5** Add same methods to `PyFilterAaLmb`, `PyFilterGaLmb`, `PyFilterPuLmb`, `PyFilterIcLmb`
- [x] **2.6** Add `get_tracks()`, `step_detailed()` to `PyFilterLmbm`
- [x] **2.7** Add same methods to `PyFilterMultisensorLmbm`
- [x] **2.8** Register new types in `src/python/mod.rs`
- [x] **2.9** Export `_TrackData` etc. in `python/__init__.py`
- [x] **2.10** Run `cargo test` and `uvx maturin develop` to verify bindings build

### Phase 3: Python Test Infrastructure - COMPLETE

- [x] **3.1** Add `make_track_data()` function to `tests/conftest.py`
- [x] **3.2** Add `load_prior_tracks()` function to `tests/conftest.py`
- [x] **3.3** Add `compare_tracks()` function to `tests/conftest.py`
- [x] **3.4** Add `compare_association_matrices()` function to `tests/conftest.py`
- [x] **3.5** Add `make_birth_model_empty()` helper to `tests/conftest.py`
- [x] **3.6** Add `make_birth_model_from_fixture()` helper to `tests/conftest.py`

### Phase 4: Update Rust Fixture Tests - SKIPPED

Rust tests already exist in `tests/lmb/matlab_equivalence.rs` and pass.

### Phase 5: Rewrite Python Equivalence Tests - COMPLETE

- [x] **5.1** Rewrite `TestLmbFixtureEquivalence` with full intermediate validation
- [x] **5.2** Rewrite `TestLmbmFixtureEquivalence` with full intermediate validation
- [x] **5.3** Rewrite `TestMultisensorLmbFixtureEquivalence` with full intermediate validation
- [x] **5.4** Rewrite `TestMultisensorLmbmFixtureEquivalence` with full intermediate validation
- [x] **5.5** Run `uv run pytest tests/ -v` - 22 pass, 3 fail (numerical divergences)

### Phase 6: Verification - PARTIAL

- [x] **6.1** Verify all Rust tests pass: `cargo test`
- [x] **6.2** Python tests: 22 pass, 3 fail with numerical divergences
- [x] **6.3** Tests properly fail on numerical divergence - verified by failures

---

## Files to Modify

| File | Status | Changes |
|------|--------|---------|
| `src/lmb/types.rs` | DONE | Add `StepDetailedOutput`, `CardinalityEstimate` structs |
| `src/lmb/singlesensor/lmb.rs` | DONE | Add `set_tracks()`, `get_tracks()`, `step_detailed()` |
| `src/lmb/singlesensor/lmbm.rs` | DONE | Add `set_hypotheses()`, `get_tracks()`, `step_detailed()` |
| `src/lmb/multisensor/lmb.rs` | DONE | Add `set_tracks()`, `get_tracks()`, `step_detailed()` |
| `src/lmb/multisensor/lmbm.rs` | DONE | Add `set_hypotheses()`, `get_tracks()`, `step_detailed()` |
| `src/python/intermediate.rs` | DONE | Add `to_track()`, `_TrackData.__new__()`, Option fields |
| `src/python/filters.rs` | DONE | Add `set_tracks()`, `get_tracks()`, `step_detailed()` to all 7 filters |
| `tests/conftest.py` | TODO | Add comparison and loader functions |
| `tests/test_equivalence.py` | TODO | Rewrite all tests |

---

## Progress Log

_Updates will be added here as implementation proceeds._
