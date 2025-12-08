# Implementation Plan: CI, Python Bindings, and Publishing

This plan upgrades `multisensor-lmb-filters-rs` (ported from MATLAB) to include CI/CD, Python bindings, comprehensive tests, and publishing to cargo + PyPI.

**Reference**: `../norfair-rust/` for patterns

---

## Phase 0: Rename Crate ✅ DONE

- [x] **Cargo.toml**: Change `name = "prak"` to `name = "multisensor-lmb-filters-rs"`
- [x] **Cargo.toml**: Change `[lib] name = "prak"` to `name = "multisensor_lmb_filters_rs"`
- [x] **tests/*.rs**: Replace all `use prak::` with `use multisensor_lmb_filters_rs::`
- [x] **benches/*.rs**: Replace all `use prak::` with `use multisensor_lmb_filters_rs::`
- [x] **README.md**: Update any `prak` references
- [x] **Makefile**: Update any `prak` references
- [x] **docs/*.md**: Update any `prak` references
- [x] Verify: `cargo build` succeeds after rename
- [x] Verify: `cargo test` passes after rename

---

## Phase 1: CI/CD Setup (Rust Only) ✅ DONE

### 1.1 Create `.github/workflows/ci.yml`
Reference: `../norfair-rust/.github/workflows/ci.yml`

- [x] Create `.github/workflows/` directory
- [x] Create `ci.yml` with jobs:
  - [x] `rust-lint`: cargo fmt --check, cargo clippy -- -D warnings
  - [x] `rust-test`: cargo test --release (with rust-cache)
  - [x] `pre-commit`: run pre-commit hooks

### 1.2 Create `.github/workflows/release.yml`
Reference: `../norfair-rust/.github/workflows/release.yml`

- [x] Create `release.yml` triggered on `v*` tags with jobs:
  - [x] `version-check`: Validate Cargo.toml version matches git tag
  - [x] `rust-publish`: crates.io via OIDC
  - [x] `github-release`: Create release

### 1.3 Create `.pre-commit-config.yaml`
Reference: `../norfair-rust/.pre-commit-config.yaml`

- [x] Create `.pre-commit-config.yaml` with:
  - [x] Local hooks: cargo fmt, cargo clippy

---

## Phase 2: Python Bindings (Rust Side)

### 2.1 Update `Cargo.toml`
Reference: `../norfair-rust/Cargo.toml`

- [ ] Add `rust-version = "1.70"`
- [ ] Add repository, keywords, categories metadata
- [ ] Change `crate-type = ["rlib"]` to `crate-type = ["cdylib", "rlib"]`
- [ ] Add `[features]` section with `python = ["dep:pyo3", "dep:numpy"]`
- [ ] Add pyo3 dependency: `pyo3 = { version = "0.23", optional = true, features = ["extension-module", "abi3-py312"] }`
- [ ] Add numpy dependency: `numpy = { version = "0.23", optional = true }`
- [ ] Add `[profile.release]` with `lto = true`, `codegen-units = 1`

### 2.2 Create `src/python/mod.rs`
Reference: `../norfair-rust/src/python/mod.rs`

- [ ] Create `src/python/` directory
- [ ] Create `mod.rs` with `#[pymodule] fn _multisensor_lmb_filters_rs`
- [ ] Export version: `__version__`

### 2.3 Create `src/python/types.rs`
Reference: `../norfair-rust/src/python/detection.rs`, `tracked_object.rs`

- [ ] Create PyO3 wrapper for `Track` → `PyTrack`
- [ ] Create PyO3 wrapper for `TrackLabel` → `PyTrackLabel`
- [ ] Create PyO3 wrapper for `GaussianComponent` → `PyGaussianComponent`
- [ ] Create PyO3 wrapper for `LmbmHypothesis` → `PyLmbmHypothesis`

### 2.4 Create `src/python/config.rs`
Reference: `../norfair-rust/src/python/filters.rs`

- [ ] Create `PyMotionModel` with factory methods (`constant_velocity_2d`, etc.)
- [ ] Create `PySensorModel` with factory methods (`position_sensor_2d`, etc.)
- [ ] Create `PyBirthModel`
- [ ] Create `PyBirthLocation`
- [ ] Create `PyAssociationConfig`
- [ ] Create `PyFilterThresholds`
- [ ] Create `PyLmbmConfig`
- [ ] Create `PyMultisensorConfig`

### 2.5 Create `src/python/filters.rs`
Reference: `../norfair-rust/src/python/tracker.rs`

- [ ] Create `PyLmbFilter` with `step()`, `state()`, `reset()` methods
- [ ] Create `PyLmbmFilter` with `step()`, `state()`, `reset()` methods
- [ ] Create `PyAaLmbFilter` with `step()` method
- [ ] Create `PyGaLmbFilter` with `step()` method
- [ ] Create `PyPuLmbFilter` with `step()` method
- [ ] Create `PyIcLmbFilter` with `step()` method
- [ ] Create `PyMultisensorLmbmFilter` with `step()` method

### 2.6 Create `src/python/associators.rs`

- [ ] Create `PyLbpAssociator`
- [ ] Create `PyGibbsAssociator`
- [ ] Create `PyMurtyAssociator`

### 2.7 Create `src/python/output.rs`

- [ ] Create `PyStateEstimate` with numpy array conversion
- [ ] Create `PyEstimatedTrack`
- [ ] Create `PyFilterOutput`
- [ ] Create `PyTrajectory`

### 2.8 Update `src/lib.rs`

- [ ] Add `#[cfg(feature = "python")] pub mod python;`
- [ ] Verify: `cargo build --features python` succeeds

---

## Phase 3: Python Package

### 3.1 Create `pyproject.toml`
Reference: `../norfair-rust/pyproject.toml`

- [ ] Create `pyproject.toml` with:
  - [ ] `[build-system]` using maturin
  - [ ] `[project]` metadata (name, dynamic version, description, license, requires-python)
  - [ ] `[project.optional-dependencies]` for dev
  - [ ] `[dependency-groups]` for dev
  - [ ] `[tool.maturin]` with features, python-source, module-name
  - [ ] `[tool.pytest.ini_options]`
  - [ ] `[tool.ruff]` configuration

### 3.2 Create `python/multisensor_lmb_filters_rs/__init__.py`
Reference: `../norfair-rust/python/norfair_rs/__init__.py`

- [ ] Create `python/multisensor_lmb_filters_rs/` directory
- [ ] Create `__init__.py` with re-exports from native module
- [ ] Define `__all__` with all public symbols
- [ ] Export: Track, TrackLabel, GaussianComponent
- [ ] Export: MotionModel, SensorModel, BirthModel, AssociationConfig
- [ ] Export: LmbFilter, LmbmFilter, AaLmbFilter, GaLmbFilter, PuLmbFilter, IcLmbFilter
- [ ] Export: LbpAssociator, GibbsAssociator, MurtyAssociator
- [ ] Export: StateEstimate, EstimatedTrack, FilterOutput, Trajectory
- [ ] Export: __version__

### 3.3 Create `python/multisensor_lmb_filters_rs/_multisensor_lmb_filters_rs.pyi`
Reference: `../norfair-rust/python/norfair_rs/_norfair_rs.pyi`

- [ ] Create type stubs file with:
  - [ ] All class signatures with docstrings
  - [ ] `Literal` types for enums (e.g., `DataAssociationMethod = Literal["lbp", "gibbs", "murty"]`)
  - [ ] numpy array types using `NDArray[np.float64]`
  - [ ] Factory method signatures
  - [ ] Property types

### 3.4 Verify Python Build

- [ ] Run `uv run maturin develop --release`
- [ ] Verify: `python -c "import multisensor_lmb_filters_rs; print(multisensor_lmb_filters_rs.__version__)"`

---

## Phase 4: Python Tests (100% Equivalence)

### 4.1 Create `python/tests/conftest.py`
Reference: `../norfair-rust/python/tests/conftest.py`

- [ ] Create `python/tests/` directory
- [ ] Create `conftest.py` with:
  - [ ] `find_testdata_dir()` helper
  - [ ] `testdata_dir` fixture
  - [ ] `load_fixture` fixture factory

### 4.2 Create `python/tests/test_fixtures.py` - ALL Fixtures
Reference: `../norfair-rust/python/tests/test_fixtures.py`

Each fixture must be tested by BOTH Rust AND Python with identical results (1e-10 tolerance):

- [ ] `test_single_trial_42()` - loads `single_trial_42.json`
- [ ] `test_single_trial_42_quick()` - loads `single_trial_42_quick.json`
- [ ] `test_single_detection_trial_42_quick()` - loads `single_detection_trial_42_quick.json`
- [ ] `test_multisensor_trial_42()` - loads `multisensor_trial_42.json`
- [ ] `test_multisensor_clutter_trial_42_quick()` - loads `multisensor_clutter_trial_42_quick.json`
- [ ] `test_multisensor_detection_trial_42_quick()` - loads `multisensor_detection_trial_42_quick.json`
- [ ] `test_lmb_step_by_step_matlab_equivalence()` - loads `step_by_step/lmb_step_by_step_seed42.json`
- [ ] `test_lmbm_step_by_step_matlab_equivalence()` - loads `step_by_step/lmbm_step_by_step_seed42.json`
- [ ] `test_multisensor_lmb_step_by_step_matlab_equivalence()` - loads `step_by_step/multisensor_lmb_step_by_step_seed42.json`
- [ ] `test_multisensor_lmbm_step_by_step_matlab_equivalence()` - loads `step_by_step/multisensor_lmbm_step_by_step_seed42.json`

### 4.3 Create `python/tests/test_types.py`

- [ ] Test `Track` creation and properties
- [ ] Test `TrackLabel` creation and comparison
- [ ] Test `GaussianComponent` creation and properties
- [ ] Test `LmbmHypothesis` creation (if exposed)

### 4.4 Create `python/tests/test_config.py`

- [ ] Test `MotionModel.constant_velocity_2d()` factory
- [ ] Test `SensorModel.position_sensor_2d()` factory
- [ ] Test `BirthModel` creation with locations
- [ ] Test `BirthLocation` creation
- [ ] Test `AssociationConfig` creation and defaults
- [ ] Test `FilterThresholds` creation and defaults
- [ ] Test `LmbmConfig` creation
- [ ] Test `MultisensorConfig` creation

### 4.5 Create `python/tests/test_filters.py`

- [ ] Test `LmbFilter` - init, step(), state(), reset()
- [ ] Test `LmbmFilter` - init, step(), state(), reset()

### 4.6 Create `python/tests/test_multisensor.py`

- [ ] Test `AaLmbFilter` - init, step()
- [ ] Test `GaLmbFilter` - init, step()
- [ ] Test `PuLmbFilter` - init, step()
- [ ] Test `IcLmbFilter` - init, step()
- [ ] Test `MultisensorLmbmFilter` - init, step()

### 4.7 Create `python/tests/test_associators.py`

- [ ] Test `LbpAssociator` - creation, parameters
- [ ] Test `GibbsAssociator` - creation, parameters
- [ ] Test `MurtyAssociator` - creation, parameters

### 4.8 Create `python/tests/test_output.py`

- [ ] Test `StateEstimate` - properties, numpy conversion
- [ ] Test `EstimatedTrack` - properties
- [ ] Test `FilterOutput` - properties
- [ ] Test `Trajectory` - properties

### 4.9 Verify Tests Pass

- [ ] Run `uv run pytest python/tests -v`
- [ ] Verify ALL fixture tests pass
- [ ] Verify ALL API surface tests pass

---

## Phase 5: Final Verification

### 5.1 Rust Checks

- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo test --release` passes
- [ ] `cargo build --release` succeeds

### 5.2 Python Checks

- [ ] `uv run maturin develop --release` succeeds
- [ ] `uv run pytest python/tests -v` passes ALL tests
- [ ] `uv run ruff check python/` passes
- [ ] `uv run ruff format --check python/` passes

### 5.3 Equivalence Verification

- [ ] Python fixture tests produce IDENTICAL results to Rust fixture tests
- [ ] All 10 fixture files tested by both Rust and Python
- [ ] Numerical tolerance: 1e-10 for all comparisons

### 5.4 Pre-commit

- [ ] `pre-commit install`
- [ ] `pre-commit run --all-files` passes

---

## Phase 6: Publishing (Manual First Release)

### 6.1 Prepare Release

- [ ] Update version in `Cargo.toml` to `0.1.0`
- [ ] Update CHANGELOG.md (if exists)
- [ ] Commit all changes
- [ ] Create git tag: `git tag v0.1.0`

### 6.2 Test Release Workflow

- [ ] Push tag: `git push origin v0.1.0`
- [ ] Verify CI passes on tag
- [ ] Verify wheels are built for all platforms
- [ ] Verify crates.io publish succeeds (or test with `--dry-run`)
- [ ] Verify PyPI publish succeeds (or test with TestPyPI)

### 6.3 Post-Release

- [ ] Verify package installable: `pip install multisensor_lmb_filters_rs`
- [ ] Verify crate installable: `cargo add multisensor-lmb-filters-rs`
- [ ] Create GitHub release with changelog

---

## Reference Files

| Purpose | Reference File |
|---------|----------------|
| CI workflow | `../norfair-rust/.github/workflows/ci.yml` |
| Release workflow | `../norfair-rust/.github/workflows/release.yml` |
| Pre-commit config | `../norfair-rust/.pre-commit-config.yaml` |
| Cargo.toml | `../norfair-rust/Cargo.toml` |
| pyproject.toml | `../norfair-rust/pyproject.toml` |
| Python module | `../norfair-rust/src/python/mod.rs` |
| Python __init__.py | `../norfair-rust/python/norfair_rs/__init__.py` |
| Type stubs | `../norfair-rust/python/norfair_rs/_norfair_rs.pyi` |
| Fixture tests | `../norfair-rust/python/tests/test_fixtures.py` |
| conftest.py | `../norfair-rust/python/tests/conftest.py` |
