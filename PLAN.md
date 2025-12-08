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
  - [x] `python-test`: uv run maturin develop, pytest tests
  - [x] `pre-commit`: run pre-commit hooks (with astral-sh/setup-uv)

### 1.2 Create `.github/workflows/release.yml`
Reference: `../norfair-rust/.github/workflows/release.yml`

- [x] Create `release.yml` triggered on `v*` tags with jobs:
  - [x] `version-check`: Validate Cargo.toml version matches git tag
  - [x] `python-build`: Build wheels for Linux/macOS/Windows (x86_64 + aarch64)
  - [x] `python-sdist`: Build source distribution
  - [x] `python-publish`: PyPI via trusted publishing (OIDC)
  - [x] `rust-publish`: crates.io via OIDC
  - [x] `github-release`: Create release with wheels attached

### 1.3 Create `.pre-commit-config.yaml`
Reference: `../norfair-rust/.pre-commit-config.yaml`

- [x] Create `.pre-commit-config.yaml` with:
  - [x] Local hooks: cargo fmt, cargo clippy

---

## Phase 2: Python Bindings (Rust Side) ✅ DONE

### 2.1 Update `Cargo.toml`
Reference: `../norfair-rust/Cargo.toml`

- [x] Add repository, keywords, categories metadata
- [x] Change `crate-type = ["rlib"]` to `crate-type = ["cdylib", "rlib"]`
- [x] Add `[features]` section with `python = ["dep:pyo3", "dep:numpy"]`
- [x] Add pyo3 dependency: `pyo3 = { version = "0.23", optional = true, features = ["extension-module", "abi3-py312"] }`
- [x] Add numpy dependency: `numpy = { version = "0.23", optional = true }`
- [x] Add `[profile.release]` with `lto = true`, `codegen-units = 1`

### 2.2 Create `src/python/mod.rs`
Reference: `../norfair-rust/src/python/mod.rs`

- [x] Create `src/python/` directory
- [x] Create `mod.rs` with `#[pymodule] fn _multisensor_lmb_filters_rs`
- [x] Export version: `__version__`

### 2.3 Create `src/python/types.rs`
Reference: `../norfair-rust/src/python/detection.rs`, `tracked_object.rs`

- [x] Create PyO3 wrapper for `Track` → `PyTrack`
- [x] Create PyO3 wrapper for `TrackLabel` → `PyTrackLabel`
- [x] Create PyO3 wrapper for `GaussianComponent` → `PyGaussianComponent`

### 2.4 Create `src/python/config.rs`
Reference: `../norfair-rust/src/python/filters.rs`

- [x] Create `PyMotionModel` with factory methods (`constant_velocity_2d`, etc.)
- [x] Create `PySensorModel` with factory methods (`position_sensor_2d`, etc.)
- [x] Create `PyBirthModel`
- [x] Create `PyBirthLocation`
- [x] Create `PyAssociationConfig`
- [x] Create `PyFilterThresholds`
- [x] Create `PyLmbmConfig`
- [x] Create `PyMultisensorConfig`

### 2.5 Create `src/python/filters.rs`
Reference: `../norfair-rust/src/python/tracker.rs`

- [x] Create `PyLmbFilter` with `step()`, `reset()` methods
- [x] Create `PyLmbmFilter` with `step()`, `reset()` methods
- [x] Create `PyAaLmbFilter` with `step()` method
- [x] Create `PyGaLmbFilter` with `step()` method
- [x] Create `PyPuLmbFilter` with `step()` method
- [x] Create `PyIcLmbFilter` with `step()` method
- [x] Create `PyMultisensorLmbmFilter` with `step()` method

### 2.6 Create `src/python/output.rs`

- [x] Create `PyStateEstimate` with numpy array conversion
- [x] Create `PyEstimatedTrack`
- [x] Create `PyFilterOutput`
- [x] Create `PyTrajectory`

### 2.7 Update `src/lib.rs`

- [x] Add `#[cfg(feature = "python")] pub mod python;`
- [x] Verify: `cargo build --features python` succeeds

---

## Phase 3: Python Package ✅ DONE

### 3.1 Create `pyproject.toml`
Reference: `../norfair-rust/pyproject.toml`

- [x] Create `pyproject.toml` with:
  - [x] `[build-system]` using maturin
  - [x] `[project]` metadata (name, version, description, license, requires-python)
  - [x] `[project.optional-dependencies]` for dev
  - [x] `[tool.maturin]` with features, python-source, module-name
  - [x] `[tool.pytest.ini_options]`

### 3.2 Create `python/multisensor_lmb_filters_rs/__init__.py`
Reference: `../norfair-rust/python/norfair_rs/__init__.py`

- [x] Create `python/multisensor_lmb_filters_rs/` directory
- [x] Create `__init__.py` with re-exports from native module
- [x] Define `__all__` with all public symbols
- [x] Export: Track, TrackLabel, GaussianComponent
- [x] Export: MotionModel, SensorModel, BirthModel, AssociationConfig
- [x] Export: LmbFilter, LmbmFilter, AaLmbFilter, GaLmbFilter, PuLmbFilter, IcLmbFilter
- [x] Export: StateEstimate, EstimatedTrack, FilterOutput, Trajectory
- [x] Export: __version__

### 3.3 Create `python/multisensor_lmb_filters_rs/_multisensor_lmb_filters_rs.pyi`
Reference: `../norfair-rust/python/norfair_rs/_norfair_rs.pyi`

- [x] Create type stubs file with:
  - [x] All class signatures with docstrings
  - [x] numpy array types using `NDArray[np.float64]`
  - [x] Factory method signatures
  - [x] Property types
  - [x] Python protocols for common interfaces (HasLabel, HasMean, HasCovariance, etc.)

### 3.4 Create `python/multisensor_lmb_filters_rs/py.typed`

- [x] Create py.typed marker file for PEP 561 typing support

### 3.5 Verify Python Build

- [x] Run `uvx maturin build --release`
- [x] Verify: `python -c "import multisensor_lmb_filters_rs; print(multisensor_lmb_filters_rs.__version__)"`

---

## Phase 4: Python Tests ✅ DONE

### 4.1 Create `tests/conftest.py`
Reference: `../norfair-rust/python/tests/conftest.py`

- [x] Create `tests/` directory
- [x] Create `conftest.py` with:
  - [x] `motion_model_2d` fixture
  - [x] `sensor_model_2d` fixture
  - [x] `birth_model_2d` fixture
  - [x] `multisensor_config_2d` fixture
  - [x] `lmb_filter` and `lmbm_filter` fixtures
  - [x] `sample_measurements` fixture

### 4.2 Create `tests/test_api.py`

- [x] Test all imports work
- [x] Test `TrackLabel` - creation, comparison, hash
- [x] Test `GaussianComponent` - creation, properties
- [x] Test `Track` - creation, properties, components
- [x] Test `MotionModel` - factory methods, properties
- [x] Test `SensorModel` - factory methods, properties
- [x] Test `BirthModel` and `BirthLocation`
- [x] Test `AssociationConfig` - all methods (lbp, gibbs, murty)
- [x] Test `FilterThresholds` - creation, defaults
- [x] Test `LmbmConfig` - creation, defaults
- [x] Test `MultisensorConfig` - creation, properties

### 4.3 Create `tests/test_filters.py`

- [x] Test `LmbFilter` - create, step, reset, reproducibility, association methods
- [x] Test `LmbmFilter` - create, step, custom config
- [x] Test `AaLmbFilter` - create, step
- [x] Test `GaLmbFilter` - create, step
- [x] Test `PuLmbFilter` - create, step
- [x] Test `IcLmbFilter` - create, step
- [x] Test `MultisensorLmbmFilter` - create, step
- [x] Test `StateEstimate` - properties
- [x] Test `EstimatedTrack` - properties
- [x] Test protocol compliance (HasLabel, HasMean, HasStateDimension, etc.)

### 4.4 Create `tests/test_equivalence.py`

- [x] Test determinism - same seed produces identical results
- [x] Test reproducibility across filter types
- [x] Test numerical stability with large/small values
- [x] Test Rust equivalence - verify matrices and evolution

### 4.5 Verify Tests Pass

- [x] Run `uv run --with wheel --with pytest pytest tests -v`
- [x] Verify ALL 50 tests pass

---

## Phase 5: Final Verification 🔄 IN PROGRESS

### 5.1 Rust Checks

- [x] `cargo fmt --check` passes
- [x] `cargo clippy -- -D warnings` passes
- [x] `cargo test --release` passes
- [x] `cargo build --release` succeeds

### 5.2 Python Checks

- [x] `uvx maturin build --release` succeeds
- [x] `pytest tests -v` passes ALL tests (50 tests)
- [ ] Run tests via CI workflow

### 5.3 Fixture Equivalence Tests

- [ ] Create tests that load JSON fixtures
- [ ] Verify Python produces same results as Rust fixtures
- [ ] Test all 10 fixture files

### 5.4 Pre-commit

- [ ] `pre-commit install`
- [ ] `pre-commit run --all-files` passes

---

## Phase 6: Publishing

### 6.1 Prepare Release

- [x] Version `0.1.0` published to crates.io
- [ ] Update version for next release if needed
- [ ] Update CHANGELOG.md (if exists)

### 6.2 Test Release Workflow

- [ ] Push tag: `git push origin v0.1.x`
- [ ] Verify CI passes on tag
- [ ] Verify wheels are built for all platforms
- [ ] Verify crates.io publish succeeds
- [ ] Verify PyPI publish succeeds

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
