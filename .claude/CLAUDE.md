# Prak Development Guide

## Current Architecture

The library uses a trait-based API:

```
src/
├── types/              # Core types (Track, MotionModel, SensorModel, etc.)
├── components/         # Shared algorithms (prediction, update)
├── association/        # Data association (likelihood, builder)
├── filter/             # Filter implementations
│   ├── traits.rs       # Filter, Associator, Merger traits
│   ├── lmb.rs          # LmbFilter
│   ├── lmbm.rs         # LmbmFilter
│   ├── multisensor_lmb.rs    # AA/GA/PU/IC-LMB filters
│   └── multisensor_lmbm.rs   # MultisensorLmbmFilter
├── common/             # Low-level utilities
│   ├── association/    # LBP, Gibbs, Murty algorithms
│   ├── linalg.rs       # Linear algebra helpers
│   ├── rng.rs          # RNG trait
│   └── constants.rs    # Numerical constants
└── lmb/cardinality.rs  # MAP cardinality estimation
```

## Debugging MATLAB→Rust Equivalence Issues

### THE GOLDEN RULE

**When tests fail showing numerical differences between MATLAB and Rust:**

**DO THIS FIRST (5 minutes):**
1. Find the MATLAB function that's failing
2. Find the corresponding Rust function
3. Open them side-by-side
4. Compare line-by-line
5. Look for obvious bugs:
   - Wrong formulas (det(c*A) vs c^n*det(A))
   - Scalar vs array parameters
   - Missing loops or wrong loop bounds
   - Transposed matrices or wrong indexing

**ONLY IF THAT FAILS (rare):**
1. Create minimal debug fixture in MATLAB
2. Add targeted debug output to Rust
3. Compare intermediate values
4. Trace divergence point

### Basic Debugging Checklist

When Rust output ≠ MATLAB output:

- [ ] **Did I port the algorithm correctly?** (Read both implementations side-by-side)
- [ ] **Am I using the right parameters?** (Single vs multi-sensor, scalar vs array)
- [ ] **Are my formulas correct?** (Check mathematical properties like determinant rules)
- [ ] **Am I handling indices correctly?** (0-indexed vs 1-indexed, row vs column major)
- [ ] **Did I test with a minimal example?** (Simplest possible input that fails)

## Development Reminders

- **Run tests with --release**: `cargo test --release`
- **Question assumptions**: If tests require loose tolerances (>1e-10), there's likely a real bug
- **Push back** if you notice weird hacks to achieve MATLAB equivalence
- **Keep PROGRESS.md updated** with current work status and next steps

## Documentation

- `docs/00_io/` - Algorithm inputs/outputs documentation
- `docs/01_migration/` - Historical: MATLAB→Rust migration (archived)
- `docs/02_migration_comparison/` - MATLAB vs Rust code comparison
- `docs/03_optimisations/` - Performance improvements and changelog
- `PROGRESS.md` - Current refactoring progress and next steps

## Key Files

**Filter Implementations:**
- `src/filter/lmb.rs` - Single-sensor LMB filter
- `src/filter/lmbm.rs` - Single-sensor LMBM filter
- `src/filter/multisensor_lmb.rs` - Multi-sensor LMB variants (AA, GA, PU, IC)
- `src/filter/multisensor_lmbm.rs` - Multi-sensor LMBM

**Core Algorithms (kept from legacy):**
- `src/common/association/lbp.rs` - Loopy Belief Propagation
- `src/common/association/gibbs.rs` - Gibbs sampling
- `src/common/association/murtys.rs` - Murty's K-best
- `src/common/linalg.rs` - Linear algebra utilities

**MATLAB Equivalence Tests:**
- `tests/new_api_matlab_equivalence.rs` - Verifies new API matches MATLAB
