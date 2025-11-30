# MATLAB→Rust Port Quality Analysis — Summary

> Full analysis: [quality_analysis.md](./quality_analysis.md)

## Overall Assessment: 5.9/10

The port successfully preserves **algorithmic correctness** (9/10) and **MATLAB similarity** (9/10), but lacks **infrastructure** (logging, benchmarks, error handling) and **Rust idioms** (parallelization, traits, iterators).

**Bottom line:** The code works correctly and matches MATLAB, but is missing the tooling and abstractions needed for a production-quality Rust library.

### Score by Category Group
| Group | Score | What's Strong | What's Weak |
|-------|-------|---------------|-------------|
| Correctness | 8.5/10 | Algorithms match MATLAB exactly | Minor floating-point workarounds needed |
| Code Quality | 5.5/10 | Good module separation | No error types, code duplication |
| Performance | 4.5/10 | Cholesky over direct inverse | No parallelization, no benchmarks |
| Infrastructure | 5.0/10 | Minimal dependencies | No logging, hardcoded constants |
| Usability | 6.5/10 | Good documentation | Limited test coverage |

---

## What's Good

### 1. Excellent MATLAB Fidelity (9/10)

The algorithms match MATLAB exactly. This wasn't accidental—careful attention was paid to:

- **Variable naming**: `sigma_mt` ↔ `SigmaMT`, `phi` ↔ `phi`, `eta` ↔ `eta`
- **Loop structure**: Same nesting, same iteration order
- **Index handling**: Consistent 1-based → 0-based translation
- **Column-major ordering**: Explicitly handled in `update.rs` where MATLAB's `reshape()` behavior matters

**Example** — prediction.rs matches MATLAB almost character-for-character:

```matlab
% MATLAB (lmbPredictionStep.m:22-27)
for i = 1:numberOfObjects
    objects(i).r = model.survivalProbability * objects(i).r;
    for j = 1:objects(i).numberOfGmComponents
        objects(i).mu{j} = model.A * objects(i).mu{j} + model.u;
        objects(i).Sigma{j} = model.A * objects(i).Sigma{j} * model.A' + model.R;
    end
end
```

```rust
// Rust (prediction.rs:30-42)
for obj in &mut objects {
    obj.r = model.survival_probability * obj.r;
    for j in 0..obj.number_of_gm_components {
        obj.mu[j] = &model.a * &obj.mu[j] + &model.u;
        obj.sigma[j] = &model.a * &obj.sigma[j] * model.a.transpose() + &model.r;
    }
}
```

The only differences are Rust's `&` references and `.transpose()` vs `'`.

---

### 2. Numerical Stability (8/10)

Smart choices were made throughout:

**Cholesky decomposition** instead of direct `inv()`:
```rust
// association.rs:111-117 — Uses Cholesky, not inv()
let z_inv = match z_cov.clone().cholesky() {
    Some(chol) => chol.inverse(),
    None => { continue; }  // Handles singular matrices
};
```

**Log-space arithmetic** for weight normalization (prevents underflow):
```rust
// association.rs:145-153 — Log-sum-exp trick
let max_w = (0..num_comp).map(|j| w_log[(row, j)]).fold(f64::NEG_INFINITY, f64::max);
let sum_exp: f64 = (0..num_comp).map(|j| (w_log[(row, j)] - max_w).exp()).sum();
```

**Epsilon guards** before divisions:
```rust
// lbp.rs:69-73 — Prevents division by zero
sigma_tm[(i, j)] = if denom.abs() > 1e-15 {
    matrices.psi[(i, j)] / denom
} else {
    0.0
};
```

**Clamping** for floating-point edge cases:
```rust
// cardinality.rs:123-134 — Handles r ≈ 1.0 numerical artifacts
if ri > 1.0 - 1e-15 {
    1.0  // Clamp 0.99999999999999989 to exactly 1.0
}
```

---

### 3. Good Documentation (8/10)

Every public function has doc comments with:
- `# Arguments` — Parameter descriptions
- `# Returns` — Return value explanation
- `# Implementation Notes` — MATLAB correspondence

**Example:**
```rust
/// Compute posterior LMB spatial distributions
///
/// Completes the LMB filter's measurement update by computing each object's
/// posterior spatial distribution.
///
/// # Implementation Notes
/// Matches MATLAB computePosteriorLmbSpatialDistributions.m exactly:
/// 1. Update existence probability: r'
/// 2. Reweight measurement-updated GMs using marginal association probabilities
/// 3. Apply crude mixture reduction
```

---

## What Needs Work

### 1. No Logging or Observability (3/10)

**Problem:** When something goes wrong, there's no way to know. The `log` crate is in dependencies but never used. Silent failures make debugging nearly impossible.

**Specific issues:**

| Location | What Happens | Should Do |
|----------|--------------|-----------|
| `association.rs:113-116` | Cholesky fails → silent `continue` | `warn!("Cholesky failed for object {i}")` |
| `lbp.rs:95-99` | Convergence not reported | `debug!("LBP converged in {counter} iterations")` |
| `filter.rs:61` | No progress indication | `trace!("Processing timestep {t}")` |
| `lbp.rs:69-73` | Division guard triggers → returns 0 | `trace!("Division guard triggered")` |

**Example of the problem** — `association.rs:113-116`:
```rust
let z_inv = match z_cov.clone().cholesky() {
    Some(chol) => chol.inverse(),
    None => {
        continue;  // SILENT! Component skipped, no one knows
    }
};
```

**Fix:** Add `tracing` crate and instrument functions:
```rust
use tracing::{debug, warn, instrument};

#[instrument(skip(objects, measurements, model))]
pub fn generate_lmb_association_matrices(...) {
    // ...
    None => {
        warn!(object_idx = i, component_idx = j, "Cholesky decomposition failed");
        continue;
    }
}
```

---

### 2. No Benchmarks (3/10)

**Problem:** `criterion` is in `Cargo.toml` dev-dependencies but there are zero benchmarks. This means:
- No way to know current performance
- No way to measure if optimizations help
- No comparison with MATLAB runtime
- Can't identify actual bottlenecks

**What should be benchmarked:**

| Function | Why | Complexity |
|----------|-----|------------|
| `generate_lmb_association_matrices` | Most expensive, O(n×m×k) | High |
| `loopy_belief_propagation` | Iterative, O(I×n×m) | Medium |
| `elementary_symmetric_function` | O(n²), dominates for large n | Medium |
| `lmb_map_cardinality_estimate` | O(n log n) sorting | Low |

**Fix:** Create `benches/association.rs`:
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_association(c: &mut Criterion) {
    let mut group = c.benchmark_group("association");

    for n_objects in [10, 50, 100, 500] {
        for n_measurements in [10, 50, 100] {
            let (objects, measurements, model) = setup(n_objects, n_measurements);

            group.bench_with_input(
                BenchmarkId::new("generate", format!("{}x{}", n_objects, n_measurements)),
                &(&objects, &measurements, &model),
                |b, (o, m, mdl)| b.iter(|| generate_lmb_association_matrices(o, m, mdl)),
            );
        }
    }
}

criterion_group!(benches, bench_association);
criterion_main!(benches);
```

---

### 3. No Error Types (4/10)

**Problem:** Functions return defaults or panic instead of `Result<T, E>`. This makes it impossible for callers to handle errors gracefully and hides problems.

**Current patterns:**

| Pattern | Location | Problem |
|---------|----------|---------|
| Silent default | `lbp.rs:71-72` | Returns `0.0` on division guard |
| Silent continue | `association.rs:115` | Skips component on Cholesky fail |
| Unwrap | `cardinality.rs:160` | Panics if NaN in input |
| No validation | All files | Assumes valid inputs |

**Example** — `cardinality.rs:158-160`:
```rust
let max_cardinality_index = rho
    .iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())  // PANICS on NaN!
    .map(|(i, _)| i)
    .unwrap_or(0);
```

**Fix:** Add `thiserror` and define error types:
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LmbError {
    #[error("Singular covariance matrix at object {object_idx}, component {component_idx}")]
    SingularMatrix { object_idx: usize, component_idx: usize },

    #[error("LBP failed to converge after {iterations} iterations (delta={delta:.2e})")]
    ConvergenceFailed { iterations: usize, delta: f64 },

    #[error("Invalid probability {value} (must be in [0,1])")]
    InvalidProbability { value: f64 },

    #[error("NaN detected in input")]
    NaNDetected,
}
```

Then change function signatures:
```rust
pub fn loopy_belief_propagation(...) -> Result<LbpResult, LmbError> {
    // ...
    if counter >= max_iterations && max_delta > epsilon {
        return Err(LmbError::ConvergenceFailed {
            iterations: counter,
            delta: max_delta
        });
    }
}
```

---

### 4. No Parallelization (5/10)

**Problem:** Object loops are embarrassingly parallel but run sequentially. Each object's computation is independent—perfect for `rayon`.

**Parallelization opportunities:**

| Location | Loop | Independence | Expected Speedup |
|----------|------|--------------|------------------|
| `association.rs:74` | Objects | Full | ~Nx on N cores |
| `lbp.rs:65-75` | Rows | Full | ~2-4x |
| `update.rs:41` | Objects | Full | ~Nx on N cores |
| `filter.rs:108-114` | Objects | Full | ~Nx on N cores |

**Current code** — `association.rs:74-168`:
```rust
for i in 0..number_of_objects {
    // ~100 lines of computation per object
    // Each object is COMPLETELY INDEPENDENT
}
```

**Fix:** Add `rayon` and parallelize:
```rust
use rayon::prelude::*;

// Option 1: Parallel iterator with collect
let results: Vec<_> = (0..number_of_objects)
    .into_par_iter()
    .map(|i| compute_object_association(&objects[i], measurements, model))
    .collect();

// Option 2: Parallel for_each with mutex for shared state
let posterior_parameters = Mutex::new(Vec::with_capacity(number_of_objects));
(0..number_of_objects).into_par_iter().for_each(|i| {
    let params = compute_object_posterior(&objects[i], ...);
    posterior_parameters.lock().unwrap().push(params);
});
```

**Note:** The RNG (`&mut impl Rng`) passed to `run_lmb_filter` prevents easy parallelization of Gibbs sampling. Consider per-thread RNG for that case.

---

### 5. Code Duplication in LBP (80% duplicate)

**Problem:** `loopy_belief_propagation` (lines 42-138) and `fixed_loopy_belief_propagation` (lines 151-230) share ~100 lines of identical message-passing code. The only difference is the convergence check.

**Duplicated code:**
- Message passing: `sigma_tm` and `sigma_mt` computation (~30 lines)
- Final results: `gamma`, `q`, `W`, `r` computation (~35 lines)
- Both functions: ~95 lines each, ~80 lines identical

**Current structure:**
```rust
pub fn loopy_belief_propagation(...) -> LbpResult {
    // Initialize (5 lines)
    while not_converged {           // <-- Only difference: convergence check
        // Message passing (30 lines) — DUPLICATED
    }
    // Final results (35 lines) — DUPLICATED
}

pub fn fixed_loopy_belief_propagation(...) -> LbpResult {
    // Initialize (5 lines)
    for _ in 0..max_iterations {    // <-- Only difference: fixed iterations
        // Message passing (30 lines) — DUPLICATED
    }
    // Final results (35 lines) — DUPLICATED
}
```

**Fix:** Extract shared logic:
```rust
fn lbp_message_passing<F>(
    matrices: &AssociationMatrices,
    should_continue: F,
) -> (DMatrix<f64>, DMatrix<f64>)
where
    F: Fn(&DMatrix<f64>, &DMatrix<f64>, usize) -> bool,
{
    let mut sigma_mt = DMatrix::from_element(n, m, 1.0);
    let mut counter = 0;

    loop {
        let sigma_mt_old = sigma_mt.clone();
        // ... message passing logic ...
        counter += 1;
        if !should_continue(&sigma_mt, &sigma_mt_old, counter) {
            break;
        }
    }

    let b = matrices.psi.component_mul(&sigma_mt);
    (sigma_mt, b)
}

fn compute_final_results(matrices: &AssociationMatrices, b: &DMatrix<f64>) -> LbpResult {
    // ... gamma, q, W, r computation ...
}

// Now the public functions are thin wrappers:
pub fn loopy_belief_propagation(matrices: &AssociationMatrices, epsilon: f64, max_iter: usize) -> LbpResult {
    let (_, b) = lbp_message_passing(matrices, |new, old, count| {
        let delta = (new - old).abs().max();
        delta > epsilon && count < max_iter
    });
    compute_final_results(matrices, &b)
}

pub fn fixed_loopy_belief_propagation(matrices: &AssociationMatrices, max_iter: usize) -> LbpResult {
    let (_, b) = lbp_message_passing(matrices, |_, _, count| count < max_iter);
    compute_final_results(matrices, &b)
}
```

**Reduction:** ~190 lines → ~110 lines (40% reduction in lbp.rs)

---

## MATLAB↔Rust 1:1 Correspondence

### Overall Score: 7/10

The port preserves algorithm structure but MATLAB's concise syntax expands significantly in Rust due to:
1. **No broadcasting** — MATLAB's `A ./ b` requires explicit loops in Rust
2. **No concatenation syntax** — MATLAB's `[A B]` requires manual construction
3. **Explicit types** — Rust needs type annotations MATLAB infers
4. **Reference semantics** — Rust needs `&` where MATLAB copies implicitly

### The Broadcasting Gap

The single biggest source of code bloat. What's 1 line in MATLAB becomes 6-12 lines in Rust:

**MATLAB** — 3 lines total:
```matlab
Psi = L ./ eta;                           % Broadcast divide
SigmaTM = Psi ./ (-B + sum(B, 2) + 1);   % Broadcast with row sum
L_gibbs = [eta L];                        % Horizontal concatenation
```

**Rust** — 25+ lines:
```rust
// Psi = L ./ eta — 6 lines
let mut psi = DMatrix::zeros(number_of_objects, number_of_measurements);
for i in 0..number_of_objects {
    for j in 0..number_of_measurements {
        psi[(i, j)] = l_matrix[(i, j)] / eta[i];
    }
}

// SigmaTM = Psi ./ (-B + sum(B, 2) + 1) — 12 lines with epsilon check
let mut sigma_tm = DMatrix::zeros(n_objects, n_measurements);
for i in 0..n_objects {
    let row_sum: f64 = b.row(i).sum();
    for j in 0..n_measurements {
        let denom = -b[(i, j)] + row_sum + 1.0;
        sigma_tm[(i, j)] = if denom.abs() > 1e-15 {
            matrices.psi[(i, j)] / denom
        } else {
            0.0
        };
    }
}

// [eta L] — 5 lines
let mut l_gibbs = DMatrix::zeros(number_of_objects, number_of_measurements + 1);
l_gibbs.column_mut(0).copy_from(&eta);
l_gibbs.view_mut((0, 1), (number_of_objects, number_of_measurements))
    .copy_from(&l_matrix);
```

### Per-File Similarity Scores

| File | Score | Analysis |
|------|-------|----------|
| **prediction.rs** | 9/10 | Nearly identical. Only `&` refs and `.transpose()` differ. Best example of 1:1 porting. |
| **cardinality.rs** | 8/10 | ESF algorithm is character-for-character. `std::mem::swap` matches MATLAB's tmp variable swap. |
| **filter.rs** | 7/10 | Structure matches. Main differences: enum dispatch vs `strcmp`, helper function extraction. |
| **lbp.rs** | 7/10 | Algorithm identical but 3 MATLAB lines → 30 Rust lines due to broadcasting. |
| **update.rs** | 7/10 | Column-major ordering explicitly handled. Uses helper `prune_gaussian_mixture` not in MATLAB. |
| **association.rs** | 6/10 | Most expanded. Broadcasting + concatenation + log-space arithmetic all add verbosity. |

---

## Recommended Abstractions

These traits would make Rust look more like MATLAB and reduce code:

### 1. Broadcast Trait (~60 LOC reduction)

**Problem:** Every `L ./ eta` pattern needs 6 lines.

**Solution:**
```rust
pub trait Broadcast<Rhs> {
    type Output;
    fn broadcast_div(&self, rhs: &Rhs) -> Self::Output;
    fn broadcast_mul(&self, rhs: &Rhs) -> Self::Output;
    fn broadcast_add(&self, rhs: &Rhs) -> Self::Output;
}

impl Broadcast<DVector<f64>> for DMatrix<f64> {
    type Output = DMatrix<f64>;

    /// Matrix ./ column_vector (broadcast across columns, like MATLAB)
    fn broadcast_div(&self, col_vec: &DVector<f64>) -> DMatrix<f64> {
        let mut result = self.clone();
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                result[(i, j)] /= col_vec[i];
            }
        }
        result
    }
    // ... similar for mul, add
}

// Usage — now matches MATLAB!
let psi = l_matrix.broadcast_div(&eta);  // L ./ eta
let p = l_matrix.broadcast_div(&(&l_matrix + &eta));  // L ./ (L + eta)
```

### 2. Matrix Concatenation (~20 LOC reduction)

**Problem:** MATLAB's `[eta L]` needs 5 lines in Rust.

**Solution:**
```rust
pub trait MatrixConcat {
    fn hcat(&self, other: &Self) -> Self;           // [A B]
    fn vcat(&self, other: &Self) -> Self;           // [A; B]
    fn prepend_col(&self, col: &DVector<f64>) -> Self;  // [v M]
    fn append_col(&self, col: &DVector<f64>) -> Self;   // [M v]
}

impl MatrixConcat for DMatrix<f64> {
    fn prepend_col(&self, col: &DVector<f64>) -> DMatrix<f64> {
        let mut result = DMatrix::zeros(self.nrows(), self.ncols() + 1);
        result.column_mut(0).copy_from(col);
        result.view_mut((0, 1), (self.nrows(), self.ncols())).copy_from(self);
        result
    }
}

// Usage — matches MATLAB!
let l_gibbs = l_matrix.prepend_col(&eta);  // [eta L]
```

### 3. Safe Division (~30 LOC reduction)

**Problem:** `if denom.abs() > 1e-15` repeated everywhere.

**Solution:**
```rust
pub const EPSILON: f64 = 1e-15;

pub trait SafeDiv {
    fn safe_div(self, denom: f64) -> f64;
    fn safe_div_or(self, denom: f64, default: f64) -> f64;
}

impl SafeDiv for f64 {
    fn safe_div(self, denom: f64) -> f64 {
        if denom.abs() > EPSILON { self / denom } else { 0.0 }
    }

    fn safe_div_or(self, denom: f64, default: f64) -> f64 {
        if denom.abs() > EPSILON { self / denom } else { default }
    }
}

// Usage — cleaner code
sigma_tm[(i, j)] = matrices.psi[(i, j)].safe_div(denom);
r[i] = q[i].safe_div(denom);
```

### 4. GaussianMixture Struct (~20 LOC reduction)

**Problem:** GM operations scattered with manual index management.

**Solution:**
```rust
pub struct GaussianMixture {
    pub weights: Vec<f64>,
    pub means: Vec<DVector<f64>>,
    pub covariances: Vec<DMatrix<f64>>,
}

impl GaussianMixture {
    pub fn num_components(&self) -> usize { self.weights.len() }

    pub fn predict(&mut self, a: &DMatrix<f64>, u: &DVector<f64>, r: &DMatrix<f64>) {
        for j in 0..self.num_components() {
            self.means[j] = a * &self.means[j] + u;
            self.covariances[j] = a * &self.covariances[j] * a.transpose() + r;
        }
    }

    pub fn prune(&mut self, threshold: f64, max_components: usize) -> PruneResult {
        // Encapsulates the crude mixture reduction algorithm
    }
}

// Usage in prediction.rs — cleaner!
for obj in &mut objects {
    obj.r *= model.survival_probability;
    obj.gm.predict(&model.a, &model.u, &model.r);
}
```

### 5. DataAssociation Trait (~15 LOC reduction)

**Problem:** Match arms in filter.rs for each algorithm.

**Solution:**
```rust
pub trait DataAssociation {
    fn compute(&self, matrices: &AssociationMatrices) -> DataAssociationResult;
}

pub struct DataAssociationResult {
    pub r: DVector<f64>,   // Existence probabilities
    pub w: DMatrix<f64>,   // Association weights
}

pub struct Lbp { pub epsilon: f64, pub max_iterations: usize }
pub struct Gibbs { pub num_samples: usize }
pub struct Murty { pub num_assignments: usize }

impl DataAssociation for Lbp {
    fn compute(&self, matrices: &AssociationMatrices) -> DataAssociationResult {
        let result = loopy_belief_propagation(matrices, self.epsilon, self.max_iterations);
        DataAssociationResult { r: result.r, w: result.w }
    }
}

// Usage in filter.rs — cleaner dispatch
let result = model.data_association.compute(&association_result);
```

### Summary of Abstractions

| Abstraction | Files Affected | LOC Reduction | MATLAB Similarity |
|-------------|----------------|---------------|-------------------|
| `Broadcast` trait | association.rs, lbp.rs | ~60 lines | High — matches `./ .*` |
| `MatrixConcat` | association.rs | ~20 lines | High — matches `[A B]` |
| `SafeDiv` trait | lbp.rs, update.rs, cardinality.rs | ~30 lines | Medium — cleaner |
| `GaussianMixture` | prediction.rs, update.rs | ~20 lines | Medium — encapsulation |
| `DataAssociation` | filter.rs | ~15 lines | High — polymorphism |
| LBP refactor | lbp.rs | ~40 lines | Same — less duplication |

**Total potential reduction:** ~185 lines (25% of analyzed code)

---

## Recommended Libraries

| Library | Purpose | Effort | Impact | Reasoning |
|---------|---------|--------|--------|-----------|
| **tracing** | Structured logging | Low | High | Replace `log`. Add `warn!()` for silent failures, `debug!()` for iterations, `#[instrument]` for spans. Essential for debugging. |
| **thiserror** | Error types | Low | High | Derive `Error` trait. Define `LmbError` enum. Makes failures explicit and handleable. |
| **rayon** | Parallelization | Medium | High | Change `for i in 0..n` to `.into_par_iter()`. Object loops are embarrassingly parallel. Easy 4-8x speedup. |
| **proptest** | Property testing | Medium | Medium | Test invariants like "weights sum to 1", "probabilities in [0,1]". Catches edge cases unit tests miss. |
| **faer** | Fast linear algebra | High | High | 2-10x faster Cholesky/multiply for large matrices. Evaluate with benchmarks first. Only worth it if decomposition is bottleneck. |
| **criterion** | Benchmarking | Medium | High | Already in deps, just needs benchmarks written. Essential before any optimization work. |

### Not Recommended

| Library | Why Not |
|---------|---------|
| **ndarray** | nalgebra's API is closer to MATLAB. ndarray better for tensors, not matrices. |
| **nalgebra-lapack** | Adds C dependency complexity. Only worth it for large matrices (>100x100) if benchmarks show need. |
| **std::simd** | Requires nightly. nalgebra likely auto-vectorizes already. Only if profiling shows specific loops not vectorizing. |

---

## Migration Path

### Phase 1: Infrastructure (Do First, ~2-3 days)
These are prerequisites for everything else:

1. **Add `thiserror`** → Define `LmbError` enum with variants for singular matrices, convergence failure, invalid inputs
2. **Add `tracing`** → Instrument `generate_lmb_association_matrices`, `loopy_belief_propagation`, `run_lmb_filter` with spans and warnings
3. **Create benchmarks** → Add `benches/` with criterion benchmarks for association, LBP, ESF
4. **Refactor LBP** → Extract shared message-passing into internal function

### Phase 2: Performance (~1-2 days)
Now you can measure impact:

5. **Add `rayon`** → Parallelize object loop in `generate_lmb_association_matrices`
6. **Benchmark** → Measure improvement, identify remaining bottlenecks
7. **Evaluate `faer`** → If Cholesky is bottleneck, benchmark faer vs nalgebra

### Phase 3: Code Quality (~2-3 days)
Polish and abstraction:

8. **Add `Broadcast` trait** → Clean up association.rs and lbp.rs
9. **Add `SafeDiv` trait** → Replace epsilon checks throughout
10. **Add `proptest`** → Property-based tests for numerical invariants
11. **Add MATLAB fixtures** → Test files with known MATLAB outputs for regression

---

## Key Code Locations

| Issue | File | Lines | Description |
|-------|------|-------|-------------|
| Silent Cholesky failure | `association.rs` | 113-116 | `continue` with no logging when decomposition fails |
| Clone in hot loop | `association.rs` | 141 | `sigma_updated.clone()` inside measurement loop — should be outside |
| LBP duplication | `lbp.rs` | 42-138, 151-230 | 80% identical code between two functions |
| Hardcoded epsilon | `lbp.rs` | 69, 86, 130 | `1e-15` should be named constant |
| Long function | `association.rs` | 57-168 | 110 lines — should split into helpers |
| Unwrap on partial_cmp | `cardinality.rs` | 160 | Panics if NaN in input |
| Magic number | `cardinality.rs` | 138 | `1e-6` adjustment undocumented |
| Column-major comment | `update.rs` | 51-52 | Critical ordering — good that it's documented |

---

## Score Details

### All 22 Categories

| # | Category | Score | Key Finding |
|---|----------|-------|-------------|
| 1 | Efficiency | 6/10 | Unnecessary clones, no parallelization |
| 2 | Port Quality | 9/10 | Algorithms match MATLAB exactly |
| 3 | Rust Conventions | 5/10 | Explicit loops over iterators (intentional for MATLAB similarity) |
| 4 | Similarity to Original | 9/10 | Variable names, structure preserved |
| 5 | Numerical Stability | 8/10 | Cholesky, log-space, epsilon guards |
| 6 | Type Safety | 6/10 | No newtypes, uses `usize` for everything |
| 7 | Documentation | 8/10 | Good doc comments, missing examples |
| 8 | Testing | 7/10 | Unit tests present, limited edge cases |
| 9 | Error Handling | 4/10 | Silent failures, no Result types |
| 10 | Maintainability | 6/10 | LBP duplication, long functions |
| 11 | API Ergonomics | 6/10 | 5-param functions, no builders |
| 12 | Concurrency | 5/10 | No parallelization despite opportunity |
| 13 | Dependencies | 7/10 | Minimal, well-maintained |
| 14 | Build Performance | 6/10 | nalgebra generics slow compilation |
| 15 | Benchmarking | 3/10 | criterion in deps but no benchmarks |
| 16 | Memory Profiling | 4/10 | Allocations in loops, no profiling |
| 17 | Algorithmic Complexity | 7/10 | Correct but undocumented |
| 18 | Configuration | 4/10 | Hardcoded magic numbers |
| 19 | Logging | 3/10 | log in deps but unused |
| 20 | Interoperability | 5/10 | No FFI, no Python bindings |
| 21 | Portability | 7/10 | Pure Rust, not WASM tested |
| 22 | Security | 5/10 | No NaN/Inf validation |

### By File

| File | Overall | Strongest | Weakest |
|------|---------|-----------|---------|
| prediction.rs | 7.7/10 | Correctness (10), Similarity (10) | Error handling (5) |
| cardinality.rs | 7.4/10 | Documentation (9) | Error handling (5) |
| update.rs | 7.1/10 | Correctness (9), Stability (8) | Error handling (5) |
| filter.rs | 6.9/10 | Correctness (9), Similarity (9) | Error handling (4) |
| association.rs | 6.5/10 | Correctness (9) | Efficiency (5), Maintainability (5) |
| lbp.rs | 6.5/10 | Correctness (9), Similarity (9) | Maintainability (4) |

**Pattern:** All files excellent on correctness (9-10), all weak on error handling (4-5). This is the main systemic issue to address.
