# Possible Optimizations

Based on ideas from [CRDTs Go Brrr](https://josephg.com/blog/crdts-go-brrr/) and analysis of this codebase.

## Current Performance Baseline

From `BENCHMARK_BASELINE.md`:
- LBP 50x50: **439 µs** (main bottleneck)
- Association matrix 50x50: **90 µs**
- ESF n=100: **4.4 µs** (O(n²) scaling)
- Full filter t=50: **2.35 ms**
- Multi-sensor LMBM 100 timesteps: **~13.6s**

### Profiling Notes

Profile captured with samply (14,981 samples, 14.98s runtime). Saved to `profile.json`.
To view: `samply load profile.json`

For symbolized profiles, rebuild with debug info:
```bash
cargo build --release
# Or add to Cargo.toml:
# [profile.release]
# debug = true
```

## Codebase Statistics

| Pattern | Count | Impact |
|---------|-------|--------|
| `.clone()` calls | 207 | High - memory allocation + copy |
| Matrix ops (cholesky/inverse/det) | 41 | High - O(n³) operations |
| Nested `Vec<Vec<...>>` | 30+ | Medium - cache unfriendly |
| Explicit `for` loops | 200+ | Medium - potential SIMD opportunities |

---

## Category 1: Memory Layout Optimizations

### 1.1 Flatten Nested Vectors (Est. 2-5x speedup for affected code)

**Current (cache-unfriendly):**
```rust
pub struct Object {
    pub w: Vec<f64>,           // Pointer → heap allocation
    pub mu: Vec<DVector<f64>>, // Pointer → Vec of pointers → heap allocations
    pub sigma: Vec<DMatrix<f64>>, // Even worse
}
```

**Proposed (cache-friendly):**
```rust
pub struct Object {
    pub birth_location: usize,
    pub birth_time: usize,
    pub r: f64,
    pub number_of_gm_components: usize,
    // Inline small arrays for common case (1-4 GM components)
    pub w: SmallVec<[f64; 4]>,
    pub mu: SmallVec<[DVector<f64>; 4]>,
    pub sigma: SmallVec<[DMatrix<f64>; 4]>,
}
```

**Why it helps:** Most objects have 1-3 GM components. SmallVec stores up to N elements inline, avoiding heap allocation for the common case.

### 1.2 Struct-of-Arrays for Batch Operations (Est. 3-10x for vectorizable ops)

**Current (Array-of-Structs):**
```rust
let objects: Vec<Object> = ...;
for obj in &objects {
    // Each access follows pointer, cache miss likely
    process(obj.r, &obj.mu, &obj.sigma);
}
```

**Proposed (Struct-of-Arrays):**
```rust
pub struct ObjectBatch {
    pub birth_locations: Vec<usize>,
    pub birth_times: Vec<usize>,
    pub rs: Vec<f64>,  // Contiguous in memory!
    pub mus: Vec<DVector<f64>>,
    pub sigmas: Vec<DMatrix<f64>>,
    // Index mapping for GM components
    pub gm_offsets: Vec<usize>,
}
```

**Why it helps:** When iterating over existence probabilities `r` for all objects, data is contiguous → CPU prefetcher works, SIMD possible.

### 1.3 Pre-allocated Workspace Buffers (Est. 1.5-3x)

**Current:**
```rust
fn lbp_iteration(...) {
    let mut mu_obj_to_meas = DMatrix::zeros(n_obj, n_meas);  // Alloc every call
    let mut mu_meas_to_obj = DMatrix::zeros(n_obj, n_meas);  // Alloc every call
    // ... 5-20 iterations, allocating each time
}
```

**Proposed:**
```rust
struct LbpWorkspace {
    mu_obj_to_meas: DMatrix<f64>,
    mu_meas_to_obj: DMatrix<f64>,
    // Pre-sized for max expected dimensions
}

fn lbp_iteration(workspace: &mut LbpWorkspace, ...) {
    workspace.mu_obj_to_meas.fill(0.0);  // Reuse, don't reallocate
}
```

---

## Category 2: Clone Reduction

### 2.1 Audit 207 Clone Calls (Est. 1.2-2x overall)

Current codebase has 207 `.clone()` calls. Many are unnecessary:

**Pattern 1: Clone in loop (wasteful)**
```rust
for obj in objects.clone().iter() {  // Clones entire Vec
    // ...
}
```
**Fix:** Use `&objects` or `objects.iter()`

**Pattern 2: Clone for ownership transfer**
```rust
let result = some_fn(data.clone());  // Clone to pass ownership
```
**Fix:** Take `&data` if possible, or use `Cow<T>`

**Pattern 3: Clone in prediction step**
```rust
pub fn lmb_prediction_step(mut objects: Vec<Object>, ...) -> Vec<Object>
```
**Fix:** Consider in-place mutation or arena allocation

### 2.2 Use Copy-on-Write (Cow) for Large Structures

```rust
use std::borrow::Cow;

fn update_object(obj: Cow<'_, Object>) -> Object {
    match obj {
        Cow::Borrowed(o) if needs_modification(o) => {
            let mut owned = o.clone();
            modify(&mut owned);
            owned
        }
        Cow::Owned(mut o) => {
            modify(&mut o);
            o
        }
        Cow::Borrowed(o) => o.clone(), // Only clone if needed
    }
}
```

---

## Category 3: Algorithm-Level Optimizations

### 3.1 LBP Early Termination with Tolerance Tracking (Est. 1.3-2x)

**Current:** Fixed iteration count or simple convergence check
**Proposed:** Track per-element convergence, skip converged elements

```rust
fn lbp_with_early_exit(matrices: &AssociationMatrices) -> (DVector<f64>, DMatrix<f64>) {
    let mut converged = vec![false; n_obj];
    let mut prev_values = ...;

    for iter in 0..max_iter {
        let mut all_converged = true;
        for i in 0..n_obj {
            if converged[i] { continue; }  // Skip converged rows

            let new_val = compute_update(i, ...);
            if (new_val - prev_values[i]).abs() < tolerance {
                converged[i] = true;
            } else {
                all_converged = false;
            }
        }
        if all_converged { break; }
    }
}
```

### 3.2 Cholesky/Inverse Caching (Est. 2-5x for matrix ops)

41 calls to `.cholesky()`, `.try_inverse()`, `.determinant()`. Many are redundant:

```rust
// Cache expensive decompositions
struct CachedMatrix {
    matrix: DMatrix<f64>,
    cholesky: OnceCell<Option<Cholesky<f64>>>,
    inverse: OnceCell<Option<DMatrix<f64>>>,
    determinant: OnceCell<f64>,
}

impl CachedMatrix {
    fn cholesky(&self) -> Option<&Cholesky<f64>> {
        self.cholesky.get_or_init(|| self.matrix.clone().cholesky()).as_ref()
    }

    fn inverse(&self) -> Option<&DMatrix<f64>> {
        self.inverse.get_or_init(|| self.matrix.clone().try_inverse()).as_ref()
    }
}
```

### 3.3 Elementary Symmetric Function (ESF) Optimization

Current: O(n²) with n=100 taking 4.4µs

**Options:**
1. **Parallel prefix computation** - O(n) depth with O(n) processors
2. **Newton's identities** - Alternative formulation that may be faster for specific cases
3. **Logarithmic space** - Compute in log domain to avoid underflow, use FFT for polynomial multiplication

---

## Category 4: Parallelization with Rayon

### 4.1 Association Matrix Generation (Est. 3-8x on multi-core)

**Current:**
```rust
for (i, obj) in objects.iter().enumerate() {
    for (j, z) in measurements.iter().enumerate() {
        psi[(i, j)] = compute_likelihood(obj, z);
    }
}
```

**Proposed:**
```rust
use rayon::prelude::*;

let psi_data: Vec<f64> = (0..n_obj)
    .into_par_iter()
    .flat_map(|i| {
        let obj = &objects[i];
        (0..n_meas).map(move |j| compute_likelihood(obj, &measurements[j]))
    })
    .collect();
```

### 4.2 LBP Row Updates (Est. 2-4x)

Row updates in LBP are independent within an iteration:

```rust
// Parallel row updates
mu_obj_to_meas.par_row_iter_mut()
    .enumerate()
    .for_each(|(i, mut row)| {
        for j in 0..n_meas {
            row[j] = compute_message(i, j, ...);
        }
    });
```

### 4.3 Multi-Sensor Parallel Processing

Each sensor's update is largely independent:

```rust
let sensor_updates: Vec<_> = (0..num_sensors)
    .into_par_iter()
    .map(|s| process_sensor(s, &objects, &measurements[s]))
    .collect();
```

---

## Category 5: SIMD Vectorization

### 5.1 Nalgebra SIMD Features

Enable in `Cargo.toml`:
```toml
[dependencies]
nalgebra = { version = "0.33", features = ["simd-stable"] }
```

### 5.2 Manual SIMD for Hot Loops

For Gaussian likelihood computation (computed thousands of times):

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// SIMD-optimized Mahalanobis distance for 4D state vectors
unsafe fn mahalanobis_distance_simd(
    diff: &[f64; 4],
    inv_cov: &[[f64; 4]; 4],
) -> f64 {
    // Use AVX2 for 4 doubles at once
    let diff_vec = _mm256_loadu_pd(diff.as_ptr());
    // ... vectorized computation
}
```

### 5.3 Batch Likelihood Computation

Compute likelihoods for multiple measurements at once:

```rust
fn batch_gaussian_likelihood(
    z_batch: &[DVector<f64>],  // Multiple measurements
    mu: &DVector<f64>,
    sigma_inv: &DMatrix<f64>,
    det_factor: f64,
) -> Vec<f64> {
    // SIMD-friendly batch computation
}
```

---

## Category 6: Memory Allocation Reduction

### 6.1 Arena Allocation for Per-Timestep Data

```rust
use bumpalo::Bump;

fn process_timestep(arena: &Bump, objects: &[Object], measurements: &[DVector<f64>]) {
    // All temporary allocations in arena
    let temp_matrix = arena.alloc(DMatrix::zeros(n, m));
    // Arena freed at end of timestep, no individual frees
}
```

### 6.2 Object Pool for Gaussian Components

```rust
struct GaussianPool {
    available_4d_vectors: Vec<DVector<f64>>,
    available_4x4_matrices: Vec<DMatrix<f64>>,
}

impl GaussianPool {
    fn get_vector(&mut self) -> DVector<f64> {
        self.available_4d_vectors.pop()
            .unwrap_or_else(|| DVector::zeros(4))
    }

    fn return_vector(&mut self, v: DVector<f64>) {
        self.available_4d_vectors.push(v);
    }
}
```

---

## Estimated Impact Summary

| Optimization | Effort | Speedup | Priority |
|-------------|--------|---------|----------|
| Rayon parallelization | Medium | 2-8x | **HIGH** |
| Clone reduction audit | Low | 1.2-2x | **HIGH** |
| Pre-allocated workspaces | Low | 1.5-3x | **HIGH** |
| SmallVec for GM components | Low | 1.5-3x | Medium |
| Cholesky/inverse caching | Medium | 2-5x | Medium |
| Struct-of-Arrays refactor | High | 3-10x | Medium |
| SIMD for likelihoods | High | 2-4x | Low |
| Arena allocation | Medium | 1.3-2x | Low |

## Quick Wins (< 1 day of work)

1. **Add `#[inline]` to hot functions** - Free performance
2. **Enable LTO in release profile** - Add to Cargo.toml
3. **Audit and remove unnecessary clones** - grep for `.clone()` patterns
4. **Pre-allocate LBP workspace** - Biggest single-function impact

```toml
# Cargo.toml additions
[profile.release]
lto = "thin"
codegen-units = 1
```

## Validation Strategy

1. Run benchmarks before any change: `cargo bench -- --save-baseline before`
2. Make optimization
3. Run benchmarks after: `cargo bench -- --baseline before`
4. Verify correctness with existing tests: `cargo test --release`
5. Document speedup in commit message

---

## References

- [CRDTs Go Brrr](https://josephg.com/blog/crdts-go-brrr/) - 5000x optimization journey
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Nalgebra SIMD](https://docs.rs/nalgebra/latest/nalgebra/#optional-features)
- [Rayon Data Parallelism](https://docs.rs/rayon/latest/rayon/)
