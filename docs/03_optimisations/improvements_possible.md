# LMBM Performance Optimization - Comprehensive Analysis

**Target Algorithm**: Multisensor LMBM (33.97s, 283x slower than PU/GA)
**Profiling Data**: 88% time in association matrices, 10.7M function calls, 32.8 GB allocations

---

## Executive Summary

The LMBM algorithm suffers from **fundamental combinatorial explosion**: likelihood matrix dimensions = `∏(mᵢ + 1) × n`. With 3 sensors, 10 measurements each, 100 objects = **133,100 entries**, each calling `determine_log_likelihood_ratio()` which performs expensive matrix operations with multiple clones.

---

## Current Performance Baseline

From profiling with `hotpath-alloc`:

| Algorithm | Time | Total Alloc | Bottleneck Function | Bottleneck % |
|-----------|------|-------------|---------------------|--------------|
| **LMBM** | **33.97s** | **954 MB** | `generate_multisensor_lmbm_association_matrices` | **88%** |
| IC | 0.12s | 230 MB | `compute_posterior_lmb_spatial_distributions_multisensor` | 16% |
| PU | 0.03s | 118 MB | `loopy_belief_propagation` | 5% |
| GA | 0.03s | 95 MB | `loopy_belief_propagation` | 10% |
| AA | 0.30s | 756 MB | `compute_posterior_lmb_spatial_distributions_multisensor` | 12% |

---

## Complete Inefficiency Map

### Critical Path Analysis (88% of runtime)

| Location | Issue | Calls | Impact |
|----------|-------|-------|--------|
| `association.rs:210-213` | Pre-allocate 5B+ entries | 991 | 32.8 GB |
| `association.rs:226` | Call likelihood function | 10.7M | 88% time |
| `association.rs:95` | `Q.clone()` per sensor | 10.7M×S | 540 MB+ |
| `linalg.rs:199-211` | `robust_inverse` 3 clones | 10.7M | 1.5 TB temp! |
| `linalg.rs:326` | Redundant Cholesky | 10.7M | Wasteful |
| `association.rs:33-46` | `Vec` alloc per conversion | 10.7M | 5B allocs |
| `gibbs.rs:126` | `vec![0; S+1]` in inner loop | 300/sample | 1GB total |
| `hypothesis.rs:86,90` | Clone mu/sigma per object | U×n | 16 GB |

### Secondary Inefficiencies

| Location | Issue | Frequency |
|----------|-------|-----------|
| `filter.rs:111` | Clone hypothesis for prediction | H×T |
| `hypothesis.rs:158,170` | Clone during gating | 2×H per step |
| `prediction.rs:49-50` | Vec::resize with clones | B×H×T |
| `gibbs.rs:149` | Serial RNG calls | 3M+ |

---

## ALL OPTIMIZATION STRATEGIES - RANKED BY IMPACT

### TIER 1: CRITICAL (10-100x potential improvement)

#### 1. Sparse/Lazy Likelihood Computation (Expected: 50-100x)
**Rank: #1 - HIGHEST IMPACT**

Instead of computing ALL `∏(mᵢ+1)×n` likelihoods upfront, compute on-demand during Gibbs sampling.

```rust
// Current: Pre-compute everything
let mut l = vec![0.0; number_of_entries];  // 133,100 entries
for ell in 0..number_of_entries {
    l[ell] = determine_log_likelihood_ratio(...);  // 10.7M calls
}

// Proposed: Compute on-demand with memoization
struct LazyLikelihood {
    cache: HashMap<usize, f64>,
}
impl LazyLikelihood {
    fn get(&mut self, ell: usize, ...) -> f64 {
        *self.cache.entry(ell).or_insert_with(|| {
            determine_log_likelihood_ratio(...)
        })
    }
}
```

**Why it works**: Gibbs sampling only visits ~1000-10000 unique indices, not all 133,100+.

---

#### 2. Workspace Buffer Reuse (Expected: 5-10x allocation reduction)
**Rank: #2**

```rust
pub struct LmbmAssociationWorkspace {
    z: DVector<f64>,
    c: DMatrix<f64>,
    q: DMatrix<f64>,
    q_sensor_cache: Vec<DMatrix<f64>>,
    c_sensor_cache: Vec<DMatrix<f64>>,
    nu: DVector<f64>,
    z_matrix: DMatrix<f64>,
    z_inv: DMatrix<f64>,
    kalman_gain: DMatrix<f64>,
}
```

---

#### 3. Eliminate robust_inverse Triple Clone (Expected: 3x in that function)
**Rank: #3**

```rust
// Current (3 potential clones)
pub fn robust_inverse(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if let Some(chol) = matrix.clone().cholesky() { ... }
    if let Some(inv) = matrix.clone().try_inverse() { ... }
    let svd = matrix.clone().svd(true, true);
}

// Proposed: In-place with workspace
pub fn robust_inverse_inplace(
    matrix: &DMatrix<f64>,
    workspace: &mut InverseWorkspace,
    result: &mut DMatrix<f64>,
) -> bool
```

---

#### 4. Cache Cholesky Factor for log_gaussian_normalizing_constant (Expected: 2x)
**Rank: #4**

```rust
// Current: Compute Cholesky, then recompute for log-det
let z_inv = robust_inverse(&z_matrix)?;
let eta = log_gaussian_normalizing_constant(&z_matrix, z_dim);  // Cholesky again!

// Proposed: Return Cholesky factor from inverse, use for log-det
let (z_inv, log_det) = robust_inverse_with_log_det(&z_matrix)?;
let eta = -0.5 * (z_dim as f64 * LOG_2PI + log_det);
```

---

### TIER 2: HIGH IMPACT (2-5x improvement)

#### 5. Q Matrix Cache (Expected: 2-3x clone reduction)
**Rank: #5**

```rust
// Current: Clone Q matrix 10.7M×S times
q_blocks.push(model.get_measurement_noise(Some(s)).clone());

// Proposed: Clone once per timestep
let q_cache: Vec<DMatrix<f64>> = (0..number_of_sensors)
    .map(|s| model.get_measurement_noise(Some(s)).clone())
    .collect();
```

---

#### 6. Rayon Parallelization (Expected: 3-8x on multi-core)
**Rank: #6**

```rust
use rayon::prelude::*;

let results: Vec<_> = (0..number_of_entries)
    .into_par_iter()
    .map_with(workspace.clone(), |ws, ell| {
        determine_log_likelihood_ratio(ell, ws)
    })
    .collect();
```

---

#### 7. In-Place Matrix Operations (Expected: 1.5-2x)
**Rank: #7**

```rust
// Current (creates temporaries)
let nu = &z - &c * &hypothesis.mu[i];

// Proposed (in-place)
workspace.nu.copy_from(&workspace.z);
workspace.nu.gemv(-1.0, &workspace.c, &hypothesis.mu[i], 1.0);
```

---

#### 8. SmallVec for Cartesian Indices (Expected: 1.5-2x for index ops)
**Rank: #8**

```rust
// Current: Heap allocation every call
fn convert_from_linear_to_cartesian(...) -> Vec<usize> {
    let mut u = vec![0; m];

// Proposed: Stack allocation
fn convert_from_linear_to_cartesian<const N: usize>(...) -> [usize; N]
```

---

### TIER 3: MEDIUM IMPACT (1.2-2x improvement)

#### 9. Vectorized Posterior Parameter Extraction (Expected: 1.5x)
**Rank: #9**

```rust
// Current: 3 separate iterations
let r: Vec<f64> = ell_indices.iter().map(...).collect();
let mu: Vec<DVector<f64>> = ell_indices.iter().map(...).collect();
let sigma: Vec<DMatrix<f64>> = ell_indices.iter().map(...).collect();

// Proposed: Single iteration
for &idx in &ell_indices {
    r.push(posterior_parameters.r[idx]);
    mu.push(posterior_parameters.mu[idx].clone());
    sigma.push(posterior_parameters.sigma[idx].clone());
}
```

---

#### 10. Batched RNG (Expected: 1.3-1.5x in Gibbs)
**Rank: #10**

---

#### 11. Pre-computed Page Size Multipliers (Expected: 1.2x)
**Rank: #11**

```rust
let page_sizes: Vec<usize> = precompute_page_sizes(&dimensions);
fn determine_linear_index_fast(u: &[usize], page_sizes: &[usize]) -> usize
```

---

#### 12. LTO and Codegen Optimizations (Expected: 1.1-1.3x)
**Rank: #12**

```toml
[profile.release]
lto = "thin"
codegen-units = 1
opt-level = 3
```

---

### TIER 4: LOW IMPACT BUT EASY (1.05-1.2x)

#### 13. #[inline] on Hot Functions
**Rank: #13**

```rust
#[inline(always)]
fn determine_linear_index(...) -> usize
```

---

#### 14. Cow<T> for Conditional Cloning
**Rank: #14**

---

## ADDITIONAL 10 APPROACHES

### A1. Custom Allocator (jemalloc/mimalloc) - Expected: 1.2-1.5x
**Rank: #15**

```toml
[dependencies]
mimalloc = { version = "0.1", default-features = false }
```

---

### A2. Arena Allocation (bumpalo) - Expected: 2-3x for temporaries
**Rank: #16**

```rust
use bumpalo::Bump;

fn process_timestep(arena: &Bump, ...) {
    let temp = arena.alloc(DMatrix::zeros(n, m));
}
```

---

### A3. Struct-of-Arrays (SoA) for Hypothesis - Expected: 1.5-2x cache efficiency
**Rank: #17**

```rust
struct HypothesisBatch {
    rs: Vec<f64>,              // Contiguous
    mu_data: Vec<f64>,         // Flattened
    sigma_data: Vec<f64>,      // Flattened
    mu_offsets: Vec<usize>,
    sigma_offsets: Vec<usize>,
}
```

---

### A4. Flat Arrays vs Nested Vecs - Expected: 1.3-1.5x
**Rank: #18**

---

### A5. SIMD for Likelihood Computation - Expected: 2-4x for math
**Rank: #19**

Enable nalgebra SIMD: `nalgebra = { features = ["simd-stable"] }`

---

### A6. Memoization of Association Patterns - Expected: 1.5-2x
**Rank: #20**

---

### A7. Compile-Time Dimension Specialization - Expected: 1.2-1.5x
**Rank: #21**

---

### A8. Object Pooling for DVector/DMatrix - Expected: 1.5-2x
**Rank: #22**

---

### A9. Lazy Hypothesis Cloning (Copy-on-Write) - Expected: 1.3x
**Rank: #23**

---

### A10. Measurement Gating Before Full Computation - Expected: 2-5x
**Rank: #24**

```rust
let quick_distance = mahalanobis_quick(&z, &predicted_z, &sigma_diag);
if quick_distance > gate_threshold {
    return VERY_LOW_LIKELIHOOD;
}
```

---

## 5 OUTSIDE-THE-BOX STRATEGIES

### OOB1. GPU Acceleration (CUDA/OpenCL) - Expected: 10-100x
**Rank: #25**

---

### OOB2. Probabilistic Data Structure (Bloom Filter for Gating) - Expected: 2-3x
**Rank: #26**

---

### OOB3. Algorithmic Approximation (Variational Inference) - Expected: 10-50x
**Rank: #27**

---

### OOB4. JIT Compilation for Hot Loops (Cranelift) - Expected: 1.5-2x
**Rank: #28**

---

### OOB5. Streaming/Incremental Algorithm - Expected: 5-20x
**Rank: #29**

---

## COMPLETE RANKING TABLE

| Rank | Strategy | Expected Speedup | Effort | Risk |
|------|----------|------------------|--------|------|
| **1** | Sparse/Lazy Likelihood | 50-100x | High | Medium |
| **2** | Workspace Buffer Reuse | 5-10x alloc | Medium | Low |
| **3** | robust_inverse In-Place | 3x in function | Low | Low |
| **4** | Cache Cholesky Factor | 2x | Low | Low |
| **5** | Q Matrix Cache | 2-3x clones | Low | Low |
| **6** | Rayon Parallelization | 3-8x | Medium | Medium |
| **7** | In-Place Matrix Ops | 1.5-2x | Medium | Medium |
| **8** | SmallVec for Indices | 1.5-2x | Low | Low |
| **9** | Vectorized Extraction | 1.5x | Low | Low |
| **10** | Batched RNG | 1.3-1.5x | Low | Low |
| 11 | Pre-computed Page Sizes | 1.2x | Low | Low |
| 12 | LTO/Codegen | 1.1-1.3x | Trivial | None |
| 13 | #[inline] | 1.05-1.1x | Trivial | None |
| 14 | Cow<T> | 1.1x | Low | Low |
| 15 | Custom Allocator | 1.2-1.5x | Trivial | Low |
| 16 | Arena Allocation | 2-3x temps | Medium | Low |
| 17 | SoA Layout | 1.5-2x cache | High | Medium |
| 18 | Flat Arrays | 1.3-1.5x | Medium | Medium |
| 19 | SIMD | 2-4x math | High | Medium |
| 20 | Memoization | 1.5-2x | Medium | Low |
| 21 | Const Generics | 1.2-1.5x | Medium | Low |
| 22 | Object Pooling | 1.5-2x | Medium | Low |
| 23 | Lazy Hypothesis | 1.3x | Medium | Medium |
| 24 | Measurement Gating | 2-5x | Medium | Medium |
| **25** | **GPU (OOB)** | **10-100x** | Very High | High |
| 26 | Bloom Filter (OOB) | 2-3x | Medium | Medium |
| **27** | **Variational (OOB)** | **10-50x** | Very High | High |
| 28 | JIT Compilation (OOB) | 1.5-2x | Very High | High |
| 29 | Streaming (OOB) | 5-20x | Very High | High |

---

## Critical Files

| File | Lines | Priority |
|------|-------|----------|
| `src/multisensor_lmbm/association.rs` | 60-238 | **CRITICAL** |
| `src/common/linalg.rs` | 199-334 | HIGH |
| `src/multisensor_lmbm/gibbs.rs` | 104-165 | HIGH |
| `src/multisensor_lmbm/hypothesis.rs` | 34-105 | MEDIUM |
| `src/multisensor_lmbm/filter.rs` | 77-223 | MEDIUM |

---

## Validation Strategy

1. **Baseline**: `cargo bench` + `hotpath-alloc` profiling
2. **After each change**:
   - `cargo test --release` (all tests pass)
   - Re-profile to measure improvement
   - Verify MATLAB numerical equivalence
3. **Memory**: Use `hotpath-alloc` feature for allocation tracking

---

## References

- [CRDTs Go Brrr](https://josephg.com/blog/crdts-go-brrr/) - 5000x optimization journey
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Nalgebra SIMD](https://docs.rs/nalgebra/latest/nalgebra/#optional-features)
- [Rayon Data Parallelism](https://docs.rs/rayon/latest/rayon/)
