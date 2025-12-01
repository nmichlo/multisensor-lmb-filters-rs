# LMBM Optimization Plan - Part 2

**Current state**: 0.59s (after Phase G1 static types)
**Target**: <0.5s
**Required improvement**: ~18% (1.18x speedup)

---

## Current Bottleneck Breakdown

From hotpath profiling (0.69s with hotpath overhead):

| Component | Time | % | Key Operations |
|-----------|------|---|----------------|
| **Gibbs sampling** | 367ms | 52.7% | Index calc, L lookup, RNG, HashSet dedup |
| **Likelihood L computation** | 233ms | 33.5% | Matrix stack, Cholesky, quadratic form |
| **Posterior hypothesis** | 76ms | 10.9% | Index lookup, param extraction |
| Other | <1% | - | Prediction, cardinality, state extraction |

---

## Master Optimization Table

### Legend
- **Priority**: Lower = implement first
- **Expected Gain**: Improvement to that component
- **Overall Impact**: `Expected Gain × Component %`
- **Effort**: Low (<2h), Medium (2-4h), High (4-8h), Very High (>8h)
- **Risk**: Low (safe), Medium (needs testing), High (may affect correctness/quality)

---

### GIBBS STRUCTURE OPTIMIZATIONS

| Priority | ID | Optimization | Expected Gain | Overall Impact | Effort | Risk | Notes |
|----------|-----|-------------|---------------|----------------|--------|------|-------|
| **1** | G-S1 | **Fixed-size samples + sort dedup** | 10-15% | **5-8%** | Medium | Low | Replace `HashSet<Vec<usize>>` with `Vec<[usize; N]>` + sort + dedup |
| **2** | G-S2 | **Precompute stride table** | 8-12% | **4-6%** | Low | Low | `strides[i] = prod(dimensions[0..i])`, then `idx = base + j*stride[s]` |
| **3** | G-S3 | **Stack-allocate u vector** | 3-5% | **2-3%** | Low | Low | `let mut u = [0usize; MAX_SENSORS+1]` instead of `vec![]` in inner loop |
| 4 | G-S4 | L array prefetching | 2-5% | 1-3% | Medium | Low | `prefetch(l[next_idx])` before current computation |
| 5 | G-S5 | Batch sigmoid computation | 5-10% | 3-5% | High | Medium | SIMD `exp()` for multiple probabilities at once |
| 6 | G-S6 | wyrand/frand PRNG | 1-3% | <1% | Low | Low | Marginal gain - Xorshift64 already very fast |

**Implementation details for top 3:**

```rust
// G-S1: Fixed-size samples + sort dedup
type Sample = [usize; MAX_OBJECTS * MAX_SENSORS]; // Stack-allocated
let mut samples: Vec<Sample> = Vec::with_capacity(number_of_samples);
// ... collect samples ...
samples.sort_unstable();
samples.dedup();

// G-S2: Precompute stride table
let strides: [usize; MAX_SENSORS + 1] = precompute_strides(dimensions);
// In inner loop: idx = base_idx + (j + 1) * strides[s]  (instead of determine_linear_index)

// G-S3: Stack-allocate u vector
let mut u = [0usize; MAX_SENSORS + 1]; // Instead of vec![0; number_of_sensors + 1]
```

---

### GIBBS ALGORITHMIC OPTIMIZATIONS

| Priority | ID | Optimization | Expected Gain | Overall Impact | Effort | Risk | Notes |
|----------|-----|-------------|---------------|----------------|--------|------|-------|
| 7 | G-A1 | **Blocked Gibbs sampling** | 10-20% | 5-10% | High | Medium | Sample all sensors for one object together |
| 8 | G-A2 | **Early termination (convergence)** | 10-30% | 5-15% | Medium | Medium | Stop when unique sample rate drops below threshold |
| 9 | G-A3 | Burn-in + thinning | Variable | Variable | Low | **High** | Skip first N samples, keep every Kth - may affect tracking quality |
| 10 | G-A4 | Random scan vs systematic | 0-5% | 0-3% | Low | Low | Randomize object/sensor visitation order |

**Blocked Gibbs concept:**
```rust
// Current: For each sensor, for each object, sample
// Blocked: For each object, sample ALL sensors jointly
// Reduces index recalculation, better cache locality for per-object data
```

**Early termination concept:**
```rust
let mut unique_count_history = VecDeque::with_capacity(10);
for sample_idx in 0..max_samples {
    // ... generate sample ...
    let new_unique = unique_samples.len();
    unique_count_history.push_back(new_unique);
    if unique_count_history.len() >= 10 {
        let rate = (new_unique - unique_count_history.pop_front().unwrap()) as f64 / 10.0;
        if rate < 0.1 { break; } // Convergence detected
    }
}
```

---

### GIBBS ALTERNATIVES (Replace Gibbs entirely)

| Priority | ID | Optimization | Expected Gain | Overall Impact | Effort | Risk | Notes |
|----------|-----|-------------|---------------|----------------|--------|------|-------|
| 11 | G-ALT1 | **Murty's k-best assignments** | ??? | ??? | High | Medium | Deterministic, no RNG needed - already have Murty impl |
| 12 | G-ALT2 | Importance sampling | ??? | ??? | High | **High** | Sample from proposal distribution, weight by likelihood |
| 13 | G-ALT3 | LBP marginals directly | ??? | ??? | High | Medium | Use Loopy BP marginals instead of Gibbs samples |

**Comparison:**

| Method | Pros | Cons | Feasibility |
|--------|------|------|-------------|
| **Gibbs (current)** | Simple, proven, parallelizable | Sequential within sample | Baseline |
| **Murty's k-best** | Deterministic, finds optimal assignments | O(k·n³), memory intensive for large k | Have implementation, worth testing |
| **Importance Sampling** | Single-shot, embarrassingly parallel | Variance issues, weight degeneracy | Risky, needs careful tuning |
| **LBP Marginals** | Already computed for LBP mode | Approximate, may differ from Gibbs quality | Medium effort to integrate |

---

### LIKELIHOOD OPTIMIZATIONS

| Priority | ID | Optimization | Expected Gain | Overall Impact | Effort | Risk | Notes |
|----------|-----|-------------|---------------|----------------|--------|------|-------|
| **14** | L-1 | **Specialized 2x2/4x4/6x6 inverse** | 15-25% | **5-8%** | Low | Low | Closed-form determinant and inverse for small matrices |
| **15** | L-2 | **Pre-stacked sensor matrices** | 5-10% | **2-3%** | Low | Low | Cache stacked c,q matrices for common sensor combinations |
| 16 | L-3 | Thread-local workspaces | 3-5% | 1-2% | Low | Low | Reuse z,c,q allocations per thread |
| 17 | L-4 | SIMD small matrix multiply | 10-20% | 3-7% | High | Medium | AVX2/AVX-512 for 4x4 matrix operations |
| 18 | L-5 | f32 instead of f64 | 10-20% | 3-7% | Medium | Medium | Half memory bandwidth, 2x SIMD width |

**Specialized inverse implementation (L-1):**

```rust
/// 2x2 matrix inverse and log-determinant (1 sensor, z_dim=2)
fn inverse_2x2(m: &SMatrix<f64, 2, 2>) -> Option<(SMatrix<f64, 2, 2>, f64)> {
    let det = m[(0,0)] * m[(1,1)] - m[(0,1)] * m[(1,0)];
    if det.abs() < 1e-14 { return None; }
    let inv = SMatrix::<f64, 2, 2>::new(
        m[(1,1)] / det, -m[(0,1)] / det,
        -m[(1,0)] / det, m[(0,0)] / det,
    );
    let log_det = det.abs().ln();
    let eta = -0.5 * (2.0 * LOG_2PI + log_det);
    Some((inv, eta))
}

/// 4x4 matrix inverse (2 sensors) - use cofactor expansion or LU
/// 6x6 matrix inverse (3 sensors) - use Cholesky (already small enough)
```

---

### DATA LAYOUT OPTIMIZATIONS (SoA vs AoS)

| Priority | ID | Optimization | Expected Gain | Overall Impact | Effort | Risk | Notes |
|----------|-----|-------------|---------------|----------------|--------|------|-------|
| 19 | D-1 | **Flat sample buffer (SoA)** | 5-10% | 3-5% | Medium | Low | `samples: Vec<usize>` with stride, not `Vec<Vec<usize>>` |
| 20 | D-2 | L array reordering | 5-15% | 3-8% | High | Medium | Reorder L for sequential access patterns in Gibbs |
| 21 | D-3 | V/W as flat arrays | 2-5% | 1-3% | Low | Low | `v: [usize; N*S]` instead of `DMatrix<usize>` |
| 22 | D-4 | Hypothesis full SoA | 5-10% | <1% | High | Medium | All mu in one matrix, all sigma contiguous (not hot path) |

**Current vs SoA comparison:**

| Structure | Current (AoS) | SoA Alternative | Expected Benefit |
|-----------|---------------|-----------------|------------------|
| Samples | `HashSet<Vec<usize>>` | `Vec<usize>` flat buffer + stride | 10-15% (removes heap alloc per sample) |
| V matrix | `DMatrix<usize>` (heap) | `[usize; N*S]` stack array | 2-5% (better cache) |
| W matrix | `DMatrix<usize>` (heap) | `[usize; M*S]` stack array | 2-5% (better cache) |
| L array | `Vec<f64>` (current order) | Reordered for access pattern | 5-15% (cache hit rate) |
| Hypothesis.mu | `Vec<State4>` | Already good after G1 | - |

---

### OTHER OPTIMIZATIONS

| Priority | ID | Optimization | Expected Gain | Overall Impact | Effort | Risk | Notes |
|----------|-----|-------------|---------------|----------------|--------|------|-------|
| 23 | O-1 | More aggressive gating threshold | 5-20% | 2-7% | Low | Medium | Tighter threshold = more early exits, may miss associations |
| 24 | O-2 | Sparse L representation | ??? | ??? | High | Medium | If >90% of L values are -inf after gating |
| 25 | O-3 | GPU likelihood computation | 50-80% | 17-27% | Very High | High | CUDA/Metal for 10.7M parallel likelihood computations |
| 26 | O-4 | Memory-map L array | 0-5% | 0-2% | Medium | Low | Only useful if L doesn't fit in RAM |
| 27 | O-5 | PGO (Profile-Guided Optimization) | 5-15% | 5-15% | Medium | Low | Compile with profiling, recompile with profile data |
| 28 | O-6 | LTO (Link-Time Optimization) | 2-5% | 2-5% | Low | Low | Already using? Check Cargo.toml |

---

## Recommended Implementation Phases

### Phase 1: Low-hanging fruit (target: 0.59s → 0.50s)

| Order | ID | Optimization | Cumulative Time |
|-------|-----|-------------|-----------------|
| 1 | G-S1 | Fixed-size samples + sort dedup | 0.55s |
| 2 | G-S2 | Precompute stride table | 0.52s |
| 3 | G-S3 | Stack-allocate u vector | 0.51s |
| 4 | L-1 | Specialized small matrix inverse | **0.49s** |

**Expected result: 0.49s (target achieved)**

### Phase 2: If more improvement needed (target: <0.45s)

| Order | ID | Optimization | Cumulative Time |
|-------|-----|-------------|-----------------|
| 5 | G-A1 | Blocked Gibbs sampling | 0.46s |
| 6 | L-2 | Pre-stacked sensor matrices | 0.45s |
| 7 | G-A2 | Early termination | **0.42s** |

### Phase 3: Aggressive optimization (target: <0.35s)

| Order | ID | Optimization | Cumulative Time |
|-------|-----|-------------|-----------------|
| 8 | L-4 | SIMD matrix operations | 0.39s |
| 9 | D-2 | L array reordering | 0.36s |
| 10 | O-5 | PGO compilation | **0.34s** |

---

## Quick Wins Summary

**Implement together (all in Gibbs inner loop):**

1. **G-S1**: Replace `HashSet<Vec<usize>>` with fixed-size array + sort dedup
2. **G-S2**: Precompute strides, use incremental index calculation
3. **G-S3**: Stack-allocate the `u` vector

**Then:**

4. **L-1**: Specialized 2x2/4x4/6x6 inverse functions

These 4 changes should get us from 0.59s to ~0.49s with low risk.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/multisensor_lmbm/gibbs.rs` | G-S1, G-S2, G-S3, G-A1, G-A2 |
| `src/multisensor_lmbm/mod.rs` | G-S2 (stride precomputation) |
| `src/common/linalg.rs` | L-1 (specialized inverse functions) |
| `src/multisensor_lmbm/association.rs` | L-1, L-2, L-3 |

---

## Validation

After each change:
```bash
# Tests (without rayon for deterministic comparison)
cargo test --release --features mimalloc

# Benchmark (3 runs for stability)
for i in 1 2 3; do
  cargo run --release --features rayon,mimalloc --example multi_sensor \
    -- --seed 42 --filter-type LMBM --num-sensors 3 --data-association LBP
done
```

---

## PRNG Comparison (for reference)

| PRNG | Operations | Speed | Quality | Notes |
|------|-----------|-------|---------|-------|
| **Xorshift64 (current)** | 3 XOR + 3 shifts | ~1.5ns | Good | Simple, proven |
| SplitMix64 | 1 mul + 2 XOR + 2 shifts | ~1.2ns | Better | Used in Java SplittableRandom |
| wyrand | 1 mul + 1 XOR | ~1.0ns | Good | Very simple |
| frand | 2 mul + 3 XOR + 3 shifts | ~1.2-1.5ns | Better | Hash-based, not faster |

**Verdict**: PRNG is <1% of runtime. Not worth changing unless quality issues arise.

---

## Notes

- All "Expected Gain" values are estimates based on profiling and code analysis
- Actual gains may vary based on CPU architecture, cache sizes, etc.
- Some optimizations may have synergistic effects (cache improvements help multiple areas)
- Some optimizations may have diminishing returns when combined
- Always measure before and after each change
