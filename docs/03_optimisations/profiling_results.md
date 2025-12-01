# Profiling Results - All Algorithms

Generated using hotpath with allocation tracking (`--features hotpath,hotpath-alloc`).

**Test scenario**: 3 sensors, 100 timesteps, 10 objects, seed 42, LBP data association

## Executive Summary

| Algorithm | Time | Total Alloc | Bottleneck Function | Bottleneck % |
|-----------|------|-------------|---------------------|--------------|
| **LMBM** | **4.90s** | **954 MB** | `lazy::ensure_computed` (on-demand) | N/A |
| IC | 0.12s | 230 MB | `compute_posterior_lmb_spatial_distributions_multisensor` | 16% |
| PU | 0.03s | 118 MB | `loopy_belief_propagation` | 5% |
| GA | 0.03s | 95 MB | `loopy_belief_propagation` | 10% |
| AA | 0.30s | 756 MB | `compute_posterior_lmb_spatial_distributions_multisensor` | 12% |

### Key Findings (After Phase 20 - Lazy Likelihood)

1. **LMBM now 4.9x faster** - 23.14s baseline → 4.90s with lazy likelihood
2. **Likelihood computations reduced 96%** - 10.7M → 445K actual computations
3. **Cache hit rate 99.8%** - 205M lookups, only 445K computed
4. **IC, PU, GA remain fast** - sub-second performance

---

## 1. LMBM (Multi-Sensor LMBM) - **OPTIMIZED (Phase 20)**

**Execution time: 4.90s** (was 33.97s baseline, **4.9x faster**)
**Total allocations: 954 MB**

### Timing Breakdown (After Lazy Likelihood)

| Function | Calls | Avg | Total | Notes |
|----------|-------|-----|-------|-------|
| `lazy::ensure_computed` | 205,295,746 | - | - | Cache lookups (99.8% hit rate) |
| `lazy::compute_likelihood` | 445,370 | - | - | **96% reduction** from 10.7M |
| `gibbs::multisensor_lmbm_gibbs_sampling` | 991 | ~2 MB | 1.9 GB | On-demand computation |
| `hypothesis::determine_multisensor_posterior_hypothesis_parameters` | 991 | 235 KB | 227 MB | Reuses cached values |

### Allocation Breakdown (After Lazy Likelihood)

| Function | Calls | Avg | Total | % Total |
|----------|-------|-----|-------|---------|
| `lazy::ensure_computed` | 205,295,746 | 2.0 KB | 390.6 GB | Cumulative (cache lookups) |
| `gibbs::multisensor_lmbm_gibbs_sampling` | 991 | 2.0 MB | 1.9 GB | 207% |
| `lazy::compute_likelihood` | 445,370 | 1.7 KB | 733.7 MB | 77% |
| `hypothesis::determine_multisensor_posterior_hypothesis_parameters` | 991 | 235 KB | 227.5 MB | 24% |

**Improvements from Phase 20 (Lazy Likelihood):**
- ✅ **4.1x faster** (20.20s → 4.90s with default features)
- ✅ **96% reduction** in likelihood computations (10.7M → 445K)
- ✅ **99.8% cache hit rate** (205M lookups, 445K computed)
- ✅ Removed 10.7M iteration precomputation loop

---

## 2. IC (Iterated Corrector)

**Execution time: 0.12s** (283x faster than LMBM)
**Total allocations: 230 MB**

### Allocation Breakdown

| Function | Calls | Avg | Total | % Total |
|----------|-------|-----|-------|---------|
| `run_ic_lmb_filter` | 1 | 228 MB | 228 MB | 99% |
| `compute_posterior_lmb_spatial_distributions_multisensor` | 300 | 124 KB | 36 MB | 16% |
| `lmb_prediction_step` | 100 | 66 KB | 6.5 MB | 3% |
| `loopy_belief_propagation` | 300 | 20 KB | 5.9 MB | 3% |

---

## 3. PU (Parallel Update) - **FASTEST**

**Execution time: 0.03s** (1133x faster than LMBM)
**Total allocations: 118 MB**

### Allocation Breakdown

| Function | Calls | Avg | Total | % Total |
|----------|-------|-----|-------|---------|
| `run_parallel_update_lmb_filter` | 1 | 116 MB | 116 MB | 98% |
| `loopy_belief_propagation` | 300 | 21 KB | 6.0 MB | 5% |
| `compute_posterior_lmb_spatial_distributions_multisensor` | 300 | 12 KB | 3.6 MB | 3% |
| `lmb_prediction_step` | 100 | 18 KB | 1.7 MB | 1% |

---

## 4. GA (Geometric Average)

**Execution time: 0.03s** (1133x faster than LMBM)
**Total allocations: 95 MB**

### Allocation Breakdown

| Function | Calls | Avg | Total | % Total |
|----------|-------|-----|-------|---------|
| `run_parallel_update_lmb_filter` | 1 | 93 MB | 93 MB | 98% |
| `loopy_belief_propagation` | 300 | 33 KB | 9.7 MB | 10% |
| `compute_posterior_lmb_spatial_distributions_multisensor` | 300 | 19 KB | 5.4 MB | 6% |
| `lmb_prediction_step` | 100 | 20 KB | 1.9 MB | 2% |

---

## 5. AA (Arithmetic Average)

**Execution time: 0.30s** (113x faster than LMBM)
**Total allocations: 756 MB**

### Allocation Breakdown

| Function | Calls | Avg | Total | % Total |
|----------|-------|-----|-------|---------|
| `run_parallel_update_lmb_filter` | 1 | 754 MB | 754 MB | 100% |
| `compute_posterior_lmb_spatial_distributions_multisensor` | 300 | 304 KB | 89 MB | 12% |
| `loopy_belief_propagation` | 300 | 60 KB | 17 MB | 2% |
| `lmb_prediction_step` | 100 | 168 KB | 16 MB | 2% |

---

## Optimization Priorities

### 1. LMBM Algorithm (HIGH PRIORITY)

**Target: `generate_multisensor_lmbm_association_matrices` and `determine_log_likelihood_ratio`**

- 88% of execution time
- 10.7 MILLION function calls
- 32.8 GB cumulative allocations (!)
- Clear algorithmic inefficiency

**Action items:**
1. Reduce clone() operations in nested loops
2. Pre-allocate buffers instead of temporary allocations
3. Consider algorithm redesign to reduce combinatorial explosion
4. Profile specifically why `determine_log_likelihood_ratio` is called so many times

### 2. All Algorithms (MEDIUM PRIORITY)

**Target: Reduce overall allocations**

- LBP iterations could benefit from buffer reuse
- Prediction step allocates new matrices every time
- Consider using `&mut` buffers passed in instead of returning new allocations

### 3. AA Algorithm (LOW PRIORITY)

**Target: Higher allocations than PU/GA**

- 10x more allocations than PU (756 MB vs 118 MB)
- 2.5x slower than PU (0.30s vs 0.03s)
- Still acceptable performance, but room for improvement

---

## Environment

- **Platform**: darwin (macOS)
- **Compiler**: rustc with `--release` optimizations
- **Features**: `hotpath`, `hotpath-alloc`
- **Date**: 2025-12-01 (updated after Phase 20 - Lazy Likelihood)
- **Model**: Multi-sensor with LBP data association
- **Test scenario**: 3 sensors, 100 timesteps, 10 objects, seed 42
