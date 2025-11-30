# Profiling Results - All Algorithms

Generated using hotpath with allocation tracking (`--features hotpath,hotpath-alloc`).

**Test scenario**: 3 sensors, 100 timesteps, 10 objects, seed 42, LBP data association

## Executive Summary

| Algorithm | Time | Total Alloc | Bottleneck Function | Bottleneck % |
|-----------|------|-------------|---------------------|--------------|
| **LMBM** | **33.97s** | **954 MB** | `generate_multisensor_lmbm_association_matrices` | **88%** |
| IC | 0.12s | 230 MB | `compute_posterior_lmb_spatial_distributions_multisensor` | 16% |
| PU | 0.03s | 118 MB | `loopy_belief_propagation` | 5% |
| GA | 0.03s | 95 MB | `loopy_belief_propagation` | 10% |
| AA | 0.30s | 756 MB | `compute_posterior_lmb_spatial_distributions_multisensor` | 12% |

### Key Findings

1. **LMBM is 283x slower than PU/GA** - clearly the problematic algorithm
2. **Memory allocation is massive** - LMBM allocates 954 MB total, with 32.8 GB cumulative in one function
3. **Association matrix generation dominates** - `determine_log_likelihood_ratio` called **10.7 MILLION times** in LMBM
4. **IC, PU, GA are all fast** - sub-second performance

---

## 1. LMBM (Multi-Sensor LMBM) - **WORST PERFORMER**

**Execution time: 33.97s**
**Total allocations: 954 MB**

### Timing Breakdown

| Function | Calls | Avg | P95 | Total | % Total |
|----------|-------|-----|-----|-------|---------|
| `generate_multisensor_lmbm_association_matrices` | 991 | 12.33 ms | 33.65 ms | 12.22 s | **88.40%** |
| `multisensor_lmbm_gibbs_sampling` | 991 | 1.28 ms | 1.87 ms | 1.27 s | **9.15%** |
| `determine_multisensor_posterior_hypothesis_parameters` | 991 | 51.10 µs | 149.12 µs | 50.64 ms | **0.36%** |
| `lmbm_prediction_step` | 991 | 1.75 µs | 3.46 µs | 1.74 ms | **0.01%** |

### Allocation Breakdown

| Function | Calls | Avg | Total | % Total |
|----------|-------|-----|-------|---------|
| `generate_multisensor_lmbm_association_matrices` | 991 | 33.9 MB | **32.8 GB** | **3518%** ⚠️ |
| `determine_log_likelihood_ratio` | **10,691,594** | 3.1 KB | **31.3 GB** | **3364%** ⚠️ |
| `multisensor_lmbm_gibbs_sampling` | 991 | 1.1 MB | 1.1 GB | 116% |
| `determine_multisensor_posterior_hypothesis_parameters` | 991 | 235 KB | 227 MB | 24% |

**Critical Issues:**
- ⚠️ **10.7 MILLION calls** to `determine_log_likelihood_ratio`
- ⚠️ **32.8 GB cumulative allocations** in association matrix generation (3518% of total!)
- ⚠️ Nested loops creating massive temporary allocations

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
- **Date**: 2025-11-30
- **Model**: Multi-sensor with LBP data association
- **Test scenario**: 3 sensors, 100 timesteps, 10 objects, seed 42
