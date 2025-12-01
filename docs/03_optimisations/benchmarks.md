# LMBM Filter Benchmark Results

## Benchmark Methodology

### Command
```bash
cargo run --release --example multi_sensor -- --seed 42 --filter-type LMBM --num-sensors 3 --data-association LBP
```

### Environment
- **Platform**: darwin (macOS)
- **Date**: 2025-12-01
- **Test scenario**: 3 sensors, 100 timesteps, 10 objects, seed 42, LBP data association

### Notes
- All times are wall-clock time from the "Filter completed in Xs" output
- Each benchmark run 3 times, median reported
- Default features unless otherwise noted
- Clean build before each commit checkout: `cargo clean && cargo build --release`

---

## Results by Commit

| Commit | Description | Features | Run 1 | Run 2 | Run 3 | Median | vs Baseline |
|--------|-------------|----------|-------|-------|-------|--------|-------------|
| `3a12e85` | Baseline | default | 13.89s | 13.90s | 13.90s | **13.90s** | - |
| `441d2ab` | +lto+cache+logdet+stack | default | 12.20s | 12.20s | 12.18s | **12.20s** | -12.2% |
| `6b7db7e` | +mimalloc | mimalloc | 9.62s | 9.50s | 9.58s | **9.58s** | -31.1% |
| `ae4949b` | +rayon | rayon,mimalloc | 2.77s | 2.79s | 2.80s | **2.79s** | -79.9% |
| `b8f6871` | +buffer (REVERTED) | rayon,mimalloc | 2.76s | 2.76s | 2.73s | **2.76s** | -80.1% |
| `4b1306f` | +lazy (REVERTED) | default | 4.99s | 4.92s | 4.91s | **4.92s** | -64.6% |
| `4769c6b` | -buffer (lazy only, REVERTED) | default | 4.93s | 5.01s | 4.94s | **4.94s** | -64.5% |
| (uncommitted) | +clone elimination | rayon,mimalloc | 2.77s | 2.73s | 2.77s | **2.77s** | -80.1% |
| (uncommitted) | +Phase B: gating | rayon,mimalloc | 1.50s | 1.48s | 1.48s | **1.48s** | -89.4% |
| (uncommitted) | +Phase C: deferred params | rayon,mimalloc | 1.37s | 1.39s | 1.38s | **1.38s** | -90.1% |
| (uncommitted) | +Phase D: stack arrays | rayon,mimalloc | 1.42s | 1.38s | 1.40s | **1.40s** | -89.9% |
| (uncommitted) | +Phase F: parallel Gibbs | rayon,mimalloc | 0.68s | 0.65s | 0.65s | **0.65s** | -95.3% |

---

## Summary

### Current Best: rayon+mimalloc+Phase B+C+D+F
- **Time**: 0.65s
- **Speedup**: 21.4x vs baseline (13.90s)
- **Speedup**: 4.3x vs rayon+mimalloc (2.77s)
- **Command**: `cargo run --release --features rayon,mimalloc --example multi_sensor ...`

### Failed Optimizations

#### 1. Lazy Likelihood (REVERTED)
- **Result**: 4.94s (slower than rayon's 2.79s)
- **Why it failed**: While lazy computation reduced likelihood calls from 10.7M to ~445K (96% reduction), the serial nature meant it couldn't compete with parallel eager computation. The overhead of HashMap lookups and RefCell borrow checking also added latency.
- **Lesson**: Parallelization (rayon) provides better speedup than avoiding computation (lazy) when the computation is embarrassingly parallel.

#### 2. Workspace Buffer Reuse (REVERTED)
- **Result**: 2.76s (negligible 1% improvement over 2.79s)
- **Why it failed**: The buffer reuse provided minimal benefit because:
  1. mimalloc already handles allocation efficiently
  2. Most allocation overhead is in nalgebra temporaries, not explicit buffers
  3. Added code complexity for negligible gain
- **Lesson**: Don't optimize allocations that aren't the bottleneck.

---

## Optimization Progression

```
Baseline:           13.90s  ████████████████████████████████████████
+lto+cache+logdet:  12.20s  ███████████████████████████████████
+mimalloc:           9.58s  ███████████████████████████
+rayon:              2.79s  ████████
+clone_elimination:  2.77s  ████████
+Phase B (gating):   1.48s  ████
+Phase C (deferred): 1.38s  ████
+Phase D (stack):    1.40s  ████
+Phase F (parallel): 0.65s  ██  <- CURRENT BEST
+buffer:             2.76s  ████████  (reverted - negligible gain)
+lazy:               4.94s  ██████████████  (reverted - slower than rayon)
```

### Successful Optimizations (Phase B+C+D+F)

#### 1. Measurement Gating (Phase B)
- **Result**: 2.77s → 1.48s (1.87x improvement)
- **How it works**: Quick diagonal Mahalanobis distance check before expensive Cholesky decomposition
- **Threshold**: 50.0 (chi-squared, conservative to preserve equivalence)
- **Lesson**: O(n) gating check before O(n³) Cholesky saves enormous compute

#### 2. Deferred Posterior Params (Phase C)
- **Result**: 1.48s → 1.38s (1.07x improvement)
- **How it works**: Compute L for all 10.7M entries in parallel, but only compute (r, mu, sigma) for ~1000 unique indices actually used by Gibbs sampling
- **Lesson**: Most savings came from gating; deferred params provide incremental benefit

#### 3. Stack Arrays & Ownership (Phase D)
- **Result**: No measurable improvement (~1.40s)
- **How it works**: Replace small heap Vec allocations with stack arrays [bool; MAX_SENSORS], take ownership in robust_inverse_with_log_det
- **Lesson**: mimalloc already handles small allocations efficiently

#### 4. Parallel Gibbs Chains (Phase F)
- **Result**: 1.38s → 0.65s (2.1x improvement)
- **How it works**: Run multiple independent Gibbs chains in parallel (one per CPU core), merge unique samples
- **Lesson**: Embarrassingly parallel task gives excellent speedup with rayon

---

## Recommended Configuration

For best performance, use:
```bash
cargo run --release --features rayon,mimalloc --example multi_sensor -- --filter-type LMBM
```

This provides **21x speedup** over baseline with:
- Parallel likelihood computation via rayon (Phase B+C)
- Measurement gating to skip impossible associations (Phase B)
- Deferred posterior param computation (Phase C)
- Parallel Gibbs sampling chains (Phase F)
- Efficient memory allocation via mimalloc
