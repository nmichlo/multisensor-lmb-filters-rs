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

---

## Summary

### Current Best: rayon+mimalloc+clone_elimination
- **Time**: 2.77s
- **Speedup**: 5x vs baseline
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
+clone_elimination:  2.77s  ████████  <- CURRENT BEST
+buffer:             2.76s  ████████  (reverted - negligible gain)
+lazy:               4.94s  ██████████████  (reverted - slower than rayon)
```

---

## Recommended Configuration

For best performance, use:
```bash
cargo run --release --features rayon,mimalloc --example multi_sensor -- --filter-type LMBM
```

This provides **5x speedup** over baseline with:
- Parallel likelihood computation via rayon
- Efficient memory allocation via mimalloc
