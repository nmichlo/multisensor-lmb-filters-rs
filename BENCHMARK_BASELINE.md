# Benchmark Baseline Results

Date: 2025-11-30
Hardware: macOS Darwin 24.5.0
Rust: Release build

## Association Matrix Generation

| Objects | Measurements | Time |
|---------|--------------|------|
| 5 | 5 | 15.37 µs |
| 10 | 10 | 21.21 µs |
| 20 | 20 | 37.17 µs |
| 50 | 50 | 90.10 µs |

**Scaling:** Roughly O(n × m) - linear in both dimensions.

## Loopy Belief Propagation (LBP)

| Objects | Measurements | Time |
|---------|--------------|------|
| 5 | 5 | 2.18 µs |
| 10 | 10 | 10.45 µs |
| 20 | 20 | 47.38 µs |
| 50 | 50 | 439.40 µs |

**Scaling:** Roughly O(n × m × iterations). Convergence typically takes 5-20 iterations.

## Elementary Symmetric Function (ESF)

| n | Time |
|---|------|
| 5 | 92.95 ns |
| 10 | 129.55 ns |
| 20 | 249.85 ns |
| 50 | 1.26 µs |
| 100 | 4.42 µs |

**Scaling:** O(n²) as expected from algorithm.

## Prediction Step

| Objects | Time |
|---------|------|
| 5 | 3.07 µs |
| 10 | 3.04 µs |
| 20 | 3.04 µs |
| 50 | 3.05 µs |

**Scaling:** O(1) - dominated by birth parameter overhead, not object count.

## Full Filter (LMB with LBP)

| Timesteps | Time | Per-timestep |
|-----------|------|--------------|
| 10 | 260.01 µs | 26 µs |
| 25 | 806.16 µs | 32 µs |
| 50 | 2.35 ms | 47 µs |

**Scaling:** Linear in timesteps. Per-timestep cost increases as objects accumulate.

## Bottleneck Analysis

Based on the benchmarks, the main bottlenecks are:

1. **LBP (50x50 = 440 µs)** - Most expensive single operation for large problems
2. **Association matrix generation (50x50 = 90 µs)** - Significant for large problems
3. **ESF (n=100 = 4.4 µs)** - O(n²) scaling could become problematic

## Optimization Opportunities

1. **LBP parallelization** - Row updates are independent, could use rayon
2. **Association matrix parallelization** - Object loop is embarrassingly parallel
3. **Clone reduction** - Many unnecessary clones observed in code
4. **SIMD** - Matrix operations could benefit from explicit vectorization

## Commands

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench -- association_matrices

# Compare against this baseline
cargo bench -- --baseline main
```

## Saved Results Location

Full criterion results saved to: `target/criterion/`
