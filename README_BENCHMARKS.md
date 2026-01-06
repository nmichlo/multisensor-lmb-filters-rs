# LMB Filter Benchmark Results

*Generated: 2026-01-07 08:18:59*

## Overview

This benchmark compares implementations of the LMB (Labeled Multi-Bernoulli) filter:

| Implementation | Description |
|----------------|-------------|
| **Octave/MATLAB** | Original reference implementation (interpreted) |
| **Rust** | Native Rust binary compiled with `--release` |
| **Python** | Python calling Rust via PyO3/maturin bindings |

## Performance Summary

### Rust vs Octave Speedup

![Rust vs Octave Speedup](docs/benchmarks/speedup/rust_vs_octave.png)

### Performance by Language

| Octave | Rust | Python |
|--------|------|--------|
| ![Octave](docs/benchmarks/by_language/octave.png) | ![Rust](docs/benchmarks/by_language/rust.png) | ![Python](docs/benchmarks/by_language/python.png) |

## Methodology

- **Timeout**: 10 seconds per scenario
- **Thresholds**: existence=1e-3, gm_weight=1e-4, max_components=100, gm_merge=∞
- **Association**: LBP (100 iterations, tol 1e-6), Gibbs (1000 samples), Murty (25 assignments)
- **RNG Seed**: 42 (deterministic across all implementations)

## Results

### LMB-LBP

![LMB-LBP Performance](docs/benchmarks/by_filter/LMB-LBP.png)

| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |
|---------|---------|-------------|-------------|-----------|
| 5 | 1 | 4478.3 | 54.4 | 87.6 |
| 5 | 2 | *N/A* | *N/A* | *N/A* |
| 10 | 1 | TIMEOUT | 148.6 | 153.9 |
| 10 | 2 | *N/A* | *N/A* | *N/A* |
| 10 | 4 | *N/A* | *N/A* | *N/A* |
| 20 | 1 | SKIP | 740.9 | 749.8 |
| 20 | 2 | *N/A* | *N/A* | *N/A* |
| 20 | 4 | *N/A* | *N/A* | *N/A* |
| 20 | 8 | *N/A* | *N/A* | *N/A* |
| 50 | 8 | *N/A* | *N/A* | *N/A* |

### LMB-Gibbs

![LMB-Gibbs Performance](docs/benchmarks/by_filter/LMB-Gibbs.png)

| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |
|---------|---------|-------------|-------------|-----------|
| 5 | 1 | TIMEOUT | 45.6 | 40.8 |
| 5 | 2 | *N/A* | *N/A* | *N/A* |
| 10 | 1 | SKIP | 75.2 | 79.8 |
| 10 | 2 | *N/A* | *N/A* | *N/A* |
| 10 | 4 | *N/A* | *N/A* | *N/A* |
| 20 | 1 | SKIP | 169.0 | 179.3 |
| 20 | 2 | *N/A* | *N/A* | *N/A* |
| 20 | 4 | *N/A* | *N/A* | *N/A* |
| 20 | 8 | *N/A* | *N/A* | *N/A* |
| 50 | 8 | *N/A* | *N/A* | *N/A* |

### LMB-Murty

![LMB-Murty Performance](docs/benchmarks/by_filter/LMB-Murty.png)

| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |
|---------|---------|-------------|-------------|-----------|
| 5 | 1 | 3472.4 | 957.5 | 966.0 |
| 5 | 2 | *N/A* | *N/A* | *N/A* |
| 10 | 1 | 6737.2 | 2327.8 | 2292.0 |
| 10 | 2 | *N/A* | *N/A* | *N/A* |
| 10 | 4 | *N/A* | *N/A* | *N/A* |
| 20 | 1 | TIMEOUT | 7217.3 | 7288.4 |
| 20 | 2 | *N/A* | *N/A* | *N/A* |
| 20 | 4 | *N/A* | *N/A* | *N/A* |
| 20 | 8 | *N/A* | *N/A* | *N/A* |
| 50 | 8 | *N/A* | *N/A* | *N/A* |

### LMBM-Gibbs

![LMBM-Gibbs Performance](docs/benchmarks/by_filter/LMBM-Gibbs.png)

| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |
|---------|---------|-------------|-------------|-----------|
| 5 | 1 | TIMEOUT | TIMEOUT | 4798.1 |
| 5 | 2 | SKIP | SKIP | 4506.1 |
| 10 | 1 | SKIP | SKIP | 7631.8 |
| 10 | 2 | SKIP | SKIP | 7643.8 |
| 10 | 4 | SKIP | SKIP | 8065.0 |
| 20 | 1 | SKIP | SKIP | TIMEOUT |
| 20 | 2 | SKIP | SKIP | SKIP |
| 20 | 4 | SKIP | SKIP | SKIP |
| 20 | 8 | SKIP | SKIP | SKIP |
| 50 | 8 | SKIP | SKIP | SKIP |

### LMBM-Murty

![LMBM-Murty Performance](docs/benchmarks/by_filter/LMBM-Murty.png)

| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |
|---------|---------|-------------|-------------|-----------|
| 5 | 1 | TIMEOUT | TIMEOUT | 3025.5 |
| 5 | 2 | SKIP | SKIP | 3194.1 |
| 10 | 1 | SKIP | SKIP | 2946.1 |
| 10 | 2 | SKIP | SKIP | 2965.9 |
| 10 | 4 | SKIP | SKIP | 3482.9 |
| 20 | 1 | SKIP | SKIP | 3186.6 |
| 20 | 2 | SKIP | SKIP | 3208.1 |
| 20 | 4 | SKIP | SKIP | 3307.8 |
| 20 | 8 | SKIP | SKIP | 3290.3 |
| 50 | 8 | SKIP | SKIP | 3767.1 |

### AA-LMB-LBP

![AA-LMB-LBP Performance](docs/benchmarks/by_filter/AA-LMB-LBP.png)

| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |
|---------|---------|-------------|-------------|-----------|
| 5 | 1 | *N/A* | *N/A* | *N/A* |
| 5 | 2 | TIMEOUT | 544.4 | 544.0 |
| 10 | 1 | *N/A* | *N/A* | *N/A* |
| 10 | 2 | SKIP | 1623.4 | 1651.7 |
| 10 | 4 | SKIP | 8431.5 | 8139.6 |
| 20 | 1 | *N/A* | *N/A* | *N/A* |
| 20 | 2 | SKIP | 6410.1 | 6519.6 |
| 20 | 4 | SKIP | TIMEOUT | TIMEOUT |
| 20 | 8 | SKIP | SKIP | SKIP |
| 50 | 8 | SKIP | SKIP | SKIP |

### IC-LMB-LBP

![IC-LMB-LBP Performance](docs/benchmarks/by_filter/IC-LMB-LBP.png)

| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |
|---------|---------|-------------|-------------|-----------|
| 5 | 1 | *N/A* | *N/A* | *N/A* |
| 5 | 2 | 8061.9 | 94.3 | 95.1 |
| 10 | 1 | *N/A* | *N/A* | *N/A* |
| 10 | 2 | TIMEOUT | 267.5 | 277.5 |
| 10 | 4 | SKIP | 571.4 | 567.0 |
| 20 | 1 | *N/A* | *N/A* | *N/A* |
| 20 | 2 | SKIP | 1104.3 | 1161.6 |
| 20 | 4 | SKIP | 1893.5 | 1976.1 |
| 20 | 8 | SKIP | 4587.3 | 4669.9 |
| 50 | 8 | SKIP | TIMEOUT | TIMEOUT |

### PU-LMB-LBP

![PU-LMB-LBP Performance](docs/benchmarks/by_filter/PU-LMB-LBP.png)

| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |
|---------|---------|-------------|-------------|-----------|
| 5 | 1 | *N/A* | *N/A* | *N/A* |
| 5 | 2 | 4304.0 | 37.1 | 36.9 |
| 10 | 1 | *N/A* | *N/A* | *N/A* |
| 10 | 2 | 6628.0 | 81.6 | 84.5 |
| 10 | 4 | TIMEOUT | 1203.4 | 1210.2 |
| 20 | 1 | *N/A* | *N/A* | *N/A* |
| 20 | 2 | SKIP | 221.1 | 231.0 |
| 20 | 4 | SKIP | TIMEOUT | TIMEOUT |
| 20 | 8 | SKIP | SKIP | SKIP |
| 50 | 8 | SKIP | SKIP | SKIP |

### GA-LMB-LBP

![GA-LMB-LBP Performance](docs/benchmarks/by_filter/GA-LMB-LBP.png)

| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |
|---------|---------|-------------|-------------|-----------|
| 5 | 1 | *N/A* | *N/A* | *N/A* |
| 5 | 2 | 1851.2 | 30.3 | 29.5 |
| 10 | 1 | *N/A* | *N/A* | *N/A* |
| 10 | 2 | 4083.1 | 62.6 | 60.7 |
| 10 | 4 | 7060.5 | 104.4 | 107.2 |
| 20 | 1 | *N/A* | *N/A* | *N/A* |
| 20 | 2 | TIMEOUT | 199.1 | 203.4 |
| 20 | 4 | SKIP | 366.9 | 374.0 |
| 20 | 8 | SKIP | 696.3 | 689.9 |
| 50 | 8 | SKIP | 5968.2 | 6062.5 |

### MS-LMBM-Gibbs

![MS-LMBM-Gibbs Performance](docs/benchmarks/by_filter/MS-LMBM-Gibbs.png)

| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |
|---------|---------|-------------|-------------|-----------|
| 5 | 1 | *N/A* | *N/A* | *N/A* |
| 5 | 2 | TIMEOUT | 9443.2 | 555.4 |
| 10 | 1 | *N/A* | *N/A* | *N/A* |
| 10 | 2 | SKIP | TIMEOUT | 1007.2 |
| 10 | 4 | SKIP | SKIP | TIMEOUT |
| 20 | 1 | *N/A* | *N/A* | *N/A* |
| 20 | 2 | SKIP | SKIP | SKIP |
| 20 | 4 | SKIP | SKIP | SKIP |
| 20 | 8 | SKIP | SKIP | SKIP |
| 50 | 8 | SKIP | SKIP | SKIP |

## Notes

- **Octave/MATLAB** is interpreted and significantly slower by design
- **Rust** and **Python** run the same compiled Rust code; small differences are PyO3 overhead
- **TIMEOUT** means the benchmark exceeded the time limit
- **ERROR** indicates a runtime error (check logs for details)
- **SKIP** means a previous scenario timed out, so harder scenarios were skipped
- **-** means not applicable (e.g., single-sensor LMB on multi-sensor scenario) or not run

