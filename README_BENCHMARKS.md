# LMB Filter Benchmark Results

*Generated: 2026-01-06 18:45:51*

## Overview

This benchmark compares implementations of the LMB (Labeled Multi-Bernoulli) filter:

| Implementation | Description |
|----------------|-------------|
| **Octave/MATLAB** | Original reference implementation (interpreted) |
| **Rust** | Native Rust binary compiled with `--release` |
| **Python** | Python calling Rust via PyO3/maturin bindings |

## Methodology

- **Timeout**: 60 seconds per scenario
- **Thresholds**: existence=1e-3, gm_weight=1e-4, max_components=100, gm_merge=∞
- **Association**: LBP (100 iterations, tol 1e-6), Gibbs (1000 samples), Murty (25 assignments)
- **RNG Seed**: 42 (deterministic across all implementations)

## Results

### LMB-LBP

| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |
|---------|---------|-------------|-----------|-------------|
| 5 | 1 | 4524.3 | 55.1 | 54.3 |
| 5 | 2 | 22446.8 | 47.9 | 47.6 |
| 10 | 1 | 11499.4 | 149.3 | 147.4 |
| 10 | 2 | TIMEOUT | 157.3 | 154.2 |
| 10 | 4 | SKIP | 165.8 | 164.3 |
| 20 | 1 | SKIP | 709.6 | 706.1 |
| 20 | 2 | SKIP | 648.3 | 640.6 |
| 20 | 4 | SKIP | 569.4 | 568.9 |
| 20 | 8 | SKIP | 643.4 | 634.8 |
| 50 | 8 | SKIP | 6860.5 | 6696.9 |

### LMB-Gibbs

| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |
|---------|---------|-------------|-----------|-------------|
| 5 | 1 | TIMEOUT | 55.1 | 38.9 |
| 5 | 2 | SKIP | 56.2 | 40.5 |
| 10 | 1 | SKIP | 99.2 | 73.1 |
| 10 | 2 | SKIP | 92.8 | 74.3 |
| 10 | 4 | SKIP | 95.5 | 74.2 |
| 20 | 1 | SKIP | 191.1 | 162.8 |
| 20 | 2 | SKIP | 198.9 | 167.7 |
| 20 | 4 | SKIP | 196.3 | 169.6 |
| 20 | 8 | SKIP | 187.8 | 160.1 |
| 50 | 8 | SKIP | 1046.6 | 1031.3 |

### LMB-Murty

| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |
|---------|---------|-------------|-----------|-------------|
| 5 | 1 | 3350.5 | 949.3 | 943.7 |
| 5 | 2 | 13179.3 | 991.5 | 996.3 |
| 10 | 1 | 6981.0 | 2302.6 | 2506.2 |
| 10 | 2 | 38424.6 | 2336.0 | 2326.2 |
| 10 | 4 | 9816.7 | 1897.6 | 1879.5 |
| 20 | 1 | 16007.6 | 6807.2 | 6985.1 |
| 20 | 2 | 46846.3 | 7412.6 | 7472.1 |
| 20 | 4 | 14670.1 | 6932.7 | 7735.8 |
| 20 | 8 | 15092.8 | 6335.9 | 6219.3 |
| 50 | 8 | TIMEOUT | TIMEOUT | TIMEOUT |

### LMBM-Gibbs

| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |
|---------|---------|-------------|-----------|-------------|
| 5 | 1 | TIMEOUT | TIMEOUT | TIMEOUT |
| 5 | 2 | SKIP | SKIP | SKIP |
| 10 | 1 | SKIP | SKIP | SKIP |
| 10 | 2 | SKIP | SKIP | SKIP |
| 10 | 4 | SKIP | SKIP | SKIP |
| 20 | 1 | SKIP | SKIP | SKIP |
| 20 | 2 | SKIP | SKIP | SKIP |
| 20 | 4 | SKIP | SKIP | SKIP |
| 20 | 8 | SKIP | SKIP | SKIP |
| 50 | 8 | SKIP | SKIP | SKIP |

### LMBM-Murty

| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |
|---------|---------|-------------|-----------|-------------|
| 5 | 1 | 8924.0 | TIMEOUT | TIMEOUT |
| 5 | 2 | 26039.4 | SKIP | SKIP |
| 10 | 1 | 9505.5 | SKIP | SKIP |
| 10 | 2 | 27906.6 | SKIP | SKIP |
| 10 | 4 | 20615.9 | SKIP | SKIP |
| 20 | 1 | 41037.2 | SKIP | SKIP |
| 20 | 2 | 29078.7 | SKIP | SKIP |
| 20 | 4 | 35167.7 | SKIP | SKIP |
| 20 | 8 | 23710.2 | SKIP | SKIP |
| 50 | 8 | 51381.1 | SKIP | SKIP |

### AA-LMB-LBP

| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |
|---------|---------|-------------|-----------|-------------|
| 5 | 1 | - | - | - |
| 5 | 2 | 43688.3 | 539.2 | 526.1 |
| 10 | 1 | - | - | - |
| 10 | 2 | TIMEOUT | 1868.2 | 1569.8 |
| 10 | 4 | SKIP | 7925.0 | 7843.0 |
| 20 | 1 | - | - | - |
| 20 | 2 | SKIP | 6340.5 | 6249.6 |
| 20 | 4 | SKIP | 21201.2 | 20792.5 |
| 20 | 8 | SKIP | 57391.2 | 56080.0 |
| 50 | 8 | SKIP | TIMEOUT | TIMEOUT |

### IC-LMB-LBP

| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |
|---------|---------|-------------|-----------|-------------|
| 5 | 1 | - | - | - |
| 5 | 2 | 8263.3 | 90.6 | 92.3 |
| 10 | 1 | - | - | - |
| 10 | 2 | 22692.4 | 266.8 | 265.1 |
| 10 | 4 | 49859.0 | 561.6 | 584.6 |
| 20 | 1 | - | - | - |
| 20 | 2 | TIMEOUT | 1073.9 | 1064.4 |
| 20 | 4 | SKIP | 1910.4 | 1919.1 |
| 20 | 8 | SKIP | 4623.6 | 4374.9 |
| 50 | 8 | SKIP | 32626.0 | 32090.0 |

### PU-LMB-LBP

| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |
|---------|---------|-------------|-----------|-------------|
| 5 | 1 | - | - | - |
| 5 | 2 | 4333.7 | 35.5 | 34.8 |
| 10 | 1 | - | - | - |
| 10 | 2 | 6669.8 | 89.0 | 84.9 |
| 10 | 4 | TIMEOUT | 1121.6 | 1110.2 |
| 20 | 1 | - | - | - |
| 20 | 2 | SKIP | 217.3 | 213.9 |
| 20 | 4 | SKIP | 15051.5 | 15137.0 |
| 20 | 8 | SKIP | TIMEOUT | TIMEOUT |
| 50 | 8 | SKIP | SKIP | SKIP |

### GA-LMB-LBP

| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |
|---------|---------|-------------|-----------|-------------|
| 5 | 1 | - | - | - |
| 5 | 2 | 1858.1 | 27.6 | 41.8 |
| 10 | 1 | - | - | - |
| 10 | 2 | 4209.3 | 62.6 | 61.5 |
| 10 | 4 | 6560.6 | 101.1 | 102.0 |
| 20 | 1 | - | - | - |
| 20 | 2 | 12996.0 | 181.4 | 180.1 |
| 20 | 4 | 19063.6 | 372.3 | 341.4 |
| 20 | 8 | 37557.4 | 676.1 | 640.5 |
| 50 | 8 | TIMEOUT | 5729.0 | 5695.7 |

### MS-LMBM-Gibbs

| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |
|---------|---------|-------------|-----------|-------------|
| 5 | 1 | - | - | - |
| 5 | 2 | TIMEOUT | 9538.2 | 9882.9 |
| 10 | 1 | - | - | - |
| 10 | 2 | SKIP | 20988.1 | 26416.4 |
| 10 | 4 | SKIP | 51674.5 | 53997.3 |
| 20 | 1 | - | - | - |
| 20 | 2 | SKIP | TIMEOUT | 47984.3 |
| 20 | 4 | SKIP | SKIP | TIMEOUT |
| 20 | 8 | SKIP | SKIP | SKIP |
| 50 | 8 | SKIP | SKIP | SKIP |

## Notes

- **Octave/MATLAB** is interpreted and significantly slower by design
- **Rust** and **Python** run the same compiled Rust code; small differences are PyO3 overhead
- **TIMEOUT** means the benchmark exceeded the time limit
- **ERROR** indicates a runtime error (check logs for details)
- **SKIP** means a previous scenario timed out, so harder scenarios were skipped
- **-** means not applicable (e.g., multi-sensor filter on single-sensor scenario) or not run

