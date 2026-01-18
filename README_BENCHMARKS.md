# LMB Filter Benchmark Results

*Generated: 2026-01-08 12:43:40*

## Overview

This benchmark compares implementations of the LMB (Labeled Multi-Bernoulli) filter:

| Implementation | Description |
|----------------|-------------|
| **Octave/MATLAB** | Original reference implementation (interpreted) |
| **Rust** | Native Rust binary compiled with `--release` |
| **Python** | Python calling Rust via PyO3/maturin bindings |

## Performance Summary

### Rust vs Octave Speedup

<img alt="Rust vs Octave Speedup" src="docs/benchmarks/speedup/rust_vs_octave.png" width=640 />

### Performance by Language

<img alt="" src="docs/benchmarks/by_language/octave.png" width=640 />
</br>
<img alt="" src="docs/benchmarks/by_language/rust.png" width=640 />

### Performance by Sensor Count

| Single | Dual | Quad |
|--------|------|------|
| ![Single Sensor](docs/benchmarks/by_sensors/single_sensor.png) | ![Dual Sensor](docs/benchmarks/by_sensors/dual_sensor.png) | ![Quad Sensor](docs/benchmarks/by_sensors/quad_sensor.png) |

## Methodology

Methodology Details:
- **Simulation**: Bouncing objects in 2D space [-100, 100]^2
- **Steps**: 100 simulation steps per scenario
- **Model**: Constant velocity (std=3.0), P_d=0.98, P_s=0.99, Clutter(lambda=1.0)
- **Thresholds**: existence=1e-3, gm_weight=1e-4, max_components=100, gm_merge=∞
- **Association**: LBP (100 iterations, tol 1e-6), Gibbs (1000 samples), Murty (25 assignments)
- **RNG Seed**: 42 (deterministic across all implementations)

## Results

### LMB-LBP


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 1 | 45.41 ± 31.75 | 0.71 ± 0.70 (×64.1) | 0.94 ± 0.82 (×48.1) |
| 10 | 1 | 115.13 ± 78.16 | 1.47 ± 1.07 (×78.3) | 1.58 ± 1.15 (×72.8) |
| 20 | 1 | 582.98 ± 627.34 | 7.17 ± 6.84 (×81.3) | 7.24 ± 7.13 (×80.5) |

### LMB-Gibbs


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 1 | 771.39 ± 479.99 | 0.39 ± 0.16 (×1953.9) | 0.54 ± 0.22 (×1435.1) |
| 10 | 1 | 1207.08 ± 574.11 | 0.74 ± 0.33 (×1638.5) | 0.74 ± 0.34 (×1640.5) |
| 20 | 1 | 6430.94 ± 44421.96 | 1.65 ± 0.86 (×3887.9) | 1.69 ± 0.90 (×3812.5) |

### LMB-Murty


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 1 | 33.76 ± 11.46 | 9.24 ± 5.25 (×3.7) | 9.72 ± 5.45 (×3.5) |
| 10 | 1 | 64.85 ± 17.66 | 22.98 ± 12.46 (×2.8) | 23.91 ± 13.45 (×2.7) |
| 20 | 1 | 174.88 ± 69.51 | 74.93 ± 58.00 (×2.3) | 72.71 ± 54.07 (×2.4) |

### LMBM-Gibbs


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 1 | TIMEOUT | 43.95 ± 10.42 (N/A) | 43.77 ± 9.78 (N/A) |
| 5 | 2 | SKIP | 42.21 ± 9.43 (N/A) | 42.42 ± 9.75 (N/A) |
| 10 | 1 | SKIP | 75.28 ± 14.46 (N/A) | 74.99 ± 14.86 (N/A) |
| 10 | 2 | SKIP | 75.72 ± 12.63 (N/A) | 75.83 ± 13.92 (N/A) |
| 10 | 4 | SKIP | 80.76 ± 19.88 (N/A) | 77.12 ± 13.56 (N/A) |
| 20 | 1 | SKIP | 156.83 ± 34.49 (N/A) | 159.83 ± 51.14 (N/A) |
| 20 | 2 | SKIP | 155.19 ± 30.06 (N/A) | 162.36 ± 38.83 (N/A) |
| 20 | 4 | SKIP | 153.77 ± 29.27 (N/A) | 162.83 ± 43.37 (N/A) |
| 20 | 8 | SKIP | 155.53 ± 37.93 (N/A) | 161.65 ± 37.45 (N/A) |
| 50 | 8 | SKIP | 364.43 ± 69.76 (N/A) | 360.15 ± 71.23 (N/A) |

### LMBM-Murty


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 1 | 89.18 ± 77.95 | 29.54 ± 5.36 (×3.0) | 29.57 ± 5.36 (×3.0) |
| 5 | 2 | 257.06 ± 67.60 | 29.15 ± 4.92 (×8.8) | 29.65 ± 5.72 (×8.7) |
| 10 | 1 | 97.24 ± 112.71 | 30.11 ± 5.29 (×3.2) | 32.48 ± 7.52 (×3.0) |
| 10 | 2 | 267.24 ± 58.95 | 30.57 ± 5.18 (×8.7) | 31.25 ± 7.04 (×8.6) |
| 10 | 4 | 212.50 ± 122.16 | 32.13 ± 5.21 (×6.6) | 30.95 ± 5.34 (×6.9) |
| 20 | 1 | 447.20 ± 178.04 | 31.81 ± 5.21 (×14.1) | 32.12 ± 8.26 (×13.9) |
| 20 | 2 | 321.68 ± 100.72 | 31.89 ± 5.95 (×10.1) | 31.83 ± 5.19 (×10.1) |
| 20 | 4 | 368.29 ± 132.72 | 31.50 ± 5.48 (×11.7) | 32.15 ± 6.72 (×11.5) |
| 20 | 8 | 242.17 ± 120.09 | 33.53 ± 7.54 (×7.2) | 32.96 ± 6.72 (×7.3) |
| 50 | 8 | 537.13 ± 189.37 | 37.93 ± 7.12 (×14.2) | 38.94 ± 6.97 (×13.8) |

### AA-LMB-LBP


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 2 | 420.33 ± 458.87 | 4.94 ± 3.27 (×85.0) | 5.19 ± 3.70 (×81.0) |
| 10 | 2 | 814.99 ± 847.78 | 16.21 ± 7.98 (×50.3) | 15.90 ± 7.75 (×51.3) |
| 10 | 4 | 4884.95 ± 2894.06 | 77.79 ± 33.78 (×62.8) | 80.04 ± 34.82 (×61.0) |
| 20 | 2 | 3801.13 ± 2493.31 | 64.66 ± 26.15 (×58.8) | 67.85 ± 28.68 (×56.0) |
| 20 | 4 | TIMEOUT | 212.10 ± 72.79 (N/A) | 212.07 ± 72.68 (N/A) |
| 20 | 8 | SKIP | 579.56 ± 186.59 (N/A) | 579.33 ± 189.21 (N/A) |
| 50 | 8 | SKIP | 2584.64 ± 719.81 (N/A) | 2608.69 ± 719.57 (N/A) |

### IC-LMB-LBP


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 2 | 79.19 ± 74.66 | 0.89 ± 0.64 (×88.9) | 0.89 ± 0.61 (×88.5) |
| 10 | 2 | 224.33 ± 233.67 | 2.68 ± 1.38 (×83.8) | 2.60 ± 1.37 (×86.4) |
| 10 | 4 | 502.75 ± 391.91 | 5.53 ± 2.45 (×90.9) | 5.56 ± 2.41 (×90.5) |
| 20 | 2 | 774.18 ± 603.88 | 10.86 ± 9.34 (×71.3) | 10.79 ± 9.33 (×71.7) |
| 20 | 4 | 1344.30 ± 527.57 | 18.61 ± 7.62 (×72.2) | 19.49 ± 7.90 (×69.0) |
| 20 | 8 | 3313.94 ± 1306.22 | 44.36 ± 17.20 (×74.7) | 44.43 ± 17.38 (×74.6) |
| 50 | 8 | TIMEOUT | 329.41 ± 97.40 (N/A) | 332.44 ± 93.79 (N/A) |

### PU-LMB-LBP


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 2 | 42.43 ± 23.28 | 0.36 ± 0.12 (×119.2) | 0.35 ± 0.10 (×119.7) |
| 10 | 2 | 64.57 ± 42.55 | 0.77 ± 0.35 (×83.9) | 0.81 ± 0.38 (×79.2) |
| 10 | 4 | 1443.57 ± 1277.19 | 11.42 ± 11.37 (×126.4) | 11.60 ± 11.89 (×124.5) |
| 20 | 2 | 150.26 ± 71.14 | 2.12 ± 0.73 (×70.7) | 2.24 ± 0.78 (×67.1) |
| 20 | 4 | TIMEOUT | 155.81 ± 220.99 (N/A) | 158.31 ± 228.78 (N/A) |
| 20 | 8 | SKIP | TIMEOUT | TIMEOUT |
| 50 | 8 | SKIP | SKIP | SKIP |

### GA-LMB-LBP


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 2 | 18.07 ± 11.30 | 0.28 ± 0.15 (×64.1) | 0.29 ± 0.14 (×62.7) |
| 10 | 2 | 41.45 ± 37.66 | 0.58 ± 0.36 (×71.6) | 0.57 ± 0.38 (×72.5) |
| 10 | 4 | 66.44 ± 34.83 | 0.99 ± 0.66 (×67.3) | 0.98 ± 0.69 (×67.9) |
| 20 | 2 | 136.06 ± 116.04 | 1.79 ± 1.77 (×75.8) | 1.84 ± 1.86 (×73.9) |
| 20 | 4 | 194.18 ± 163.85 | 3.60 ± 3.68 (×54.0) | 3.74 ± 4.38 (×51.9) |
| 20 | 8 | 389.94 ± 338.37 | 6.79 ± 7.47 (×57.4) | 7.66 ± 10.17 (×50.9) |
| 50 | 8 | 23819.81 ± 135882.29 | 58.36 ± 52.12 (×408.2) | 61.68 ± 56.20 (×386.2) |

### MS-LMBM-Gibbs


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 2 | 4769.72 ± 6777.22 | 5.24 ± 1.50 (×910.7) | 5.38 ± 1.50 (×886.2) |
| 10 | 2 | TIMEOUT | 9.67 ± 2.00 (N/A) | 9.58 ± 1.88 (N/A) |
| 10 | 4 | SKIP | 545.09 ± 95.75 (N/A) | 550.69 ± 93.81 (N/A) |
| 20 | 2 | SKIP | 35.68 ± 8.89 (N/A) | 34.98 ± 8.80 (N/A) |
| 20 | 4 | SKIP | TIMEOUT | TIMEOUT |
| 20 | 8 | SKIP | SKIP | SKIP |
| 50 | 8 | SKIP | SKIP | SKIP |

## Notes

- **Octave/MATLAB** is interpreted and significantly slower by design
- **Rust** and **Python** run the same compiled Rust code; small differences are PyO3 overhead
- **TIMEOUT** means the benchmark exceeded the time limit
- **ERROR** indicates a runtime error (check logs for details)
- **SKIP** means a previous scenario timed out, so harder scenarios were skipped
- **-** means not applicable (e.g., single-sensor LMB on multi-sensor scenario) or not run

