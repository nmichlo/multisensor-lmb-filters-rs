# LMB Filter Benchmark Results

*Generated: 2026-01-07 19:55:55*

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
| 5 | 1 | 45.41 \u00b1 31.75 | 0.71 \u00b1 0.70 (\u00d764.1) | 0.94 \u00b1 0.82 (\u00d748.1) |
| 10 | 1 | 115.13 \u00b1 78.16 | 1.47 \u00b1 1.07 (\u00d778.3) | 1.58 \u00b1 1.15 (\u00d772.8) |
| 20 | 1 | 582.98 \u00b1 627.34 | 7.17 \u00b1 6.84 (\u00d781.3) | 7.24 \u00b1 7.13 (\u00d780.5) |

### LMB-Gibbs


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 1 | 771.39 \u00b1 479.99 | 0.39 \u00b1 0.16 (\u00d71953.9) | 0.54 \u00b1 0.22 (\u00d71435.1) |
| 10 | 1 | 1207.08 \u00b1 574.11 | 0.74 \u00b1 0.33 (\u00d71638.5) | 0.74 \u00b1 0.34 (\u00d71640.5) |
| 20 | 1 | 6430.94 \u00b1 44421.96 | 1.65 \u00b1 0.86 (\u00d73887.9) | 1.69 \u00b1 0.90 (\u00d73812.5) |

### LMB-Murty


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 1 | 33.76 \u00b1 11.46 | 9.24 \u00b1 5.25 (\u00d73.7) | 9.72 \u00b1 5.45 (\u00d73.5) |
| 10 | 1 | 64.85 \u00b1 17.66 | 22.98 \u00b1 12.46 (\u00d72.8) | 23.91 \u00b1 13.45 (\u00d72.7) |
| 20 | 1 | 174.88 \u00b1 69.51 | 74.93 \u00b1 58.00 (\u00d72.3) | 72.71 \u00b1 54.07 (\u00d72.4) |

### LMBM-Gibbs


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 1 | TIMEOUT | 43.95 \u00b1 10.42 (N/A) | 43.77 \u00b1 9.78 (N/A) |
| 5 | 2 | SKIP | 42.21 \u00b1 9.43 (N/A) | 42.42 \u00b1 9.75 (N/A) |
| 10 | 1 | SKIP | 75.28 \u00b1 14.46 (N/A) | 74.99 \u00b1 14.86 (N/A) |
| 10 | 2 | SKIP | 75.72 \u00b1 12.63 (N/A) | 75.83 \u00b1 13.92 (N/A) |
| 10 | 4 | SKIP | 80.76 \u00b1 19.88 (N/A) | 77.12 \u00b1 13.56 (N/A) |
| 20 | 1 | SKIP | 156.83 \u00b1 34.49 (N/A) | 159.83 \u00b1 51.14 (N/A) |
| 20 | 2 | SKIP | 155.19 \u00b1 30.06 (N/A) | 162.36 \u00b1 38.83 (N/A) |
| 20 | 4 | SKIP | 153.77 \u00b1 29.27 (N/A) | 162.83 \u00b1 43.37 (N/A) |
| 20 | 8 | SKIP | 155.53 \u00b1 37.93 (N/A) | 161.65 \u00b1 37.45 (N/A) |
| 50 | 8 | SKIP | 364.43 \u00b1 69.76 (N/A) | 360.15 \u00b1 71.23 (N/A) |

### LMBM-Murty


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 1 | 89.18 \u00b1 77.95 | 29.54 \u00b1 5.36 (\u00d73.0) | 29.57 \u00b1 5.36 (\u00d73.0) |
| 5 | 2 | 257.06 \u00b1 67.60 | 29.15 \u00b1 4.92 (\u00d78.8) | 29.65 \u00b1 5.72 (\u00d78.7) |
| 10 | 1 | 97.24 \u00b1 112.71 | 30.11 \u00b1 5.29 (\u00d73.2) | 32.48 \u00b1 7.52 (\u00d73.0) |
| 10 | 2 | 267.24 \u00b1 58.95 | 30.57 \u00b1 5.18 (\u00d78.7) | 31.25 \u00b1 7.04 (\u00d78.6) |
| 10 | 4 | 212.50 \u00b1 122.16 | 32.13 \u00b1 5.21 (\u00d76.6) | 30.95 \u00b1 5.34 (\u00d76.9) |
| 20 | 1 | 447.20 \u00b1 178.04 | 31.81 \u00b1 5.21 (\u00d714.1) | 32.12 \u00b1 8.26 (\u00d713.9) |
| 20 | 2 | 321.68 \u00b1 100.72 | 31.89 \u00b1 5.95 (\u00d710.1) | 31.83 \u00b1 5.19 (\u00d710.1) |
| 20 | 4 | 368.29 \u00b1 132.72 | 31.50 \u00b1 5.48 (\u00d711.7) | 32.15 \u00b1 6.72 (\u00d711.5) |
| 20 | 8 | 242.17 \u00b1 120.09 | 33.53 \u00b1 7.54 (\u00d77.2) | 32.96 \u00b1 6.72 (\u00d77.3) |
| 50 | 8 | 537.13 \u00b1 189.37 | 37.93 \u00b1 7.12 (\u00d714.2) | 38.94 \u00b1 6.97 (\u00d713.8) |

### AA-LMB-LBP


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 2 | 420.33 \u00b1 458.87 | 4.94 \u00b1 3.27 (\u00d785.0) | 5.19 \u00b1 3.70 (\u00d781.0) |
| 10 | 2 | 814.99 \u00b1 847.78 | 16.21 \u00b1 7.98 (\u00d750.3) | 15.90 \u00b1 7.75 (\u00d751.3) |
| 10 | 4 | 4884.95 \u00b1 2894.06 | 77.79 \u00b1 33.78 (\u00d762.8) | 80.04 \u00b1 34.82 (\u00d761.0) |
| 20 | 2 | 3801.13 \u00b1 2493.31 | 64.66 \u00b1 26.15 (\u00d758.8) | 67.85 \u00b1 28.68 (\u00d756.0) |
| 20 | 4 | TIMEOUT | 212.10 \u00b1 72.79 (N/A) | 212.07 \u00b1 72.68 (N/A) |
| 20 | 8 | SKIP | 579.56 \u00b1 186.59 (N/A) | 579.33 \u00b1 189.21 (N/A) |
| 50 | 8 | SKIP | 2584.64 \u00b1 719.81 (N/A) | 2608.69 \u00b1 719.57 (N/A) |

### IC-LMB-LBP


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 2 | 79.19 \u00b1 74.66 | 0.89 \u00b1 0.64 (\u00d788.9) | 0.89 \u00b1 0.61 (\u00d788.5) |
| 10 | 2 | 224.33 \u00b1 233.67 | 2.68 \u00b1 1.38 (\u00d783.8) | 2.60 \u00b1 1.37 (\u00d786.4) |
| 10 | 4 | 502.75 \u00b1 391.91 | 5.53 \u00b1 2.45 (\u00d790.9) | 5.56 \u00b1 2.41 (\u00d790.5) |
| 20 | 2 | 774.18 \u00b1 603.88 | 10.86 \u00b1 9.34 (\u00d771.3) | 10.79 \u00b1 9.33 (\u00d771.7) |
| 20 | 4 | 1344.30 \u00b1 527.57 | 18.61 \u00b1 7.62 (\u00d772.2) | 19.49 \u00b1 7.90 (\u00d769.0) |
| 20 | 8 | 3313.94 \u00b1 1306.22 | 44.36 \u00b1 17.20 (\u00d774.7) | 44.43 \u00b1 17.38 (\u00d774.6) |
| 50 | 8 | TIMEOUT | 329.41 \u00b1 97.40 (N/A) | 332.44 \u00b1 93.79 (N/A) |

### PU-LMB-LBP


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 2 | 42.43 \u00b1 23.28 | 0.36 \u00b1 0.12 (\u00d7119.2) | 0.35 \u00b1 0.10 (\u00d7119.7) |
| 10 | 2 | 64.57 \u00b1 42.55 | 0.77 \u00b1 0.35 (\u00d783.9) | 0.81 \u00b1 0.38 (\u00d779.2) |
| 10 | 4 | 1443.57 \u00b1 1277.19 | 11.42 \u00b1 11.37 (\u00d7126.4) | 11.60 \u00b1 11.89 (\u00d7124.5) |
| 20 | 2 | 150.26 \u00b1 71.14 | 2.12 \u00b1 0.73 (\u00d770.7) | 2.24 \u00b1 0.78 (\u00d767.1) |
| 20 | 4 | TIMEOUT | 155.81 \u00b1 220.99 (N/A) | 158.31 \u00b1 228.78 (N/A) |
| 20 | 8 | SKIP | ERROR | ERROR |
| 50 | 8 | SKIP | ERROR | ERROR |

### GA-LMB-LBP


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 2 | 18.07 \u00b1 11.30 | 0.28 \u00b1 0.15 (\u00d764.1) | 0.29 \u00b1 0.14 (\u00d762.7) |
| 10 | 2 | 41.45 \u00b1 37.66 | 0.58 \u00b1 0.36 (\u00d771.6) | 0.57 \u00b1 0.38 (\u00d772.5) |
| 10 | 4 | 66.44 \u00b1 34.83 | 0.99 \u00b1 0.66 (\u00d767.3) | 0.98 \u00b1 0.69 (\u00d767.9) |
| 20 | 2 | 136.06 \u00b1 116.04 | 1.79 \u00b1 1.77 (\u00d775.8) | 1.84 \u00b1 1.86 (\u00d773.9) |
| 20 | 4 | 194.18 \u00b1 163.85 | 3.60 \u00b1 3.68 (\u00d754.0) | 3.74 \u00b1 4.38 (\u00d751.9) |
| 20 | 8 | 389.94 \u00b1 338.37 | 6.79 \u00b1 7.47 (\u00d757.4) | 7.66 \u00b1 10.17 (\u00d750.9) |
| 50 | 8 | 23819.81 \u00b1 135882.29 | 58.36 \u00b1 52.12 (\u00d7408.2) | 61.68 \u00b1 56.20 (\u00d7386.2) |

### MS-LMBM-Gibbs


| Objects | Sensors | Octave (ms/step) | Python (ms/step) | Rust (ms/step) |
|---------|---------|------------------|------------------|----------------|
| 5 | 2 | 4769.72 \u00b1 6777.22 | 5.24 \u00b1 1.50 (\u00d7910.7) | 5.38 \u00b1 1.50 (\u00d7886.2) |
| 10 | 2 | TIMEOUT | 9.67 \u00b1 2.00 (N/A) | 9.58 \u00b1 1.88 (N/A) |
| 10 | 4 | SKIP | 545.09 \u00b1 95.75 (N/A) | 550.69 \u00b1 93.81 (N/A) |
| 20 | 2 | SKIP | 35.68 \u00b1 8.89 (N/A) | 34.98 \u00b1 8.80 (N/A) |
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

