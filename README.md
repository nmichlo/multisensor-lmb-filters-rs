## multisensor-lmb-filters-rs

**Multi-object tracking using Labeled Multi-Bernoulli filters for Rust**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE-MIT)
[![Crates.io](https://img.shields.io/crates/v/multisensor-lmb-filters-rs.svg)](https://crates.io/crates/multisensor-lmb-filters-rs)
[![PyPI](https://img.shields.io/pypi/v/multisensor-lmb-filters-rs.svg)](https://pypi.org/project/multisensor-lmb-filters-rs/)
[![Python](https://img.shields.io/pypi/pyversions/multisensor-lmb-filters-rs.svg)](https://pypi.org/project/multisensor-lmb-filters-rs/)
[![Rust Version](https://img.shields.io/badge/Rust-1.70%2B-orange?logo=rust)](https://www.rust-lang.org)

---

> **Disclaimer:** This is an unofficial Rust port of the MATLAB [multisensor-lmb-filters](https://github.com/scjrobertson/multisensor-lmb-filters) library by Stuart Robertson. This project is **NOT** affiliated with or endorsed by the original author. All credit for the original design and algorithms goes to Stuart Robertson and the referenced academic papers.

---

## What is Multi-Object Tracking?

Imagine you have a security camera watching a parking lot. Cars come and go, and you want to:
- **Detect** each car in every video frame
- **Track** each car across frames (Car A in frame 1 is the same as Car A in frame 100)
- **Handle uncertainty** (cars get occluded, detections are noisy, new cars appear)

This library solves exactly that problem using **Labeled Multi-Bernoulli (LMB) filters** - a probabilistic approach that:
- Maintains a probability distribution over "how many objects exist and where are they"
- Assigns unique labels to each tracked object
- Handles object birth (new objects appearing) and death (objects leaving)
- Fuses data from multiple sensors (cameras, radars, lidars)

### Related Projects

- **[multisensor-lmb-filters](https://github.com/nmichlo/multisensor-lmb-filters)** - MATLAB reference implementation with deterministic fixtures
- **[Original repository](https://github.com/scjrobertson/multisensor-lmb-filters)** - Stuart Robertson's original MATLAB implementation

---

## Overview

**multisensor-lmb-filters-rs** brings LMB tracking to Rust with:

- **Verified correctness:** Numerical equivalence with original MATLAB at 1e-10 tolerance
- **High performance:** Zero-cost abstractions
- **Type safety:** Compile-time validation of configurations
- **Python bindings:** Use from Python with `pip install multisensor-lmb-filters-rs`
- **Modular design:** Swap components via traits for custom implementations

---

## Quick Decision Guide

**Don't know where to start? Follow this:**

```
Q1: How many sensors do you have?
    ├─► ONE sensor ──────────► Use FilterLmb + LBP (simplest, fastest)
    └─► MULTIPLE sensors ────► Use FilterIcLmb + LBP (robust default)

Q2: Are objects frequently crossing paths or occluding each other?
    ├─► YES ─► Consider FilterLmbm (single) or Gibbs sampling (better accuracy)
    └─► NO ──► Stick with FilterLmb/FilterIcLmb + LBP

Q3: Do you need deterministic results (same input = same output)?
    ├─► YES ─► Use LBP or Murty (avoid Gibbs)
    └─► NO ──► Any association method works
```

**TL;DR for beginners:** Start with `FilterLmb` + `LBP` for single sensor, or `FilterIcLmb` + `LBP` for multiple sensors.

---

## All Available Combinations

The library is modular: choose a **Filter** and an **Association Method** independently.

### Filters (7 options)

| Filter | Sensors | Description | When to Use |
|--------|---------|-------------|-------------|
| `FilterLmb` | 1 | Standard LMB with marginal updates | **Default for single sensor.** Fast, good accuracy |
| `FilterLmbm` | 1 | LMB-Mixture with hypothesis tracking | Objects cross paths frequently, need track identity |
| `FilterIcLmb` | 2+ | Iterated Corrector (sequential) | **Default for multi-sensor.** Robust, handles different sensor types |
| `FilterPuLmb` | 2+ | Parallel Update | Sensors are independent, need maximum speed |
| `FilterGaLmb` | 2+ | Geometric Average fusion | Some sensors may be unreliable |
| `FilterAaLmb` | 2+ | Arithmetic Average fusion | Sensors have similar quality |
| `FilterMultisensorLmbm` | 2+ | Multi-sensor LMBM | Highest accuracy, offline processing |

### Association Methods (3 options)

| Method | Deterministic | Speed | Accuracy | When to Use |
|--------|---------------|-------|----------|-------------|
| `LBP` | Yes | Fast | Good | **Default.** Works for most scenarios |
| `Gibbs` | No | Medium | Excellent | Dense scenes with many overlapping objects |
| `Murty` | Yes | Slow | Exact | Small problems (<20 objects), must be reproducible |

All filters work with all association methods:

| | LBP | Gibbs | Murty |
|---|:---:|:---:|:---:|
| **FilterLmb** | ✓ | ✓ | ✓ |
| **FilterLmbm** | ✓ | ✓ | ✓ |
| **FilterIcLmb** | ✓ | ✓ | ✓ |
| **FilterPuLmb** | ✓ | ✓ | ✓ |
| **FilterGaLmb** | ✓ | ✓ | ✓ |
| **FilterAaLmb** | ✓ | ✓ | ✓ |
| **FilterMultisensorLmbm** | — | ✓ | ✓ |

*Note: FilterMultisensorLmbm uses Gibbs internally for hypothesis management*

---

## Installation

### Rust

```toml
[dependencies]
multisensor-lmb-filters-rs = { git = "https://github.com/nmichlo/multisensor-lmb-filters-rs" }
nalgebra = "0.32"
rand = "0.8"
```

### Python

```bash
uv add multisensor-lmb-filters-rs
```

---

## Quick Start

### Python

```python
import numpy as np
from multisensor_lmb_filters_rs import (
    FilterLmb, MotionModel, SensorModel, BirthModel, AssociatorConfig
)

# 1. Define how objects move (constant velocity model)
motion = MotionModel.constant_velocity_2d(
    dt=1.0,              # Time between frames (seconds)
    process_noise=0.1,   # How unpredictable is motion
    survival_prob=0.99   # Probability object persists to next frame
)

# 2. Define your sensor characteristics
sensor = SensorModel.position_2d(
    noise_std=1.0,       # Measurement noise (pixels or meters)
    detection_prob=0.9,  # Probability of detecting an object
    clutter_rate=10.0    # Expected false detections per frame
)

# 3. Define where new objects can appear
birth = BirthModel.uniform_2d(
    region=[0, 100, 0, 100],  # [x_min, x_max, y_min, y_max]
    birth_prob=0.1            # Probability of new object per location
)

# 4. Choose association method
association = AssociatorConfig.lbp(max_iterations=1000, tolerance=1e-6)

# 5. Create filter
filter = FilterLmb(motion, sensor, birth, association, seed=42)

# 6. Process each frame
for t in range(100):
    # Your detector gives you measurements: [[x1,y1], [x2,y2], ...]
    measurements = np.array([[10.5, 20.3], [50.1, 60.8]])  # Example

    output = filter.step(measurements, timestep=t)

    print(f"Frame {t}: {len(output.tracks)} tracked objects")
    for track in output.tracks:
        print(f"  ID={track.label}: position=({track.state[0]:.1f}, {track.state[2]:.1f})")
```

### Multi-Sensor Example (Python)

```python

...

# Define each sensor with its characteristics
sensors = [
    SensorModel.position_2d(noise_std=1.0, detection_prob=0.9, clutter_rate=10.0),  # Camera
    SensorModel.position_2d(noise_std=2.0, detection_prob=0.95, clutter_rate=5.0),  # Radar
]

...

# Choose a multi-sensor compatible filter
filter = FilterIcLmb(motion, sensors, birth, association, seed=42)

for t in range(100):
    # Each sensor provides its own measurements
    camera_detections = get_camera_detections(t)
    radar_detections = get_radar_detections(t)

    output = filter.step([camera_detections, radar_detections], timestep=t)
    print(f"Frame {t}: {len(output.tracks)} fused tracks")
```

<details>
<summary><b>Rust Example</b></summary>

```rust
use multisensor_lmb_filters_rs::lmb::{
    FilterLmb, MotionModel, SensorModel, BirthModel, BirthLocation,
    AssociationConfig, Filter,
};
use nalgebra::{DVector, DMatrix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);
    let sensor = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);

    let birth_loc = BirthLocation::new(
        0,
        DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
        DMatrix::identity(4, 4) * 100.0,
    );
    let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);
    let association = AssociationConfig::default();

    let mut filter = FilterLmb::new(motion, sensor, birth, association);
    let mut rng = rand::thread_rng();

    for t in 0..100 {
        let measurements = get_measurements(t);
        let output = filter.step(&mut rng, &measurements, t)?;
        println!("Frame {}: {} tracks", t, output.tracks.len());
    }

    Ok(())
}
```

</details>

---

## Architecture & Extensibility

The library uses a **trait-based modular design**. Each component can be swapped independently:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              YOUR APPLICATION                               │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Filter (trait)                              │    │
│  │  Orchestrates the tracking pipeline for each timestep               │    │
│  │  Implementations:                                                   │    │
│  │  • FilterLmb, FilterLmbm (single-sensor)                            │    │
│  │  • FilterIcLmb, FilterPuLmb, FilterGaLmb, FilterAaLmb (multi)       │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│           ┌─────────────────────┼─────────────────────┐                     │
│           ▼                     ▼                     ▼                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐            │
│  │ Predictor       │   │ Associator      │   │ Merger          │            │
│  │ (trait)         │   │ (trait)         │   │ (trait)         │            │
│  │                 │   │                 │   │                 │            │
│  │ Predicts state  │   │ Matches tracks  │   │ Fuses multi-    │            │
│  │ to next frame   │   │ to measurements │   │ sensor results  │            │
│  │                 │   │                 │   │                 │            │
│  │ • KalmanPredict │   │ • LbpAssociator │   │ • ICMerger      │            │
│  │ • (custom)      │   │ • GibbsAssoc.   │   │ • PUMerger      │            │
│  │                 │   │ • MurtyAssoc.   │   │ • GAMerger      │            │
│  │                 │   │ • (custom)      │   │ • AAMerger      │            │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘            │
│                                 │                                           │
│                                 ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    AssociationBuilder                                │   │
│  │  Computes likelihood matrices from tracks + measurements             │   │
│  │  (Uses MotionModel + SensorModel internally)                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Traits

| Trait | Purpose | Implement When... |
|-------|---------|-------------------|
| `Filter` | Main entry point. Runs prediction → association → update loop | Creating entirely new filter types |
| `Associator` | Solves measurement-to-track assignment | Adding new assignment algorithms |
| `Merger` | Combines results from multiple sensors | Adding new fusion strategies |
| `Predictor` | Predicts track state forward in time | Using non-Kalman prediction |
| `Updater` | Updates track state with measurements | Custom measurement updates |

### Key Data Structures

| Type | Description |
|------|-------------|
| `Track` | Single tracked object with state, covariance, existence probability |
| `GaussianComponent` | Single Gaussian in a mixture (mean, covariance, weight) |
| `MotionModel` | State transition matrix (A), process noise (R), survival probability |
| `SensorModel` | Observation matrix (C), measurement noise (Q), detection probability, clutter |
| `BirthModel` | Where/how new tracks can appear |
| `AssociationMatrices` | Likelihood, cost, and probability matrices for assignment |

### Extending the Library

**Custom Associator Example:**

```rust
use multisensor_lmb_filters_rs::lmb::{Associator, AssociationResult, AssociationConfig};
use multisensor_lmb_filters_rs::association::AssociationMatrices;

struct MyCustomAssociator;

impl Associator for MyCustomAssociator {
    fn associate<R: rand::Rng>(
        &self,
        matrices: &AssociationMatrices,
        config: &AssociationConfig,
        rng: &mut R,
    ) -> Result<AssociationResult, FilterError> {
        // Your custom assignment algorithm here
        // Input: matrices.cost, matrices.likelihood, etc.
        // Output: existence probabilities + marginal weights
    }
}
```

**Custom Merger Example:**

```rust
use multisensor_lmb_filters_rs::lmb::{Merger, Track, SensorModel};

struct MyCustomMerger;

impl Merger for MyCustomMerger {
    fn merge(&self, sensor_tracks: &[Vec<Track>], sensors: &[&SensorModel]) -> Vec<Track> {
        // Your custom fusion logic here
        // Input: tracks from each sensor
        // Output: fused tracks
    }

    fn is_sequential(&self) -> bool {
        false  // true if sensors should be processed sequentially
    }
}
```

### Directory Structure

```
src/
├── lmb/                        # Main filter implementations
│   ├── singlesensor/           # FilterLmb, FilterLmbm
│   ├── multisensor/            # FilterIcLmb, FilterAaLmb, FilterGaLmb, FilterPuLmb
│   ├── traits.rs               # Filter, Associator, Merger, Predictor, Updater
│   └── types.rs                # Track, MotionModel, SensorModel, BirthModel
├── association/                # Association matrix building
│   ├── builder.rs              # AssociationBuilder
│   └── likelihood.rs           # Gaussian likelihood computation
├── common/                     # Shared algorithms
│   ├── association/            # LBP, Gibbs, Murty implementations
│   ├── linalg.rs               # Linear algebra utilities
│   └── rng.rs                  # MATLAB-compatible RNG
├── components/                 # Reusable filter components
│   ├── prediction.rs           # Kalman prediction
│   └── update.rs               # Existence/weight updates
└── python/                     # PyO3 bindings
```

---

## Configuration Reference

### Association Methods

```python
from multisensor_lmb_filters_rs import AssociatorConfig

# Loopy Belief Propagation - RECOMMENDED DEFAULT
config = AssociatorConfig.lbp(
    max_iterations=1000,  # Max LBP iterations
    tolerance=1e-6        # Convergence threshold
)

# Gibbs Sampling - for dense/ambiguous scenes
config = AssociatorConfig.gibbs(
    num_samples=1000      # Number of samples to draw
)

# Murty's K-best - exact solution for small problems
config = AssociatorConfig.murty(
    k_best=100            # Number of top hypotheses
)
```

### Detailed Output

```python
# Get intermediate results for debugging/analysis
output = filter.step_detailed(measurements, timestep=t)

# Association matrices (cost, likelihood, etc.)
if output.association_matrices:
    print("Cost matrix shape:", output.association_matrices.cost.shape)

# Association result (existence probs, marginals)
if output.association_result:
    print("Existence probabilities:", output.association_result.existence_probabilities)

# Per-sensor data (multi-sensor filters only)
if output.sensor_updates:
    for update in output.sensor_updates:
        print(f"Sensor {update.sensor_index}: {len(update.updated_tracks)} tracks")
```

---

## Testing

```bash
# Rust tests
cargo test --release

# Python tests
uv run pytest tests/ -v
```

---

## References

1. Vo, B.-T., & Vo, B.-N. (2013). Labeled Random Finite Sets and Multi-Object Conjugate Priors. *IEEE Trans. Signal Processing*.
2. Reuter, S., et al. (2014). The Labeled Multi-Bernoulli Filter. *IEEE Trans. Signal Processing*.
3. Vo, B.-N., et al. (2017). Multi-Sensor Multi-Object Tracking with the Generalized Labeled Multi-Bernoulli Filter. *IEEE Trans. Signal Processing*.

---

## License & Attribution

**multisensor-lmb-filters-rs** is re-licensed under the [MIT License](LICENSE-MIT). See the license file for more information on the license grant from the original author.

This Rust port is based on the original [multisensor-lmb-filters](https://github.com/scjrobertson/multisensor-lmb-filters) by Stuart Robertson.

---

**Contributing:** Issues and pull requests welcome!
