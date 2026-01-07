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


## Overview

**multisensor-lmb-filters-rs** brings LMB tracking to Rust with:

- **Verified correctness:** Numerical equivalence with original MATLAB at 1e-10 tolerance
- **High performance:** Zero-cost abstractions
- **Type safety:** Compile-time validation of configurations
- **Python bindings:** Use from Python with `pip install multisensor-lmb-filters-rs`
- **Modular design:** Swap components via traits for custom implementations

### Speedup vs MATLAB/Octave

See extensive benchmarks at [README_BENCHMARKS.md](README_BENCHMARKS.md) run on an M4 Pro MacBook
- Results are from 50-1000x faster over the original implementations, bringing tracking times into the 10s-100s of milliseconds range for typical scenarios.
- _optimizations are still ongoing too for complex cases!_

### Related Projects

- **[multisensor-lmb-filters](https://github.com/nmichlo/multisensor-lmb-filters)** - MATLAB reference implementation with deterministic fixtures
- **[Original repository](https://github.com/scjrobertson/multisensor-lmb-filters)** - Stuart Robertson's original MATLAB implementation


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

> **Performance Note:** In Rust, Gibbs (40ms) can be faster than LBP (88ms) due to simple sampling loops that compile well, while LBP uses complex iterative matrix operations. However, in Octave/MATLAB, Gibbs (~75s) is much slower than LBP (~4s) because interpreted loops have high overhead while BLAS-optimized matrix operations stay fast. Murty's algorithm (960ms Rust, timeout Octave) is intentionally slow—it's O(K×n⁴) for exact K-best assignments and only suitable for small problems (<20 objects) where determinism is critical.

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

<details>
<summary><b>📊 Future Work & Related Algorithms</b></summary>

### What's Implemented

#### Filter Algorithms

| Algorithm | Status | This Library | Notes |
|-----------|:------:|--------------|-------|
| PHD/CPHD | ❌ | — | Foundation filters ([paper](https://ieeexplore.ieee.org/document/1710358)) |
| GLMB | ⚠️ Partial | — | LMB is an approximation ([paper](https://ieeexplore.ieee.org/document/6863850)) |
| **LMB** | ✅ | `FilterLmb` | Core single-sensor filter |
| **LMB-Mixture** | ✅ | `FilterLmbm` | Hypothesis-based for track identity |
| PMBM | ❌ | — | Current SOTA? ([paper](https://ieeexplore.ieee.org/document/8289395)) |

#### Multi-Sensor Fusion Strategies

| Strategy | Status | This Library | Description |
|----------|:------:|--------------|-------------|
| **Iterated Corrector (IC)** | ✅ | `FilterIcLmb` | Sequential sensor updates |
| **Parallel Update (PU)** | ✅ | `FilterPuLmb` | Information-form with decorrelation |
| **Geometric Average (GA)** | ✅ | `FilterGaLmb` | Covariance intersection |
| **Arithmetic Average (AA)** | ✅ | `FilterAaLmb` | Simple weighted average |
| **Multi-sensor LMBM** | ✅ | `FilterMultisensorLmbm` | Hypothesis tracking across sensors |

#### Association Methods

| Method | Status | Notes |
|--------|:------:|-------|
| **LBP** | ✅ | Fast, deterministic |
| **Gibbs Sampling** | ✅ | Stochastic, accurate |
| **Murty's K-best** | ✅ | Exact, slow |
| **Hungarian** | ✅ | Internal (used by Murty) |
| GNN | ❌ | Simple greedy |
| JPDA | ❌ | Joint probabilistic |

### Not Implemented (Potential Future Work)

| Extension | Description | Reference |
|-----------|-------------|-----------|
| **PMBM Filter** | Current SOTA, elegant undetected object handling | [Williams 2015](https://ieeexplore.ieee.org/document/7272821) |
| **Full GLMB** | Theoretically optimal (computationally expensive) | [Vo & Vo 2013](https://ieeexplore.ieee.org/document/6863850) |
| **Track-Before-Detect** | Raw sensor data without detector | [2024 paper](https://www.sciencedirect.com/science/article/abs/pii/S1051200424002434) |
| **Distributed LMB** | Sensor network fusion | [2024 paper](https://link.springer.com/article/10.1631/FITEE.2400582) |
| **Multiple-Model** | Maneuvering targets with mode switching | [MM-GLMB](https://www.sciencedirect.com/science/article/abs/pii/S0165168421001572) |
| **Extended Target** | Objects generating multiple detections | PMBM-based |
| **Appearance Features** | Re-identification across views | Would need CNN integration |

### Related Trackers & Libraries

#### RFS-Based (Same Paradigm)

| Library | Language | Algorithms | Link |
|---------|----------|------------|------|
| **This library** | Rust/Python | LMB, LMBM, IC/PU/GA/AA | You're here! |
| Original MATLAB | MATLAB | LMB, LMBM, multi-sensor | [scjrobertson/multisensor-lmb-filters](https://github.com/scjrobertson/multisensor-lmb-filters) |
| Stone Soup | Python | PHD, CPHD, GM-PHD, JPDA | [dstl/Stone-Soup](https://github.com/dstl/Stone-Soup) |
| PMBM Tracker | Python | PMBM for autonomous driving | [chisyliu/PMBM](https://github.com/chisyliu/PMBM) |

#### Heuristic Kalman-Based

| Library | Language | Approach | Link |
|---------|----------|----------|------|
| SORT | Python | Kalman + Hungarian | [abewley/sort](https://github.com/abewley/sort) |
| DeepSORT | Python | SORT + appearance CNN | [nwojke/deep_sort](https://github.com/nwojke/deep_sort) |
| ByteTrack | Python | Two-stage confidence matching | [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) |
| OC-SORT | Python | Observation-centric Kalman | [noahcao/OC_SORT](https://github.com/noahcao/OC_SORT) |
| StrongSORT | Python | Enhanced DeepSORT | [dyhBUPT/StrongSORT](https://github.com/dyhBUPT/StrongSORT) |
| BoT-SORT | Python | ByteTrack + camera motion | [NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT) |
| Norfair | Python | Flexible Kalman | [tryolabs/norfair](https://github.com/tryolabs/norfair) |
| norfair-rs | Rust | Norfair port | [nmichlo/norfair-rs](https://github.com/nmichlo/norfair-rs) |

#### Transformer-Based

| Library | Language | Approach | Link |
|---------|----------|----------|------|
| TrackFormer | Python | DETR-based tracking | [timmeinhardt/trackformer](https://github.com/timmeinhardt/trackformer) |
| MOTR | Python | Query-based tracking | [megvii-research/MOTR](https://github.com/megvii-research/MOTR) |
| MOTRv2 | Python | Enhanced MOTR | [megvii-research/MOTRv2](https://github.com/megvii-research/MOTRv2) |

</details>

<details>
<summary><b>Trade-offs</b></summary>

### The Hard Truths

**On "Principled Uncertainty" (This Library's Strength):**
> The Bayesian RFS framework gives you *mathematically rigorous* uncertainty quantification. But here's what papers don't tell you: **most production systems don't use this uncertainty downstream**. If you're just drawing boxes on video, you don't need a full posterior distribution. The value comes when you're making *decisions* based on tracking (autonomous vehicles, safety systems) or when you need to *explain* why the tracker made a choice.

**On "Multi-Sensor Fusion" (This Library's Strength):**
> This is where RFS-based trackers genuinely shine. SORT/ByteTrack have no principled way to combine camera + radar + lidar. You'd be writing ad-hoc fusion code. IC-LMB, PU-LMB, GA-LMB give you *theoretically grounded* fusion with different trade-offs. If you have multiple sensors, this library saves you from reinventing a worse wheel.

**On "Gaussian Assumption" (This Library's Weakness):**
> LMB assumes Gaussian state distributions. Real objects don't always move in Gaussian ways (think: a car at an intersection could go straight, left, or right - that's multi-modal, not Gaussian). The Multiple-Model extension handles this, but it's not implemented here. For most tracking scenarios (pedestrians, vehicles on highways), Gaussian is fine. For complex maneuvers, you'll hit limits.

**On Speed (Heuristic Trackers' Strength):**
> ByteTrack at 200+ FPS vs LMB at 20 FPS sounds like a 10x difference. But consider: your detector (YOLO, etc.) probably runs at 30-100 FPS. The tracker is rarely the bottleneck. The speed difference matters for: (1) embedded systems with tight latency budgets, (2) batch processing massive video archives, (3) real-time with many objects (>100). For typical "track 10-20 objects in real-time video," both are fast enough.

**On "ID Switches" (Everyone's Problem):**
> All trackers struggle when objects cross paths with similar appearance. DeepSORT/StrongSORT use appearance features, which helps but isn't magic (similar-looking people still get swapped). LMBM maintains multiple hypotheses, which is more principled but also not magic. **The honest truth: if your detector misses an object for 10 frames, any tracker will struggle.** Tracker quality is bounded by detector quality.

**On Transformers (The Hype Check):**
> Transformer trackers get SOTA numbers on MOT benchmarks. But: (1) they're 10-100x slower, (2) they're black boxes you can't debug, (3) benchmark performance doesn't always transfer to your domain, (4) they need GPU. Use them if you're publishing papers or have unlimited compute. For production, heuristic or RFS-based trackers are more practical.

**On Norfair (The Practical Middle Ground):**
> Norfair's strength is *flexibility* and *ease of use*. Custom distance functions, easy detector integration, Pythonic API. It's not theoretically principled, but it's practical. If you're prototyping or need something working in an afternoon, Norfair is great. If you need multi-sensor fusion or principled uncertainty, this library is better.

</details>

---

## License & Attribution

**multisensor-lmb-filters-rs** is re-licensed under the [MIT License](LICENSE-MIT). See the license file for more information on the license grant from the original author.

This Rust port is based on the original [multisensor-lmb-filters](https://github.com/scjrobertson/multisensor-lmb-filters) by Stuart Robertson.

---

**Contributing:** Issues and pull requests welcome!
