# Prak - Multi-Object Tracking Library

A Rust implementation of Labeled Multi-Bernoulli (LMB) and LMB Mixture (LMBM) filters for multi-object tracking. Ported from the MATLAB [multisensor-lmb-filters](https://github.com/example/multisensor-lmb-filters) library with **100% numerical equivalence**.

## What is Multi-Object Tracking?

Multi-object tracking (MOT) solves the problem of simultaneously tracking multiple targets from noisy sensor measurements, where:
- **Objects appear and disappear** at unknown times
- **Measurements contain clutter** (false detections)
- **Detections may be missed** (targets not always observed)
- **Data association is uncertain** (which measurement came from which object?)

This library implements **random finite set (RFS)** based filters that elegantly handle all these challenges in a unified probabilistic framework.

## When to Use This Library

Use **prak** when you need to:

- Track multiple objects with **unknown and time-varying cardinality**
- Handle **high clutter** environments (many false detections)
- Deal with **missed detections** (low detection probability)
- Maintain **track identity** across time (labeled tracking)
- **Fuse measurements** from multiple sensors
- Need **100% deterministic, reproducible** results

Common applications include:
- Radar/sonar target tracking
- Video surveillance and pedestrian tracking
- Autonomous vehicle perception
- Air traffic control
- Robotics and SLAM

## Available Algorithms

### Single-Sensor Filters

| Filter | Description | Use When |
|--------|-------------|----------|
| **LMB** | Labeled Multi-Bernoulli | Standard single-sensor tracking. Fast, good for moderate clutter. |
| **LMBM** | LMB Mixture | Higher accuracy through hypothesis tracking. Better for high clutter but slower. |

### Multi-Sensor Filters (Sensor Fusion)

| Filter | Description | Use When |
|--------|-------------|----------|
| **IC-LMB** | Iterated Corrector | Sequential sensor processing. Exact inference, best accuracy. |
| **PU-LMB** | Parallel Update | Parallel sensor fusion with decorrelation. Fast, good accuracy. |
| **GA-LMB** | Geometric Average | Weighted geometric fusion. Robust to outlier sensors. |
| **AA-LMB** | Arithmetic Average | Simple weighted average fusion. Fast, less robust. |
| **Multi-LMBM** | Multi-sensor LMBM | Hypothesis-based multi-sensor fusion. Highest accuracy, slowest. |

### Data Association Methods

| Method | Description | Trade-off |
|--------|-------------|-----------|
| **LBP** | Loopy Belief Propagation | Fast approximate inference. Default choice. |
| **Gibbs** | Gibbs Sampling | Stochastic, converges to exact. Good for complex scenarios. |
| **Murty** | Murty's K-best | Exact K-best assignments. Deterministic, can be slow. |

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
prak = { git = "https://github.com/yourname/prak" }
```

### Basic Usage

```rust
use prak::common::model::generate_model;
use prak::common::ground_truth::generate_ground_truth;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::filter::run_lmb_filter;

fn main() {
    // Initialize deterministic RNG
    let mut rng = SimpleRng::new(42);

    // Create tracking model
    let model = generate_model(
        &mut rng,
        10.0,   // expected clutter per timestep
        0.95,   // detection probability
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,   // use default birth locations
    );

    // Generate simulated ground truth and measurements
    let gt = generate_ground_truth(&mut rng, &model, None);

    // Run LMB filter
    let results = run_lmb_filter(&mut rng, &model, &gt.measurements);

    // Access results
    for (t, labels) in results.labels.iter().enumerate() {
        println!("Time {}: {} objects detected", t, labels.len());
    }
}
```

### Multi-Sensor Tracking

```rust
use prak::common::model::generate_multisensor_model;
use prak::common::ground_truth::generate_multisensor_ground_truth;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::multisensor_lmb::parallel_update::{run_parallel_update_lmb_filter, ParallelUpdateMode};

fn main() {
    let mut rng = SimpleRng::new(42);
    let num_sensors = 3;

    // Create multi-sensor model
    let model = generate_multisensor_model(
        &mut rng,
        num_sensors,
        vec![5.0; num_sensors],           // clutter rates per sensor
        vec![0.67, 0.70, 0.73],           // detection probabilities
        vec![4.0, 3.0, 2.0],              // sensor quality (Q values)
        ParallelUpdateMode::PU,
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    // Generate multi-sensor measurements
    let gt = generate_multisensor_ground_truth(&mut rng, &model, None);

    // Run PU-LMB filter
    let results = run_parallel_update_lmb_filter(
        &mut rng,
        &model,
        &gt.measurements,
        num_sensors,
        ParallelUpdateMode::PU,
    );
}
```

## Architecture

```
prak/
├── src/
│   ├── lib.rs                    # Library root, re-exports
│   ├── common/                   # Shared utilities
│   │   ├── types.rs              # Core data structures (Model, Object, Measurement)
│   │   ├── model.rs              # Model generation
│   │   ├── ground_truth.rs       # Simulation and measurement generation
│   │   ├── rng.rs                # Deterministic RNG (SimpleRng/Xorshift64)
│   │   ├── linalg.rs             # Linear algebra (Kalman, Gaussian PDF)
│   │   ├── metrics.rs            # OSPA, Hellinger, KL divergence
│   │   ├── utils.rs              # ESF, factorial, binomial
│   │   └── association/          # Data association algorithms
│   │       ├── hungarian.rs      # Hungarian algorithm
│   │       ├── lbp.rs            # Loopy Belief Propagation
│   │       ├── gibbs.rs          # Gibbs sampling
│   │       └── murtys.rs         # Murty's K-best algorithm
│   ├── lmb/                      # Single-sensor LMB filter
│   │   ├── filter.rs             # Main filter loop
│   │   ├── prediction.rs         # Prediction step
│   │   ├── association.rs        # Association matrix generation
│   │   ├── data_association.rs   # LBP/Gibbs/Murty wrappers
│   │   ├── update.rs             # Measurement update
│   │   └── cardinality.rs        # MAP cardinality estimation
│   ├── lmbm/                     # Single-sensor LMBM filter
│   │   ├── filter.rs             # Main filter loop
│   │   ├── prediction.rs         # Prediction step
│   │   ├── association.rs        # Association matrices + Gibbs
│   │   └── hypothesis.rs         # Hypothesis management
│   ├── multisensor_lmb/          # Multi-sensor LMB filters
│   │   ├── parallel_update.rs    # PU/GA/AA-LMB filters
│   │   ├── iterated_corrector.rs # IC-LMB filter
│   │   ├── merging.rs            # Track merging (PU/GA/AA)
│   │   └── association.rs        # Per-sensor association
│   └── multisensor_lmbm/         # Multi-sensor LMBM filter
│       ├── filter.rs             # Main filter loop
│       ├── association.rs        # Multi-sensor association
│       ├── hypothesis.rs         # Hypothesis management
│       └── gibbs.rs              # Multi-sensor Gibbs sampling
├── examples/
│   ├── single_sensor.rs          # Single-sensor CLI example
│   └── multi_sensor.rs           # Multi-sensor CLI example
└── tests/                        # Integration tests (8000+ lines)
```

## Running Examples

### Single-Sensor Example

```bash
# Default LMB filter
cargo run --release --example single_sensor

# LMBM filter with Gibbs sampling
cargo run --release --example single_sensor -- --lmbm -a Gibbs

# High clutter scenario
cargo run --release --example single_sensor -- --clutter-rate 50 -p 0.8

# Full options
cargo run --release --example single_sensor -- --help
```

### Multi-Sensor Example

```bash
# Default PU-LMB with 3 sensors
cargo run --release --example multi_sensor

# IC-LMB (Iterated Corrector)
cargo run --release --example multi_sensor -- --filter-type IC

# GA-LMB (Geometric Average)
cargo run --release --example multi_sensor -- --filter-type GA

# Multi-sensor LMBM
cargo run --release --example multi_sensor -- --filter-type LMBM

# Full options
cargo run --release --example multi_sensor -- --help
```

## Algorithm Selection Guide

### Choosing a Filter

```
                    ┌─────────────────────┐
                    │ How many sensors?   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │ 1              │                │ 2+
              ▼                │                ▼
    ┌─────────────────┐        │      ┌─────────────────┐
    │ Need hypothesis │        │      │ Need highest    │
    │ tracking?       │        │      │ accuracy?       │
    └────────┬────────┘        │      └────────┬────────┘
             │                 │               │
      ┌──────┼──────┐          │        ┌──────┼──────┐
      │ No   │ Yes  │          │        │ Yes  │ No   │
      ▼      ▼      │          │        ▼      ▼
    ┌───┐  ┌────┐   │          │      ┌─────┐ ┌─────┐
    │LMB│  │LMBM│   │          │      │IC-LMB│ │Speed│
    └───┘  └────┘   │          │      └─────┘ │prio?│
                    │          │              └──┬──┘
                    │          │          ┌─────┼─────┐
                    │          │          │ Yes │ No  │
                    │          │          ▼     ▼
                    │          │        ┌────┐ ┌─────┐
                    │          │        │PU  │ │GA or│
                    │          │        │LMB │ │AA   │
                    │          │        └────┘ └─────┘
```

### Choosing Data Association

| Scenario | Recommended Method |
|----------|-------------------|
| General purpose, fast | **LBP** |
| Complex associations, need accuracy | **Gibbs** (10K samples) |
| Need deterministic results | **Murty** (K=1000) |
| Very high clutter (50+) | **LBP** or **Gibbs** |

## Key Concepts

### The LMB Filter

The LMB filter represents the multi-object state as a set of labeled Bernoulli components:

```
LMB = { (r¹, p¹, ℓ¹), (r², p², ℓ²), ..., (rⁿ, pⁿ, ℓⁿ) }
```

Where for each component:
- `r` = existence probability (0 to 1)
- `p` = spatial density (Gaussian mixture)
- `ℓ` = unique label (track identity)

The filter performs:
1. **Prediction**: Propagate state estimates forward in time
2. **Update**: Incorporate measurements via data association
3. **Extraction**: Output state estimates for existing objects

### Multi-Sensor Fusion

Multi-sensor filters combine information from multiple sensors:

- **IC-LMB**: Process sensors sequentially, exact but order-dependent
- **PU-LMB**: Parallel information fusion, decorrelates common prior
- **GA-LMB**: Geometric average of posteriors, robust to outliers
- **AA-LMB**: Arithmetic average, simple but less robust

### Deterministic RNG

All randomness uses `SimpleRng` (Xorshift64), enabling:
- **100% reproducible** results across runs
- **Cross-language equivalence** with MATLAB
- **Deterministic testing** without statistical validation

## Performance

Benchmarks on M1 MacBook Pro (100 timesteps, 3 sensors):

| Filter | Time | Notes |
|--------|------|-------|
| LMB-LBP | ~50ms | Fastest single-sensor |
| LMBM | ~200ms | 10 timesteps |
| IC-LMB | ~150ms | Sequential sensors |
| PU-LMB | ~100ms | Parallel fusion |
| GA-LMB | ~100ms | Information form |
| Multi-LMBM | ~500ms | 10 timesteps |

## Testing

```bash
# Run all tests
cargo test --release

# Run specific test suite
cargo test --release --test numerical_equivalence_multi_sensor

# Run with output
cargo test --release -- --nocapture
```

**Test coverage**: 150+ tests, 8000+ lines, verifying 100% numerical equivalence with MATLAB.

## References

1. Vo, B.-T., & Vo, B.-N. (2013). Labeled Random Finite Sets and Multi-Object Conjugate Priors. *IEEE TSP*.
2. Reuter, S., et al. (2014). The Labeled Multi-Bernoulli Filter. *IEEE TSP*.
3. Vo, B.-N., et al. (2017). Multi-Sensor Multi-Object Tracking with the Generalized Labeled Multi-Bernoulli Filter. *IEEE TSP*.

## License

MIT OR Apache-2.0

## Contributing

Contributions welcome! Please ensure:
- All tests pass (`cargo test --release`)
- Code follows existing style
- New features include tests
