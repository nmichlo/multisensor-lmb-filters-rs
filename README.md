# Prak - Multi-Object Tracking Library

A Rust implementation of Labeled Multi-Bernoulli (LMB) and LMB Mixture (LMBM) filters for multi-object tracking. Ported from the MATLAB [multisensor-lmb-filters](https://github.com/example/multisensor-lmb-filters) library.

## Features

- **Single-sensor filters**: LMB and LMBM
- **Multi-sensor fusion**: IC-LMB, PU-LMB, GA-LMB, AA-LMB, Multi-LMBM
- **Data association**: LBP, Gibbs sampling, Murty's K-best
- **Trait-based design**: Easy to extend and customize
- **MATLAB-equivalent**: Verified numerical equivalence at 1e-12 tolerance

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
prak = { git = "https://github.com/yourname/prak" }
nalgebra = "0.32"
rand = "0.8"
```

### Basic LMB Filter

```rust
use prak::filter::{Filter, LmbFilter};
use prak::types::{MotionModel, SensorModel, BirthModel, BirthLocation, AssociationConfig};
use nalgebra::{DVector, DMatrix};

fn main() {
    // Define motion model (constant velocity 2D)
    let motion = MotionModel::constant_velocity_2d(
        1.0,   // timestep
        0.1,   // process noise
        0.99,  // survival probability
    );

    // Define sensor model (position measurements)
    let sensor = SensorModel::position_sensor_2d(
        1.0,   // measurement noise std
        0.9,   // detection probability
        10.0,  // clutter rate
        100.0, // observation space volume
    );

    // Define birth model
    let birth_loc = BirthLocation::new(
        0,
        DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
        DMatrix::identity(4, 4) * 100.0,
    );
    let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);

    // Create filter with default LBP association
    let association = AssociationConfig::default();
    let mut filter = LmbFilter::new(motion, sensor, birth, association);

    // Process measurements
    let mut rng = rand::thread_rng();
    for t in 0..100 {
        let measurements = get_measurements(t); // Your measurement source
        let estimate = filter.step(&mut rng, &measurements, t).unwrap();

        println!("Time {}: {} tracks", t, estimate.tracks.len());
        for track in &estimate.tracks {
            println!("  Track {}: pos=({:.2}, {:.2})",
                     track.label, track.state[0], track.state[2]);
        }
    }
}
```

### Multi-Sensor Fusion

```rust
use prak::filter::{Filter, AaLmbFilter, ArithmeticAverageMerger};
use prak::types::{MotionModel, SensorModel, MultisensorConfig, BirthModel, AssociationConfig};

fn main() {
    let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);

    // Define multiple sensors with different characteristics
    let sensor1 = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);
    let sensor2 = SensorModel::position_sensor_2d(1.5, 0.85, 8.0, 100.0);
    let sensors = MultisensorConfig::new(vec![sensor1, sensor2]);

    let birth = BirthModel::new(vec![/* birth locations */], 0.1, 0.01);
    let association = AssociationConfig::default();
    let merger = ArithmeticAverageMerger::uniform(2, 100);

    // Create AA-LMB filter (Arithmetic Average fusion)
    let mut filter: AaLmbFilter = AaLmbFilter::new(
        motion, sensors, birth, association, merger
    );

    let mut rng = rand::thread_rng();
    for t in 0..100 {
        // Measurements from each sensor
        let measurements = vec![
            get_sensor1_measurements(t),
            get_sensor2_measurements(t),
        ];

        let estimate = filter.step(&mut rng, &measurements, t).unwrap();
        println!("Time {}: {} tracks", t, estimate.tracks.len());
    }
}
```

## Available Filters

### Single-Sensor

| Type | Description |
|------|-------------|
| `LmbFilter` | Standard LMB filter with marginal reweighting |
| `LmbmFilter` | LMBM filter with hypothesis tracking |

### Multi-Sensor

| Type | Description |
|------|-------------|
| `AaLmbFilter` | Arithmetic Average fusion |
| `GaLmbFilter` | Geometric Average fusion |
| `PuLmbFilter` | Parallel Update fusion |
| `IcLmbFilter` | Iterated Corrector (sequential) |
| `MultisensorLmbmFilter` | Multi-sensor LMBM with Gibbs sampling |

### Data Association Methods

| Method | Description |
|--------|-------------|
| `Lbp` | Loopy Belief Propagation (default, fast) |
| `Gibbs` | Gibbs sampling (stochastic, accurate) |
| `Murty` | Murty's K-best (deterministic) |

## Architecture

```
src/
├── types/              # Core data types
│   ├── track.rs        # Track, GaussianComponent, TrackLabel
│   ├── config.rs       # MotionModel, SensorModel, BirthModel
│   └── output.rs       # StateEstimate, FilterOutput
├── components/         # Shared algorithms
│   ├── prediction.rs   # Kalman prediction
│   └── update.rs       # Existence probability updates
├── association/        # Data association
│   ├── likelihood.rs   # Likelihood computation
│   └── builder.rs      # Association matrix construction
├── filter/             # Filter implementations
│   ├── traits.rs       # Filter, Associator, Merger traits
│   ├── lmb.rs          # LmbFilter
│   ├── lmbm.rs         # LmbmFilter
│   ├── multisensor_lmb.rs    # Multi-sensor LMB variants
│   └── multisensor_lmbm.rs   # MultisensorLmbmFilter
├── common/             # Low-level utilities
│   ├── association/    # LBP, Gibbs, Murty algorithms
│   ├── linalg.rs       # Linear algebra helpers
│   └── rng.rs          # RNG traits
└── lmb/
    └── cardinality.rs  # MAP cardinality estimation
```

## Customization

### Custom Associator

```rust
use prak::filter::{Associator, AssociationResult};
use prak::association::AssociationMatrices;

struct MyAssociator;

impl Associator for MyAssociator {
    fn associate<R: rand::Rng>(
        &self,
        matrices: &AssociationMatrices,
        config: &AssociationConfig,
        rng: &mut R,
    ) -> Result<AssociationResult, FilterError> {
        // Your custom association logic
    }
}

// Use with LmbFilter
let filter = LmbFilter::with_associator_type::<MyAssociator>(motion, sensor, birth, config);
```

### Custom Merger (Multi-Sensor)

```rust
use prak::filter::Merger;

struct MyMerger;

impl Merger for MyMerger {
    fn merge(&self, sensor_tracks: &[Vec<Track>], sensors: &[&SensorModel]) -> Vec<Track> {
        // Your custom fusion logic
    }
}
```

## Testing

```bash
# Run all tests
cargo test --release

# Run specific test
cargo test --release test_new_api_lbp_marginals_equivalence

# Run with output
cargo test --release -- --nocapture
```

## Performance

Typical performance on modern hardware (100 timesteps):

| Filter | Time |
|--------|------|
| LMB (LBP) | ~50ms |
| LMBM | ~200ms |
| Multi-sensor LMB | ~100-150ms |

## References

1. Vo, B.-T., & Vo, B.-N. (2013). Labeled Random Finite Sets and Multi-Object Conjugate Priors. *IEEE TSP*.
2. Reuter, S., et al. (2014). The Labeled Multi-Bernoulli Filter. *IEEE TSP*.
3. Vo, B.-N., et al. (2017). Multi-Sensor Multi-Object Tracking with the Generalized Labeled Multi-Bernoulli Filter. *IEEE TSP*.

## License

MIT OR Apache-2.0
