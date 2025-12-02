/*!
# Prak - Multi-object tracking library

Rust implementation of multi-object tracking algorithms based on
Labelled Multi-Bernoulli (LMB) filters and their variants.

## Features

- Single-sensor LMB and LMBM filters
- Multi-sensor fusion algorithms (PU-LMB, IC-LMB, GA-LMB, AA-LMB)
- Multiple data association methods (LBP, Gibbs, Murty's algorithm)

## Modules

- [`types`] - Core types: `Track`, `FilterParams`, `StateEstimate`
- [`components`] - Shared algorithms: prediction, update
- [`association`] - Data association: likelihood computation, matrix building
- [`filter`] - Filter trait and implementations

## Example

```rust,no_run
use prak::filter::{Filter, LmbFilter};
use prak::types::{MotionModel, SensorModel, BirthModel, BirthLocation, AssociationConfig};
use nalgebra::{DVector, DMatrix};

// Create filter configuration
let motion = MotionModel::constant_velocity_2d(1.0, 0.1, 0.99);
let sensor = SensorModel::position_sensor_2d(1.0, 0.9, 10.0, 100.0);

// Define a birth location
let birth_loc = BirthLocation::new(
    0,
    DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
    DMatrix::identity(4, 4) * 100.0,
);
let birth = BirthModel::new(vec![birth_loc], 0.1, 0.01);
let association = AssociationConfig::default();

// Create filter
let mut filter = LmbFilter::new(motion, sensor, birth, association);

// Process measurements
let mut rng = rand::thread_rng();
let measurements = vec![DVector::from_vec(vec![1.0, 2.0])];
let estimate = filter.step(&mut rng, &measurements, 0).unwrap();
```
*/

// Core modules
pub mod types;
pub mod components;
pub mod association;
pub mod filter;

// Internal utilities (exposed for advanced use cases)
pub mod common;
pub mod lmb;

// Re-export commonly used types
pub use types::{
    Track, TrackLabel, GaussianComponent, LmbmHypothesis,
    MotionModel, SensorModel, FilterParams, BirthModel, BirthLocation,
    StateEstimate, EstimatedTrack, FilterOutput,
};

pub use filter::{Filter, Associator, Merger, FilterError};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
