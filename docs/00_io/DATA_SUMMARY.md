# Multi-Sensor LMB/LMBM Tracking Library - Quick Reference

## What It Does
Tracks multiple objects in 2D space using sensor measurements (positions), handling:
- Uncertain detections (objects may be missed)
- Clutter (false alarms)
- Unknown number of objects (births/deaths)
- Multiple sensors with different noise characteristics

## Core Concepts

**State**: 4D vector `[x, vx, y, vy]` - position and velocity in 2D
**Measurement**: 2D vector `[x, y]` - observed position only
**Existence Probability**: `r ∈ [0,1]` - confidence an object exists
**Label**: `(birth_time, birth_location)` - unique object identifier

## Input/Output

### Input: Measurements
```rust
// Single-sensor: Vec<Vec<DVector<f64>>>
let measurements = vec![
    vec![DVector::from_vec(vec![10.5, 20.3])],  // t=0: 1 detection at (10.5, 20.3)
    vec![DVector::from_vec(vec![11.0, 21.0])],  // t=1: 1 detection at (11.0, 21.0)
];

// Multi-sensor: Vec<Vec<Vec<DVector<f64>>>>
// measurements[sensor][time][detection]
```

### Output: State Estimates
```rust
pub struct LmbStateEstimates {
    pub labels: Vec<DMatrix<usize>>,      // Object IDs: [birth_time; birth_location]
    pub mu: Vec<Vec<DVector<f64>>>,       // States: mu[t][obj] = [x, vx, y, vy]
    pub sigma: Vec<Vec<DMatrix<f64>>>,    // Covariances: 4×4 uncertainty matrices
    pub objects: Vec<Object>,             // Long trajectories (>20 timesteps)
}

// Access position at timestep t for object i:
let x = estimates.mu[t][i][0];     // x position (meters)
let y = estimates.mu[t][i][2];     // y position (meters)
let pos_std_x = estimates.sigma[t][i][(0,0)].sqrt();  // uncertainty (meters)
```

## Minimal Working Example

```rust
use prak::common::{model::generate_model, ground_truth::generate_ground_truth, rng::SimpleRng};
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::filter::run_lmb_filter;

fn main() {
    let mut rng = SimpleRng::new(42);

    // 1. Configure tracking model
    let model = generate_model(
        &mut rng,
        10.0,   // clutter_rate: 10 false alarms/frame
        0.95,   // detection_probability: 95% chance of detecting objects
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    // 2. Get measurements (from sensors or simulation)
    let ground_truth = generate_ground_truth(&mut rng, &model, None);

    // 3. Run tracker
    let estimates = run_lmb_filter(&mut rng, &model, &ground_truth.measurements);

    // 4. Extract tracked positions
    for t in 0..estimates.mu.len() {
        for i in 0..estimates.mu[t].len() {
            let x = estimates.mu[t][i][0];
            let y = estimates.mu[t][i][2];
            println!("t={}, obj={}: pos=({:.2}, {:.2}) m", t, i, x, y);
        }
    }
}
```

## Which Filter to Use?

| Scenario | Recommended Filter | Code |
|----------|-------------------|------|
| **Single sensor, general use** | LMB + LBP | `run_lmb_filter(..., DataAssociationMethod::LBP)` |
| **High clutter (>20/frame)** | LMB + Gibbs | `run_lmb_filter(..., DataAssociationMethod::Gibbs)` |
| **Multiple sensors** | PU-LMB | `run_parallel_update_lmb_filter(..., ParallelUpdateMode::PU)` |
| **Need exact solution** | LMB + Murty | `run_lmb_filter(..., DataAssociationMethod::Murty)` (slow!) |
| **Hypothesis tracking** | LMBM + Gibbs | `run_lmbm_filter(..., DataAssociationMethod::Gibbs)` (slow!) |

**Avoid**: Multi-sensor LMBM (extremely slow), LBPFixed (inaccurate), Murty for >5 objects

## Filter Performance

- **LMB-LBP**: Fast, deterministic, good accuracy (default choice)
- **LMB-Gibbs**: Slower, better for complex scenarios
- **LMBM**: 10× slower than LMB, only if you need hypothesis tracking
- **PU-LMB**: Optimal multi-sensor fusion, 2-3× slower than single-sensor

## Key Parameters

```rust
// Model configuration
clutter_rate: 10.0              // Expected false alarms per frame
detection_probability: 0.95     // P(detect | exists)
survival_probability: 0.95      // P(survive | exists)
existence_threshold: 0.01       // Prune objects with r < 0.01

// Motion model (constant velocity, Δt=1.0 sec)
process_noise_std: 5.0          // m/s² (acceleration uncertainty)
measurement_noise_std: 3.0      // m (position measurement noise)

// Observation space
[-100, 100] × [-100, 100] meters
```

See `DATA.md` for complete documentation with examples for all 10 filter variants.
