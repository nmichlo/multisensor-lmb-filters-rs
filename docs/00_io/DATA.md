# Multi-Sensor LMB/LMBM Tracking Library - Data Structures and Usage Guide

This document provides a comprehensive guide to using the multi-sensor tracking library, including algorithm outputs, data structures, and complete I/O examples for 2D position tracking.

---

## Table of Contents

1. [Algorithm Output Data Structures](#algorithm-output-data-structures)
2. [Input/Output Data Flow for 2D XY Tracking](#inputoutput-data-flow-for-2d-xy-tracking)
3. [Filter Comparison and Recommendations](#filter-comparison-and-recommendations)

---

## Algorithm Output Data Structures

### 1. LMB Filter Outputs

**Source**: `src/lmb/filter.rs` (lines 46-184)

**Function**: `run_lmb_filter(rng: &mut impl Rng, model: &Model, measurements: &[Vec<DVector<f64>>]) -> LmbStateEstimates`

**Returns**: `LmbStateEstimates` struct (lines 16-26):

```rust
pub struct LmbStateEstimates {
    /// Labels for each time-step (birth_time, birth_location)
    pub labels: Vec<DMatrix<usize>>,

    /// Mean estimates for each time-step
    pub mu: Vec<Vec<DVector<f64>>>,

    /// Covariance estimates for each time-step
    pub sigma: Vec<Vec<DMatrix<f64>>>,

    /// All objects (including discarded long trajectories)
    pub objects: Vec<Object>,
}
```

**Field Descriptions**:

- **`labels`**: Vector of 2×N matrices, one per timestep
  - Row 0: Birth time of each tracked object
  - Row 1: Birth location index of each tracked object
  - Example: `labels[t][(0, i)]` = birth time of object i at timestep t

- **`mu`**: State mean estimates
  - Type: `Vec<Vec<DVector<f64>>>`
  - `mu[t][i]` = 4D state vector for object i at timestep t: `[x, vx, y, vy]`
    - `mu[t][i][0]`: Position in x-direction (meters)
    - `mu[t][i][1]`: Velocity in x-direction (m/s)
    - `mu[t][i][2]`: Position in y-direction (meters)
    - `mu[t][i][3]`: Velocity in y-direction (m/s)

- **`sigma`**: State covariance estimates
  - Type: `Vec<Vec<DMatrix<f64>>>`
  - `sigma[t][i]` = 4×4 covariance matrix for object i at timestep t
  - Represents uncertainty in state estimate

- **`objects`**: Complete object trajectories
  - Type: `Vec<Object>`
  - Contains all tracked objects with long trajectories (>20 timesteps by default)
  - Each object includes historical states and timestamps

---

### 2. The Object Struct

**Source**: `src/common/types.rs` (lines 59-102)

```rust
pub struct Object {
    /// Birth location label (0-indexed)
    pub birth_location: usize,

    /// Birth time step (0-indexed)
    pub birth_time: usize,

    /// Existence probability (0.0 to 1.0)
    pub r: f64,

    /// Number of Gaussian mixture components
    pub number_of_gm_components: usize,

    /// Gaussian mixture component weights (sum to 1.0)
    pub w: Vec<f64>,

    /// Gaussian mixture component means (state vectors)
    pub mu: Vec<DVector<f64>>,

    /// Gaussian mixture component covariances
    pub sigma: Vec<DMatrix<f64>>,

    /// Length of trajectory (number of timesteps tracked)
    pub trajectory_length: usize,

    /// Historical states (4 × trajectory_length matrix)
    /// Each column is a state vector at one timestep
    pub trajectory: DMatrix<f64>,

    /// Timestamps for each trajectory point
    pub timestamps: Vec<usize>,
}
```

**Key Concepts**:

- **Existence probability (`r`)**: Probability that the object truly exists (0-1)
  - Objects with `r < 0.01` (default threshold) are pruned
  - High `r` (e.g., 0.95) indicates confident detection

- **Labels**: Unique identifier combining `(birth_time, birth_location)`
  - Example: `(5, 2)` means object born at timestep 5, location 2

- **Gaussian Mixtures**: Each object represented as weighted sum of Gaussians
  - Handles uncertainty in data association and state estimation
  - Most objects have 1-3 components after merging

---

### 3. LMBM Filter Outputs

**Source**: `src/lmbm/filter.rs` (lines 51-222)

**Function**: `run_lmbm_filter(rng: &mut impl Rng, model: &Model, measurements: &[Vec<DVector<f64>>]) -> LmbmStateEstimates`

**Returns**: `LmbmStateEstimates` struct (lines 17-27):

```rust
pub struct LmbmStateEstimates {
    /// Labels for each time-step
    pub labels: Vec<DMatrix<usize>>,

    /// Mean estimates for each time-step
    pub mu: Vec<Vec<DVector<f64>>>,

    /// Covariance estimates for each time-step
    pub sigma: Vec<Vec<DMatrix<f64>>>,

    /// Complete trajectories (uses Trajectory type, not Object)
    pub objects: Vec<Trajectory>,
}
```

**Difference from LMB**: LMBM maintains multiple **hypotheses** about which objects exist. Each hypothesis is a different interpretation of the measurement data.

---

### 4. The Hypothesis Struct

**Source**: `src/common/types.rs` (lines 104-135)

```rust
pub struct Hypothesis {
    /// Birth locations for all objects in this hypothesis
    pub birth_location: Vec<usize>,

    /// Birth times for all objects in this hypothesis
    pub birth_time: Vec<usize>,

    /// Hypothesis weight (sum over all hypotheses = 1.0)
    pub w: f64,

    /// Existence probabilities for each object in hypothesis
    pub r: Vec<f64>,

    /// State means for each object
    pub mu: Vec<DVector<f64>>,

    /// State covariances for each object
    pub sigma: Vec<DMatrix<f64>>,
}
```

**Additional Information**:
- LMBM maintains up to 25 hypotheses (default `maximum_number_of_posterior_hypotheses`)
- Hypotheses with weight < 0.001 are pruned
- Final output uses highest-weighted hypothesis (MAP) or weighted combination (EAP)

**Example Hypothesis**:
```rust
// Hypothesis 1: Two objects exist (weight = 0.7)
Hypothesis {
    birth_location: vec![0, 2],
    birth_time: vec![1, 3],
    w: 0.7,
    r: vec![0.95, 0.85],
    mu: vec![state1, state2],
    sigma: vec![cov1, cov2],
}

// Hypothesis 2: Three objects exist (weight = 0.3)
Hypothesis {
    birth_location: vec![0, 1, 2],
    birth_time: vec![1, 2, 3],
    w: 0.3,
    r: vec![0.90, 0.60, 0.80],
    mu: vec![state1, state2, state3],
    sigma: vec![cov1, cov2, cov3],
}
```

---

### 5. Multi-Sensor LMB Filters

**Source**: `src/multisensor_lmb/`

All multi-sensor LMB variants return `ParallelUpdateStateEstimates` (lines 28-39 in `parallel_update.rs`):

```rust
pub struct ParallelUpdateStateEstimates {
    pub labels: Vec<DMatrix<usize>>,
    pub mu: Vec<Vec<DVector<f64>>>,
    pub sigma: Vec<Vec<DMatrix<f64>>>,
    pub objects: Vec<Trajectory>,  // Note: Trajectory, not Object
}
```

**IC-LMB** (`iterated_corrector.rs`, lines 40-232):
- **Function**: `run_ic_lmb_filter()`
- **Method**: Sequential sensor updates (iterated corrector approach)
- **Returns**: Same format as single-sensor LMB

**PU-LMB, GA-LMB, AA-LMB** (`parallel_update.rs`, lines 144-361):
- **Function**: `run_parallel_update_lmb_filter(..., mode: ParallelUpdateMode)`
- **Modes**:
  - `ParallelUpdateMode::PU`: Parallel Update (information form fusion)
  - `ParallelUpdateMode::GA`: Geometric Average (geometric mean of densities)
  - `ParallelUpdateMode::AA`: Arithmetic Average (weighted average)
- **Returns**: Same structure as IC-LMB

---

### 6. Multi-Sensor LMBM Filter

**Source**: `src/multisensor_lmbm/filter.rs` (lines 60-237)

**Function**: `run_multisensor_lmbm_filter()`

**Returns**: `MultisensorLmbmStateEstimates` (same structure as `LmbmStateEstimates`)

**Warning** (lines 34-36): "This filter is impossibly slow and very memory intensive. Use only for small problems."

---

### 7. State Extraction Methods

#### MAP Estimation (Maximum A Posteriori)

**Source**: `src/lmb/cardinality.rs` (lines 76-118)

```rust
pub fn lmb_map_cardinality_estimate(r: &[f64]) -> (usize, Vec<usize>)
```

- Finds the **most likely number of objects** given existence probabilities
- Uses Elementary Symmetric Functions (ESF) and Mahler's algorithm
- Returns: `(estimated_cardinality, indices_of_selected_objects)`
- Objects are ranked by existence probability (highest first)

**When to use**: Default for most applications. Provides single best estimate.

#### EAP Estimation (Expected A Posteriori)

**Source**: `src/lmbm/hypothesis.rs` (lines 245-289)

```rust
pub fn lmbm_state_extraction(
    hypotheses: &[Hypothesis],
    use_eap_on_lmbm: bool
) -> (Vec<DMatrix<usize>>, Vec<Vec<DVector<f64>>>, Vec<Vec<DMatrix<f64>>>)
```

- Estimates cardinality as `floor(sum of all existence probabilities)`
- More conservative than MAP (tends to estimate fewer objects)
- Default for LMBM: use MAP (not EAP)

**When to use**: When you prefer conservative estimates (lower false positive rate).

#### Practical Extraction Example

```rust
// Access position at timestep t for object i
let position_x = estimates.mu[t][i][0];
let position_y = estimates.mu[t][i][2];
let velocity_x = estimates.mu[t][i][1];
let velocity_y = estimates.mu[t][i][3];

// Access uncertainty (standard deviation)
let covariance = &estimates.sigma[t][i];
let pos_x_std = covariance[(0, 0)].sqrt();  // meters
let pos_y_std = covariance[(2, 2)].sqrt();  // meters

// Access label
let birth_time = estimates.labels[t][(0, i)];
let birth_location = estimates.labels[t][(1, i)];
let label = (birth_time, birth_location);
```

---

## Input/Output Data Flow for 2D XY Tracking

### 1. Input Measurement Format

#### Single-Sensor Measurements

**Type**: `Vec<Vec<DVector<f64>>>`

**Structure**: `measurements[time][measurement_index]`

**Example**: 2D position measurements over 3 timesteps

```rust
use nalgebra::DVector;

// Timestep 0: Two detections at (10.5, 20.3) and (15.2, 18.7)
// Timestep 1: One detection at (11.0, 21.0)
// Timestep 2: Three detections
let measurements = vec![
    vec![
        DVector::from_vec(vec![10.5, 20.3]),  // Detection 1 at t=0
        DVector::from_vec(vec![15.2, 18.7]),  // Detection 2 at t=0
    ],
    vec![
        DVector::from_vec(vec![11.0, 21.0]),  // Detection 1 at t=1
    ],
    vec![
        DVector::from_vec(vec![11.5, 21.5]),  // Detection 1 at t=2
        DVector::from_vec(vec![16.0, 19.2]),  // Detection 2 at t=2
        DVector::from_vec(vec![8.3, 15.1]),   // Detection 3 at t=2 (clutter)
    ],
];
```

#### Multi-Sensor Measurements

**Type**: `Vec<Vec<Vec<DVector<f64>>>>`

**Structure**: `measurements[sensor][time][measurement_index]`

**Example**: 2 sensors over 2 timesteps

```rust
let measurements = vec![
    // Sensor 0 (higher quality: lower noise)
    vec![
        vec![DVector::from_vec(vec![10.5, 20.3])],  // t=0: 1 detection
        vec![DVector::from_vec(vec![11.0, 21.0])],  // t=1: 1 detection
    ],
    // Sensor 1 (different viewpoint)
    vec![
        vec![
            DVector::from_vec(vec![10.7, 20.1]),    // t=0: 2 detections
            DVector::from_vec(vec![15.0, 18.5]),
        ],
        vec![],  // t=1: No detections (missed)
    ],
];
```

---

### 2. Model Setup and Configuration

#### The Model Struct

**Source**: `src/common/types.rs` (lines 165-302)

**Key Parameters for 2D Tracking**:

```rust
pub struct Model {
    // State and measurement dimensions
    pub x_dimension: usize,        // 4 (state: [x, vx, y, vy])
    pub z_dimension: usize,        // 2 (measurement: [x, y])
    pub t: f64,                    // Sampling period (1.0 sec)

    // Motion model (constant velocity)
    pub a: DMatrix<f64>,           // 4×4 state transition matrix
    pub r: DMatrix<f64>,           // 4×4 process noise covariance

    // Observation model
    pub c: DMatrix<f64>,           // 2×4 observation matrix
    pub q: DMatrix<f64>,           // 2×2 measurement noise covariance

    // Detection and clutter parameters
    pub detection_probability: f64,      // P(detect | exists), e.g., 0.95
    pub clutter_rate: f64,              // Expected false alarms per frame, e.g., 10.0
    pub clutter_per_unit_volume: f64,   // Spatial density of clutter
    pub survival_probability: f64,       // P(survive | exists), e.g., 0.95

    // Birth model
    pub mu_b: Vec<DVector<f64>>,        // Birth location means
    pub sigma_b: Vec<DMatrix<f64>>,     // Birth location covariances
    pub r_b: Vec<f64>,                  // Birth existence probabilities

    // Observation space
    pub x_min: f64,  // -100.0
    pub x_max: f64,  // 100.0
    pub y_min: f64,  // -100.0
    pub y_max: f64,  // 100.0

    // Filter parameters
    pub existence_threshold: f64,              // 0.01 (prune if r < threshold)
    pub data_association_method: DataAssociationMethod,
    pub maximum_number_of_gm_components: usize,  // 25

    // Multi-sensor specific
    pub number_of_sensors: Option<usize>,
    pub ga_sensor_weights: Option<Vec<f64>>,   // For GA fusion
    pub aa_sensor_weights: Option<Vec<f64>>,   // For AA fusion
}
```

#### Creating a Model

**Source**: `src/common/model.rs` (lines 23-227)

**Single-Sensor Example**:

```rust
use prak::common::model::generate_model;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::common::rng::SimpleRng;

let mut rng = SimpleRng::new(42);

let model = generate_model(
    &mut rng,
    10.0,                          // clutter_rate (false alarms per frame)
    0.95,                          // detection_probability
    DataAssociationMethod::LBP,    // Data association method
    ScenarioType::Fixed,           // Scenario type
    None,                          // num_birth_locations (for Random scenario)
);
```

**Multi-Sensor Example**:

```rust
use prak::common::model::generate_multisensor_model;
use prak::multisensor_lmb::parallel_update::ParallelUpdateMode;

let model = generate_multisensor_model(
    &mut rng,
    3,                                    // number_of_sensors
    vec![5.0, 5.0, 5.0],                 // clutter_rates per sensor
    vec![0.67, 0.70, 0.73],              // detection_probs per sensor
    vec![4.0, 3.0, 2.0],                 // measurement_noise_std per sensor
    ParallelUpdateMode::PU,              // Fusion mode: PU, GA, or AA
    DataAssociationMethod::LBP,
    ScenarioType::Fixed,
    None,
);
```

#### Model Configuration Details

**State Transition Matrix** (Constant Velocity Model):

```
A = [1  Δt  0   0 ]     [1  1  0  0]
    [0  1   0   0 ]  =  [0  1  0  0]  (for Δt = 1.0 sec)
    [0  0   1  Δt ]     [0  0  1  1]
    [0  0   0   1 ]     [0  0  0  1]
```

Prediction: `x_{k+1} = A * x_k + w_k`

**Process Noise Covariance** (Continuous White Noise Acceleration):

Based on standard deviation σ_v = 5.0 m/s²:

```
R = σ_v² * [Δt³/3   Δt²/2   0       0    ]
           [Δt²/2   Δt      0       0    ]
           [0       0       Δt³/3   Δt²/2]
           [0       0       Δt²/2   Δt   ]
```

**Observation Matrix** (Position-Only Measurements):

```
C = [1  0  0  0]
    [0  0  1  0]
```

Measurement: `z_k = C * x_k + v_k`

**Measurement Noise Covariance**:

Default σ_z = 3.0 meters:

```
Q = [σ_z²  0  ]     [9.0  0.0]
    [0     σ_z²]  =  [0.0  9.0]
```

**Observation Space**:
- X range: [-100, 100] meters
- Y range: [-100, 100] meters
- Area: 40,000 m²

**Clutter Spatial Density**:
```
λ_c = clutter_rate / area = 10.0 / 40000 = 0.00025 per m²
```

---

### 3. Initialization and Birth Parameters

Birth parameters are automatically configured in `generate_model()` based on scenario type.

#### Fixed Scenario (Default)

**4 Birth Locations**:

| Location | Position (x, y) | Initial Velocity (vx, vy) |
|----------|----------------|---------------------------|
| 0        | (-80, -20)     | (0, 0)                    |
| 1        | (-20, 80)      | (0, 0)                    |
| 2        | (0, 0)         | (0, 0)                    |
| 3        | (40, -60)      | (0, 0)                    |

**Birth Covariance**: 10×10 meters² in position, 1×1 (m/s)² in velocity

**Birth Probabilities**:
- LMB: `r_b = 0.03` per location per timestep
- LMBM: `r_b = 0.045` per location per timestep

**Interpretation**: At each timestep, each location has a 3% (LMB) or 4.5% (LMBM) chance of spawning a new object.

#### Random Scenario

**Configurable Number of Locations**: Specified via `num_birth_locations` parameter

**Random Positions**: Uniformly sampled from observation space

**Same Birth Probabilities**: 0.03 (LMB) or 0.045 (LMBM)

---

### 4. Complete Running Examples

#### Single-Sensor LMB Filter

**Source**: `examples/single_sensor.rs`

```rust
use prak::common::model::generate_model;
use prak::common::ground_truth::generate_ground_truth;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::filter::run_lmb_filter;

fn main() {
    // Initialize RNG for reproducibility
    let mut rng = SimpleRng::new(42);

    // Step 1: Create model
    let model = generate_model(
        &mut rng,
        10.0,   // clutter_rate: 10 false alarms per frame
        0.95,   // detection_probability: 95% chance of detecting existing objects
        DataAssociationMethod::LBP,  // Fast approximate data association
        ScenarioType::Fixed,         // Use predefined birth locations
        None,                        // Not needed for Fixed scenario
    );

    // Step 2: Generate ground truth and measurements
    // This simulates the true object trajectories and sensor measurements
    let ground_truth = generate_ground_truth(
        &mut rng,
        &model,
        None  // Use default 100 timesteps
    );

    // Step 3: Run LMB filter
    let estimates = run_lmb_filter(
        &mut rng,
        &model,
        &ground_truth.measurements
    );

    // Step 4: Process and display results
    println!("Filter completed {} timesteps", estimates.labels.len());
    println!("Tracked {} long trajectories\n", estimates.objects.len());

    // Display results for each timestep
    for t in 0..estimates.labels.len() {
        let n_objects = estimates.mu[t].len();
        println!("Timestep {}: {} objects detected", t + 1, n_objects);

        for i in 0..n_objects {
            let state = &estimates.mu[t][i];
            let x = state[0];
            let vx = state[1];
            let y = state[2];
            let vy = state[3];

            let cov = &estimates.sigma[t][i];
            let pos_x_std = cov[(0, 0)].sqrt();
            let pos_y_std = cov[(2, 2)].sqrt();

            let birth_time = estimates.labels[t][(0, i)];
            let birth_loc = estimates.labels[t][(1, i)];

            println!(
                "  Object {}: pos=({:.2}, {:.2}) m, vel=({:.2}, {:.2}) m/s, σ=({:.2}, {:.2}) m, label=({}, {})",
                i, x, y, vx, vy, pos_x_std, pos_y_std, birth_time, birth_loc
            );
        }
    }

    // Display long trajectories
    println!("\nLong Trajectories (>20 timesteps):");
    for (idx, obj) in estimates.objects.iter().enumerate() {
        println!(
            "  Trajectory {}: {} timesteps, birth=({}, {})",
            idx, obj.trajectory_length, obj.birth_time, obj.birth_location
        );
    }
}
```

#### Single-Sensor LMBM Filter

```rust
use prak::lmbm::filter::run_lmbm_filter;

fn main() {
    let mut rng = SimpleRng::new(42);

    let model = generate_model(
        &mut rng,
        10.0,
        0.95,
        DataAssociationMethod::Gibbs,  // Gibbs often better for LMBM
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_ground_truth(&mut rng, &model, None);

    // Run LMBM filter
    let estimates = run_lmbm_filter(
        &mut rng,
        &model,
        &ground_truth.measurements
    );

    // Same output structure as LMB
    for t in 0..estimates.labels.len() {
        println!("Timestep {}: {} objects", t + 1, estimates.mu[t].len());
    }
}
```

#### Multi-Sensor Parallel Update LMB Filter

**Source**: `examples/multi_sensor.rs`

```rust
use prak::common::model::generate_multisensor_model;
use prak::common::ground_truth::generate_multisensor_ground_truth;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::multisensor_lmb::parallel_update::{
    run_parallel_update_lmb_filter, ParallelUpdateMode
};

fn main() {
    let mut rng = SimpleRng::new(42);
    let number_of_sensors = 3;

    // Step 1: Create multi-sensor model
    let model = generate_multisensor_model(
        &mut rng,
        number_of_sensors,
        vec![5.0, 5.0, 5.0],         // Lower clutter per sensor
        vec![0.67, 0.70, 0.73],      // Different detection probabilities
        vec![4.0, 3.0, 2.0],         // Different measurement noise levels
        ParallelUpdateMode::PU,      // Information form fusion (optimal)
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    // Step 2: Generate multi-sensor measurements
    let ground_truth = generate_multisensor_ground_truth(
        &mut rng,
        &model,
        None
    );

    // ground_truth.measurements is Vec<Vec<Vec<DVector<f64>>>>
    // Structure: [sensor][time][measurement]

    // Step 3: Run PU-LMB filter
    let estimates = run_parallel_update_lmb_filter(
        &mut rng,
        &model,
        &ground_truth.measurements,
        number_of_sensors,
        ParallelUpdateMode::PU,  // Can also use GA or AA
    );

    // Step 4: Display results (same structure as single-sensor)
    for t in 0..estimates.labels.len() {
        let n_objects = estimates.mu[t].len();
        println!("Timestep {}: {} objects fused from {} sensors",
                 t + 1, n_objects, number_of_sensors);

        for i in 0..n_objects {
            let x = estimates.mu[t][i][0];
            let y = estimates.mu[t][i][2];
            println!("  Object {}: ({:.2}, {:.2})", i, x, y);
        }
    }
}
```

#### Multi-Sensor Iterated Corrector LMB Filter

```rust
use prak::multisensor_lmb::iterated_corrector::run_ic_lmb_filter;

fn main() {
    let mut rng = SimpleRng::new(42);

    let model = generate_multisensor_model(
        &mut rng,
        3,                          // number_of_sensors
        vec![5.0, 5.0, 5.0],
        vec![0.67, 0.70, 0.73],
        vec![4.0, 3.0, 2.0],
        ParallelUpdateMode::PU,     // Not used by IC-LMB, but required for model
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    let ground_truth = generate_multisensor_ground_truth(&mut rng, &model, None);

    // Run IC-LMB filter (simpler than PU, but suboptimal)
    let estimates = run_ic_lmb_filter(
        &mut rng,
        &model,
        &ground_truth.measurements,
        3,  // number_of_sensors
    );

    // Process results...
}
```

---

### 5. Getting Results from Filter Outputs

#### Extract Current Object Positions

```rust
// For timestep t, iterate through all tracked objects
for t in 0..estimates.labels.len() {
    for i in 0..estimates.mu[t].len() {
        let state = &estimates.mu[t][i];

        // Extract 2D position
        let position = (state[0], state[2]);  // (x, y) in meters

        // Extract 2D velocity
        let velocity = (state[1], state[3]);  // (vx, vy) in m/s

        // Extract full state
        let x = state[0];   // x position
        let vx = state[1];  // x velocity
        let y = state[2];   // y position
        let vy = state[3];  // y velocity

        println!("t={}, obj={}: pos=({:.2}, {:.2}), vel=({:.2}, {:.2})",
                 t, i, x, y, vx, vy);
    }
}
```

#### Extract Uncertainty (Covariances)

```rust
for t in 0..estimates.sigma.len() {
    for i in 0..estimates.sigma[t].len() {
        let cov = &estimates.sigma[t][i];  // 4×4 covariance matrix

        // Position uncertainties
        let var_x = cov[(0, 0)];           // x variance (m²)
        let var_y = cov[(2, 2)];           // y variance (m²)
        let std_x = var_x.sqrt();          // x standard deviation (m)
        let std_y = var_y.sqrt();          // y standard deviation (m)

        // Velocity uncertainties
        let var_vx = cov[(1, 1)];          // vx variance (m²/s²)
        let var_vy = cov[(3, 3)];          // vy variance (m²/s²)

        // Position-velocity correlation
        let cov_x_vx = cov[(0, 1)];

        println!("t={}, obj={}: σ_pos=({:.2}, {:.2}) m", t, i, std_x, std_y);
    }
}
```

#### Extract Track Labels/IDs

```rust
// Labels are 2×n matrices: [birth_time; birth_location]
for t in 0..estimates.labels.len() {
    let n_objects = estimates.labels[t].ncols();

    for i in 0..n_objects {
        let birth_time = estimates.labels[t][(0, i)];
        let birth_location = estimates.labels[t][(1, i)];

        // Unique label for this object
        let label = (birth_time, birth_location);

        println!("Object {} has label {:?}", i, label);
    }
}
```

#### Extract Complete Trajectories

```rust
// For LMB filters: use estimates.objects
for (track_id, obj) in estimates.objects.iter().enumerate() {
    println!("Track {}: {} timesteps", track_id, obj.trajectory_length);
    println!("  Birth: time={}, location={}", obj.birth_time, obj.birth_location);
    println!("  Existence probability: {:.3}", obj.r);

    // Access historical states
    for j in 0..obj.trajectory_length {
        let state = obj.trajectory.column(j);  // 4×1 state vector
        let time = obj.timestamps[j];

        let x = state[0];
        let y = state[2];

        println!("  t={}: ({:.2}, {:.2})", time, x, y);
    }
}

// For LMBM/multi-sensor filters: trajectories stored differently
// Access via mu/sigma vectors instead of trajectory matrix
```

#### Track Individual Objects Across Time

```rust
use std::collections::HashMap;

// Build a map of object labels to their states over time
let mut tracks: HashMap<(usize, usize), Vec<(usize, f64, f64)>> = HashMap::new();

for t in 0..estimates.labels.len() {
    let n_objects = estimates.labels[t].ncols();

    for i in 0..n_objects {
        let birth_time = estimates.labels[t][(0, i)];
        let birth_loc = estimates.labels[t][(1, i)];
        let label = (birth_time, birth_loc);

        let x = estimates.mu[t][i][0];
        let y = estimates.mu[t][i][2];

        tracks.entry(label)
            .or_insert_with(Vec::new)
            .push((t, x, y));
    }
}

// Print each track
for (label, states) in tracks.iter() {
    println!("Track {:?}: {} observations", label, states.len());
    for (t, x, y) in states {
        println!("  t={}: ({:.2}, {:.2})", t, x, y);
    }
}
```

---

### 6. Filter Loop Structure (Internal)

Understanding what happens inside the filter:

#### Prediction Step (Automatic)

For each existing object and birth location:

1. **State Prediction** (Kalman prediction):
   ```
   x̂_{k|k-1} = A * x_{k-1|k-1}
   P_{k|k-1} = A * P_{k-1|k-1} * A^T + R
   ```

2. **Existence Prediction**:
   ```
   r_{k|k-1} = survival_probability * r_{k-1|k-1}
   ```

3. **Add Birth Objects**:
   - Create new objects at each birth location
   - Initial state: `x_b ~ N(μ_b, Σ_b)`
   - Initial existence: `r_b = 0.03` (or 0.045 for LMBM)

#### Update Step (Automatic)

For each measurement at timestep k:

1. **Data Association**:
   - Compute association probabilities between objects and measurements
   - Methods:
     - **LBP**: Loopy Belief Propagation (fast, approximate)
     - **Gibbs**: Gibbs sampling (slower, stochastic)
     - **Murty**: Murty's algorithm (exact, expensive)

2. **Measurement Update** (Kalman update for each association):
   ```
   Innovation: y = z - C * x̂_{k|k-1}
   Innovation covariance: S = C * P_{k|k-1} * C^T + Q
   Kalman gain: K = P_{k|k-1} * C^T * S^{-1}
   Updated state: x_{k|k} = x̂_{k|k-1} + K * y
   Updated covariance: P_{k|k} = (I - K * C) * P_{k|k-1}
   ```

3. **Existence Update**:
   ```
   If detected: r_{k|k} = r_{k|k-1} * P_D * likelihood / normalization
   If missed: r_{k|k} = r_{k|k-1} * (1 - P_D) / normalization
   ```

4. **Pruning**:
   - Remove objects with `r < 0.01` (existence threshold)
   - Merge similar Gaussian components
   - Cap total components at 25 (default)

5. **State Extraction**:
   - Use MAP or EAP to determine number of objects
   - Select objects with highest existence probabilities

---

## Filter Comparison and Recommendations

### Algorithm Comparison Table

| Filter | Sensors | Hypotheses | Speed | Memory | Best For |
|--------|---------|------------|-------|--------|----------|
| **LMB-LBP** | Single | No | Very Fast | Low | General single-sensor tracking |
| **LMB-Gibbs** | Single | No | Fast | Low | Complex scenarios, better accuracy |
| **LMB-Murty** | Single | No | Slow | Low | Small problems, need exact solution |
| **LMBM-Gibbs** | Single | Yes | Slow | Medium | High uncertainty, need hypotheses |
| **LMBM-Murty** | Single | Yes | Very Slow | Medium | Small problems with uncertainty |
| **IC-LMB** | Multi | No | Fast | Low | Simple multi-sensor fusion |
| **PU-LMB** | Multi | No | Medium | Medium | **Optimal multi-sensor fusion** |
| **GA-LMB** | Multi | No | Medium | Medium | Robust to noise, conservative |
| **AA-LMB** | Multi | No | Medium | Low | Simple fusion, can be overconfident |
| **LMBM-MS** | Multi | Yes | **Extremely Slow** | **Very High** | Research only, impractical |

### Data Association Method Comparison

| Method | Speed | Accuracy | Deterministic | Best For |
|--------|-------|----------|---------------|----------|
| **LBP** | Fast | Good | Yes | Default choice for most applications |
| **LBPFixed** | Very Fast | Fair | Yes | Real-time applications, can sacrifice accuracy |
| **Gibbs** | Slow | Better | No (stochastic) | Complex scenarios, many objects |
| **Murty** | Very Slow | **Exact** | Yes | Small problems (<5 objects), benchmarking |

### Recommendations by Use Case

#### General Single-Sensor Tracking
**Use**: LMB + LBP
- Best balance of speed and accuracy
- Deterministic results
- Low memory footprint

```rust
let model = generate_model(&mut rng, 10.0, 0.95,
                          DataAssociationMethod::LBP,
                          ScenarioType::Fixed, None);
let estimates = run_lmb_filter(&mut rng, &model, &measurements);
```

#### High-Clutter Environments
**Use**: LMB + Gibbs
- Better handling of ambiguous associations
- Improved accuracy in complex scenarios

```rust
let model = generate_model(&mut rng, 50.0, 0.90,
                          DataAssociationMethod::Gibbs,
                          ScenarioType::Fixed, None);
let estimates = run_lmb_filter(&mut rng, &model, &measurements);
```

#### Multi-Sensor Optimal Fusion
**Use**: PU-LMB + LBP
- Information-theoretic optimal fusion
- Best multi-sensor accuracy
- Handles decorrelation properly

```rust
let model = generate_multisensor_model(&mut rng, 3,
                                      vec![5.0, 5.0, 5.0],
                                      vec![0.67, 0.70, 0.73],
                                      vec![4.0, 3.0, 2.0],
                                      ParallelUpdateMode::PU,
                                      DataAssociationMethod::LBP,
                                      ScenarioType::Fixed, None);
let estimates = run_parallel_update_lmb_filter(&mut rng, &model,
                                               &measurements, 3,
                                               ParallelUpdateMode::PU);
```

#### Conservative Multi-Sensor Fusion
**Use**: GA-LMB + LBP
- More robust to sensor noise
- Conservative uncertainty estimates
- Prevents overconfidence

```rust
// Same as PU-LMB but with ParallelUpdateMode::GA
let estimates = run_parallel_update_lmb_filter(&mut rng, &model,
                                               &measurements, 3,
                                               ParallelUpdateMode::GA);
```

#### Hypothesis Tracking (Rare)
**Use**: LMBM + Gibbs
- Only when you need to track multiple interpretations
- High computational cost
- Not recommended for real-time

```rust
let model = generate_model(&mut rng, 10.0, 0.95,
                          DataAssociationMethod::Gibbs,
                          ScenarioType::Fixed, None);
let estimates = run_lmbm_filter(&mut rng, &model, &measurements);
```

### When NOT to Use Certain Filters

**Avoid LMBFixed**:
- Inaccurate for complex scenarios
- Only use if speed is critical and accuracy can be sacrificed

**Avoid LMBM for General Tracking**:
- Unnecessarily slow
- LMB performs nearly as well in most cases
- Only use if you specifically need hypothesis tracking

**Avoid Multi-Sensor LMBM**:
- Computationally prohibitive
- "Impossibly slow and very memory intensive" (src/multisensor_lmbm/filter.rs:34)
- Research tool only

**Avoid Murty's Algorithm for Large Problems**:
- Exponential complexity in number of objects
- Use only for <5 objects or benchmarking
- Switch to LBP or Gibbs for practical applications

---

## Quick Start Template

```rust
use prak::common::model::generate_model;
use prak::common::ground_truth::generate_ground_truth;
use prak::common::rng::SimpleRng;
use prak::common::types::{DataAssociationMethod, ScenarioType};
use prak::lmb::filter::run_lmb_filter;

fn main() {
    // 1. Initialize
    let mut rng = SimpleRng::new(42);

    // 2. Configure model
    let model = generate_model(
        &mut rng,
        10.0,   // clutter_rate
        0.95,   // detection_probability
        DataAssociationMethod::LBP,
        ScenarioType::Fixed,
        None,
    );

    // 3. Get measurements (either from generate_ground_truth or your sensors)
    let ground_truth = generate_ground_truth(&mut rng, &model, None);
    let measurements = ground_truth.measurements;

    // 4. Run filter
    let estimates = run_lmb_filter(&mut rng, &model, &measurements);

    // 5. Extract results
    for t in 0..estimates.mu.len() {
        for i in 0..estimates.mu[t].len() {
            let x = estimates.mu[t][i][0];
            let y = estimates.mu[t][i][2];
            println!("t={}, obj={}: ({:.2}, {:.2})", t, i, x, y);
        }
    }
}
```

---

## References

- **LMB Filter**: Vo, B.-T., & Vo, B.-N. (2013). "Labeled Random Finite Sets and Multi-Object Conjugate Priors"
- **LMBM Filter**: Reuter, S., et al. (2014). "The Labeled Multi-Bernoulli Filter"
- **Multi-Sensor Fusion**: García-Fernández, Á. F., et al. (2013). "Poisson Multi-Bernoulli Mixture Conjugate Prior for Multiple Extended Object Filtering"
- **Source Code**: `src/` directory in this repository
- **MATLAB Reference**: `../multisensor-lmb-filters/` (authoritative implementation)

---

**Last Updated**: 2025-11-16
**Version**: 1.0
**Validation Status**: 100% numerical equivalence verified against MATLAB for all single-sensor algorithms via fixture tests
