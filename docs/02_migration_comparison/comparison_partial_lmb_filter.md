# Comparison: runLmbFilter.m → filter.rs

## Overview
- **MATLAB**: `../multisensor-lmb-filters/lmb/runLmbFilter.m` (102 lines)
- **Rust**: `src/lmb/filter.rs` (184 lines for run_lmb_filter(), excluding tests)
- **Purpose**: Main LMB filter loop - orchestrates prediction, association, update, and cardinality estimation

## Line-by-Line Mapping

### Function Signature & Documentation

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 1 | `function [rng, stateEstimates] = runLmbFilter(rng, model, measurements)` | 46-50 | `pub fn run_lmb_filter(rng: &mut impl crate::common::rng::Rng, model: &Model, measurements: &[Vec<DVector<f64>>]) -> LmbStateEstimates` | Function signature. Rust returns single struct |
| 2 | `% RUNLMBFILTER -- Run the LMB filter for a given simulated scenario.` | 28-30 | `/// Run the LMB filter` | Doc title |
| 3 | `%   [rng, stateEstimates] = runLmbFilter(rng, model, measurements)` | - | - | Usage example |
| 4 | `%` | - | - | Blank |
| 5 | `%   Determine the objects' state estimates using the LMB filter.` | 30 | `/// Determines the objects' state estimates using the LMB filter.` | Purpose |
| 6 | `%` | - | - | Blank |
| 7 | `%   See also generateModel, generateGroundTruth, lmbPredictionStep,` | - | - | See also |
| 8 | `%   generateLmbAssociationMatrices, loopyBeliefPropagation, lmbGibbsSampling,` | - | - | See also continued |
| 9 | `%   lmbMurtysAlgorithm, computePosteriorLmbSpatialDistributions, lmbMapCardinalityEstimate` | - | - | See also continued |
| 10 | `%` | - | - | Blank |
| 11 | `%   Inputs` | 32 | `/// # Arguments` | Args header |
| 12 | `%       rng - SimpleRng object. Random number generator (for Gibbs sampling).` | 33 | `/// * \`rng\` - Random number generator` | Input doc |
| 13 | `%       model - struct. A struct with the fields declared in generateModel.` | 34 | `/// * \`model\` - Model parameters` | Input doc |
| 14 | `%       measurements - cell array. An array containing the measurements for` | 35 | `/// * \`measurements\` - Measurements for each time-step` | Input doc |
| 15 | `%           each time-step of the simulation. See also generateModel.` | - | - | Doc continued |
| 16 | `%` | - | - | Blank |
| 17 | `%   Output` | 37 | `/// # Returns` | Returns header |
| 18 | `%       rng - SimpleRng object. Updated random number generator state.` | - | - | Output (RNG mutation handled by Rust &mut) |
| 19 | `%       stateEstimates - struct. A struct containing the LMB filter's` | 38 | `/// LmbStateEstimates containing MAP estimates and trajectories` | Output doc |
| 20 | `%           approximate MAP estimate for each time-step of the simulation, as` | - | - | Doc continued |
| 21 | `%           well as the objects' trajectories.` | - | - | Doc continued |
| 22 | (blank) | - | - | Whitespace |

### Variable Initialization

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 23 | `%% Initialise variables` | 53 | `// Initialize` | Section comment |
| 24 | `simulationLength = length(measurements);` | 51 | `let simulation_length = measurements.len();` | Get length |
| 25 | `% Struct containing objects' Bernoulli parameters and metadata` | - | - | Comment |
| 26 | `objects = model.object;` | 54 | `let mut objects = model.object.clone();` | Initialize objects. **MATLAB uses model.object, Rust clones** |
| 27 | `% Output struct` | - | - | Comment |
| 28 | `stateEstimates.labels = cell(simulationLength, 1);` | 55 | `let mut labels = Vec::with_capacity(simulation_length);` | Init labels |
| 29 | `stateEstimates.mu = cell(simulationLength, 1);` | 56 | `let mut mu_estimates = Vec::with_capacity(simulation_length);` | Init mu |
| 30 | `stateEstimates.Sigma = cell(simulationLength, 1);` | 57 | `let mut sigma_estimates = Vec::with_capacity(simulation_length);` | Init Sigma |
| 31 | `stateEstimates.objects = objects;` | 58 | `let mut all_objects = Vec::new();` | Init objects storage |

### Progress Display (MATLAB only)

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 32 | `%% Run the LMB filter` | 60 | `// Run filter` | Section comment |
| 33 | `showProgress = (simulationLength >= 1);  % Only show progress for long simulations` | - | - | **Not in Rust** (console output) |
| 34 | `fprintf(' (%d)', simulationLength);` | - | - | **Not in Rust** |
| 35 | `fflush(stdout);` | - | - | **Not in Rust** |
| 36 | (blank) | - | - | Whitespace |

### Main Loop

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 37 | `for t = 1:simulationLength` | 61 | `for t in 0..simulation_length {` | Main loop. **MATLAB 1-indexed, Rust 0-indexed** |
| 38 | `    % Show progress every 1 timesteps (LMBM is slow, so show more frequently)` | - | - | **Not in Rust** |
| 39 | `    fprintf(' %d', t);` | - | - | **Not in Rust** |
| 40 | `    fflush(stdout);` | - | - | **Not in Rust** |
| 41 | (blank) | - | - | Whitespace |

### Prediction Step

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 42 | `    %% Prediction` | 62 | `// Prediction` | Section comment |
| 43 | `    objects = lmbPredictionStep(objects, model, t);` | 63 | `objects = lmb_prediction_step(objects, model, t + 1);` | Call prediction. **t+1 to match MATLAB 1-indexed time** |

### Measurement Update (with measurements)

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 44 | `    %% Measurement update` | 65 | `// Measurement update` | Section comment |
| 45 | `    if (numel(measurements{t}))` | 66 | `if !measurements[t].is_empty() {` | Check for measurements. **MATLAB {t}, Rust [t]** |
| 46 | `        % Populate the association matrices required by the data association algorithms` | 67 | `// Generate association matrices` | Comment |
| 47 | `        [associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(objects, measurements{t}, model);` | 68-69 | `let association_result = generate_lmb_association_matrices(&objects, &measurements[t], model);` | Call association. Rust returns single struct |
| 48 | `        if (strcmp(model.dataAssociationMethod, 'LBP'))` | 72-74 | `let (r, w) = match model.data_association_method { DataAssociationMethod::LBP => { ... }` | LBP branch |
| 49 | `            % Data association by way of loopy belief propagation` | 73 | (in match arm) | Comment |
| 50 | `            [r, W] = loopyBeliefPropagation(associationMatrices, model.lbpConvergenceTolerance, model.maximumNumberOfLbpIterations);` | 74 | `lmb_lbp(&association_result, model.lbp_convergence_tolerance, model.maximum_number_of_lbp_iterations)` | Call LBP |
| 51 | `        elseif(strcmp(model.dataAssociationMethod, 'LBPFixed'))` | 76-77 | `DataAssociationMethod::LBPFixed => { ... }` | LBPFixed branch |
| 52 | `            [r, W] = fixedLoopyBeliefPropagation(associationMatrices, model.maximumNumberOfLbpIterations);` | 77 | `lmb_lbp_fixed(&association_result, model.maximum_number_of_lbp_iterations)` | Call fixed LBP |
| 53 | `        elseif(strcmp(model.dataAssociationMethod, 'Gibbs'))` | 79-80 | `DataAssociationMethod::Gibbs => { ... }` | Gibbs branch |
| 54 | `            % Data association by way of Gibbs sampling` | 79 | (in match arm) | Comment |
| 55 | `            [rng, r, W] = lmbGibbsSampling(rng, associationMatrices, model.numberOfSamples);` | 80 | `lmb_gibbs(rng, &association_result, model.number_of_samples)` | Call Gibbs. **Rust mutates rng via &mut** |
| 56 | `        else` | 82-84 | `DataAssociationMethod::Murty => { ... }` | Murty branch |
| 57 | `            % Data association by way of Murty's algorithm` | 82 | (in match arm) | Comment |
| 58 | `            [r, W] = lmbMurtysAlgorithm(associationMatrices, model.numberOfAssignments);` | 83-84 | `let (r, w, _v) = lmb_murtys(&association_result, model.number_of_assignments); (r, w)` | Call Murty. **Rust discards 3rd output** |
| 59 | `        end` | 86 | `};` | End match |
| 60 | `        % Compute posterior spatial distributions` | 88 | `// Compute posterior spatial distributions` | Comment |
| 61 | `        objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);` | 89-95 | `objects = compute_posterior_lmb_spatial_distributions(objects, &r, &w, &association_result.posterior_parameters, model);` | Call update |

### Measurement Update (no measurements)

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 62 | `    else` | 96 | `} else {` | No measurements branch |
| 63 | `        % No measurements collected` | 97 | `// No measurements` | Comment |
| 64 | `        for i = 1:numel(objects)` | 98 | `objects = update_no_measurements(objects, model.detection_probability);` | **Rust uses helper function** |
| 65 | `            objects(i).r = (objects(i).r * (1-model.detectionProbability)) / (1 - objects(i).r * model.detectionProbability);` | (in update.rs:108-111) | `obj.r = (obj.r * (1.0 - detection_probability)) / (1.0 - obj.r * detection_probability);` | Update existence probability |
| 66 | `        end` | 98 | (in helper) | End loop |
| 67 | `    end` | 99 | `}` | End if |

### Gate Tracks

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 68 | `    %% Gate tracks` | 101 | `// Gate tracks` | Section comment |
| 69 | `    % Determine which objects have high existence probabilities` | - | - | Comment |
| 70 | `    objectsLikelyToExist = [objects.r] > model.existenceThreshold;` | 102-105 | `let objects_likely_to_exist = gate_objects_by_existence(&objects.iter().map(\|obj\| obj.r).collect::<Vec<_>>(), model.existence_threshold);` | Threshold check. **Rust uses helper** |
| 71 | `    % Objects with low existence probabilities and long trajectories are worth exporting` | 107 | `// Extract discarded objects with long trajectories` | Comment |
| 72 | `    discardedObjects = objects(~objectsLikelyToExist & ([objects.trajectoryLength] > model.minimumTrajectoryLength));` | 108-114 | `for (i, obj) in objects.iter().enumerate() { if !objects_likely_to_exist[i] && obj.trajectory_length > model.minimum_trajectory_length { all_objects.push(obj.clone()); } }` | Extract discarded with long trajectories |
| 73 | `    stateEstimates.objects(end+1:end+numel(discardedObjects)) =  discardedObjects;` | 112 | `all_objects.push(obj.clone());` | Store discarded |
| 74 | `    % Keep objects with high existence probabilities` | 116 | `// Keep only likely objects` | Comment |
| 75 | `    objects = objects(objectsLikelyToExist);` | 117-127 | `objects = objects.into_iter().enumerate().filter_map(\|(i, obj)\| { if objects_likely_to_exist[i] { Some(obj) } else { None } }).collect();` | Filter objects |

### MAP Cardinality Extraction

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 76 | `    %% MAP cardinality extraction` | 129 | `// MAP cardinality extraction` | Section comment |
| 77 | `    % Determine approximate MAP estimate of the posterior LMB` | - | - | Comment |
| 78 | `    [nMap, mapIndices] = lmbMapCardinalityEstimate([objects.r]);` | 130-131 | `let existence_probs: Vec<f64> = objects.iter().map(\|obj\| obj.r).collect(); let (n_map, map_indices) = lmb_map_cardinality_estimate(&existence_probs);` | Call cardinality estimate |
| 79 | `    % Extract RFS state estimate` | 133 | `// Extract RFS state estimate` | Comment |
| 80 | `    stateEstimates.labels{t} = zeros(2, nMap);` | 134 | `let mut labels_t = DMatrix::zeros(2, n_map);` | Init labels |
| 81 | `    stateEstimates.mu{t} = cell(1, nMap);` | 135 | `let mut mu_t = Vec::with_capacity(n_map);` | Init mu |
| 82 | `    stateEstimates.Sigma{t} = cell(1, nMap);` | 136 | `let mut sigma_t = Vec::with_capacity(n_map);` | Init Sigma |
| 83 | `    for i = 1:nMap` | 138 | `for (i, &j) in map_indices.iter().enumerate() {` | Loop over MAP objects |
| 84 | `        j = mapIndices(i);` | 138 | (in for pattern) | Get index |
| 85 | `        % Gaussians in the posterior GM are sorted according to weight` | - | - | Comment |
| 86 | `        stateEstimates.labels{t}(:, i) = [objects(j).birthTime; objects(j).birthLocation];` | 139-140 | `labels_t[(0, i)] = objects[j].birth_time; labels_t[(1, i)] = objects[j].birth_location;` | Set labels |
| 87 | `        stateEstimates.mu{t}{i} = objects(j).mu{1};` | 141 | `mu_t.push(objects[j].mu[0].clone());` | Set mu. **MATLAB mu{1}, Rust mu[0]** |
| 88 | `        stateEstimates.Sigma{t}{i} = objects(j).Sigma{1};` | 142 | `sigma_t.push(objects[j].sigma[0].clone());` | Set Sigma |
| 89 | `    end` | 143 | `}` | End loop |

### Store Timestep Results

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| - | (implicit storage in MATLAB) | 145-147 | `labels.push(labels_t); mu_estimates.push(mu_t); sigma_estimates.push(sigma_t);` | **Rust explicitly pushes to output vectors** |

### Update Trajectories

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 90 | `    %% Update each object's trajectory` | 149 | `// Update trajectories` | Section comment |
| 91 | `    for i = 1:numel(objects)` | 150 | `for obj in &mut objects {` | Loop over objects |
| 92 | `        j = objects(i).trajectoryLength;` | 151 | `let j = obj.trajectory_length;` | Get current length |
| 93 | `        objects(i).trajectoryLength = j + 1;` | 152 | `obj.trajectory_length = j + 1;` | Increment length |
| 94 | `        objects(i).trajectory(:, j+1) = objects(i).mu{1};` | 154-162 | See below | Add to trajectory |

**Line 94 expansion (trajectory resize logic):**

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 94a | `trajectory(:, j+1) = mu{1}` | 155-160 | `if obj.trajectory.ncols() < j + 1 { let mut new_traj = DMatrix::zeros(...); new_traj.view_mut(...).copy_from(&obj.trajectory); obj.trajectory = new_traj; }` | **Rust needs explicit resize** |
| 94b | (same line) | 162 | `obj.trajectory.column_mut(j).copy_from(&obj.mu[0]);` | Copy mu to trajectory column |

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 95 | `        objects(i).timestamps(j+1) = t;` | 164-167 | `if obj.timestamps.len() <= j { obj.timestamps.resize(j + 1, 0); } obj.timestamps[j] = t + 1;` | Set timestamp. **Rust needs resize, t+1 for 1-indexed time** |
| 96 | `    end` | 168 | `}` | End loop |
| 97 | `end` | 169 | `}` | End main loop |

### Extract Remaining Trajectories

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 98 | `%% Get any long trajectories that weren't extracted` | 171 | `// Get any long trajectories that weren't extracted` | Section comment |
| 99 | `discardedObjects = objects(([objects.trajectoryLength] > model.minimumTrajectoryLength));` | 172-175 | `for obj in &objects { if obj.trajectory_length > model.minimum_trajectory_length { all_objects.push(obj.clone()); } }` | Extract remaining long trajectories |
| 100 | `numberOfDiscardedObjects = numel(discardedObjects);` | - | - | **Not needed in Rust** |
| 101 | `stateEstimates.objects(end+1:end+numberOfDiscardedObjects) =  discardedObjects;` | 174 | `all_objects.push(obj.clone());` | Store objects |
| 102 | `end` | 178-183 | `LmbStateEstimates { labels, mu: mu_estimates, sigma: sigma_estimates, objects: all_objects }` | Return result struct |

---

## Key Translation Notes

1. **Time Indexing**:
   - MATLAB: `t = 1:simulationLength` (1-indexed)
   - Rust: `t in 0..simulation_length` (0-indexed)
   - When passing to prediction: Rust uses `t + 1` to match MATLAB's 1-indexed time

2. **Data Association Dispatch**:
   - MATLAB: `if strcmp(model.dataAssociationMethod, 'LBP')` string comparison
   - Rust: `match model.data_association_method { DataAssociationMethod::LBP => ... }` enum match

3. **No-Measurements Update**:
   - MATLAB: Inline loop (lines 64-66)
   - Rust: Uses `update_no_measurements()` helper function

4. **Array Filtering**:
   - MATLAB: `objects(objectsLikelyToExist)` logical indexing
   - Rust: `objects.into_iter().enumerate().filter_map(...)` iterator chain

5. **Dynamic Array Growth**:
   - MATLAB: `trajectory(:, j+1) = mu{1}` auto-expands
   - Rust: Must explicitly check size and resize matrix

6. **Progress Display**:
   - MATLAB: `fprintf` and `fflush` for console output
   - Rust: Not implemented (no console output)

7. **Return Type**:
   - MATLAB: `[rng, stateEstimates]` returns modified RNG
   - Rust: Returns `LmbStateEstimates`, RNG is mutated via `&mut`
