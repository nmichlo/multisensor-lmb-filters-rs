# Comparison: lmbPredictionStep.m → prediction.rs

## Overview
- **MATLAB**: `../multisensor-lmb-filters/lmb/lmbPredictionStep.m` (33 lines)
- **Rust**: `src/lmb/prediction.rs` (52 lines, excluding tests)
- **Purpose**: LMB filter prediction step using Chapman-Kolmogorov equation

## Line-by-Line Mapping

### Function Signature & Documentation

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 1 | `function objects = lmbPredictionStep(objects, model, t)` | 28 | `pub fn lmb_prediction_step(mut objects: Vec<Object>, model: &Model, t: usize) -> Vec<Object>` | Function signature. MATLAB returns modified `objects`, Rust takes ownership and returns. |
| 2 | `% LMBPREDICTIONSTEP -- Complete the LMB filter's prediction step.` | 8-9 | `/// LMB prediction step` | Doc comment title |
| 3 | `%   objects = lmbPredictionStep(objects, model, t)` | - | - | Usage example (not in Rust) |
| 4 | `%` | - | - | Blank doc line |
| 5 | `%   Computes predicted prior for the current time-step using the` | 10-11 | `/// Computes predicted prior for the current time-step using the Chapman-Kolmogorov` | Purpose description |
| 6 | `%   Chapman-Kolmogorov equation, assuming an LMB prior and the standard` | 11-12 | `/// equation, assuming an LMB prior and the standard multi-object motion model.` | Purpose continued |
| 7 | `%   multi-object motion model.` | 12 | (merged above) | Purpose continued |
| 8 | `%` | - | - | Blank doc line |
| 9 | `%   See also runLmbFilter, generateModel` | - | - | See also (not in Rust) |
| 10 | `%` | - | - | Blank doc line |
| 11 | `%   Inputs` | 14 | `/// # Arguments` | Arguments section header |
| 12 | `%       objects - struct. A struct containing the posterior LMB's Bernoulli components.` | 15 | `/// * \`objects\` - Vector of posterior LMB Bernoulli components from previous time` | objects parameter doc |
| 13 | `%       model - struct. A struct with the fields declared in generateModel.` | 16 | `/// * \`model\` - Model parameters` | model parameter doc |
| 14 | `%       t - integer. An integer representing the simulation's current` | 17 | `/// * \`t\` - Current time-step` | t parameter doc |
| 15 | `%           time-step` | 17 | (merged above) | t parameter continued |
| 16 | `%` | - | - | Blank doc line |
| 17 | `%   Output` | 19 | `/// # Returns` | Returns section header |
| 18 | `%       objects - struct. A struct containing the prior LMB's Bernoulli components.` | 20 | `/// Vector of prior LMB Bernoulli components (surviving + newly born objects)` | Return value doc |
| 19 | (blank) | - | - | Whitespace |

### Motion Model Section

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 20 | `%% Put existing Bernoulli componenents through the motion model` | 29 | `// Put existing Bernoulli components through the motion model` | Section comment |
| 21 | `numberOfObjects = numel(objects);` | - | - | Not needed in Rust (iterator handles length) |
| 22 | `for i = 1:numberOfObjects` | 30 | `for obj in &mut objects {` | Loop over objects. MATLAB 1-indexed, Rust iterator |
| 23 | `    objects(i).r = model.survivalProbability * objects(i).r;` | 32 | `obj.r = model.survival_probability * obj.r;` | Predict existence probability |
| 24 | `    for j = 1:objects(i).numberOfGmComponents` | 35 | `for j in 0..obj.number_of_gm_components {` | Loop over GM components. MATLAB 1-indexed, Rust 0-indexed |
| 25 | `        objects(i).mu{j} = model.A * objects(i).mu{j} + model.u;` | 37 | `obj.mu[j] = &model.a * &obj.mu[j] + &model.u;` | Predict mean. Rust uses references for nalgebra ops |
| 26 | `        objects(i).Sigma{j} = model.A * objects(i).Sigma{j} * model.A' + model.R;` | 40 | `obj.sigma[j] = &model.a * &obj.sigma[j] * model.a.transpose() + &model.r;` | Predict covariance. `A'` → `.transpose()` |
| 27 | `    end` | 41 | `}` | End inner loop |
| 28 | `end` | 42 | `}` | End outer loop |

### Birth Section

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 29 | `%% Add in Bernoulli components for newly appearing objects` | 44 | `// Add Bernoulli components for newly appearing objects` | Section comment |
| 30 | `newNumberOfObjects = numberOfObjects + model.numberOfBirthLocations;` | - | - | Not needed in Rust (push handles allocation) |
| 31 | `objects(numberOfObjects+1:newNumberOfObjects) = model.birthParameters;` | 45-48 | `for birth_obj in &model.birth_parameters { let mut new_obj = birth_obj.clone(); ... objects.push(new_obj); }` | Add birth objects. MATLAB array extension vs Rust loop+push |
| 32 | `[objects(numberOfObjects+1:newNumberOfObjects).birthTime] = deal(t);` | 47 | `new_obj.birth_time = t;` | Set birth time. MATLAB `deal()` vs Rust direct assignment |
| 33 | `end` | 52 | `objects` | End function. Rust returns `objects` implicitly |

## Key Translation Notes

1. **Indexing**: MATLAB uses 1-based indexing (`1:numberOfObjects`), Rust uses 0-based (`0..obj.number_of_gm_components`)

2. **Matrix Operations**:
   - MATLAB: `A'` for transpose
   - Rust: `.transpose()` method

3. **Memory Management**:
   - MATLAB: `objects(n+1:m) = ...` extends array automatically
   - Rust: Uses `.push()` in a loop with `.clone()`

4. **Field Naming**:
   - MATLAB: `camelCase` (`survivalProbability`, `numberOfGmComponents`)
   - Rust: `snake_case` (`survival_probability`, `number_of_gm_components`)

5. **Iteration**:
   - MATLAB: Index-based `for i = 1:n`
   - Rust: Iterator `for obj in &mut objects`
