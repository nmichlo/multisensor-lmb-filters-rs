# Comparison: computePosteriorLmbSpatialDistributions.m → update.rs

## Overview
- **MATLAB**: `../multisensor-lmb-filters/lmb/computePosteriorLmbSpatialDistributions.m` (52 lines)
- **Rust**: `src/lmb/update.rs` (92 lines for compute_posterior_lmb_spatial_distributions(), excluding tests)
- **Purpose**: Compute posterior spatial distributions for LMB filter measurement update

## Line-by-Line Mapping

### Function Signature & Documentation

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 1 | `function objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model)` | 34-40 | `pub fn compute_posterior_lmb_spatial_distributions(mut objects: Vec<Object>, r: &DVector<f64>, w: &DMatrix<f64>, posterior_parameters: &[PosteriorParameters], model: &Model) -> Vec<Object>` | Function signature |
| 2 | `% COMPUTEPOSTERIORLMBSPATIALDISTRIUBUTIONS -- Complete the LMB filter's measurement update` | 11-14 | `/// Compute posterior LMB spatial distributions` | Doc title |
| 3 | `%    objects = computePosteriorLmbSpatialDistributions(...)` | - | - | Usage example |
| 4 | `%` | - | - | Blank |
| 5 | `%   This function computes each object's posterior spatial distrubtion.` | 14 | `/// Completes the LMB filter's measurement update by computing each object's posterior spatial distribution.` | Purpose |
| 6 | `%` | - | - | Blank |
| 7 | `%   See also generateModel, runLmbFilter, lmbPredictionStep,` | - | - | See also |
| 8 | `%            loopyBeliefPropagation, generateLmbAssociationMatrices` | - | - | See also continued |
| 9 | `%` | - | - | Blank |
| 10 | `%   Inputs` | 16 | `/// # Arguments` | Args header |
| 11 | `%       objects - struct...` | 17 | `/// * \`objects\` - Prior LMB Bernoulli components` | Input doc |
| 12 | `%           components. This struct is produced by lmbPredictionStep.` | - | - | Doc continued |
| 13 | `%       r - array. Each object's posterior existence probability.` | 18 | `/// * \`r\` - Posterior existence probabilities (n x 1)` | Input doc |
| 14 | `%       W - array. An array of marginal association probabilities...` | 19 | `/// * \`w\` - Marginal association probabilities (n x (m+1))` | Input doc |
| 15 | `%           each row is an object's marginal association probabilities.` | - | - | Doc continued |
| 16 | `%       posteriorParameters - struct...` | 20 | `/// * \`posterior_parameters\` - Posterior parameters from association step` | Input doc |
| 17 | `%           posterior spatial distribution parameters.` | - | - | Doc continued |
| 18 | `%       model - struct...` | 21 | `/// * \`model\` - Model parameters` | Input doc |
| 19 | `%` | - | - | Blank |
| 20 | `%   Output` | 23 | `/// # Returns` | Returns header |
| 21 | `%       objects - struct. A struct containing the posterior LMB's Bernoulli` | 24 | `/// Updated objects with posterior spatial distributions` | Output doc |
| 22 | `%           components.` | - | - | Doc continued |
| 23 | (blank) | - | - | Whitespace |

### Main Loop

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 24 | `for i = 1:numel(objects)` | 41 | `for i in 0..objects.len() {` | Loop over objects. **MATLAB 1-indexed, Rust 0-indexed** |

### Update Existence Probability

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 25 | `    %% Update posterior existence probability` | 42 | `// Update posterior existence probability` | Section comment |
| 26 | `    objects(i).r = r(i);` | 43 | `objects[i].r = r[i];` | Direct assignment |

### Reweight Gaussian Mixtures

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 27 | `    %% Reweight each the measurement-updated Gaussian mixtures...` | 45 | `// Reweight measurement-updated Gaussian mixtures` | Section comment |
| 28 | `    numberOfPosteriorComponents = numel(posteriorParameters(i).w);` | 46-47 | `let num_posterior_components = posterior_parameters[i].w.ncols(); let num_meas_plus_one = posterior_parameters[i].w.nrows();` | Get dimensions. MATLAB single numel, Rust needs both dims |
| 29 | `    posteriorWeights = reshape(W(i, :)' .* posteriorParameters(i).w, 1, numberOfPosteriorComponents);` | 48-59 | See below | Compute posterior weights |

**Line 29 expansion (critical column-major ordering):**

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 29a | `W(i, :)'` | 56 | `w[(i, meas_idx)]` | Get association weights for object i |
| 29b | `.* posteriorParameters(i).w` | 56 | `* posterior_parameters[i].w[(meas_idx, comp_idx)]` | Element-wise multiply |
| 29c | `reshape(..., 1, n)` | 51-58 | `// IMPORTANT: MATLAB uses COLUMN-MAJOR ordering when reshaping! for comp_idx in 0..num_posterior_components { for meas_idx in 0..num_meas_plus_one { posterior_weights.push(w[(i, meas_idx)] * posterior_parameters[i].w[(meas_idx, comp_idx)]); } }` | **CRITICAL: Column-major loop order (comp outer, meas inner)** |

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 30 | `    posteriorWeights = posteriorWeights ./ sum(posteriorWeights);` | 62-66 | `let sum: f64 = posterior_weights.iter().sum(); if sum > 1e-15 { for weight in &mut posterior_weights { *weight /= sum; } }` | Normalize weights |

### Mixture Reduction

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 31 | `    %% Crude mixture reduction algorithm` | 69 | `// Crude mixture reduction algorithm` | Section comment |
| 32 | `    % Sort the weights` | - | - | Comment |
| 33 | `    [posteriorWeights, sortedIndices] = sort(posteriorWeights, 'descend');` | 70-71 | `let pruned = prune_gaussian_mixture(&posterior_weights, model.gm_weight_threshold, model.maximum_number_of_gm_components);` | **Rust uses helper function that combines sort, threshold, and cap** |
| 34 | `    % Discard insignificant components` | - | - | Comment |
| 35 | `    significantComponents = posteriorWeights > model.gmWeightThreshold;` | 70-71 | (in prune_gaussian_mixture) | Threshold check |
| 36 | `    significantWeights = posteriorWeights(significantComponents);` | 70-71 | (in prune_gaussian_mixture) | Filter weights |
| 37 | `    objects(i).w = significantWeights ./ sum(significantWeights);` | 73-74 | `objects[i].number_of_gm_components = pruned.num_components; objects[i].w = pruned.weights.clone();` | Assign normalized weights |
| 38 | `    sortedIndices = sortedIndices(significantComponents);` | 70-71 | (in prune_gaussian_mixture) | Filter indices |
| 39 | `    objects(i).numberOfGmComponents = numel(objects(i).w);` | 73 | (merged above) | Set component count |

### Hard Limit Check

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 40 | `    % Impose hard limit if there are too many components` | - | - | Comment |
| 41 | `    if (objects(i).numberOfGmComponents > model.maximumNumberOfGmComponents)` | 70-71 | (handled in prune_gaussian_mixture) | **Rust helper handles this internally** |
| 42 | `        objects(i).w = objects(i).w(1:model.maximumNumberOfGmComponents);` | 70-71 | (in prune_gaussian_mixture) | Take first N |
| 43 | `        objects(i).w = objects(i).w ./ sum(objects(i).w);` | 70-71 | (in prune_gaussian_mixture) | Renormalize |
| 44 | `        sortedIndices = sortedIndices(1:model.maximumNumberOfGmComponents);` | 70-71 | (in prune_gaussian_mixture) | Take first N indices |
| 45 | `        objects(i).numberOfGmComponents = model.maximumNumberOfGmComponents;` | 73 | (in pruned.num_components) | Cap count |
| 46 | `    end` | - | - | End if |

### Select Mixture Components

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 47 | `    %% Select the mixture components with the largest weights` | 76 | `// Extract corresponding mu and sigma using sorted indices` | Section comment |
| 48 | `    objects(i).mu = reshape(posteriorParameters(i).mu(sortedIndices), 1, objects(i).numberOfGmComponents);` | 77-88 | See below | Select mu components |

**Line 48 expansion (critical column-major indexing):**

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 48a | `posteriorParameters(i).mu(sortedIndices)` | 80-87 | `for &original_idx in &pruned.indices { let comp_idx = original_idx / num_meas_plus_one; let meas_idx = original_idx % num_meas_plus_one; objects[i].mu.push(posterior_parameters[i].mu[meas_idx][comp_idx].clone()); }` | **Column-major index conversion: flat_idx = meas_idx + comp_idx * num_rows** |

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 49 | `    objects(i).Sigma = reshape(posteriorParameters(i).Sigma(sortedIndices), 1, objects(i).numberOfGmComponents);` | 87 | `objects[i].sigma.push(posterior_parameters[i].sigma[meas_idx][comp_idx].clone());` | Select sigma components (same index conversion) |
| 50 | `end` | 89 | `}` | End main loop |
| 51 | (blank) | - | - | Whitespace |
| 52 | `end` | 91 | `objects` | Return objects |

---

## Key Translation Notes

1. **Column-Major Ordering (CRITICAL)**:
   - MATLAB `reshape()` uses column-major order
   - When flattening `posteriorParameters(i).w` (m+1 x n matrix):
     - MATLAB reads column-by-column
     - Rust must iterate: outer loop over components, inner loop over measurements
   - Index conversion: `flat_idx = meas_idx + comp_idx * num_rows`

2. **Helper Function**:
   - MATLAB: Inline sort, threshold, and cap logic (lines 33-46)
   - Rust: Uses `prune_gaussian_mixture()` helper that encapsulates all three operations

3. **Weight Normalization**:
   - Both normalize weights after filtering and after capping
   - Rust adds `sum > 1e-15` check to avoid division by near-zero

4. **Index Conversion (Column-Major to 2D)**:
   ```rust
   // Column-major: flat_idx = row + col * num_rows
   let comp_idx = original_idx / num_meas_plus_one;  // col
   let meas_idx = original_idx % num_meas_plus_one;  // row
   ```

5. **Field Naming**:
   - MATLAB: `numberOfGmComponents`, `gmWeightThreshold`
   - Rust: `number_of_gm_components`, `gm_weight_threshold`
