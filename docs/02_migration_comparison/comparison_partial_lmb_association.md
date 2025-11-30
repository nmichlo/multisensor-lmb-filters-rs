# Comparison: generateLmbAssociationMatrices.m → association.rs

## Overview
- **MATLAB**: `../multisensor-lmb-filters/lmb/generateLmbAssociationMatrices.m` (83 lines)
- **Rust**: `src/lmb/association.rs` (228 lines for generate_lmb_association_matrices(), excluding tests)
- **Purpose**: Compute association matrices for LBP, Gibbs, and Murty's algorithms, plus posterior spatial distribution parameters

## Line-by-Line Mapping

### Function Signature & Documentation

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 1 | `function [associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(objects, z, model)` | 57-61 | `pub fn generate_lmb_association_matrices(objects: &[Object], measurements: &[DVector<f64>], model: &Model) -> LmbAssociationResult` | Function signature |
| 2 | `% GENERATELMBASSOCIATIONMATRICES -- Compute the association matrices...` | 37-56 | `/// Generate LMB association matrices` | Doc title |
| 3 | `%   [associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(...)` | - | - | Usage |
| 4 | `%` | - | - | Blank |
| 5 | `%   This function computes the association matrices required by the LBP,` | 39-41 | `/// Computes the association matrices required by LBP, Gibbs sampler, and Murty's algorithm.` | Purpose |
| 6 | `%   Gibbs sampler, and Murty's algorithms. It also determines...` | 41-42 | `/// Also determines measurement-updated components for posterior spatial distribution.` | Purpose continued |
| 7 | `%   are used to determine each object's posterior spatial distribution.` | - | - | Purpose continued |
| 8 | `%` | - | - | Blank |
| 9 | `%   See also runLmbFilter, generateModel, loopyBeliefPropagation...` | - | - | See also |
| 10 | `%` | - | - | Blank |
| 11 | `%   Inputs` | 44 | `/// # Arguments` | Args header |
| 12 | `%       objects - struct...` | 45 | `/// * \`objects\` - Prior LMB Bernoulli components` | Input doc |
| 13 | `%       z - cell array. A cell array of measurements...` | 46 | `/// * \`measurements\` - Measurements at current time-step` | Input doc |
| 14 | `%           current time-step.` | - | - | Doc continued |
| 15 | `%       model - struct...` | 47 | `/// * \`model\` - Model parameters` | Input doc |
| 16 | `%` | - | - | Blank |
| 17 | `%   Output` | 49 | `/// # Returns` | Returns header |
| 18 | `%       associationMatrices - struct...` | 50 | `/// LmbAssociationResult with all association matrices and posterior parameters` | Output doc |
| 19 | `%           by the various data association algorithms.` | - | - | Doc continued |
| 20 | `%       posteriorParameters - struct...` | - | - | Output doc |
| 21 | `%           posterior spatial distribution parameters.` | - | - | Doc continued |
| 22 | `%` | - | - | Blank |
| 23 | (blank) | - | - | Whitespace |

### Variable Declaration

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 24 | `%% Declare output structs` | - | - | Section comment |
| 25 | `numberOfObjects = numel(objects);` | 62 | `let number_of_objects = objects.len();` | Get object count |
| 26 | `numberOfMeasurements = numel(z);` | 63 | `let number_of_measurements = measurements.len();` | Get measurement count |
| 27 | `% Auxillary matrices` | 65 | `// Auxiliary matrices` | Comment |
| 28 | `L = zeros(numberOfObjects, numberOfMeasurements);` | 66 | `let mut l_matrix = DMatrix::zeros(number_of_objects, number_of_measurements);` | L matrix |
| 29 | `phi = zeros(numberOfObjects, 1);` | 67 | `let mut phi = DVector::zeros(number_of_objects);` | phi vector |
| 30 | `eta = zeros(numberOfObjects, 1);` | 68 | `let mut eta = DVector::zeros(number_of_objects);` | eta vector |
| 31 | `% Updated components for the objects' posterior spatial distributions` | 70 | `// Posterior parameters for each object` | Comment |
| 32 | `posteriorParameters.w = [];` | 71 | `let mut posterior_parameters = Vec::with_capacity(number_of_objects);` | Init posterior params |
| 33 | `posteriorParameters.mu = {};` | - | (in struct) | Init mu |
| 34 | `posteriorParameters.Sigma = {};` | - | (in struct) | Init Sigma |
| 35 | `posteriorParameters = repmat(posteriorParameters, 1, numberOfObjects);` | 71 | (handled by Vec::with_capacity and push) | Replicate struct |

### Main Object Loop

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 36 | `%% Populate the LBP arrays, and compute posterior components` | 73 | `// Populate arrays and compute posterior components` | Section comment |
| 37 | `for i = 1:numberOfObjects` | 74 | `for i in 0..number_of_objects {` | Object loop. **MATLAB 1-indexed, Rust 0-indexed** |

### Initialize Posterior Parameters

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 38 | `    % Predeclare the object's posterior components, and include missed detection event` | 77 | `// Predeclare object's posterior components (m+1 rows, num_components cols)` | Comment |
| 39 | `    posteriorParameters(i).w = repmat(log(objects(i).w * (1 - model.detectionProbability)), numberOfMeasurements + 1, 1);` | 79, 84-85 | `let mut w_log = DMatrix::zeros(number_of_measurements + 1, num_comp); ... w_log[(0, j)] = (obj.w[j] * (1.0 - model.detection_probability)).ln();` | Init log-weights. Rust initializes row 0 (miss) explicitly |
| 40 | `    posteriorParameters(i).mu  = repmat(objects(i).mu, numberOfMeasurements + 1, 1);` | 80, 86 | `let mut mu_posterior = vec![vec![DVector::zeros(model.x_dimension); num_comp]; number_of_measurements + 1]; mu_posterior[0][j] = obj.mu[j].clone();` | Init mu |
| 41 | `    posteriorParameters(i).Sigma = repmat(objects(i).Sigma, numberOfMeasurements + 1, 1);` | 81, 87 | `let mut sigma_posterior = vec![...]; sigma_posterior[0][j] = obj.sigma[j].clone();` | Init Sigma |

### Auxiliary LBP Parameters

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 42 | `    % Populate auxiliary LBP parameters` | 90 | `// Auxiliary LBP parameters` | Comment |
| 43 | `    phi(i) = (1 -  model.detectionProbability) * objects(i).r;` | 91 | `phi[i] = (1.0 - model.detection_probability) * obj.r;` | Compute phi |
| 44 | `    eta(i) = 1 - model.detectionProbability * objects(i).r;` | 92 | `eta[i] = 1.0 - model.detection_probability * obj.r;` | Compute eta |

### GM Component Loop

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 45 | `    %% Determine marginal likelihood ratio of the object generating each measurement` | 94 | `// Determine marginal likelihood ratio for each measurement` | Comment |
| 46 | `    for j = 1:objects(i).numberOfGmComponents` | 95 | `for j in 0..num_comp {` | Component loop. **MATLAB 1-indexed, Rust 0-indexed** |

### Measurement Prediction

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 47 | `        % Update components for a mixture component` | 96 | `// Predicted measurement and innovation covariance` | Comment |
| 48 | `        muZ = model.C * objects(i).mu{j};` | 97 | `let mu_z = &model.c * &obj.mu[j];` | Predicted measurement |
| 49 | `        Z = model.C * objects(i).Sigma{j} * model.C' + model.Q;` | 98 | `let z_cov = &model.c * &obj.sigma[j] * model.c.transpose() + &model.q;` | Innovation covariance. `C'` → `.transpose()` |
| 50 | `        logGaussianNormalisingConstant = - (0.5 * model.zDimension) * log(2 * pi) - 0.5 * log(det(Z));` | 101-102 | `let log_gaussian_norm = -(0.5 * model.z_dimension as f64) * (2.0 * std::f64::consts::PI).ln() - 0.5 * z_cov.determinant().ln();` | Log normalizing constant |
| 51 | `        logLikelihoodRatioTerms = log(objects(i).r) + log(model.detectionProbability) + log(objects(i).w(j)) - log(model.clutterPerUnitVolume);` | 105-108 | `let log_likelihood_ratio_terms = obj.r.ln() + model.detection_probability.ln() + obj.w[j].ln() - model.clutter_per_unit_volume.ln();` | Log-likelihood ratio |

### Kalman Gain and Updated Covariance

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 52 | `        ZInv = inv(Z);` | 111-117 | `let z_inv = match z_cov.clone().cholesky() { Some(chol) => chol.inverse(), None => { continue; } };` | Matrix inverse. **Rust uses Cholesky decomposition with fallback** |
| 53 | `        K = objects(i).Sigma{j} * model.C' * ZInv;` | 119 | `let k = &obj.sigma[j] * model.c.transpose() * &z_inv;` | Kalman gain |
| 54 | `        SigmaUpdated = (eye(model.xDimension) - K * model.C) * objects(i).Sigma{j};` | 120-121 | `let sigma_updated = (DMatrix::identity(model.x_dimension, model.x_dimension) - &k * &model.c) * &obj.sigma[j];` | Updated covariance |

### Measurement Loop

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 55 | `        % Determine total marginal likelihood, and determine posterior components` | 123 | `// Process each measurement` | Comment |
| 56 | `        for k = 1:numberOfMeasurements` | 124 | `for (meas_idx, z) in measurements.iter().enumerate() {` | Measurement loop. Rust uses enumerate |
| 57 | `            % Determine marginal likelihood ratio` | 125 | `// Innovation` | Comment |
| 58 | `            nu = z{k} - muZ;` | 126 | `let nu = z - &mu_z;` | Innovation vector |
| 59 | `            gaussianLogLikelihood = logGaussianNormalisingConstant - 0.5 * nu' * ZInv * nu;` | 129 | `let gaussian_log_likelihood = log_gaussian_norm - 0.5 * nu.dot(&(&z_inv * &nu));` | Gaussian log-likelihood. `nu' * M * nu` → `nu.dot(&(M * &nu))` |
| 60 | `            L(i, k) = L(i, k) + exp(logLikelihoodRatioTerms + gaussianLogLikelihood);` | 132 | `l_matrix[(i, meas_idx)] += (log_likelihood_ratio_terms + gaussian_log_likelihood).exp();` | Update L matrix |
| 61 | `            % Determine updated mean and covariance for each mixture component` | 134 | `// Posterior component parameters` | Comment |
| 62 | `            posteriorParameters(i).w(k+1, j) = log(objects(i).w(j)) + gaussianLogLikelihood + log(model.detectionProbability) - log(model.clutterPerUnitVolume);` | 135-138 | `w_log[(meas_idx + 1, j)] = obj.w[j].ln() + gaussian_log_likelihood + model.detection_probability.ln() - model.clutter_per_unit_volume.ln();` | Posterior weight. **k+1 (MATLAB) → meas_idx+1 (Rust)** |
| 63 | `            posteriorParameters(i).mu{k+1, j} = objects(i).mu{j} + K * nu;` | 140 | `mu_posterior[meas_idx + 1][j] = &obj.mu[j] + &k * &nu;` | Posterior mean |
| 64 | `            posteriorParameters(i).Sigma{k+1, j} = SigmaUpdated;` | 141 | `sigma_posterior[meas_idx + 1][j] = sigma_updated.clone();` | Posterior covariance |
| 65 | `        end` | 142 | `}` | End measurement loop |
| 66 | `    end` | 143 | `}` | End component loop |

### Weight Normalization

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 67 | `    % Normalise weights` | 145 | `// Normalize weights using log-sum-exp` | Comment |
| 68 | `    maximumWeights = max(posteriorParameters(i).w, [], 2);` | 147 | `let max_w = (0..num_comp).map(\|j\| w_log[(row, j)]).fold(f64::NEG_INFINITY, f64::max);` | Row-wise max |
| 69 | `    offsetWeights = posteriorParameters(i).w - maximumWeights;` | 148, 151 | `let sum_exp: f64 = (0..num_comp).map(\|j\| (w_log[(row, j)] - max_w).exp()).sum();` | Offset for numerical stability |
| 70 | `    posteriorParameters(i).w = exp(offsetWeights) ./ sum(exp(offsetWeights), 2);` | 150-152 | `w_log[(row, j)] = ((w_log[(row, j)] - max_w).exp() / sum_exp).ln();` | Normalize (keep in log space) |
| 71 | `end` | 168 | `}` | End object loop |

### Convert Log-Weights to Linear (Rust only)

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| - | (implicit in MATLAB) | 156-161 | `let mut w_normalized = DMatrix::zeros(...); for row in ... { for j in ... { w_normalized[(row, j)] = w_log[(row, j)].exp(); } }` | **Rust stores in log, converts at end** |

### Build Association Matrices

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 72 | `%% Output association matrices` | 170 | `// Build association matrices` | Section comment |
| 73 | `% LBP association matrices` | 172 | `// LBP matrices: Psi = L ./ eta (broadcast division)` | Comment |
| 74 | `associationMatrices.Psi = L ./ eta;` | 173-178 | `let mut psi = DMatrix::zeros(...); for i ... { for j ... { psi[(i, j)] = l_matrix[(i, j)] / eta[i]; } }` | Psi = L/eta. Rust explicit loops for broadcast |
| 75 | `associationMatrices.phi = phi;` | 184 | `let lbp = AssociationMatrices { psi, phi, eta };` | Store phi |
| 76 | `associationMatrices.eta = eta;` | 184 | (merged above) | Store eta |
| 77 | `% Gibbs sampler association matrices` | 186 | `// Gibbs matrices: P = L ./ (L + eta) (broadcast division)` | Comment |
| 78 | `associationMatrices.P = L./ (L + eta);` | 187-192 | `let mut p = DMatrix::zeros(...); for i ... { for j ... { p[(i, j)] = l_matrix[(i, j)] / (l_matrix[(i, j)] + eta_gibbs[i]); } }` | P matrix |
| 79 | `associationMatrices.L = [eta L];` | 195-198 | `let mut l_gibbs = DMatrix::zeros(n, m+1); l_gibbs.column_mut(0).copy_from(&eta_gibbs); l_gibbs.view_mut((0, 1), (n, m)).copy_from(&l_matrix);` | L = [eta, L]. Column concatenation |
| 80 | `associationMatrices.R = [(phi ./ eta) ones(numberOfObjects, numberOfMeasurements)];` | 201-207 | `let mut r_gibbs = DMatrix::zeros(n, m+1); for i ... { r_gibbs[(i, 0)] = phi_gibbs[i] / eta_gibbs[i]; for j in 1..=m { r_gibbs[(i, j)] = 1.0; } }` | R matrix. First col = phi/eta, rest = 1 |
| 81 | `% Murty's algorithm association matrices` | 216 | `// Murty's cost matrix: C = -log(L)` | Comment |
| 82 | `associationMatrices.C = -log(L);` | 220 | `let cost = l_matrix.map(\|val\| -val.ln());` | Cost matrix |
| 83 | `end` | 222-227 | `LmbAssociationResult { lbp, gibbs, cost, posterior_parameters }` | Return result struct |

---

## Key Translation Notes

1. **Matrix Broadcasting**:
   - MATLAB: `L ./ eta` broadcasts eta across columns
   - Rust: Explicit nested loops required

2. **Matrix Inverse**:
   - MATLAB: `inv(Z)` direct inverse
   - Rust: Uses Cholesky decomposition with `continue` on failure (more numerically stable)

3. **Quadratic Form**:
   - MATLAB: `nu' * ZInv * nu` (row vector × matrix × column vector)
   - Rust: `nu.dot(&(z_inv * &nu))` using dot product

4. **Log-Space Arithmetic**:
   - Both use log-sum-exp trick for numerical stability
   - Rust explicitly converts back to linear space at the end

5. **Cell Array vs Vec<Vec>**:
   - MATLAB: `posteriorParameters(i).mu{k+1, j}` cell array indexing
   - Rust: `mu_posterior[meas_idx + 1][j]` nested Vec indexing

6. **Matrix Concatenation**:
   - MATLAB: `[eta L]` horizontal concatenation
   - Rust: Create matrix, copy columns using `column_mut()` and `view_mut()`

7. **Return Type**:
   - MATLAB: Returns two separate structs
   - Rust: Returns single `LmbAssociationResult` containing all outputs
