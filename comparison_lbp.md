# Comparison: loopyBeliefPropagation.m → lbp.rs

## Overview
- **MATLAB**: `../multisensor-lmb-filters/common/loopyBeliefPropagation.m` (48 lines)
- **Rust**: `src/common/association/lbp.rs` (138 lines for loopy_belief_propagation(), excluding tests)
- **Purpose**: Loopy Belief Propagation for data association - computes posterior existence probabilities and marginal association weights

## Line-by-Line Mapping

### Function Signature & Documentation

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 1 | `function [r, W] = loopyBeliefPropagation(associationMatrices, epsilon, maximumNumberOfLbpIterations)` | 42-46 | `pub fn loopy_belief_propagation(matrices: &AssociationMatrices, epsilon: f64, max_iterations: usize) -> LbpResult` | Function signature. MATLAB returns two vars, Rust returns struct |
| 2 | `% LOOPYBELIEFPROPAGATION -- Determine posterior existence probabilities...` | 30-33 | `/// Loopy Belief Propagation with convergence check` | Doc title |
| 3 | `%   [r, W] = loopyBeliefPropagation(...)` | - | - | Usage example |
| 4 | `%` | - | - | Blank |
| 5 | `%   This function determines each object's posterior existence and marginal` | 32-33 | `/// Determines posterior existence probabilities and marginal association` | Purpose |
| 6 | `%   association probabilities using loopy belief propagation (LBP).` | 34 | `/// probabilities using loopy belief propagation.` | Purpose continued |
| 7 | `%` | - | - | Blank |
| 8 | `%   See also runLmbFilter, generateLmbAssociationMatrices,` | - | - | See also |
| 9 | `%   computePosteriorLmbSpatialDistributions, lmbGibbsSampling,` | - | - | See also continued |
| 10 | `%   lmbMurtysAlgorithm` | - | - | See also continued |
| 11 | `%` | - | - | Blank |
| 12 | `%   Inputs` | 36 | `/// # Arguments` | Args header |
| 13 | `%       associationMatrices - struct...` | 37 | `/// * \`matrices\` - Association matrices (Psi, phi, eta)` | Input doc |
| 14 | `%           by the various data association algorithms.` | - | - | Doc continued |
| 15 | `%       epsilon - double. The convergence tolerance...` | 38 | `/// * \`epsilon\` - Convergence tolerance` | Input doc |
| 16 | `%       maximumNumberOfLbpIterations - integer...` | 39 | `/// * \`max_iterations\` - Maximum number of LBP iterations` | Input doc |
| 17 | `%           of LBP iterations.` | - | - | Doc continued |
| 18 | `%` | - | - | Blank |
| 19 | `%   Output` | 41 | `/// # Returns` | Returns header |
| 20 | `%       r - (n, 1) array. Each object's posterior existence probability.` | 42 | `/// LbpResult with posterior existence probabilities and association weights` | Output doc |
| 21 | `%       W - (n, m) array. An array of marginal association probabilities...` | 42 | (merged above) | Output doc |
| 22 | `%           each row is an object's marginal association probabilities.` | - | - | Doc continued |
| 23 | (blank) | - | - | Whitespace |

### Variable Initialization

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 24 | `%% Declare variables` | - | - | Section comment |
| 25 | `SigmaMT = ones(size(associationMatrices.Psi));` | 51 | `let mut sigma_mt = DMatrix::from_element(n_objects, n_measurements, 1.0);` | Init messages to 1. MATLAB ones(), Rust from_element |
| 26 | `notConverged = true;` | 52 | `let mut not_converged = true;` | Convergence flag |
| 27 | `counter = 0;` | 53 | `let mut counter = 0;` | Iteration counter |

### Message Passing Loop

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 28 | `%% Loopy belief propagation` | - | - | Section comment |
| 29 | `while notConverged` | 56 | `while not_converged {` | Main loop |
| 30 | `    % Cache previous iteration's messages` | 57 | `// Cache previous iteration's messages` | Comment |
| 31 | `    SigmaMTOld = SigmaMT;` | 58 | `let sigma_mt_old = sigma_mt.clone();` | Cache old values |
| 32 | `    % Pass messages from the object to the measurement clusters` | 60 | `// Pass messages from object to measurement clusters` | Comment |
| 33 | `    B = associationMatrices.Psi .* SigmaMT;` | 61 | `let b = matrices.psi.component_mul(&sigma_mt);` | Element-wise multiply. MATLAB `.*`, Rust `.component_mul()` |
| 34 | `    SigmaTM = associationMatrices.Psi ./ (-B + sum(B, 2) + 1);` | 64-75 | See below | Object→measurement messages |

**Line 34 expansion:**

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 34a | `sum(B, 2)` | 66 | `let row_sum: f64 = b.row(i).sum();` | Row-wise sum |
| 34b | `-B + sum(B, 2) + 1` | 68 | `let denom = -b[(i, j)] + row_sum + 1.0;` | Denominator |
| 34c | `Psi ./ (...)` | 70-73 | `sigma_tm[(i, j)] = if denom.abs() > 1e-15 { matrices.psi[(i, j)] / denom } else { 0.0 };` | Division with safety check |

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 35 | `    % Pass messages from the measurement to the object clusters` | 77 | `// Pass messages from measurement to object clusters` | Comment |
| 36 | `    SigmaMT = 1./ (-SigmaTM + sum(SigmaTM, 1) + 1);` | 79-92 | See below | Measurement→object messages |

**Line 36 expansion:**

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 36a | `sum(SigmaTM, 1)` | 79-81 | `let col_sums: Vec<f64> = (0..n_measurements).map(\|j\| sigma_tm.column(j).sum()).collect();` | Column-wise sum |
| 36b | `-SigmaTM + sum(...) + 1` | 85 | `let denom = -sigma_tm[(i, j)] + col_sums[j] + 1.0;` | Denominator |
| 36c | `1 ./ (...)` | 86-90 | `sigma_mt[(i, j)] = if denom.abs() > 1e-15 { 1.0 / denom } else { 0.0 };` | Division with safety |

### Convergence Check

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 37 | `    % Check for convergence` | 94 | `// Check for convergence` | Comment |
| 38 | `    counter = counter + 1;` | 95 | `counter += 1;` | Increment counter |
| 39 | `    delta = abs(SigmaMT - SigmaMTOld);` | 96 | `let delta = (&sigma_mt - &sigma_mt_old).abs();` | Compute delta |
| 40 | `    notConverged = (max(delta(:)) > epsilon) && (counter < maximumNumberOfLbpIterations);` | 97-98 | `let max_delta = delta.iter().cloned().fold(0.0, f64::max); not_converged = max_delta > epsilon && counter < max_iterations;` | Check convergence. MATLAB `delta(:)` flattens, Rust `.iter()` |
| 41 | `end` | 99 | `}` | End while loop |

### Compute Gamma and q

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 42 | `Gamma = [associationMatrices.phi B .* associationMatrices.eta];` | 102-111 | See below | Build Gamma matrix |

**Line 42 expansion:**

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 42a | `[phi ...]` | 105-107 | `let mut gamma = DMatrix::zeros(n_objects, n_measurements + 1); for i in 0..n_objects { gamma[(i, 0)] = matrices.phi[i]; ... }` | First column is phi |
| 42b | `B .* eta` | 108-110 | `for j in 0..n_measurements { gamma[(i, j + 1)] = b[(i, j)] * matrices.eta[i]; }` | Remaining columns: B .* eta |

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 43 | `q = sum(Gamma, 2);` | 114 | `let q: Vec<f64> = (0..n_objects).map(\|i\| gamma.row(i).sum()).collect();` | Row sums |

### Association Probabilities

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 44 | `%% Determine association probabilities` | - | - | Section comment |
| 45 | `W = Gamma ./ q;` | 117-124 | `let mut w = DMatrix::zeros(n_objects, n_measurements + 1); for i in 0..n_objects { if q[i].abs() > 1e-15 { for j in 0..(n_measurements + 1) { w[(i, j)] = gamma[(i, j)] / q[i]; } } }` | Normalize rows. MATLAB broadcast division, Rust explicit loops with safety check |

### Existence Probabilities

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 46 | `%% Determine existence probabilities` | - | - | Section comment |
| 47 | `r = q ./ (associationMatrices.eta + q - associationMatrices.phi);` | 127-135 | `let mut r = DVector::zeros(n_objects); for i in 0..n_objects { let denom = matrices.eta[i] + q[i] - matrices.phi[i]; r[i] = if denom.abs() > 1e-15 { q[i] / denom } else { 0.0 }; }` | Compute existence. Rust adds division safety |
| 48 | `end` | 137 | `LbpResult { r, w }` | Return result struct |

---

## Key Translation Notes

1. **Matrix Operations**:
   - MATLAB: `.*` element-wise multiply, `./` element-wise divide
   - Rust: `.component_mul()`, manual loops with indexing

2. **Broadcasting**:
   - MATLAB: Implicit broadcasting in `Gamma ./ q` (divides each row)
   - Rust: Explicit loops required

3. **Sum Directions**:
   - MATLAB: `sum(X, 2)` = row sums, `sum(X, 1)` = column sums
   - Rust: Manual iteration with `.row(i).sum()` or `.column(j).sum()`

4. **Safety Checks**:
   - Rust adds `if denom.abs() > 1e-15` checks before division
   - MATLAB relies on IEEE inf/nan handling

5. **Return Type**:
   - MATLAB: Returns two separate arrays `[r, W]`
   - Rust: Returns `LbpResult` struct with `r` and `w` fields

6. **Flattening**:
   - MATLAB: `delta(:)` flattens to column vector
   - Rust: `.iter()` iterates over all elements
