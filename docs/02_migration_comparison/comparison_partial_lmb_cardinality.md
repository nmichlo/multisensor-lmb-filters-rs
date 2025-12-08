# Comparison: esf.m + lmbMapCardinalityEstimate.m → cardinality.rs

## Overview
- **MATLAB**:
  - `../multisensor-lmb-filters/common/esf.m` (40 lines)
  - `../multisensor-lmb-filters/common/lmbMapCardinalityEstimate.m` (29 lines)
- **Rust**: `src/lmb/cardinality.rs` (56 lines for both functions, excluding tests)
- **Purpose**: Elementary Symmetric Function (ESF) and MAP cardinality estimation

---

## Part 1: esf.m → elementary_symmetric_function()

### Function Signature & Documentation

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 1 | `function s = esf(Z)` | 23 | `pub fn elementary_symmetric_function(z: &[f64]) -> Vec<f64>` | Function signature. MATLAB column vector, Rust slice |
| 2 | `% ESF -- Caculate an elementary symmetric function using Mahler's recursive` | 6-9 | `/// Elementary Symmetric Function (ESF)` | Doc comment |
| 3 | `%   formula.` | 8 | `/// Calculates elementary symmetric function using Mahler's recursive formula.` | Doc continued |
| 4 | `%   s = esf(Z)` | - | - | Usage example |
| 5 | `%` | - | - | Blank |
| 6 | `%   This is Vo and Vo's code that calculates an elemnetary sysmmteric` | 9 | `/// This is Vo and Vo's code ported to Rust.` | Attribution |
| 7 | `%   function using Mahler's rursive formula.` | - | - | Merged above |
| 8 | `%` | - | - | Blank |
| 9 | `%  Inputs` | 11 | `/// # Arguments` | Args header |
| 10 | `%       Z - array.` | 12 | `/// * \`z\` - Input array` | Input doc |
| 11 | `%` | - | - | Blank |
| 12 | `%   Output` | 14 | `/// # Returns` | Returns header |
| 13 | `%       s - array. The elementary symmetric function of Z.` | 15-17 | `/// Vector of ESF values [e_0, e_1, ..., e_n]` | Output doc |
| 14 | (blank) | - | - | Whitespace |

### Empty Check

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 15 | `if isempty(Z)` | 24 | `if z.is_empty() {` | Empty check |
| 16 | `    s= 1;` | 25 | `return vec![1.0];` | Return 1 for empty |
| 17 | `    return;` | 25 | (merged above) | Early return |
| 18 | `end` | 26 | `}` | End if |
| 19 | (blank) | - | - | Whitespace |

### Variable Initialization

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 20 | `n_z = length(Z);` | 28 | `let n_z = z.len();` | Get length |
| 21 | `F = zeros(2,n_z);` | 29 | `let mut f = vec![vec![0.0; n_z]; 2];` | 2xn_z matrix. MATLAB zeros(), Rust vec of vecs |
| 22 | (blank) | - | - | Whitespace |
| 23 | `i_n = 1;` | 31 | `let mut i_n = 0;` | Row index. **MATLAB 1-indexed, Rust 0-indexed** |
| 24 | `i_nminus = 2;` | 32 | `let mut i_nminus = 1;` | Other row index. **MATLAB 2→Rust 1** |
| 25 | (blank) | - | - | Whitespace |

### Main Loop

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 26 | `for n = 1:n_z` | 34 | `for n in 0..n_z {` | Main loop. **MATLAB 1:n_z, Rust 0..n_z** |
| 27 | `    F(i_n,1) = F(i_nminus,1) + Z(n);` | 36 | `f[i_n][0] = f[i_nminus][0] + z[n];` | First column update. **Column index: MATLAB 1→Rust 0** |
| 28 | `    for k = 2:n` | 38 | `for k in 1..=n {` | Inner loop. **MATLAB 2:n, Rust 1..=n** |
| 29 | `        if k==n` | 39 | `if k == n {` | Diagonal case |
| 30 | `            F(i_n,k) = Z(n)*F(i_nminus,k-1);` | 41 | `f[i_n][k] = z[n] * f[i_nminus][k - 1];` | Diagonal formula |
| 31 | `        else` | 42 | `} else {` | Non-diagonal case |
| 32 | `            F(i_n,k) = F(i_nminus,k) + Z(n)*F(i_nminus,k-1);` | 44 | `f[i_n][k] = f[i_nminus][k] + z[n] * f[i_nminus][k - 1];` | General formula |
| 33 | `        end` | 45 | `}` | End if |
| 34 | `    end` | 46 | `}` | End inner loop |

### Index Swap

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 35 | `    tmp = i_n;` | 49 | `std::mem::swap(&mut i_n, &mut i_nminus);` | Swap indices. MATLAB manual swap, Rust std::mem::swap |
| 36 | `    i_n = i_nminus;` | 49 | (merged above) | Part of swap |
| 37 | `    i_nminus = tmp;` | 49 | (merged above) | Part of swap |
| 38 | `end` | 50 | `}` | End main loop |
| 39 | (blank) | - | - | Whitespace |

### Return Result

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 40 | `s= [1; F(i_nminus,:)'];` | 53-55 | `let mut result = vec![1.0]; result.extend_from_slice(&f[i_nminus][..n_z]); result` | Build result. MATLAB column concat, Rust vec extend |

---

## Part 2: lmbMapCardinalityEstimate.m → lmb_map_cardinality_estimate()

### Function Signature & Documentation

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 1 | `function [nMap, mapIndices] = lmbMapCardinalityEstimate(r)` | 76 | `pub fn lmb_map_cardinality_estimate(r: &[f64]) -> (usize, Vec<usize>)` | Function signature. Returns tuple |
| 2 | `% LMBMAPCARDINALITYESTIMATE -- Determine approximate LMB MAP estimate` | 58-60 | `/// LMB MAP cardinality estimate` | Doc comment |
| 3 | `%   [nMap, mapIndices] = lmbMapCardinalityEstimate(r)` | - | - | Usage |
| 4 | `%` | - | - | Blank |
| 5 | `%   This function computes an approximate MAP estimate for the LMB filter.` | 62 | `/// Determines approximate MAP estimate for LMB filter using Mahler's algorithm.` | Purpose |
| 6 | `%` | - | - | Blank |
| 7 | `%   See also runLmbFilter.` | - | - | See also |
| 8 | `%` | - | - | Blank |
| 9 | `%   Inputs` | 64 | `/// # Arguments` | Args header |
| 10 | `%       r - array. Each object's posterior existence probability.` | 65 | `/// * \`r\` - Posterior existence probabilities for each object` | Input doc |
| 11 | `%` | - | - | Blank |
| 12 | `%   Output` | 67 | `/// # Returns` | Returns header |
| 13 | `%       nMap - integer. The MAP estimate for the LMB` | 68-69 | `/// Tuple of (n_map, map_indices)` | Output doc |
| 14 | `%           cardinality estimate.` | 69 | `/// - n_map: MAP estimate for cardinality` | Continued |
| 15 | `%       mapIndices - array. The indices of the nMap greatest indices` | 70 | `/// - map_indices: Indices of the n_map objects with highest existence probabilities` | Output doc |
| 16 | `%           of r` | 70 | (merged above) | Continued |
| 17 | (blank) | - | - | Whitespace |

### Empty Check (Rust only)

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| - | (implicit in MATLAB) | 77-79 | `if r.is_empty() { return (0, Vec::new()); }` | Empty guard (Rust adds explicit check) |

### Clamping (Rust only - Bug #17 fix)

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| - | (not needed in MATLAB) | 123-134 | `let r_clamped: Vec<f64> = r.iter().map(\|&ri\| { if ri > 1.0 - 1e-15 { 1.0 } ... }).collect();` | **Bug #17 fix**: Clamp near-1.0 values to exactly 1.0 |

### Cardinality Distribution Computation

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 18 | `% Use Mahler's aglorithm to determine the LMB cardinality distribution` | - | - | Comment |
| 19 | `r = r - 1e-6; % Does not work with unit existence probabilities` | 138 | `let r_adjusted: Vec<f64> = r_clamped.iter().map(\|&ri\| ri - 1e-6).collect();` | Subtract 1e-6 to avoid unit probabilities |
| 20 | `rho = prod(1 - r)*esf(r./(1-r));` | 141-152 | See below | Compute rho |

**Line 20 expansion:**

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 20a | `prod(1 - r)` | 141, 145 | `let mut prod_1_minus_r = 1.0; ... prod_1_minus_r *= 1.0 - ri;` | Compute product |
| 20b | `r./(1-r)` | 142, 146 | `let mut r_ratio = Vec::...; r_ratio.push(ri / (1.0 - ri));` | Compute ratio |
| 20c | `esf(...)` | 149 | `let esf_values = elementary_symmetric_function(&r_ratio);` | Call ESF |
| 20d | `prod(...)*esf(...)` | 152 | `let rho: Vec<f64> = esf_values.iter().map(\|&e\| prod_1_minus_r * e).collect();` | Final rho |

### Find Maximum

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 21 | `% Determine the MAP estimate of the distribution` | - | - | Comment |
| 22 | `[~, maxCardinalityIndex] = max(rho);` | 155-160 | `let max_cardinality_index = rho.iter().enumerate().max_by(...).map(\|(i, _)\| i).unwrap_or(0);` | Find max index. MATLAB `max()` returns value and index |
| 23 | `% The MAP estimate cannot be larger than the number of objects` | - | - | Comment |
| 24 | `nMap = min(maxCardinalityIndex - 1, length(r));` | 163 | `let n_map = std::cmp::min(max_cardinality_index, r.len());` | Cap at num objects. **MATLAB subtracts 1 (1-indexed), Rust doesn't (0-indexed)** |

### Sort and Select

| MATLAB Line | MATLAB Code | Rust Line | Rust Code | Notes |
|-------------|-------------|-----------|-----------|-------|
| 25 | `% Sort r in descending order` | - | - | Comment |
| 26 | `[~, sortedIndices] = sort(-r);` | 169-179 | `let mut indexed_r: Vec<(usize, f64)> = r_adjusted.iter()...; indexed_r.sort_by(...);` | Sort descending. MATLAB `sort(-r)`, Rust custom comparator |
| 27 | `% Choose the nMap largest indices of r` | - | - | Comment |
| 28 | `mapIndices = sortedIndices(1:nMap);` | 182 | `let map_indices: Vec<usize> = indexed_r.iter().take(n_map).map(\|(i, _)\| *i).collect();` | Select top n_map indices |
| 29 | `end` | 184 | `(n_map, map_indices)` | Return tuple |

---

## Key Translation Notes

1. **Index Offset (Bug #17)**:
   - MATLAB: `maxCardinalityIndex - 1` because `max()` returns 1-indexed
   - Rust: `max_cardinality_index` directly (0-indexed enumerate)

2. **Clamping (Rust addition)**:
   - Rust adds clamping at lines 123-134 to handle floating-point precision issues where `r` can be `0.99999999999999989` instead of `1.0` after Murty marginal computation

3. **Sorting**:
   - MATLAB: `sort(-r)` negates for descending sort
   - Rust: Custom comparator with `b.partial_cmp(a)` for descending

4. **Array Building**:
   - MATLAB: `[1; F(i_nminus,:)']` vertical concatenation
   - Rust: `vec![1.0]` then `extend_from_slice()`

5. **Return Values**:
   - MATLAB: Two output variables `[nMap, mapIndices]`
   - Rust: Returns tuple `(usize, Vec<usize>)`
