# MATLAB to Rust Migration Plan - 100% Equivalence

**Goal**: Achieve 100% equivalence between the MATLAB implementation at `../multisensor-lmb-filters` and this Rust implementation.

**Ground Truth**: MATLAB code is the authoritative reference. Rust must contain NOTHING more and NOTHING less.

---

## Repository Overview

### MATLAB Repository: `/Users/nathanmichlo/Desktop/active/multisensor-lmb-filters`
- **Total MATLAB files**: 57 .m files
- **Total lines**: ~5,091 lines
- **Additional files**: 7 MEX binaries + 2 C/C++ source files for Hungarian assignment
- **Purpose**: Multi-sensor LMB/LMBM filters with various data association methods

### Rust Repository: `/Users/nathanmichlo/Desktop/active/prak`
- **Total Rust files**: 40 .rs files
- **Total lines**: ~8,404 lines (including tests)
- **Status**: ~70% complete overall, ~95% for core algorithms
- **Tests**: 44 tests passing, embedded in source files

---

## Current Status Summary

### ✅ FULLY IMPLEMENTED (Core Algorithms - ~95%)

1. **Common Utilities** (100%)
   - ✅ Hungarian assignment (pure Rust, no MEX)
   - ✅ Loopy Belief Propagation (LBP)
   - ✅ Gibbs sampling framework
   - ✅ Murty's algorithm
   - ✅ Model & ground truth generation
   - ✅ OSPA metrics (Euclidean & Hellinger)
   - ✅ Linear algebra (Kalman, Gaussian PDF, etc.)

2. **Single-Sensor LMB Filter** (100%)
   - ✅ Prediction step
   - ✅ Association matrices
   - ✅ Data association (LBP/Gibbs/Murty's)
   - ✅ Posterior computation
   - ✅ Cardinality estimation
   - ✅ Main filter loop

3. **Single-Sensor LMBM Filter** (100%)
   - ✅ Prediction step
   - ✅ Association matrices
   - ✅ Hypothesis management
   - ✅ Gibbs sampling
   - ✅ Main filter loop
   - ✅ State extraction (EAP and MAP)

4. **Multi-Sensor LMB Filters** (100%)
   - ✅ Parallel Update (PU-LMB)
   - ✅ Iterated Corrector (IC-LMB)
   - ✅ Geometric Average (GA-LMB)
   - ✅ Arithmetic Average (AA-LMB)
   - ✅ Track merging (all 3 variants)
   - ✅ Association matrices

5. **Multi-Sensor LMBM Filter** (100%)
   - ✅ Main filter loop
   - ✅ Association matrices
   - ✅ Hypothesis management
   - ✅ Gibbs sampling

### ⚠️ MISSING IMPLEMENTATIONS

1. **Gibbs Sampling Variant** (0%)
   - ❌ `lmbGibbsFrequencySampling.m` - Frequency-counting Gibbs sampler
   - Current Rust only implements unique-sample approach
   - MATLAB has BOTH variants

2. **Examples** (0/2)
   - ❌ `runFilters.m` → `examples/single_sensor.rs`
   - ❌ `runMultisensorFilters.m` → `examples/multi_sensor.rs`

3. **Validation Tests** (0/5)
   - ❌ `evaluateSmallExamples.m` - LBP vs Murty's validation
   - ❌ `evaluateMarginalDistributions.m`
   - ❌ `evaluateMarginalDistrubtionsVariableObjects.m`
   - ❌ `generateAssociationMatrices.m` (test utility)
   - ❌ `generateSimplifiedModel.m` (test utility)

4. **Performance Trials** (1/7 - 14%)
   - ✅ `lmbFilterTimeTrials.m` → `benches/lmb_performance.rs` (partial)
   - ❌ `singleSensorAccuracyTrial.m`
   - ❌ `singleSensorClutterTrial.m`
   - ❌ `singleSensorDetectionProbabilityTrial.m`
   - ❌ `multiSensorAccuracyTrial.m`
   - ❌ `multiSensorClutterTrial.m`
   - ❌ `multiSensorDetectionProbabilityTrial.m`

### ⚠️ FILES TO REMOVE (4 empty stubs)

1. `src/lmb/gibbs_sampling.rs` (2 lines) - functionality in `data_association.rs`
2. `src/lmb/murtys.rs` (2 lines) - functionality in `data_association.rs`
3. `src/lmbm/update.rs` (2 lines) - functionality in `filter.rs`/`hypothesis.rs`
4. `src/multisensor_lmbm/update.rs` (2 lines) - functionality in `filter.rs`/`hypothesis.rs`

### ❌ INTENTIONALLY NOT PORTED (Visualization)

- `plotResults.m` - MATLAB-specific visualization
- `plotMultisensorResults.m` - MATLAB-specific visualization
- `setPath.m` - MATLAB path management

---

## Migration Plan - Step by Step

### Phase 1: Cleanup (REMOVE)

**Priority: HIGH | Effort: LOW | RNG: No**

#### Task 1.1: Remove empty stub files
- [ ] Delete `src/lmb/gibbs_sampling.rs` (2 lines)
- [ ] Delete `src/lmb/murtys.rs` (2 lines)
- [ ] Delete `src/lmbm/update.rs` (2 lines)
- [ ] Delete `src/multisensor_lmbm/update.rs` (2 lines)
- [ ] Update module references if needed

**Rationale**: These files contain only comment headers and serve no purpose. Functionality is already implemented in other modules.

---

### Phase 2: Missing Algorithm Implementation (ADD)

**Priority: HIGH | Effort: MEDIUM | RNG: Yes (inherently stochastic)**

#### Task 2.1: Implement frequency-based Gibbs sampling

**MATLAB Reference**: `lmbGibbsFrequencySampling.m` (47 lines)

**Missing**: Alternative Gibbs implementation that counts sample frequencies instead of unique samples.

- [ ] Add `lmb_gibbs_frequency_sampling()` to `src/common/association/gibbs.rs`
- [ ] Key difference: Uses tally approach instead of unique() deduplication
- [ ] Lines 34-37: `ell = n * v + eta; Sigma(ell) = Sigma(ell) + (1 / numberOfSamples)`
- [ ] Cannot be deterministically tested due to RNG
- [ ] Add unit tests with approximate validation

**Implementation Notes**:
```matlab
% MATLAB approach (frequency counting):
for i = 1:numberOfSamples
    ell = n * v + eta;  % Compute linear index
    Sigma(ell) = Sigma(ell) + (1 / numberOfSamples);  % Tally frequencies
    [v, w] = generateGibbsSample(associationMatrices.P, v, w);
end
```

Current Rust approach (unique samples):
```rust
// Rust approach (unique deduplication):
for _ in 0..num_samples {
    (v, w) = generate_gibbs_sample(&matrices.p, v, w);
    v_samples_vec.push(v.clone());
}
// Find unique samples
let mut unique_samples: HashMap<Vec<usize>, usize> = HashMap::new();
```

**Testing Strategy**:
- ⚠️ Cannot create deterministic fixtures (RNG-dependent)
- Use large sample counts and statistical validation
- Compare frequency vs unique approaches with tolerance

---

### Phase 3: Examples (ADD)

**Priority: MEDIUM | Effort: MEDIUM | RNG: Yes (ground truth generation)**

#### Task 3.1: Create single-sensor example

**MATLAB Reference**: `runFilters.m` (19 lines)

- [ ] Create `examples/single_sensor_lmb.rs`
- [ ] Port lines 1-19 of `runFilters.m`
- [ ] Generate model with `generate_model(10, 0.95, 'LBP', 'Fixed')`
- [ ] Generate ground truth and measurements
- [ ] Run LMB or LMBM filter based on flag
- [ ] Output results to console (skip plotting)
- [ ] Add CLI argument for LMB vs LMBM selection

**Implementation**:
```rust
// examples/single_sensor_lmb.rs
use prak::common::model::generate_model;
use prak::common::ground_truth::generate_ground_truth;
use prak::lmb::filter::run_lmb_filter;
use prak::lmbm::filter::run_lmbm_filter;

fn main() {
    // Port runFilters.m logic here
}
```

**Testing Strategy**:
- ⚠️ Ground truth generation uses RNG
- Use seeded RNG for deterministic output
- Create fixture with known seed for regression testing

#### Task 3.2: Create multi-sensor example

**MATLAB Reference**: `runMultisensorFilters.m` (29 lines)

- [ ] Create `examples/multi_sensor.rs`
- [ ] Port lines 1-29 of `runMultisensorFilters.m`
- [ ] Support filter type selection: 'IC', 'PU', 'LMBM'
- [ ] Generate multi-sensor model with 3 sensors
- [ ] Run selected filter
- [ ] Output results to console

**Testing Strategy**: Same as Task 3.1

---

### Phase 4: Integration Tests (ADD)

**Priority: MEDIUM | Effort: HIGH | RNG: Partial**

#### Task 4.1: LBP vs Murty's validation test

**MATLAB Reference**: `evaluateSmallExamples.m` (117 lines)

**Purpose**: Validate LBP approximation against exact Murty's marginals.

- [ ] Create `tests/marginal_evaluations.rs`
- [ ] Port the core validation logic (lines 30-68)
- [ ] Generate association matrices for n=1..7 objects
- [ ] Run LBP to get approximate marginals
- [ ] Run Murty's to exhaustively compute exact marginals
- [ ] Compute KL divergence and Hellinger distance errors
- [ ] Assert errors are within acceptable bounds

**Implementation Notes**:
```matlab
% Key MATLAB logic:
associationMatrices = generateAssociationMatrices(model);
[rLbp, WLbp] = loopyBeliefPropagation(associationMatrices, ...);
[~, ~, V] = lmbMurtysAlgorithm(associationMatrices, numberOfEvents);
% Compute exact marginals from V
% Compare LBP vs Murty's
rKl(t, n) = averageKullbackLeiblerDivergence([1-rMurty rMurty], [1-rLbp rLbp]);
WKl(t, n) = averageKullbackLeiblerDivergence(WMurty, WLbp);
```

**Testing Strategy**:
- ⚠️ Partially RNG-dependent (model generation)
- Can compare LBP vs Murty deterministically for same input
- Use fixed scenarios for regression testing
- Helper functions needed:
  - [ ] `calculate_number_of_association_events()`
  - [ ] `average_kullback_leibler_divergence()`
  - [ ] `average_hellinger_distance()`

#### Task 4.2: Accuracy trial tests

**MATLAB References**:
- `singleSensorAccuracyTrial.m` (125 lines)
- `multiSensorAccuracyTrial.m` (132 lines)

- [ ] Create `tests/accuracy_trials.rs`
- [ ] Port single-sensor accuracy trial
- [ ] Port multi-sensor accuracy trial
- [ ] Run filters over multiple scenarios
- [ ] Compute OSPA metrics
- [ ] Assert mean OSPA within bounds

**Testing Strategy**:
- ⚠️ Heavily RNG-dependent (ground truth + clutter)
- Need pre-generated MATLAB fixtures OR
- Statistical validation over many runs with tolerance

#### Task 4.3: Clutter sensitivity tests

**MATLAB References**:
- `singleSensorClutterTrial.m` (113 lines)
- `multiSensorClutterTrial.m` (95 lines)

- [ ] Create `tests/clutter_trials.rs`
- [ ] Vary clutter rates: 0, 5, 10, 15, 20, 25
- [ ] Measure filter performance degradation
- [ ] Assert OSPA increases monotonically with clutter

**Testing Strategy**: Same as Task 4.2

#### Task 4.4: Detection probability tests

**MATLAB References**:
- `singleSensorDetectionProbabilityTrial.m` (111 lines)
- `multiSensorDetectionProbabilityTrial.m` (93 lines)

- [ ] Create `tests/detection_trials.rs`
- [ ] Vary detection probability: 0.2 to 1.0
- [ ] Measure filter performance improvement
- [ ] Assert OSPA decreases with higher detection probability

**Testing Strategy**: Same as Task 4.2

---

### Phase 5: Detailed Verification (FIX/VERIFY)

**Priority: CRITICAL | Effort: VERY HIGH | RNG: Mixed**

#### Task 5.1: File-by-file logic comparison

For EACH of the 40+ corresponding MATLAB/Rust file pairs, perform detailed comparison:

**Common Utilities (18 MATLAB → 12 Rust)**:
- [ ] Hungarian.m ↔ hungarian.rs
- [ ] munkres.m ↔ hungarian.rs (merged)
- [ ] loopyBeliefPropagation.m ↔ lbp.rs
- [ ] fixedLoopyBeliefPropagation.m ↔ lbp.rs (merged)
- [ ] generateGibbsSample.m ↔ gibbs.rs
- [ ] initialiseGibbsAssociationVectors.m ↔ gibbs.rs (merged)
- [ ] murtysAlgorithm.m ↔ murtys.rs
- [ ] murtysAlgorithmWrapper.m ↔ murtys.rs (merged)
- [ ] generateModel.m ↔ model.rs
- [ ] generateMultisensorModel.m ↔ model.rs (merged)
- [ ] generateGroundTruth.m ↔ ground_truth.rs
- [ ] generateMultisensorGroundTruth.m ↔ ground_truth.rs (merged)
- [ ] ospa.m ↔ metrics.rs
- [ ] computeSimulationOspa.m ↔ metrics.rs (merged)
- [ ] esf.m ↔ utils.rs
- [ ] lmbMapCardinalityEstimate.m ↔ cardinality.rs

**LMB Filter (6 MATLAB → 7 Rust)**:
- [ ] runLmbFilter.m ↔ filter.rs
- [ ] lmbPredictionStep.m ↔ prediction.rs
- [ ] generateLmbAssociationMatrices.m ↔ association.rs
- [ ] computePosteriorLmbSpatialDistributions.m ↔ update.rs
- [ ] lmbGibbsSampling.m ↔ data_association.rs (via gibbs)
- [ ] lmbGibbsFrequencySampling.m ↔ **MISSING**
- [ ] lmbMurtysAlgorithm.m ↔ data_association.rs (via murtys)

**LMBM Filter (7 MATLAB → 5 Rust)**:
- [ ] runLmbmFilter.m ↔ filter.rs
- [ ] lmbmPredictionStep.m ↔ prediction.rs
- [ ] generateLmbmAssociationMatrices.m ↔ association.rs
- [ ] determinePosteriorHypothesisParameters.m ↔ hypothesis.rs
- [ ] lmbmGibbsSampling.m ↔ association.rs (merged)
- [ ] lmbmNormalisationAndGating.m ↔ hypothesis.rs (merged)
- [ ] lmbmStateExtraction.m ↔ hypothesis.rs (merged as function)

**Multi-Sensor LMB (6 MATLAB → 5 Rust)**:
- [ ] runParallelUpdateLmbFilter.m ↔ parallel_update.rs
- [ ] runIcLmbFilter.m ↔ iterated_corrector.rs
- [ ] puLmbTrackMerging.m ↔ merging.rs
- [ ] gaLmbTrackMerging.m ↔ merging.rs (merged)
- [ ] aaLmbTrackMerging.m ↔ merging.rs (merged)
- [ ] generateLmbSensorAssociationMatrices.m ↔ association.rs

**Multi-Sensor LMBM (5 MATLAB → 5 Rust)**:
- [ ] runMultisensorLmbmFilter.m ↔ filter.rs
- [ ] generateMultisensorLmbmAssociationMatrices.m ↔ association.rs
- [ ] determineMultisensorPosteriorHypothesisParameters.m ↔ hypothesis.rs
- [ ] multisensorLmbmGibbsSampling.m ↔ gibbs.rs
- [ ] generateMultisensorAssociationEvent.m ↔ association.rs (merged)

#### Task 5.2: Numerical equivalence testing

**Strategy**: Generate fixtures from MATLAB with known RNG seeds, then verify Rust produces identical output.

- [ ] Create MATLAB fixture generator script
- [ ] Use `rng(42, 'twister')` for deterministic seeding
- [ ] Generate ground truth scenarios (5-10 different seeds)
- [ ] Save to JSON/CSV fixtures
- [ ] Create Rust fixture loader
- [ ] Run Rust filters on same inputs
- [ ] Assert numerical equivalence within tolerance (1e-10 for floats)

**Fixture Coverage**:
- [ ] Single-sensor LMB with LBP
- [ ] Single-sensor LMB with Gibbs
- [ ] Single-sensor LMB with Murty's
- [ ] Single-sensor LMBM
- [ ] Multi-sensor IC-LMB
- [ ] Multi-sensor PU-LMB
- [ ] Multi-sensor GA-LMB
- [ ] Multi-sensor AA-LMB
- [ ] Multi-sensor LMBM

#### Task 5.3: Cross-algorithm validation

**Purpose**: Verify different data association algorithms converge to similar results.

- [ ] Run LBP, Gibbs, and Murty's on identical scenarios
- [ ] Compare posterior existence probabilities
- [ ] Compare marginal association weights
- [ ] Assert LBP/Gibbs are close to Murty's (exact) within tolerance
- [ ] Document expected error bounds (from MATLAB evaluation)

---

## RNG Analysis - Testing Implications

### Files Using Random Number Generation

**Ground Truth Generation (CRITICAL)**:
1. `generateGroundTruth.m` (111 lines)
   - Line 57: `randn(numberOfObjects, 2)` - Initial velocities
   - Line 71: `rand(model.zDimension, 1)` - Clutter measurements
   - Line 98: `randn(1, model.zDimension)` - Measurement noise

2. `generateMultisensorGroundTruth.m` (130 lines)
   - Line 57: `randn(numberOfObjects, 2)` - Initial velocities
   - Line 72: `rand(model.zDimension, 1)` - Clutter measurements
   - Line 98: `rand(model.numberOfSensors, 1)` - Detection success
   - Line 108: `randn(1, model.zDimension)` - Measurement noise

**Model Generation**:
3. `generateModel.m` (182 lines)
   - Line 93: `rand(model.zDimension, numberOfBirthLocations)` - Birth locations

4. `generateMultisensorModel.m` (194 lines)
   - Line 103: `rand(model.zDimension, numberOfBirthLocations)` - Birth locations

**Data Association (STOCHASTIC)**:
5. `generateGibbsSample.m` (42 lines)
   - Line 30: `rand() < P(i, j)` - Sampling decisions

6. `generateMultisensorAssociationEvent.m` (49 lines)
   - Line 27: `rand() < P` - Association sampling

### RNG Mitigation Strategies

#### ✅ Already Implemented
- Rust uses custom Octave-compatible MT19937 RNG
- Ground truth generation can be seeded identically to MATLAB
- Produces 100% identical sequences when seeded the same

#### ⚠️ Testing Limitations
1. **Cannot test Gibbs sampling deterministically**
   - Inherently stochastic algorithm
   - Use large sample counts (1000+) for convergence
   - Compare against Murty's (exact) with tolerance

2. **Examples require fixtures**
   - Generate MATLAB output with known seeds
   - Use as regression test baselines
   - Update fixtures when algorithms change

3. **Trial scripts need statistical validation**
   - Run 100+ trials per configuration
   - Compare mean/variance of OSPA metrics
   - Use t-test or similar for equivalence testing

---

## Detailed File Mapping

### MATLAB Files → Rust Files

#### Common Utilities (18→12)
| MATLAB | Lines | Rust | Lines | Status |
|--------|-------|------|-------|--------|
| Hungarian.m | 280 | hungarian.rs | 447 | ✅ Complete |
| munkres.m | 132 | hungarian.rs | 447 | ✅ Merged |
| loopyBeliefPropagation.m | 47 | lbp.rs | 282 | ✅ Complete |
| fixedLoopyBeliefPropagation.m | 40 | lbp.rs | 282 | ✅ Merged |
| generateGibbsSample.m | 42 | gibbs.rs | 268 | ✅ Complete |
| initialiseGibbsAssociationVectors.m | 29 | gibbs.rs | 268 | ✅ Merged |
| murtysAlgorithm.m | 127 | murtys.rs | 268 | ✅ Complete |
| murtysAlgorithmWrapper.m | 59 | murtys.rs | 268 | ✅ Merged |
| generateModel.m | 182 | model.rs | 485 | ✅ Complete |
| generateMultisensorModel.m | 194 | model.rs | 485 | ✅ Merged |
| generateGroundTruth.m | 111 | ground_truth.rs | 584 | ✅ Complete |
| generateMultisensorGroundTruth.m | 130 | ground_truth.rs | 584 | ✅ Merged |
| ospa.m | 72 | metrics.rs | 263 | ✅ Complete |
| computeSimulationOspa.m | 35 | metrics.rs | 263 | ✅ Merged |
| esf.m | 39 | utils.rs | 289 | ✅ Complete |
| lmbMapCardinalityEstimate.m | 28 | cardinality.rs | 190 | ✅ Complete |
| plotResults.m | 277 | - | - | ❌ Skipped (viz) |
| plotMultisensorResults.m | 291 | - | - | ❌ Skipped (viz) |

#### Single-Sensor LMB (6→9)
| MATLAB | Lines | Rust | Lines | Status |
|--------|-------|------|-------|--------|
| runLmbFilter.m | 91 | filter.rs | 230 | ✅ Complete |
| lmbPredictionStep.m | 32 | prediction.rs | 144 | ✅ Complete |
| generateLmbAssociationMatrices.m | 82 | association.rs | 314 | ✅ Complete |
| computePosteriorLmbSpatialDistributions.m | 51 | update.rs | 208 | ✅ Complete |
| lmbGibbsSampling.m | 51 | data_association.rs | 280 | ✅ Complete |
| lmbGibbsFrequencySampling.m | 46 | - | - | ❌ MISSING |
| lmbMurtysAlgorithm.m | 39 | data_association.rs | 280 | ✅ Complete |
| - | - | cardinality.rs | 190 | ✅ Extra |
| - | - | gibbs_sampling.rs | 2 | ⚠️ REMOVE |
| - | - | murtys.rs | 2 | ⚠️ REMOVE |

#### Single-Sensor LMBM (7→6)
| MATLAB | Lines | Rust | Lines | Status |
|--------|-------|------|-------|--------|
| runLmbmFilter.m | 91 | filter.rs | 266 | ✅ Complete |
| lmbmPredictionStep.m | 33 | prediction.rs | 132 | ✅ Complete |
| generateLmbmAssociationMatrices.m | 63 | association.rs | 352 | ✅ Complete |
| determinePosteriorHypothesisParameters.m | 47 | hypothesis.rs | 404 | ✅ Complete |
| lmbmGibbsSampling.m | 35 | association.rs | 352 | ✅ Merged |
| lmbmNormalisationAndGating.m | 55 | hypothesis.rs | 404 | ✅ Merged |
| lmbmStateExtraction.m | 32 | hypothesis.rs | 404 | ✅ Merged |
| - | - | update.rs | 2 | ⚠️ REMOVE |

#### Multi-Sensor LMB (6→5)
| MATLAB | Lines | Rust | Lines | Status |
|--------|-------|------|-------|--------|
| runParallelUpdateLmbFilter.m | 105 | parallel_update.rs | 389 | ✅ Complete |
| runIcLmbFilter.m | 91 | iterated_corrector.rs | 260 | ✅ Complete |
| puLmbTrackMerging.m | 93 | merging.rs | 447 | ✅ Complete |
| gaLmbTrackMerging.m | 73 | merging.rs | 447 | ✅ Merged |
| aaLmbTrackMerging.m | 40 | merging.rs | 447 | ✅ Merged |
| generateLmbSensorAssociationMatrices.m | 84 | association.rs | 273 | ✅ Complete |

#### Multi-Sensor LMBM (5→6)
| MATLAB | Lines | Rust | Lines | Status |
|--------|-------|------|-------|--------|
| runMultisensorLmbmFilter.m | 95 | filter.rs | 266 | ✅ Complete |
| generateMultisensorLmbmAssociationMatrices.m | 113 | association.rs | 277 | ✅ Complete |
| determineMultisensorPosteriorHypothesisParameters.m | 60 | hypothesis.rs | 200 | ✅ Complete |
| multisensorLmbmGibbsSampling.m | 40 | gibbs.rs | 225 | ✅ Complete |
| generateMultisensorAssociationEvent.m | 49 | association.rs | 277 | ✅ Merged |
| - | - | update.rs | 2 | ⚠️ REMOVE |

#### Tests/Validation (12→1)
| MATLAB | Lines | Rust | Lines | Status |
|--------|-------|------|-------|--------|
| evaluateSmallExamples.m | 117 | - | - | ❌ Missing |
| evaluateMarginalDistributions.m | 148 | - | - | ❌ Missing |
| evaluateMarginalDistrubtionsVariableObjects.m | 123 | - | - | ❌ Missing |
| generateAssociationMatrices.m | 65 | - | - | ❌ Missing |
| generateSimplifiedModel.m | 61 | - | - | ❌ Missing |
| lmbFilterTimeTrials.m | 203 | lmb_performance.rs | ~50 | ⚠️ Partial |
| singleSensorAccuracyTrial.m | 125 | - | - | ❌ Missing |
| singleSensorClutterTrial.m | 113 | - | - | ❌ Missing |
| singleSensorDetectionProbabilityTrial.m | 111 | - | - | ❌ Missing |
| multiSensorAccuracyTrial.m | 132 | - | - | ❌ Missing |
| multiSensorClutterTrial.m | 95 | - | - | ❌ Missing |
| multiSensorDetectionProbabilityTrial.m | 93 | - | - | ❌ Missing |

#### Examples (2→0)
| MATLAB | Lines | Rust | Lines | Status |
|--------|-------|------|-------|--------|
| runFilters.m | 19 | - | - | ❌ Missing |
| runMultisensorFilters.m | 29 | - | - | ❌ Missing |

---

## Key Differences Between MATLAB and Rust

### 1. File Organization
- **MATLAB**: Flat structure, one function per file
- **Rust**: Modular structure, multiple related functions per file
- **Impact**: Several MATLAB files merged into single Rust files
- **Verification**: Must ensure ALL MATLAB functions have Rust equivalents

### 2. MEX Binaries
- **MATLAB**: Uses MEX (C/C++) for Hungarian assignment (`assignmentoptimal.c/cc`)
- **Rust**: Pure Rust implementation of Hungarian algorithm
- **Impact**: Must verify numerical equivalence
- **Status**: ✅ Verified by existing unit tests

### 3. Gibbs Sampling Variants
- **MATLAB**: TWO implementations
  - `lmbGibbsSampling.m` - Uses unique() to deduplicate samples
  - `lmbGibbsFrequencySampling.m` - Uses frequency counting (faster in non-MATLAB)
- **Rust**: ONE implementation (unique samples approach)
- **Impact**: ❌ Missing frequency variant
- **Action**: Implement frequency variant in Phase 2

### 4. Testing Approach
- **MATLAB**: Separate trial/validation scripts
- **Rust**: Inline unit tests + separate integration tests
- **Impact**: Must port trial scripts as integration tests
- **Status**: ❌ Only 1/7 partially done

### 5. Visualization
- **MATLAB**: Built-in plotting with `plotResults.m`
- **Rust**: No visualization (out of scope)
- **Impact**: None (intentionally skipped)

---

## Critical Observations & Concerns

### 1. Empty Stub Files
**Issue**: 4 files with only comment headers (2 lines each)
- `src/lmb/gibbs_sampling.rs`
- `src/lmb/murtys.rs`
- `src/lmbm/update.rs`
- `src/multisensor_lmbm/update.rs`

**Action**: DELETE in Phase 1 (cleanup)

### 2. Missing Gibbs Variant
**Issue**: `lmbGibbsFrequencySampling.m` not implemented
**Impact**: MATLAB has 2 Gibbs variants, Rust has only 1
**Action**: IMPLEMENT in Phase 2

### 3. No Examples
**Issue**: Entry points (`runFilters.m`, `runMultisensorFilters.m`) not ported
**Impact**: No executable demos of the library
**Action**: CREATE in Phase 3

### 4. No Integration Tests
**Issue**: 12 MATLAB test/trial scripts not ported
**Impact**: Cannot verify MATLAB↔Rust numerical equivalence
**Action**: CREATE in Phase 4

### 5. RNG Compatibility Critical
**Issue**: Ground truth generation MUST produce identical results
**Status**: ✅ Octave-compatible MT19937 already implemented
**Action**: VERIFY with fixture tests in Phase 5

---

## Testing Strategy Summary

### Deterministic Tests (Can achieve 100% equivalence)
1. **Pure algorithms** (no RNG):
   - Hungarian assignment
   - Loopy Belief Propagation (given fixed input)
   - Murty's algorithm
   - Kalman filter operations
   - OSPA metrics

2. **Fixed scenario tests**:
   - Pre-generate ground truth with known MATLAB seed
   - Save to fixtures
   - Run Rust with same fixtures
   - Assert bit-for-bit equivalence (within float tolerance)

### Statistical Tests (Approximate equivalence)
1. **Gibbs sampling**:
   - Run with large sample count (1000+)
   - Compare marginals vs Murty's (exact)
   - Assert KL divergence < threshold (from MATLAB evaluation)

2. **Trial scripts**:
   - Run 100+ Monte Carlo trials
   - Compare mean OSPA metrics
   - Use statistical tests (t-test) for equivalence

### Regression Tests
1. **Fixture-based**:
   - Golden outputs from MATLAB
   - Verify Rust matches on update
   - Update fixtures when algorithms intentionally change

---

## Completion Criteria

### Phase 1: Cleanup
- [ ] All 4 stub files deleted
- [ ] No broken module references
- [ ] All tests still pass

### Phase 2: Missing Algorithm
- [ ] `lmb_gibbs_frequency_sampling()` implemented
- [ ] Matches MATLAB `lmbGibbsFrequencySampling.m` logic
- [ ] Unit tests pass (with tolerance for RNG)
- [ ] Documentation added

### Phase 3: Examples
- [ ] `examples/single_sensor.rs` runs successfully
- [ ] `examples/multi_sensor.rs` runs successfully
- [ ] README updated with usage instructions
- [ ] Example output matches MATLAB (with seeded RNG)

### Phase 4: Integration Tests
- [ ] `tests/marginal_evaluations.rs` validates LBP vs Murty's
- [ ] `tests/accuracy_trials.rs` runs and passes
- [ ] `tests/clutter_trials.rs` runs and passes
- [ ] `tests/detection_trials.rs` runs and passes
- [ ] All trial metrics within expected bounds

### Phase 5: Verification
- [ ] All 40+ file pairs compared line-by-line
- [ ] Numerical fixtures generated from MATLAB
- [ ] All fixtures pass with <1e-10 error
- [ ] Cross-algorithm validation complete
- [ ] Documentation updated with differences/limitations

### Final Deliverable
- [ ] 100% MATLAB functionality ported (excluding visualization)
- [ ] All core algorithms numerically equivalent
- [ ] Comprehensive test coverage (unit + integration)
- [ ] Examples demonstrate usage
- [ ] Migration complete ✅

---

## Appendix: Full File Paths

### MATLAB Files by Category

**Common - Association (8 files)**:
- `../multisensor-lmb-filters/common/Hungarian.m`
- `../multisensor-lmb-filters/common/munkres.m`
- `../multisensor-lmb-filters/common/loopyBeliefPropagation.m`
- `../multisensor-lmb-filters/common/fixedLoopyBeliefPropagation.m`
- `../multisensor-lmb-filters/common/generateGibbsSample.m`
- `../multisensor-lmb-filters/common/initialiseGibbsAssociationVectors.m`
- `../multisensor-lmb-filters/common/murtysAlgorithm.m`
- `../multisensor-lmb-filters/common/murtysAlgorithmWrapper.m`

**Common - Model/Ground Truth (4 files)**:
- `../multisensor-lmb-filters/common/generateModel.m`
- `../multisensor-lmb-filters/common/generateMultisensorModel.m`
- `../multisensor-lmb-filters/common/generateGroundTruth.m`
- `../multisensor-lmb-filters/common/generateMultisensorGroundTruth.m`

**Common - Metrics/Utils (4 files)**:
- `../multisensor-lmb-filters/common/ospa.m`
- `../multisensor-lmb-filters/common/computeSimulationOspa.m`
- `../multisensor-lmb-filters/common/esf.m`
- `../multisensor-lmb-filters/common/lmbMapCardinalityEstimate.m`

**LMB Filter (6 files)**:
- `../multisensor-lmb-filters/lmb/runLmbFilter.m`
- `../multisensor-lmb-filters/lmb/lmbPredictionStep.m`
- `../multisensor-lmb-filters/lmb/generateLmbAssociationMatrices.m`
- `../multisensor-lmb-filters/lmb/computePosteriorLmbSpatialDistributions.m`
- `../multisensor-lmb-filters/lmb/lmbGibbsSampling.m`
- `../multisensor-lmb-filters/lmb/lmbGibbsFrequencySampling.m` ⚠️ MISSING in Rust
- `../multisensor-lmb-filters/lmb/lmbMurtysAlgorithm.m`

**LMBM Filter (7 files)**:
- `../multisensor-lmb-filters/lmbm/runLmbmFilter.m`
- `../multisensor-lmb-filters/lmbm/lmbmPredictionStep.m`
- `../multisensor-lmb-filters/lmbm/generateLmbmAssociationMatrices.m`
- `../multisensor-lmb-filters/lmbm/determinePosteriorHypothesisParameters.m`
- `../multisensor-lmb-filters/lmbm/lmbmGibbsSampling.m`
- `../multisensor-lmb-filters/lmbm/lmbmNormalisationAndGating.m`
- `../multisensor-lmb-filters/lmbm/lmbmStateExtraction.m`

**Multi-Sensor LMB (6 files)**:
- `../multisensor-lmb-filters/multisensorLmb/runParallelUpdateLmbFilter.m`
- `../multisensor-lmb-filters/multisensorLmb/runIcLmbFilter.m`
- `../multisensor-lmb-filters/multisensorLmb/puLmbTrackMerging.m`
- `../multisensor-lmb-filters/multisensorLmb/gaLmbTrackMerging.m`
- `../multisensor-lmb-filters/multisensorLmb/aaLmbTrackMerging.m`
- `../multisensor-lmb-filters/multisensorLmb/generateLmbSensorAssociationMatrices.m`

**Multi-Sensor LMBM (5 files)**:
- `../multisensor-lmb-filters/multisensorLmbm/runMultisensorLmbmFilter.m`
- `../multisensor-lmb-filters/multisensorLmbm/generateMultisensorLmbmAssociationMatrices.m`
- `../multisensor-lmb-filters/multisensorLmbm/determineMultisensorPosteriorHypothesisParameters.m`
- `../multisensor-lmb-filters/multisensorLmbm/multisensorLmbmGibbsSampling.m`
- `../multisensor-lmb-filters/multisensorLmbm/generateMultisensorAssociationEvent.m`

**Marginal Evaluations (5 files)**:
- `../multisensor-lmb-filters/marginalEvalulations/evaluateMarginalDistributions.m`
- `../multisensor-lmb-filters/marginalEvalulations/evaluateMarginalDistrubtionsVariableObjects.m`
- `../multisensor-lmb-filters/marginalEvalulations/evaluateSmallExamples.m`
- `../multisensor-lmb-filters/marginalEvalulations/generateAssociationMatrices.m`
- `../multisensor-lmb-filters/marginalEvalulations/generateSimplifiedModel.m`

**Trials (7 files)**:
- `../multisensor-lmb-filters/trials/lmbFilterTimeTrials.m`
- `../multisensor-lmb-filters/trials/singleSensorAccuracyTrial.m`
- `../multisensor-lmb-filters/trials/singleSensorClutterTrial.m`
- `../multisensor-lmb-filters/trials/singleSensorDetectionProbabilityTrial.m`
- `../multisensor-lmb-filters/trials/multiSensorAccuracyTrial.m`
- `../multisensor-lmb-filters/trials/multiSensorClutterTrial.m`
- `../multisensor-lmb-filters/trials/multiSensorDetectionProbabilityTrial.m`

**Entry Points (3 files)**:
- `../multisensor-lmb-filters/runFilters.m`
- `../multisensor-lmb-filters/runMultisensorFilters.m`
- `../multisensor-lmb-filters/setPath.m` (MATLAB-specific, skip)

### Rust Files Complete List

**src/common/ (12 files)**:
- `src/common/mod.rs`
- `src/common/types.rs`
- `src/common/model.rs`
- `src/common/ground_truth.rs`
- `src/common/linalg.rs`
- `src/common/metrics.rs`
- `src/common/utils.rs`
- `src/common/association/mod.rs`
- `src/common/association/hungarian.rs`
- `src/common/association/lbp.rs`
- `src/common/association/gibbs.rs`
- `src/common/association/murtys.rs`

**src/lmb/ (9 files, 2 to remove)**:
- `src/lmb/mod.rs`
- `src/lmb/filter.rs`
- `src/lmb/prediction.rs`
- `src/lmb/association.rs`
- `src/lmb/data_association.rs`
- `src/lmb/update.rs`
- `src/lmb/cardinality.rs`
- `src/lmb/gibbs_sampling.rs` ⚠️ REMOVE (stub)
- `src/lmb/murtys.rs` ⚠️ REMOVE (stub)

**src/lmbm/ (6 files, 1 to remove)**:
- `src/lmbm/mod.rs`
- `src/lmbm/filter.rs`
- `src/lmbm/prediction.rs`
- `src/lmbm/association.rs`
- `src/lmbm/hypothesis.rs`
- `src/lmbm/update.rs` ⚠️ REMOVE (stub)

**src/multisensor_lmb/ (5 files)**:
- `src/multisensor_lmb/mod.rs`
- `src/multisensor_lmb/parallel_update.rs`
- `src/multisensor_lmb/iterated_corrector.rs`
- `src/multisensor_lmb/merging.rs`
- `src/multisensor_lmb/association.rs`

**src/multisensor_lmbm/ (6 files, 1 to remove)**:
- `src/multisensor_lmbm/mod.rs`
- `src/multisensor_lmbm/filter.rs`
- `src/multisensor_lmbm/association.rs`
- `src/multisensor_lmbm/hypothesis.rs`
- `src/multisensor_lmbm/gibbs.rs`
- `src/multisensor_lmbm/update.rs` ⚠️ REMOVE (stub)

**Other**:
- `src/lib.rs`
- `benches/lmb_performance.rs`

---

## Summary Statistics

| Category | MATLAB Files | MATLAB Lines | Rust Files | Rust Lines | Completeness |
|----------|--------------|--------------|------------|------------|--------------|
| Common utilities | 18 | ~1,800 | 12 | ~2,781 | ✅ 100% |
| LMB filter | 7 | ~438 | 9 | ~1,367 | ⚠️ 86% (missing freq Gibbs) |
| LMBM filter | 7 | ~356 | 6 | ~1,155 | ✅ 100% |
| Multi-sensor LMB | 6 | ~486 | 5 | ~1,369 | ✅ 100% |
| Multi-sensor LMBM | 5 | ~357 | 6 | ~985 | ✅ 100% |
| Tests/Trials | 12 | ~1,350 | 1 | ~50 | ❌ 8% |
| Examples | 2 | ~48 | 0 | 0 | ❌ 0% |
| Visualization | 2 | ~568 | 0 | 0 | ✅ 0% (N/A) |
| **TOTAL FUNCTIONAL** | **45** | **~3,485** | **39** | **~7,707** | **~92%** |
| **TOTAL WITH TESTS** | **57** | **~5,091** | **40** | **~8,404** | **~70%** |

**Core Algorithms**: ~92% complete (one Gibbs variant missing)
**Testing Infrastructure**: ~8% complete (critical gap)
**Examples**: 0% complete (should be added)

---

## Next Steps

1. **Start with Phase 1** (cleanup) - Quick win, reduces confusion
2. **Proceed to Phase 2** (missing algorithm) - Achieves feature parity
3. **Phase 3** (examples) - Makes library usable
4. **Phase 4** (integration tests) - Validates correctness
5. **Phase 5** (detailed verification) - Ensures 100% equivalence

Each phase builds on the previous, ensuring incremental progress toward the goal of 100% MATLAB equivalence.
