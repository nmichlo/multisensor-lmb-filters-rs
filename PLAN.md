# Fixture Coverage Analysis - Full Value Comparison TODOs

## Current Status

**Python tests**: 31 passed
**Rust tests**: 44 passed (+ 2 ignored)

## Coverage Matrix

### LMB FIXTURE (lmb_step_by_step_seed42.json)

| Field | Python | Rust | Action |
|-------|--------|------|--------|
| step1.predicted_objects | ✓ values | ✓ values | COMPLETE |
| step2.C (cost) | ✓ values | ✓ values | COMPLETE |
| step2.L (likelihood) | ✓ values | ✓ values | COMPLETE |
| step2.R (miss prob) | ✓ values | ✓ values | COMPLETE |
| step2.P (sampling) | ✓ values | ✓ values | COMPLETE |
| step2.eta | ✓ values | ✓ values | COMPLETE |
| step2.posteriorParameters.w | ✓ values | ✗ | TODO: Add Rust value comparison |
| step2.posteriorParameters.mu | ✓ values | ✗ | TODO: Add Rust value comparison |
| step2.posteriorParameters.Sigma | ✓ values | ✗ | TODO: Add Rust value comparison |
| step3a_lbp.r | ✓ values | ✓ values | COMPLETE |
| step3a_lbp.W | ✓ values | ✓ values | COMPLETE |
| step3b_gibbs.r | ✓ values | ✗ | TODO: Add Rust value comparison |
| step3b_gibbs.W | ✓ values | ✗ | TODO: Add Rust value comparison |
| step3c_murtys.r | ✓ values | ✗ | TODO: Add Rust value comparison |
| step3c_murtys.W | ✓ values | ✗ | TODO: Add Rust value comparison |
| step4.posterior_objects | ✓ values | ✗ | TODO: Add Rust value comparison |
| step5.n_estimated | ✓ values | ✗ | TODO: Add Rust value comparison |
| step5.map_indices | ✓ values | ✗ | TODO: Add Rust value comparison |

### LMBM FIXTURE (lmbm_step_by_step_seed42.json)

| Field | Python | Rust | Action |
|-------|--------|------|--------|
| step1.predicted_hypothesis.w | ✗ | ✗ | TODO: Add BOTH |
| step1.predicted_hypothesis.r | ✗ | ✓ values | TODO: Add Python |
| step1.predicted_hypothesis.mu | ✗ | ✗ | TODO: Add BOTH |
| step1.predicted_hypothesis.Sigma | ✗ | ✗ | TODO: Add BOTH |
| step1.predicted_hypothesis.birthTime | ✗ | ✓ values | TODO: Add Python |
| step1.predicted_hypothesis.birthLocation | ✗ | ✗ | TODO: Add BOTH |
| step2.C | ✓ values | ✗ | TODO: Add Rust value comparison |
| step2.L | ✓ values | ✗ | TODO: Add Rust value comparison |
| step2.P | ✓ values | ✓ values | COMPLETE |
| step2.posteriorParameters.r | ✗ | ✓ values | TODO: Add Python |
| step2.posteriorParameters.mu | ✗ | ✗ | TODO: Add BOTH |
| step2.posteriorParameters.Sigma | ✗ | ✗ | TODO: Add BOTH |
| step3a_gibbs.V | ✓ values | ✓ structure | TODO: Add Rust value comparison |
| step3b_murtys.V | ✗ | ✓ structure | TODO: Add BOTH value comparison |
| step4.new_hypotheses.w | ✓ values | ✗ | TODO: Add Rust value comparison |
| step4.new_hypotheses.r | ✓ values | ✗ | TODO: Add Rust value comparison |
| step4.new_hypotheses.mu | ✓ values | ✗ | TODO: Add Rust value comparison |
| step4.new_hypotheses.Sigma | ✓ values | ✗ | TODO: Add Rust value comparison |
| step4.new_hypotheses.birthTime | ✓ values | ✗ | TODO: Add Rust value comparison |
| step4.new_hypotheses.birthLocation | ✓ values | ✗ | TODO: Add Rust value comparison |
| step5.normalized_hypotheses.w | ✓ sum only | ✓ sum only | TODO: Add BOTH full value comparison |
| step5.normalized_hypotheses.r | ✗ | ✗ | TODO: Add BOTH |
| step5.normalized_hypotheses.mu | ✗ | ✗ | TODO: Add BOTH |
| step5.normalized_hypotheses.Sigma | ✗ | ✗ | TODO: Add BOTH |
| step5.objects_likely_to_exist | ✓ values | ✗ | TODO: Add Rust value comparison |
| step6.cardinality_estimate | ✓ values | ✓ structure | TODO: Add Rust value comparison |
| step6.extraction_indices | ✓ values | ✓ structure | TODO: Add Rust value comparison |

### MULTISENSOR LMB FIXTURE (multisensor_lmb_step_by_step_seed42.json)

| Field | Python | Rust | Action |
|-------|--------|------|--------|
| step1.predicted_objects | ✓ values | ✓ values | COMPLETE |
| sensorUpdates[0].association.C | ✗ | ✓ values | TODO: Add Python |
| sensorUpdates[0].association.L | ✗ | ✗ | TODO: Add BOTH |
| sensorUpdates[0].association.R | ✗ | ✗ | TODO: Add BOTH |
| sensorUpdates[0].association.P | ✗ | ✓ values | TODO: Add Python |
| sensorUpdates[0].association.eta | ✗ | ✓ values | TODO: Add Python |
| sensorUpdates[0].posteriorParameters.w | ✗ | ✗ | TODO: Add BOTH |
| sensorUpdates[0].posteriorParameters.mu | ✗ | ✗ | TODO: Add BOTH |
| sensorUpdates[0].posteriorParameters.Sigma | ✗ | ✗ | TODO: Add BOTH |
| sensorUpdates[0].dataAssociation.r | ✗ | ✗ | TODO: Add BOTH |
| sensorUpdates[0].dataAssociation.W | ✗ | ✗ | TODO: Add BOTH |
| sensorUpdates[0].output.updated_objects | ✗ | ✓ structure | TODO: Add BOTH value comparison |
| sensorUpdates[1].* | same as [0] | same as [0] | Same TODOs as sensor 0 |
| stepFinal.n_estimated | ✓ values | ✓ structure | TODO: Add Rust value comparison |
| stepFinal.map_indices | ✗ | ✓ structure | TODO: Add BOTH value comparison |

### MULTISENSOR LMBM FIXTURE (multisensor_lmbm_step_by_step_seed42.json)

| Field | Python | Rust | Action |
|-------|--------|------|--------|
| step1.predicted_hypothesis.w | ✗ | ✗ | TODO: Add BOTH |
| step1.predicted_hypothesis.r | ✗ | ✓ values | TODO: Add Python |
| step1.predicted_hypothesis.mu | ✗ | ✗ | TODO: Add BOTH |
| step1.predicted_hypothesis.Sigma | ✗ | ✗ | TODO: Add BOTH |
| step1.predicted_hypothesis.birthTime | ✗ | ✓ values | TODO: Add Python |
| step1.predicted_hypothesis.birthLocation | ✗ | ✗ | TODO: Add BOTH |
| step2.L | ✗ | ✓ structure | TODO: Add BOTH value comparison |
| step2.posteriorParameters.r | ✗ | ✓ structure | TODO: Add BOTH value comparison |
| step2.posteriorParameters.mu | ✗ | ✗ | TODO: Add BOTH |
| step2.posteriorParameters.Sigma | ✗ | ✗ | TODO: Add BOTH |
| step3_gibbs.A | ✗ | ✓ structure | TODO: Add BOTH value comparison |
| step4.new_hypotheses.w | ✗ | ✓ structure | TODO: Add BOTH value comparison |
| step4.new_hypotheses.r | ✗ | ✓ structure | TODO: Add BOTH value comparison |
| step4.new_hypotheses.mu | ✗ | ✗ | TODO: Add BOTH |
| step4.new_hypotheses.Sigma | ✗ | ✗ | TODO: Add BOTH |
| step5.normalized_hypotheses.w | ✗ | ✓ sum only | TODO: Add BOTH full value comparison |
| step5.normalized_hypotheses.r | ✗ | ✗ | TODO: Add BOTH |
| step5.objects_likely_to_exist | ✗ | ✗ | TODO: Add BOTH |
| step6.cardinality_estimate | ✗ | ✓ structure | TODO: Add BOTH value comparison |
| step6.extraction_indices | ✗ | ✓ structure | TODO: Add BOTH value comparison |

---

## Prioritized TODO List

### HIGH PRIORITY - No Value Comparison Exists

1. **LMBM posteriorParameters.mu and Sigma** - Not compared in EITHER Python or Rust
2. **LMBM step5.normalized_hypotheses** - Only weight sum checked, not individual r/mu/Sigma
3. **Multisensor LMB sensorUpdates L, R, posteriorParams, dataAssoc** - Not tested
4. **Multisensor LMBM** - Most tests are structure-only

### MEDIUM PRIORITY - Missing in Rust (Python has it)

5. **LMB step2.posteriorParameters** - Python has it, add to Rust
6. **LMB step3b_gibbs, step3c_murtys** - Python has it, add to Rust
7. **LMB step4.posterior_objects** - Python has it, add to Rust
8. **LMB step5 cardinality** - Python has it, add to Rust
9. **LMBM step2.C, L** - Python has it, add to Rust
10. **LMBM step4/step5/step6** - Python has it (partial), add full to Rust

### LOW PRIORITY - Missing in Python (Rust has it)

11. **Multisensor LMB association.C, P, eta** - Rust has it, add to Python
12. **LMBM step2.posteriorParameters.r** - Rust has it, add to Python
13. **LMBM/Multisensor-LMBM prediction step** - Rust has it, add to Python

---

## Implementation Notes

### What "✓ values" means
- Actual numerical values are compared against fixture expected values
- Uses TOLERANCE = 1e-10 per CLAUDE.md
- Raises AssertionError on first mismatch

### What "✓ structure" means
- Only checks dimensions, non-null, basic validity
- Does NOT compare actual numerical values
- These need to be upgraded to value comparisons

### What "✓ sum only" means
- For normalized_hypotheses.w: only checks weights sum to 1.0
- Does not compare individual hypothesis weights

### Required Changes for Full Coverage

**Rust tests need to:**
1. Actually run the filter with fixture input (not just load fixture)
2. Get intermediate outputs from filter
3. Compare each value against fixture expected values

**Python tests already have the infrastructure but need:**
1. More comprehensive tests for multisensor fixtures
2. Tests for LMBM posteriorParameters.mu/Sigma
