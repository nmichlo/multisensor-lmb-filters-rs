# Claude Development Guide for multisensor-lmb-filters-rs

## CRITICAL: Test Integrity Rules

### NEVER Relax Tolerances to Make Tests Pass

**If a test fails, the code is wrong. Fix the code, not the test.**

Red flags that indicate you're doing the wrong thing:
- Tolerance > 1e-6 for numerical comparisons
- Adding "relaxed tolerance" comments
- Documenting "known differences" instead of fixing them
- Test name says "equivalence" but tolerance allows significant deviation

### Self-Check Before Committing Test Changes

Before modifying any test, ask yourself:

1. **Am I weakening the test?** If yes, STOP. Fix the code instead.
2. **Would this test catch the bug I just "fixed"?** If no, the test is now useless.
3. **Is the test name still accurate?** "test_equivalence" with 2% tolerance is a lie.
4. **Am I documenting a bug as a "known difference"?** That's not documentation, that's denial.

### Acceptable vs Unacceptable Responses to Failing Tests

**UNACCEPTABLE:**
```python
# Relaxed tolerance due to algorithmic differences
TOLERANCE = 1.5  # BAD: This makes the test meaningless
```

**UNACCEPTABLE:**
```python
"""Known difference: MATLAB merges, Rust prunes."""
# BAD: This documents the bug instead of fixing it
```

**ACCEPTABLE:**
```python
@pytest.mark.skip(reason="TODO: Implement GM merging - see issue #123")
def test_lmb_update_equivalence():
    # Keep tight tolerance, skip until algorithm is correct
```

**ACCEPTABLE:**
```rust
// TODO: Implement GM component merging to match MATLAB
// Reference: gm_merge_by_mahalanobis() in MATLAB
// Issue: #123
```

### The Tolerance Scale

| Tolerance | Meaning | Action |
|-----------|---------|--------|
| 1e-10 to 1e-14 | Floating point precision | Acceptable |
| 1e-6 to 1e-9 | Minor numerical differences | Investigate, usually acceptable |
| 1e-3 to 1e-5 | Suspicious | Likely a bug, investigate thoroughly |
| > 1e-2 | Test is useless | NEVER acceptable, fix the algorithm |

## THE GOLDEN RULE

**When tests fail showing numerical differences between MATLAB and Rust:**

**DO THIS FIRST (5 minutes):**
1. Find the MATLAB function that's failing (fixtures are the source of truth!)
2. Find the corresponding Rust function
3. Open them side-by-side
4. Compare line-by-line
5. Look for obvious bugs:
   - Wrong formulas (det(c*A) vs c^n*det(A))
   - Scalar vs array parameters
   - Missing loops or wrong loop bounds
   - Transposed matrices or wrong indexing

**ONLY IF THAT FAILS (rare):**
1. Create minimal debug fixture in MATLAB
2. Add targeted debug output to Rust
3. Compare intermediate values
4. Trace divergence point

**NEVER DO THIS:**
- Relax tolerance until test passes
- Document the difference as "known" or "expected"
- Claim different algorithms are "equivalent enough"

## Key Principle

**Fixtures are the source of truth!** The MATLAB-generated fixtures represent correct behavior.
If Rust produces different results, Rust is wrong (not the fixtures).

## Debugging Checklist

When Rust output != MATLAB fixture output:

- [ ] **Did I port the algorithm correctly?** (Read both implementations side-by-side)
- [ ] **Am I using the right parameters?** (Single vs multi-sensor, scalar vs array)
- [ ] **Are my formulas correct?** (Check mathematical properties like determinant rules)
- [ ] **Am I handling indices correctly?** (0-indexed vs 1-indexed, row vs column major)
- [ ] **Did I test with a minimal example?** (Simplest possible input that fails)

## Development Reminders

- **Run tests with --release**: `cargo test --release`
- **Tight tolerances are correct**: If tests require loose tolerances (>1e-6), there's a real bug
- **Push back** if you notice weird hacks to achieve MATLAB equivalence
- **Fixtures are truth**: Always trust fixture values over Rust outputs when debugging
- **Skip, don't weaken**: If you can't fix a test, skip it with a TODO, don't relax it

## Current Technical Debt

### GM Component Merging (NOT IMPLEMENTED)

**Status**: Rust uses weight-based pruning only. MATLAB uses Mahalanobis-distance merging.

**Impact**: `test_lmb_update_equivalence` currently uses an unacceptably loose tolerance (1.5).

**Required fix**: Implement `gm_merge_by_mahalanobis()` in Rust to match MATLAB behavior.

**Tracking**: This should be a GitHub issue, not a "known difference" we accept.
