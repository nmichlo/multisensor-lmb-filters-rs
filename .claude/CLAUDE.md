# Debugging MATLAB→Rust Equivalence Issues

## ⚠️ THE GOLDEN RULE ⚠️

**When tests fail showing numerical differences between MATLAB and Rust:**

### DO THIS FIRST (5 minutes):
```
1. Find the MATLAB function that's failing
2. Find the corresponding Rust function
3. Open them side-by-side
4. Compare line-by-line
5. Look for obvious bugs:
   - Wrong formulas (det(c*A) vs c^n*det(A))
   - Scalar vs array parameters (detection_probability vs detection_probability_multisensor)
   - Single-sensor vs multi-sensor parameters (c vs c_multisensor)
   - Missing loops or wrong loop bounds
   - Transposed matrices or wrong indexing
```

### ONLY IF THAT FAILS (rare):
```
1. Create minimal debug fixture in MATLAB
2. Add targeted debug output to Rust
3. Compare intermediate values
4. Trace divergence point
```

## Why This Matters

**Bug #19 Example** - Wasted 3+ hours on runtime debugging when bugs were obvious in code:
- ❌ Used: `eta = -0.5 * (2*pi * det(Z)).ln()`
- ✅ Should be: `eta = -0.5 * ((2*pi)^n * det(Z)).ln()`
- ❌ Used: `model.detection_probability` (scalar)
- ✅ Should be: `model.detection_probability_multisensor[s]` (per-sensor array)

**These were visible in the source code.** No execution tracing needed.

## Basic Debugging Checklist

When Rust output ≠ MATLAB output:

- [ ] **Did I port the algorithm correctly?** (Read both implementations side-by-side)
- [ ] **Am I using the right parameters?** (Single vs multi-sensor, scalar vs array)
- [ ] **Are my formulas correct?** (Check mathematical properties like determinant rules)
- [ ] **Am I handling indices correctly?** (0-indexed vs 1-indexed, row vs column major)
- [ ] **Did I test with a minimal example?** (Simplest possible input that fails)

Only after checking all these should you resort to execution tracing.

## Other Important Reminders

- **Push back** if you notice weird hacks to achieve MATLAB equivalence
- **Push back** if 100% equivalence requires unreasonable complexity
- **Run full tests** after changing common/shared Rust code (might break other algorithms)
- **Use correct tools**: `rg` for search, NOT grep/find/cat/xargs
- **Question assumptions**: If tests require loose tolerances (>1e-10), there's likely a real bug
