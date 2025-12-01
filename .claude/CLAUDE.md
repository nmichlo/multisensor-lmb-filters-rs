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

## Documents

- .md docs containing history of work comleted have been moved to `docs`
  - docs/00_io gives info on all the algorithms expected inputs and outputs
  - docs/01_migration/migration_plan.md was the original migration plan from octave/matlab to rust (this is now complete, you probably DONT need to look at this)
  - docs/o1_migration/migration_history.md is a useful summary of how everything was migrated
  - docs/02_migration_comparison/comparison_analysis_summary.md is a useful comparison of original matlab to migrated code, this shows where we could have improved the rust API
  - docs/03_optimisations/improvements_possible.md ranks various possible improvements to the rust code including performance wise and api wise
  - docs/03_optimisations/profiling_results.md gives profiling results over the various algorithms --> this should be updated over time as we make improvements, with a history of improvements kept in `docs/03_optimisations/changelog.md` this changelog should include any future changes we make.
  
- if you come across code and notice any code that can be de-duplicated or should be turned into traits make not of it in ./docs/03_optimisations/todos.md
- use hotpath when running benchmarks with the flag --features='hotpath,hotpath-alloc'