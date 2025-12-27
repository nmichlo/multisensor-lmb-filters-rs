
# ALWAYS USE `uv run python` instead (BANNED: direct `python3` / `python` usage)

# ALWAYS USE `fd` or `rg` instead (BANNED: grep/find/xargs/cat/tree)

# AVOID `uv run python << 'EOF'` as far as possible, however, if for any reason you need to run this, then think HARD before doing it to minimize the number of times you need to run scripts this way.



## Debugging MATLAB→Rust Equivalence Issues

### THE GOLDEN RULE

**When tests fail showing numerical differences between MATLAB and Rust:**

**DO THIS FIRST (5 minutes):**
1. Find the MATLAB function that's failing
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

### Basic Debugging Checklist

When Rust output ≠ MATLAB output:

- [ ] **Did I port the algorithm correctly?** (Read both implementations side-by-side)
- [ ] **Am I using the right parameters?** (Single vs multi-sensor, scalar vs array)
- [ ] **Are my formulas correct?** (Check mathematical properties like determinant rules)
- [ ] **Am I handling indices correctly?** (0-indexed vs 1-indexed, row vs column major)
- [ ] **Did I test with a minimal example?** (Simplest possible input that fails)

## Development Reminders

- **Run tests with --release**: `cargo test --release`
- **Question assumptions**: If tests require loose tolerances (>1e-10), there's likely a real bug
- **Push back** if you notice weird hacks to achieve MATLAB equivalence
- **Keep PLAN.md updated** with current work status and next steps, this must be formatted as a todo list. Keep it up-to-date after making changes. Also add missing todos and descriptions if new work is started or new todos are identified.
