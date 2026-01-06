# Quick Start Guide - LMB Filter Benchmarks

## TL;DR

```bash
# Run everything (MATLAB + Python)
./benchmarks/run_all_benchmarks.sh

# OR run Python only (faster)
./benchmarks/run_quick_benchmark.sh

# View results
cat benchmarks/results/BENCHMARK_REPORT.md
```

---

## What Gets Benchmarked?

### Implementations
- **MATLAB** (reference implementation) - Single-sensor LMB only
- **Python** (via Rust bindings) - All filter types
- **Rust** (optional, very slow) - All filter types with statistical analysis

### Filter Types
- **LMB** - Basic labeled multi-Bernoulli filter
- **LMBM** - LMB with multi-Bernoulli components
- **AA/GA/PU/IC-LMB** - Multi-sensor fusion variants
- **MS-LMBM** - Multi-sensor LMBM

### Associators (Data Association Methods)
- **LBP** - Loopy Belief Propagation (fastest)
- **Gibbs** - Gibbs sampling (moderate)
- **Murty** - Murty's algorithm (slowest in Python)

### Scenarios
10 test scenarios with varying complexity:
- **n5, n10, n20, n50** - Number of objects to track
- **s1, s2, s4, s8** - Number of sensors

---

## Understanding the Output

### Result Files

After running benchmarks, find results in `benchmarks/results/`:

```
results/
├── BENCHMARK_REPORT.md      ← Read this first! Full analysis
├── QUICK_SUMMARY.txt         ← Quick stats and highlights
├── comparison_summary.md     ← Full comparison table
├── all_results.csv           ← Raw data for analysis
├── comparison_data.json      ← Structured data
├── python_benchmarks.txt     ← Python raw output
└── matlab_benchmarks.txt     ← MATLAB raw output
```

### Sample Results

From `comparison_summary.md`:

```markdown
| Scenario | Filter | MATLAB (ms) | Python (ms) | Speedup |
|----------|--------|-------------|-------------|---------|
| n5_s1    | LMB-LBP | 20.9 | 52.9 | 0.40× |
```

**Interpretation:**
- MATLAB: 20.9ms
- Python: 52.9ms
- Python is **2.5× slower** (0.40× speed = 1/2.5)

### Status Codes

- ✓ **OK** - Completed successfully
- ⏱ **TIMEOUT** - Exceeded 10s limit
- ⏭ **SKIP** - Auto-skipped (harder scenarios after timeout)
- ✗ **ERROR** - Runtime error
- **N/A** - Not available (e.g., MATLAB doesn't support multi-sensor)

---

## Common Scenarios

### "I just want quick Python benchmarks"

```bash
./benchmarks/run_quick_benchmark.sh 5
# Uses 5-second timeout, skips MATLAB
```

### "I want full comparison with MATLAB"

```bash
./benchmarks/run_all_benchmarks.sh
# ~10-20 minutes total
```

### "I only care about LBP associator"

```bash
uv run python benchmarks/run_python.py --assoc LBP --timeout 10 > results.txt
```

### "I want to test one specific scenario"

```bash
uv run python benchmarks/run_python.py --scenario n5_s1 --filter LMB --assoc LBP
```

---

## Key Insights from Latest Results

### MATLAB vs Python Performance

**Single-Sensor LMB Filters:**
- LBP: Python is ~2.5-7× slower
- Gibbs: Python is ~2-8× slower
- Murty: Python is **50-400× slower** ⚠️

**Why is MATLAB faster?**
1. Highly optimized matrix operations
2. JIT compilation for hot loops
3. Decades of performance tuning

### Multi-Sensor Filters (Python Only)

Best performers for complex scenarios:
- **GA-LMB-LBP**: 62ms (n5_s1) → 6.8s (n50_s8)
- **PU-LMB-LBP**: 56ms (n5_s1) → 208ms (n20_s2)
- **MS-LMBM-Murty**: 7ms (n5_s1) → 170ms (n10_s2)

### Smart Timeout Handling

Once a filter times out, harder scenarios are automatically skipped:

```
n5_s1  | LMBM-Gibbs | TIMEOUT
n5_s2  | LMBM-Gibbs | SKIP      ← Auto-skipped
n10_s1 | LMBM-Gibbs | SKIP      ← Auto-skipped
...
```

**Result:** Saved ~13 hours of benchmark time! 🎉

---

## Troubleshooting

### "Python benchmarks are taking forever"

**Normal!** Complex scenarios (n20+, s4+) with Gibbs/Murty can take 5-10s each.
Use shorter timeout:
```bash
./benchmarks/run_quick_benchmark.sh 5
```

### "All MATLAB results show ERROR"

**Expected!** MATLAB reference implementation only supports single-sensor scenarios.
Multi-sensor scenarios (s2, s4, s8) will error out - this is normal.

### "Rust benchmarks are too slow"

**Very normal!** Criterion runs extensive statistical analysis (10+ iterations).
- For quick checks: Use Python benchmarks
- For detailed profiling: Run specific scenarios only

---

## Next Steps

1. **View the report**: `cat benchmarks/results/BENCHMARK_REPORT.md`
2. **Check specific results**: `cat benchmarks/results/comparison_summary.md | grep n5_s1`
3. **Analyze data**: Open `benchmarks/results/all_results.csv` in Excel/Python
4. **Read full docs**: `cat benchmarks/README.md`

---

## Questions?

See `benchmarks/README.md` for full documentation including:
- Manual benchmark commands
- Adding new scenarios
- Customizing filters
- Understanding the code structure
- Debugging tips
