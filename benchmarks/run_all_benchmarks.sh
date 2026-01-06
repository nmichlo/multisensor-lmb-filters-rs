#!/usr/bin/env bash
#
# Run all benchmarks (MATLAB + Python), consolidate results, and generate report.
# Usage: ./benchmarks/run_all_benchmarks.sh
#

set -e  # Exit on error

# Configuration
TIMEOUT=10
MATLAB_DIR="../multisensor-lmb-filters"
RESULTS_DIR="benchmarks/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "======================================="
echo "LMB Filter Benchmark Suite"
echo "======================================="
echo ""

# Clean previous results
echo "🧹 Cleaning previous results..."
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
echo "✓ Cleaned"
echo ""

# Run MATLAB benchmarks
echo "🔬 Running MATLAB benchmarks..."
if [ -d "$MATLAB_DIR" ]; then
    cd "$MATLAB_DIR"
    octave --eval "addpath('benchmarks'); run_benchmarks" 2>&1 | tee temp_matlab_results.txt

    # Clean MATLAB output (remove progress indicators)
    python3 -c "
import re
with open('temp_matlab_results.txt') as f:
    content = f.read()
cleaned = re.sub(r' \(100\) [\d ]+', '', content)
with open('temp_matlab_results.txt', 'w') as f:
    f.write(cleaned)
"

    mv temp_matlab_results.txt "$PROJECT_ROOT/$RESULTS_DIR/matlab_benchmarks.txt"
    cd "$PROJECT_ROOT"
    echo "✓ MATLAB benchmarks complete"
else
    echo "⚠ MATLAB directory not found, skipping MATLAB benchmarks"
    touch "$RESULTS_DIR/matlab_benchmarks.txt"
fi
echo ""

# Run Python benchmarks
echo "🐍 Running Python benchmarks..."
uv run python benchmarks/run_python.py --timeout "$TIMEOUT" 2>&1 | tee "$RESULTS_DIR/python_benchmarks.txt"
echo "✓ Python benchmarks complete"
echo ""

# Consolidate results
echo "📊 Consolidating results..."
uv run python benchmarks/consolidate_results.py \
    --python "$RESULTS_DIR/python_benchmarks.txt" \
    --matlab "$RESULTS_DIR/matlab_benchmarks.txt" \
    --output "$RESULTS_DIR/"
echo "✓ Results consolidated"
echo ""

# Generate summary report
echo "📝 Generating final report..."
uv run python - <<'PYTHON_SCRIPT'
import json
import re
from pathlib import Path

# Load consolidated data
results_dir = Path("benchmarks/results")
with open(results_dir / "comparison_data.json") as f:
    data = json.load(f)

# Count stats
matlab_completed = sum(1 for v in data.values() if v.get("matlab", {}).get("status") == "ok")
matlab_error = sum(1 for v in data.values() if v.get("matlab", {}).get("status") == "error")
python_completed = sum(1 for v in data.values() if v.get("python", {}).get("status") == "ok")
python_timeout = sum(1 for v in data.values() if v.get("python", {}).get("status") == "timeout")
python_skip = sum(1 for v in data.values() if v.get("python", {}).get("status") == "skip")

# Find best/worst performances
single_sensor_comparisons = []
for key, impls in data.items():
    scenario, filter_name = key.split("/")
    matlab = impls.get("matlab", {})
    python = impls.get("python", {})

    if (matlab.get("status") == "ok" and python.get("status") == "ok" and
        matlab.get("mean_ms") and python.get("mean_ms")):
        ratio = python["mean_ms"] / matlab["mean_ms"]
        single_sensor_comparisons.append({
            "scenario": scenario,
            "filter": filter_name,
            "matlab_ms": matlab["mean_ms"],
            "python_ms": python["mean_ms"],
            "ratio": ratio
        })

# Generate quick summary
summary = f"""
# Benchmark Summary

**Date:** {Path("benchmarks/results/comparison_summary.md").stat().st_mtime}
**Total Benchmarks:** {len(data)}

## Results

### MATLAB
- ✓ Completed: {matlab_completed}
- ✗ Error: {matlab_error}

### Python
- ✓ Completed: {python_completed}
- ⏱ Timeout: {python_timeout}
- ⏭ Skipped: {python_skip}

## Performance Highlights

### Best Python Performance (vs MATLAB)
"""

if single_sensor_comparisons:
    best = sorted(single_sensor_comparisons, key=lambda x: x["ratio"])[:5]
    for comp in best:
        summary += f"- **{comp['filter']}** on {comp['scenario']}: {comp['ratio']:.1f}× slower ({comp['matlab_ms']:.1f}ms → {comp['python_ms']:.1f}ms)\n"

    summary += "\n### Worst Python Performance (vs MATLAB)\n"
    worst = sorted(single_sensor_comparisons, key=lambda x: x["ratio"], reverse=True)[:5]
    for comp in worst:
        summary += f"- **{comp['filter']}** on {comp['scenario']}: {comp['ratio']:.1f}× slower ({comp['matlab_ms']:.1f}ms → {comp['python_ms']:.1f}ms)\n"

summary += """

## Files Generated
- `comparison_summary.md` - Full comparison table
- `all_results.csv` - Raw data
- `comparison_data.json` - Structured JSON
- `BENCHMARK_REPORT.md` - Detailed analysis
- `QUICK_SUMMARY.txt` - This file

See BENCHMARK_REPORT.md for detailed analysis and recommendations.
"""

with open(results_dir / "QUICK_SUMMARY.txt", "w") as f:
    f.write(summary)

print("✓ Quick summary generated")
PYTHON_SCRIPT

echo ""
echo "======================================="
echo "✅ All benchmarks complete!"
echo "======================================="
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "📄 View results:"
echo "  - Quick summary:  cat $RESULTS_DIR/QUICK_SUMMARY.txt"
echo "  - Full report:    cat $RESULTS_DIR/BENCHMARK_REPORT.md"
echo "  - Comparison:     cat $RESULTS_DIR/comparison_summary.md"
echo ""
