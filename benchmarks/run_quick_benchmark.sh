#!/usr/bin/env bash
#
# Quick Python-only benchmark (skip MATLAB).
# Usage: ./benchmarks/run_quick_benchmark.sh [timeout_seconds]
#

set -e

TIMEOUT="${1:-10}"
RESULTS_DIR="benchmarks/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "======================================="
echo "Quick Python Benchmark (no MATLAB)"
echo "Timeout: ${TIMEOUT}s per benchmark"
echo "======================================="
echo ""

# Clean previous results
echo "🧹 Cleaning previous results..."
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Create empty MATLAB results for consolidation
touch "$RESULTS_DIR/matlab_benchmarks.txt"
echo ""

# Run Python benchmarks
echo "🐍 Running Python benchmarks..."
uv run python benchmarks/run_python.py --timeout "$TIMEOUT" 2>&1 | tee "$RESULTS_DIR/python_benchmarks.txt"
echo ""

# Consolidate results
echo "📊 Consolidating results..."
uv run python benchmarks/consolidate_results.py \
    --python "$RESULTS_DIR/python_benchmarks.txt" \
    --output "$RESULTS_DIR/"
echo ""

# Show quick stats
echo "======================================="
echo "✅ Python benchmarks complete!"
echo "======================================="
echo ""
echo "Quick stats:"
grep -c "TIMEOUT" "$RESULTS_DIR/python_benchmarks.txt" | xargs echo "  Timeouts:"
grep -c "SKIP" "$RESULTS_DIR/python_benchmarks.txt" | xargs echo "  Skipped:"
grep -c "ERROR" "$RESULTS_DIR/python_benchmarks.txt" | xargs echo "  Errors:"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "View: cat $RESULTS_DIR/comparison_summary.md"
echo ""
