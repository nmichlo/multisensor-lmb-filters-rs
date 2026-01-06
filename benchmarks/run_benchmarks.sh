#!/usr/bin/env bash
#
# Unified benchmark runner for LMB filter performance comparison.
# Centralized orchestration with minimal per-language runners.
#
# On macOS: brew install coreutils (for gtimeout)
#
# Usage:
#   ./benchmarks/run_benchmarks.sh                    # Full suite
#   ./benchmarks/run_benchmarks.sh --quick            # Python only, short timeout
#   ./benchmarks/run_benchmarks.sh --timeout 30       # Custom timeout
#   ./benchmarks/run_benchmarks.sh --lang rust        # Single language
#   ./benchmarks/run_benchmarks.sh --filter LMB-LBP   # Single filter
#

set -e

# Force C locale for consistent decimal point handling
export LC_ALL=C

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results"
RUNNERS_DIR="$PROJECT_ROOT/benchmarks/runners"
SCENARIOS_DIR="$PROJECT_ROOT/benchmarks/scenarios"
MATLAB_DIR="$PROJECT_ROOT/../multisensor-lmb-filters"

# Defaults
TIMEOUT=30
LANGUAGES="octave,rust,python"
FILTER_PATTERN="all"
VERBOSE=0

# =============================================================================
# Filter Configuration Matrix
# Format: name|is_multi|octave_support
# =============================================================================

declare -a FILTER_CONFIGS=(
    # Single-sensor LMB
    "LMB-LBP|false|true"
    "LMB-Gibbs|false|true"
    "LMB-Murty|false|true"
    # Single-sensor LMBM
    "LMBM-Gibbs|false|true"
    "LMBM-Murty|false|true"
    # Multi-sensor LMB variants
    "AA-LMB-LBP|true|true"
    "IC-LMB-LBP|true|true"
    "PU-LMB-LBP|true|true"
    "GA-LMB-LBP|true|true"
    # Multi-sensor LMBM
    "MS-LMBM-Gibbs|true|true"
)

# =============================================================================
# Argument Parsing
# =============================================================================

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --quick           Run Python benchmarks only with 5s timeout"
    echo "  --timeout N       Set timeout per benchmark in seconds (default: 30)"
    echo "  --lang LANGS      Comma-separated list: octave,rust,python (default: all)"
    echo "  --filter PATTERN  Filter name pattern to run (default: all)"
    echo "  --verbose         Show verbose output"
    echo "  --help, -h        Show this help"
    echo ""
    echo "Available filters:"
    for config in "${FILTER_CONFIGS[@]}"; do
        IFS='|' read -r name is_multi octave_support <<< "$config"
        multi_tag=""
        [[ "$is_multi" == "true" ]] && multi_tag=" (multi-sensor)"
        octave_tag=""
        [[ "$octave_support" == "false" ]] && octave_tag=" [no octave]"
        echo "  $name$multi_tag$octave_tag"
    done
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            LANGUAGES="python"
            TIMEOUT=5
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --lang)
            LANGUAGES="$2"
            shift 2
            ;;
        --filter)
            FILTER_PATTERN="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_DIR"

# Find timeout command (gtimeout on macOS with coreutils)
if command -v gtimeout &> /dev/null; then
    TIMEOUT_CMD="gtimeout"
elif command -v timeout &> /dev/null; then
    TIMEOUT_CMD="timeout"
else
    echo "Error: timeout command not found. On macOS, run: brew install coreutils"
    exit 1
fi

echo "======================================="
echo "LMB Filter Benchmark Suite"
echo "======================================="
echo "Timeout: ${TIMEOUT}s"
echo "Languages: $LANGUAGES"
echo "Filter: $FILTER_PATTERN"
echo ""

# =============================================================================
# Skip Tracking (per filter+language)
# Uses a simple file-based approach for bash 3.x compatibility
# =============================================================================

SKIP_FILE=$(mktemp)
trap "rm -f $SKIP_FILE" EXIT

should_skip() {
    local filter_name="$1"
    local lang="$2"
    local key="${filter_name}:${lang}"
    grep -qxF "$key" "$SKIP_FILE" 2>/dev/null
}

mark_timed_out() {
    local filter_name="$1"
    local lang="$2"
    local key="${filter_name}:${lang}"
    echo "$key" >> "$SKIP_FILE"
}

# =============================================================================
# Scenario Discovery & Sorting
# =============================================================================

get_sorted_scenarios() {
    # Find all bouncing_*.json files and sort by (n, s)
    for path in "$SCENARIOS_DIR"/bouncing_*.json; do
        [[ -f "$path" ]] || continue
        name=$(basename "$path" .json)
        n=$(echo "$name" | grep -oE 'n[0-9]+' | grep -oE '[0-9]+')
        s=$(echo "$name" | grep -oE 's[0-9]+' | grep -oE '[0-9]+')
        printf "%04d%04d %s\n" "$n" "$s" "$path"
    done | sort -n | cut -d' ' -f2
}

# =============================================================================
# Filter Applicability
# =============================================================================

is_filter_applicable() {
    local filter_name="$1"
    local num_sensors="$2"
    local lang="$3"

    for config in "${FILTER_CONFIGS[@]}"; do
        IFS='|' read -r name is_multi octave_support <<< "$config"
        if [[ "$name" == "$filter_name" ]]; then
            # Check multi-sensor compatibility
            if [[ "$is_multi" == "true" && "$num_sensors" -eq 1 ]]; then
                return 1  # Multi-sensor filter on single-sensor scenario
            fi
            # Check Octave support
            if [[ "$lang" == "octave" && "$octave_support" == "false" ]]; then
                return 1
            fi
            return 0
        fi
    done
    return 1
}

get_num_sensors() {
    local scenario_path="$1"
    # Extract num_sensors from JSON using grep (avoid jq dependency)
    grep -oE '"num_sensors"[[:space:]]*:[[:space:]]*[0-9]+' "$scenario_path" | grep -oE '[0-9]+$'
}

# =============================================================================
# Runner Functions
# =============================================================================

run_benchmark() {
    local scenario_path="$1"
    local filter_name="$2"
    local lang="$3"
    local result
    local exit_code

    case "$lang" in
        octave)
            # Run Octave with timeout, LMB_SILENT suppresses progress output
            # Capture both stdout and exit code separately
            local octave_output
            octave_output=$(LMB_SILENT=1 $TIMEOUT_CMD "${TIMEOUT}s" octave --no-gui --path "$RUNNERS_DIR" \
                --eval "run_octave('$scenario_path', '$filter_name')" 2>&1) || exit_code=$?
            # Extract the timing number from output (last line matching a number)
            result=$(echo "$octave_output" | grep -oE '^[0-9]+\.?[0-9]*$' | tail -1)
            ;;
        rust)
            result=$($TIMEOUT_CMD "${TIMEOUT}s" "$PROJECT_ROOT/target/release/benchmark_single" \
                --scenario "$scenario_path" --filter "$filter_name" 2>/dev/null) || exit_code=$?
            ;;
        python)
            result=$($TIMEOUT_CMD "${TIMEOUT}s" uv run python \
                "$RUNNERS_DIR/run_python.py" \
                --scenario "$scenario_path" --filter "$filter_name" 2>/dev/null) || exit_code=$?
            ;;
    esac

    # Check for timeout (exit code 124)
    if [[ "${exit_code:-0}" -eq 124 ]]; then
        echo "TIMEOUT"
        return
    fi

    # Validate result is a number
    if [[ "$result" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "$result"
    else
        echo "ERROR"
    fi
}

# =============================================================================
# Build Rust if needed
# =============================================================================

if [[ "$LANGUAGES" == *"rust"* ]]; then
    echo "Building Rust benchmark runner..."
    cargo build --release --bin benchmark_single 2>&1 | tail -3
    echo ""
fi

# =============================================================================
# Build Python bindings if needed
# =============================================================================

if [[ "$LANGUAGES" == *"python"* ]]; then
    echo "Building Python bindings..."
    uv run maturin develop --release 2>&1 | tail -3
    echo ""
fi

# =============================================================================
# Check Octave/MATLAB availability
# =============================================================================

if [[ "$LANGUAGES" == *"octave"* ]]; then
    if ! command -v octave &> /dev/null; then
        echo "Warning: Octave not found, skipping Octave benchmarks"
        LANGUAGES="${LANGUAGES//octave/}"
        LANGUAGES="${LANGUAGES//,,/,}"
        LANGUAGES="${LANGUAGES#,}"
        LANGUAGES="${LANGUAGES%,}"
    elif [[ ! -d "$MATLAB_DIR" ]]; then
        echo "Warning: MATLAB library not found at $MATLAB_DIR, skipping Octave benchmarks"
        LANGUAGES="${LANGUAGES//octave/}"
        LANGUAGES="${LANGUAGES//,,/,}"
        LANGUAGES="${LANGUAGES#,}"
        LANGUAGES="${LANGUAGES%,}"
    fi
fi

# =============================================================================
# Main Benchmark Loop
# =============================================================================

RESULTS_FILE="$RESULTS_DIR/benchmarks_$(date +%Y%m%d_%H%M%S).csv"

# Print CSV header (easier to parse for README generation)
echo "objects,sensors,filter,lang,time_ms" > "$RESULTS_FILE"

# Also print human-readable header to console
printf "%-8s | %-8s | %-18s | %-8s | %9s\n" "Objects" "Sensors" "Filter" "Lang" "Time(ms)"
printf "%s\n" "$(printf '%.0s-' {1..60})"

IFS=',' read -ra LANG_ARRAY <<< "$LANGUAGES"

# Iterate through scenarios
for scenario_path in $(get_sorted_scenarios); do
    scenario_name=$(basename "$scenario_path" .json)
    num_sensors=$(get_num_sensors "$scenario_path")

    # Extract n and s from scenario name
    num_objects=$(echo "$scenario_name" | grep -oE 'n[0-9]+' | grep -oE '[0-9]+')

    # Iterate through filters
    for filter_config in "${FILTER_CONFIGS[@]}"; do
        IFS='|' read -r filter_name is_multi octave_support <<< "$filter_config"

        # Check if filter matches --filter arg
        if [[ "$FILTER_PATTERN" != "all" && "$filter_name" != *"$FILTER_PATTERN"* ]]; then
            continue
        fi

        # Iterate through languages
        for lang in "${LANG_ARRAY[@]}"; do
            # Check applicability
            if ! is_filter_applicable "$filter_name" "$num_sensors" "$lang"; then
                continue
            fi

            # Check skip
            if should_skip "$filter_name" "$lang"; then
                echo "$num_objects,$num_sensors,$filter_name,$lang,SKIP" >> "$RESULTS_FILE"
                printf "%-8s | %-8s | %-18s | %-8s | %9s\n" \
                    "$num_objects" "$num_sensors" "$filter_name" "$lang" "SKIP"
                continue
            fi

            # Run benchmark
            [[ $VERBOSE -eq 1 ]] && echo "Running: n=$num_objects s=$num_sensors / $filter_name / $lang" >&2
            result=$(run_benchmark "$scenario_path" "$filter_name" "$lang")

            if [[ "$result" == "TIMEOUT" ]]; then
                mark_timed_out "$filter_name" "$lang"
                echo "$num_objects,$num_sensors,$filter_name,$lang,TIMEOUT" >> "$RESULTS_FILE"
                printf "%-8s | %-8s | %-18s | %-8s | %9s\n" \
                    "$num_objects" "$num_sensors" "$filter_name" "$lang" "TIMEOUT"
            elif [[ "$result" == "ERROR" ]]; then
                echo "$num_objects,$num_sensors,$filter_name,$lang,ERROR" >> "$RESULTS_FILE"
                printf "%-8s | %-8s | %-18s | %-8s | %9s\n" \
                    "$num_objects" "$num_sensors" "$filter_name" "$lang" "ERROR"
            else
                echo "$num_objects,$num_sensors,$filter_name,$lang,$result" >> "$RESULTS_FILE"
                printf "%-8s | %-8s | %-18s | %-8s | %9.1f\n" \
                    "$num_objects" "$num_sensors" "$filter_name" "$lang" "$result"
            fi
        done
    done
done

# =============================================================================
# Generate Summary
# =============================================================================

echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

# Count results
total=$(tail -n +2 "$RESULTS_FILE" | wc -l | tr -d ' ')
timeouts=$(grep -c 'TIMEOUT' "$RESULTS_FILE" 2>/dev/null || echo 0)
errors=$(grep -c 'ERROR' "$RESULTS_FILE" 2>/dev/null || echo 0)
skips=$(grep -c 'SKIP' "$RESULTS_FILE" 2>/dev/null || echo 0)
successful=$((total - timeouts - errors - skips))

echo "Summary:"
echo "  Successful: $successful"
echo "  Timeouts:   $timeouts"
echo "  Errors:     $errors"
echo "  Skipped:    $skips"
echo ""

# =============================================================================
# Generate README_BENCHMARKS.md
# =============================================================================

README_FILE="$PROJECT_ROOT/README_BENCHMARKS.md"

generate_readme() {
    cat << 'HEADER'
# LMB Filter Benchmark Results

HEADER
    echo "*Generated: $(date '+%Y-%m-%d %H:%M:%S')*"
    echo ""
    cat << 'OVERVIEW'
## Overview

This benchmark compares implementations of the LMB (Labeled Multi-Bernoulli) filter:

| Implementation | Description |
|----------------|-------------|
| **Octave/MATLAB** | Original reference implementation (interpreted) |
| **Rust** | Native Rust binary compiled with `--release` |
| **Python** | Python calling Rust via PyO3/maturin bindings |

## Methodology

OVERVIEW
    echo "- **Timeout**: ${TIMEOUT} seconds per scenario"
    cat << 'METHOD'
- **Thresholds**: existence=1e-3, gm_weight=1e-4, max_components=100, gm_merge=∞
- **Association**: LBP (100 iterations, tol 1e-6), Gibbs (1000 samples), Murty (25 assignments)
- **RNG Seed**: 42 (deterministic across all implementations)

## Results

METHOD

    # Format a result value
    format_result() {
        local val="$1"
        if [[ -z "$val" ]]; then
            echo "-"
        elif [[ "$val" == "TIMEOUT" || "$val" == "ERROR" || "$val" == "SKIP" ]]; then
            echo "$val"
        else
            printf "%.1f" "$val"
        fi
    }

    # Get all unique (n,s) combinations from scenarios directory
    all_scenarios=""
    for path in "$SCENARIOS_DIR"/bouncing_*.json; do
        [[ -f "$path" ]] || continue
        name=$(basename "$path" .json)
        n=$(echo "$name" | grep -oE 'n[0-9]+' | grep -oE '[0-9]+')
        s=$(echo "$name" | grep -oE 's[0-9]+' | grep -oE '[0-9]+')
        all_scenarios="$all_scenarios$n,$s\n"
    done
    sorted_scenarios=$(echo -e "$all_scenarios" | sort -t',' -k1,1n -k2,2n -u | grep -v '^$')

    # Iterate through all filters from config
    for filter_config in "${FILTER_CONFIGS[@]}"; do
        IFS='|' read -r filter_name is_multi octave_support <<< "$filter_config"

        echo "### $filter_name"
        echo ""
        echo "| Objects | Sensors | Octave (ms) | Rust (ms) | Python (ms) |"
        echo "|---------|---------|-------------|-----------|-------------|"

        # Iterate through all scenarios
        echo "$sorted_scenarios" | while IFS=',' read -r n s; do
            [[ -z "$n" ]] && continue

            # Check if this filter applies to this scenario
            # Multi-sensor filters need s > 1, single-sensor filters work on any
            applicable="yes"
            if [[ "$is_multi" == "true" && "$s" -eq 1 ]]; then
                applicable="no"
            fi

            if [[ "$applicable" == "no" ]]; then
                # Filter not applicable to this scenario
                echo "| $n | $s | - | - | - |"
            else
                # Look up results from CSV
                octave_result=$(grep "^$n,$s,$filter_name,octave," "$RESULTS_FILE" 2>/dev/null | cut -d',' -f5 | head -1)
                rust_result=$(grep "^$n,$s,$filter_name,rust," "$RESULTS_FILE" 2>/dev/null | cut -d',' -f5 | head -1)
                python_result=$(grep "^$n,$s,$filter_name,python," "$RESULTS_FILE" 2>/dev/null | cut -d',' -f5 | head -1)

                oct_fmt=$(format_result "$octave_result")
                rust_fmt=$(format_result "$rust_result")
                py_fmt=$(format_result "$python_result")

                echo "| $n | $s | $oct_fmt | $rust_fmt | $py_fmt |"
            fi
        done
        echo ""
    done

    cat << 'NOTES'
## Notes

- **Octave/MATLAB** is interpreted and significantly slower by design
- **Rust** and **Python** run the same compiled Rust code; small differences are PyO3 overhead
- **TIMEOUT** means the benchmark exceeded the time limit
- **ERROR** indicates a runtime error (check logs for details)
- **SKIP** means a previous scenario timed out, so harder scenarios were skipped
- **-** means not applicable (e.g., multi-sensor filter on single-sensor scenario) or not run

NOTES
}

echo "Generating README_BENCHMARKS.md..."
generate_readme > "$README_FILE"
echo "README saved to: $README_FILE"
echo ""
echo "Done!"
