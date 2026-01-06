#!/usr/bin/env bash
# Requires bash 4+ for associative arrays. On macOS, use: brew install bash
#
# Unified benchmark runner for LMB filter performance comparison.
# Centralized orchestration with minimal per-language runners.
#
# Usage:
#   ./benchmarks/run_benchmarks.sh                    # Full suite
#   ./benchmarks/run_benchmarks.sh --quick            # Python only, short timeout
#   ./benchmarks/run_benchmarks.sh --timeout 30       # Custom timeout
#   ./benchmarks/run_benchmarks.sh --lang rust        # Single language
#   ./benchmarks/run_benchmarks.sh --filter LMB-LBP   # Single filter
#

set -e

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
    # Multi-sensor LMB variants (no Octave support)
    "AA-LMB-LBP|true|false"
    "IC-LMB-LBP|true|false"
    "PU-LMB-LBP|true|false"
    "GA-LMB-LBP|true|false"
    # Multi-sensor LMBM
    "MS-LMBM-Gibbs|true|false"
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
            # Run Octave with timeout, capture only the last numeric line
            result=$(timeout "${TIMEOUT}s" octave --no-gui --path "$RUNNERS_DIR" \
                --eval "run_octave('$scenario_path', '$filter_name')" 2>/dev/null | \
                grep -oE '^[0-9]+\.[0-9]+$' | tail -1) || exit_code=$?
            ;;
        rust)
            result=$(timeout "${TIMEOUT}s" "$PROJECT_ROOT/target/release/benchmark_single" \
                --scenario "$scenario_path" --filter "$filter_name" 2>/dev/null) || exit_code=$?
            ;;
        python)
            result=$(timeout "${TIMEOUT}s" uv run python \
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

RESULTS_FILE="$RESULTS_DIR/benchmarks_$(date +%Y%m%d_%H%M%S).txt"

# Print header
printf "%-22s | %-18s | %-8s | %9s\n" "Scenario" "Filter" "Lang" "Time(ms)" | tee "$RESULTS_FILE"
printf "%s\n" "$(printf '%.0s-' {1..65})" | tee -a "$RESULTS_FILE"

IFS=',' read -ra LANG_ARRAY <<< "$LANGUAGES"

# Iterate through scenarios
for scenario_path in $(get_sorted_scenarios); do
    scenario_name=$(basename "$scenario_path" .json)
    num_sensors=$(get_num_sensors "$scenario_path")

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
                printf "%-22s | %-18s | %-8s | %9s\n" \
                    "$scenario_name" "$filter_name" "$lang" "SKIP" | tee -a "$RESULTS_FILE"
                continue
            fi

            # Run benchmark
            [[ $VERBOSE -eq 1 ]] && echo "Running: $scenario_name / $filter_name / $lang" >&2
            result=$(run_benchmark "$scenario_path" "$filter_name" "$lang")

            if [[ "$result" == "TIMEOUT" ]]; then
                mark_timed_out "$filter_name" "$lang"
                printf "%-22s | %-18s | %-8s | %9s\n" \
                    "$scenario_name" "$filter_name" "$lang" "TIMEOUT" | tee -a "$RESULTS_FILE"
            elif [[ "$result" == "ERROR" ]]; then
                printf "%-22s | %-18s | %-8s | %9s\n" \
                    "$scenario_name" "$filter_name" "$lang" "ERROR" | tee -a "$RESULTS_FILE"
            else
                printf "%-22s | %-18s | %-8s | %9.1f\n" \
                    "$scenario_name" "$filter_name" "$lang" "$result" | tee -a "$RESULTS_FILE"
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
total=$(grep -c '|' "$RESULTS_FILE" 2>/dev/null || echo 0)
timeouts=$(grep -c 'TIMEOUT' "$RESULTS_FILE" 2>/dev/null || echo 0)
errors=$(grep -c 'ERROR' "$RESULTS_FILE" 2>/dev/null || echo 0)
skips=$(grep -c 'SKIP' "$RESULTS_FILE" 2>/dev/null || echo 0)
successful=$((total - timeouts - errors - skips - 1))  # -1 for header

echo "Summary:"
echo "  Successful: $successful"
echo "  Timeouts:   $timeouts"
echo "  Errors:     $errors"
echo "  Skipped:    $skips"
echo ""
echo "Done!"
