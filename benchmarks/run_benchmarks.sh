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
#   ./benchmarks/run_benchmarks.sh --get-config       # Compare configs across languages
#   ./benchmarks/run_benchmarks.sh --get-config --skip-run  # Config only, no benchmarks
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
RUNNERS_DIR="$PROJECT_ROOT/benchmarks/run_benchmarks"
SCENARIOS_DIR="$PROJECT_ROOT/tests/fixtures"
MATLAB_DIR="$PROJECT_ROOT/vendor/multisensor-lmb-filters"

# Defaults
TIMEOUT=30
LANGUAGES="octave,rust,python"
FILTER_PATTERN="all"
VERBOSE=0
USE_CACHE=1
CONTINUE_MODE=0
GET_CONFIG=0
SKIP_RUN=0

# Cache file for persistent results
CACHE_FILE="$RESULTS_DIR/cache.csv"

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
    echo "  --no-cache        Ignore cached results, run everything fresh"
    echo "  --continue        Continue from first timeout per filter/lang (for longer runs)"
    echo "  --get-config      Print filter configs for comparison (JSON)"
    echo "  --skip-run        Skip running benchmarks (useful with --get-config)"
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
        --no-cache)
            USE_CACHE=0
            shift
            ;;
        --continue)
            CONTINUE_MODE=1
            shift
            ;;
        --get-config)
            GET_CONFIG=1
            shift
            ;;
        --skip-run)
            SKIP_RUN=1
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
[[ $USE_CACHE -eq 1 ]] && echo "Cache: enabled" || echo "Cache: disabled"
[[ $CONTINUE_MODE -eq 1 ]] && echo "Mode: continue from timeouts"
[[ $GET_CONFIG -eq 1 ]] && echo "Config: will print filter configs"
[[ $SKIP_RUN -eq 1 ]] && echo "Skip run: benchmarks will be skipped"
echo ""

# =============================================================================
# Cache Functions
# =============================================================================

init_cache() {
    if [[ ! -f "$CACHE_FILE" ]]; then
        echo "objects,sensors,filter,lang,timeout,time_ms,status,timestamp" > "$CACHE_FILE"
    fi
}

# Check cache for a result. Returns:
#   "OK:time_ms" if we have a valid cached result to use
#   "CACHED_TIMEOUT" if already timed out at same or higher timeout
#   "RUN" if we need to run this benchmark
check_cache() {
    local objects="$1"
    local sensors="$2"
    local filter="$3"
    local lang="$4"
    local current_timeout="$5"

    if [[ $USE_CACHE -eq 0 ]]; then
        echo "RUN"
        return
    fi

    # Look for cached entry
    local cached_line
    cached_line=$(grep "^${objects},${sensors},${filter},${lang}," "$CACHE_FILE" 2>/dev/null | tail -1)

    if [[ -z "$cached_line" ]]; then
        echo "RUN"
        return
    fi

    # Parse cached entry: objects,sensors,filter,lang,timeout,time_ms,status,timestamp
    local cached_timeout cached_time_ms cached_status
    cached_timeout=$(echo "$cached_line" | cut -d',' -f5)
    cached_time_ms=$(echo "$cached_line" | cut -d',' -f6)
    cached_status=$(echo "$cached_line" | cut -d',' -f7)

    if [[ "$cached_status" == "OK" ]]; then
        # Have a successful result - use it
        echo "OK:$cached_time_ms"
    elif [[ "$cached_status" == "TIMEOUT" ]]; then
        if [[ "$current_timeout" -gt "$cached_timeout" ]]; then
            # Current timeout is higher, might succeed now
            echo "RUN"
        else
            # Already timed out at same or higher timeout
            echo "CACHED_TIMEOUT"
        fi
    else
        # ERROR or unknown status - re-run
        echo "RUN"
    fi
}

# Update cache with a new result
update_cache() {
    local objects="$1"
    local sensors="$2"
    local filter="$3"
    local lang="$4"
    local timeout="$5"
    local time_ms="$6"
    local status="$7"
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S')

    # Remove old entries for this combination
    local temp_file
    temp_file=$(mktemp)
    grep -v "^${objects},${sensors},${filter},${lang}," "$CACHE_FILE" > "$temp_file" 2>/dev/null || true
    mv "$temp_file" "$CACHE_FILE"

    # Add new entry
    echo "${objects},${sensors},${filter},${lang},${timeout},${time_ms},${status},${timestamp}" >> "$CACHE_FILE"
}

# Initialize cache file
init_cache

# =============================================================================
# Skip Tracking (per filter+language within current run)
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
    # Find all scenario_*.json files and sort by (n, s)
    for path in "$SCENARIOS_DIR"/scenario_*.json; do
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
            # Check single-sensor LMB limitation (LMB filters only support 1 sensor)
            if [[ "$name" =~ ^LMB- && "$num_sensors" -gt 1 ]]; then
                return 1
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

get_filter_config() {
    local scenario_path="$1"
    local filter_name="$2"
    local lang="$3"

    case "$lang" in
        octave)
            LMB_SILENT=1 octave --no-gui --path "$RUNNERS_DIR" \
                --eval "run_octave('$scenario_path', '$filter_name', true, true)" 2>/dev/null
            ;;
        rust)
            "$PROJECT_ROOT/target/release/benchmark_single" \
                --scenario "$scenario_path" --filter "$filter_name" --get-config --skip-run 2>/dev/null
            ;;
        python)
            uv run python "$RUNNERS_DIR/run_python.py" \
                --scenario "$scenario_path" --filter "$filter_name" --get-config --skip-run 2>/dev/null
            ;;
    esac
}

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
# Config Comparison Mode (--get-config)
# =============================================================================

if [[ $GET_CONFIG -eq 1 ]]; then
    echo "======================================="
    echo "Filter Configuration Comparison"
    echo "======================================="
    echo ""

    # Pick first scenario for config comparison
    first_scenario=$(get_sorted_scenarios | head -1)
    first_scenario_name=$(basename "$first_scenario" .json)
    num_sensors=$(get_num_sensors "$first_scenario")
    echo "Using scenario: $first_scenario_name (sensors=$num_sensors)"
    echo ""

    IFS=',' read -ra LANG_ARRAY <<< "$LANGUAGES"

    for filter_config in "${FILTER_CONFIGS[@]}"; do
        IFS='|' read -r filter_name is_multi octave_support <<< "$filter_config"

        # Check if filter matches --filter arg
        if [[ "$FILTER_PATTERN" != "all" && "$filter_name" != *"$FILTER_PATTERN"* ]]; then
            continue
        fi

        # Check applicability to scenario
        if ! is_filter_applicable "$filter_name" "$num_sensors" "rust"; then
            continue
        fi

        echo "=== $filter_name ==="
        echo ""

        for lang in "${LANG_ARRAY[@]}"; do
            if ! is_filter_applicable "$filter_name" "$num_sensors" "$lang"; then
                echo "--- $lang: N/A ---"
                continue
            fi

            echo "--- $lang ---"
            config_output=$(get_filter_config "$first_scenario" "$filter_name" "$lang" 2>&1)
            if [[ -n "$config_output" ]]; then
                echo "$config_output"
            else
                echo "(no output or error)"
            fi
            echo ""
        done
        echo ""
    done
fi

# =============================================================================
# Main Benchmark Loop
# =============================================================================

if [[ $SKIP_RUN -eq 1 ]]; then
    echo "Skipping benchmark run (--skip-run specified)"
    exit 0
fi

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

            # Check skip (from current run's timeouts)
            if should_skip "$filter_name" "$lang"; then
                echo "$num_objects,$num_sensors,$filter_name,$lang,SKIP" >> "$RESULTS_FILE"
                printf "%-8s | %-8s | %-18s | %-8s | %9s\n" \
                    "$num_objects" "$num_sensors" "$filter_name" "$lang" "SKIP"
                continue
            fi

            # Check cache
            cache_result=$(check_cache "$num_objects" "$num_sensors" "$filter_name" "$lang" "$TIMEOUT")

            if [[ "$cache_result" == OK:* ]]; then
                # Use cached result
                cached_time=${cache_result#OK:}
                echo "$num_objects,$num_sensors,$filter_name,$lang,$cached_time" >> "$RESULTS_FILE"
                printf "%-8s | %-8s | %-18s | %-8s | %8.1f*\n" \
                    "$num_objects" "$num_sensors" "$filter_name" "$lang" "$cached_time"
                continue
            elif [[ "$cache_result" == "CACHED_TIMEOUT" ]]; then
                # Already timed out at this timeout level
                mark_timed_out "$filter_name" "$lang"
                echo "$num_objects,$num_sensors,$filter_name,$lang,TIMEOUT" >> "$RESULTS_FILE"
                printf "%-8s | %-8s | %-18s | %-8s | %9s\n" \
                    "$num_objects" "$num_sensors" "$filter_name" "$lang" "TIMEOUT*"
                continue
            fi

            # Run benchmark
            [[ $VERBOSE -eq 1 ]] && echo "Running: n=$num_objects s=$num_sensors / $filter_name / $lang" >&2
            result=$(run_benchmark "$scenario_path" "$filter_name" "$lang")

            if [[ "$result" == "TIMEOUT" ]]; then
                mark_timed_out "$filter_name" "$lang"
                update_cache "$num_objects" "$num_sensors" "$filter_name" "$lang" "$TIMEOUT" "" "TIMEOUT"
                echo "$num_objects,$num_sensors,$filter_name,$lang,TIMEOUT" >> "$RESULTS_FILE"
                printf "%-8s | %-8s | %-18s | %-8s | %9s\n" \
                    "$num_objects" "$num_sensors" "$filter_name" "$lang" "TIMEOUT"
            elif [[ "$result" == "ERROR" ]]; then
                update_cache "$num_objects" "$num_sensors" "$filter_name" "$lang" "$TIMEOUT" "" "ERROR"
                echo "$num_objects,$num_sensors,$filter_name,$lang,ERROR" >> "$RESULTS_FILE"
                printf "%-8s | %-8s | %-18s | %-8s | %9s\n" \
                    "$num_objects" "$num_sensors" "$filter_name" "$lang" "ERROR"
            else
                update_cache "$num_objects" "$num_sensors" "$filter_name" "$lang" "$TIMEOUT" "$result" "OK"
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

# Count results (ensure clean integers)
total=$(tail -n +2 "$RESULTS_FILE" | wc -l | tr -d ' \n')
timeouts=$(grep -c 'TIMEOUT' "$RESULTS_FILE" 2>/dev/null | tr -d ' \n' || echo 0)
errors=$(grep -c 'ERROR' "$RESULTS_FILE" 2>/dev/null | tr -d ' \n' || echo 0)
skips=$(grep -c 'SKIP' "$RESULTS_FILE" 2>/dev/null | tr -d ' \n' || echo 0)
# Default to 0 if empty
[[ -z "$timeouts" ]] && timeouts=0
[[ -z "$errors" ]] && errors=0
[[ -z "$skips" ]] && skips=0
successful=$((total - timeouts - errors - skips))

echo "Summary:"
echo "  Successful: $successful"
echo "  Timeouts:   $timeouts"
echo "  Errors:     $errors"
echo "  Skipped:    $skips"
echo ""

# =============================================================================
# Generate Plots
# =============================================================================

PLOTS_DIR="$PROJECT_ROOT/docs/benchmarks"

echo "Generating benchmark plots..."
if uv run "$RUNNERS_DIR/generate_plots.py" \
    --cache-file "$CACHE_FILE" \
    --output-dir "$PLOTS_DIR" 2>&1; then
    PLOTS_GENERATED=1
    echo "Plots saved to: $PLOTS_DIR"
else
    PLOTS_GENERATED=0
    echo "Warning: Plot generation failed (matplotlib may not be installed)"
fi
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

## Performance Summary

### Rust vs Octave Speedup

<img alt="Rust vs Octave Speedup" src="docs/benchmarks/speedup/rust_vs_octave.png" width=640 />

### Performance by Language

<img alt="" src="docs/benchmarks/by_language/octave.png" width=640 />
</br>
<img alt="" src="docs/benchmarks/by_language/rust.png" width=640 />

### Performance by Sensor Count

| Single | Dual | Quad |
|--------|------|------|
| ![Single Sensor](docs/benchmarks/by_sensors/single_sensor.png) | ![Dual Sensor](docs/benchmarks/by_sensors/dual_sensor.png) | ![Quad Sensor](docs/benchmarks/by_sensors/quad_sensor.png) |

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

    # Format result with speedup (baseline / compare)
    format_result_with_speedup() {
        local val="$1"
        local baseline="$2"

        if [[ -z "$val" ]]; then
            echo "-"
        elif [[ "$val" == "TIMEOUT" || "$val" == "ERROR" || "$val" == "SKIP" ]]; then
            echo "$val"
        elif [[ -z "$baseline" || "$baseline" == "TIMEOUT" || "$baseline" == "ERROR" || "$baseline" == "SKIP" ]]; then
            # No baseline, just show value
            printf "%.1f (N/A)" "$val"
        else
            # Calculate speedup
            local speedup
            speedup=$(awk "BEGIN {printf \"%.1f\", $baseline / $val}")
            printf "%.1f (×%.1f)" "$val" "$speedup"
        fi
    }

    # Get all unique (n,s) combinations from scenarios directory
    all_scenarios=""
    for path in "$SCENARIOS_DIR"/scenario_*.json; do
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
        echo ""
        echo "| Objects | Sensors | Octave (ms) | Python (ms) | Rust (ms) |"
        echo "|---------|---------|-------------|-------------|-----------|"

        # Iterate through all scenarios
        echo "$sorted_scenarios" | while IFS=',' read -r n s; do
            [[ -z "$n" ]] && continue

            # Check if this filter applies to this scenario
            # Multi-sensor filters need s > 1, single-sensor filters work on any
            applicable="yes"
            if [[ "$is_multi" == "true" && "$s" -eq 1 ]]; then
                applicable="no"
            fi
            # Check single-sensor LMB limitation (LMB filters only support 1 sensor)
            if [[ "$filter_name" =~ ^LMB- && "$s" -gt 1 ]]; then
                applicable="no"
            fi

            if [[ "$applicable" == "no" ]]; then
                # Filter not applicable to this scenario
                # echo "| $n | $s | - | - | - |"
                true
            else
                # Look up results from CSV
                octave_result=$(grep "^$n,$s,$filter_name,octave," "$RESULTS_FILE" 2>/dev/null | cut -d',' -f5 | head -1)
                rust_result=$(grep "^$n,$s,$filter_name,rust," "$RESULTS_FILE" 2>/dev/null | cut -d',' -f5 | head -1)
                python_result=$(grep "^$n,$s,$filter_name,python," "$RESULTS_FILE" 2>/dev/null | cut -d',' -f5 | head -1)

                oct_fmt=$(format_result "$octave_result")
                rust_fmt=$(format_result_with_speedup "$rust_result" "$octave_result")
                py_fmt=$(format_result_with_speedup "$python_result" "$octave_result")

                echo "| $n | $s | $oct_fmt | $py_fmt | $rust_fmt |"
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
- **-** means not applicable (e.g., single-sensor LMB on multi-sensor scenario) or not run

NOTES
}

echo "Generating README_BENCHMARKS.md..."
generate_readme > "$README_FILE"
echo "README saved to: $README_FILE"
echo ""
echo "Done!"
