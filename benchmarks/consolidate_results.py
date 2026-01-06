#!/usr/bin/env python3
"""
Consolidate benchmark results from Rust (Criterion), Python, and MATLAB.

Usage:
    uv run python benchmarks/consolidate_results.py \
      --criterion target/criterion \
      --python benchmarks/results/python_benchmarks.txt \
      --matlab ../multisensor-lmb-filters/benchmarks/matlab_results.txt \
      --output benchmarks/results/
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Single benchmark result for a (scenario, filter) pair."""

    scenario: str
    filter: str
    impl: str  # 'rust', 'python', 'matlab'
    mean_ms: float | None = None
    std_dev_ms: float | None = None
    ci_lower_ms: float | None = None
    ci_upper_ms: float | None = None
    ospa: float | None = None
    status: str = "ok"  # 'ok', 'timeout', 'error', 'missing'


def parse_criterion_results(criterion_dir: Path) -> list[BenchmarkResult]:
    """Parse Criterion JSON results from target/criterion/."""
    results = []

    # Criterion structure: target/criterion/<benchmark_group>/<filter_name>/estimates.json
    # Look for patterns like: lmb_filters/<scenario>/<filter>/estimates.json
    for estimates_file in criterion_dir.glob("lmb_filters/*/*/new/estimates.json"):
        try:
            # Extract scenario and filter from path
            # Path structure: .../lmb_filters/<filter>/<scenario>/new/estimates.json
            parts = estimates_file.parts
            scenario = parts[-3]
            filter_name = parts[-4]

            with open(estimates_file) as f:
                data = json.load(f)

            # Criterion stores times in nanoseconds
            # Extract mean, std_dev, and confidence interval
            mean_ns = data.get("mean", {}).get("point_estimate")
            std_dev_ns = data.get("std_dev", {}).get("point_estimate")
            ci_lower_ns = data.get("mean", {}).get("confidence_interval", {}).get("lower_bound")
            ci_upper_ns = data.get("mean", {}).get("confidence_interval", {}).get("upper_bound")

            if mean_ns is not None:
                results.append(
                    BenchmarkResult(
                        scenario=scenario,
                        filter=filter_name,
                        impl="rust",
                        mean_ms=mean_ns / 1_000_000,  # ns to ms
                        std_dev_ms=std_dev_ns / 1_000_000 if std_dev_ns else None,
                        ci_lower_ms=ci_lower_ns / 1_000_000 if ci_lower_ns else None,
                        ci_upper_ms=ci_upper_ns / 1_000_000 if ci_upper_ns else None,
                    )
                )
        except Exception as e:
            print(f"Warning: Failed to parse {estimates_file}: {e}")

    return results


def parse_ascii_table(text: str, impl: str, has_ospa: bool = False) -> list[BenchmarkResult]:
    """Parse ASCII benchmark table (Python or MATLAB format)."""
    results = []

    # Skip header and separator lines
    lines = text.strip().split("\n")
    for line in lines:
        # Skip headers and separators
        if not line.strip() or line.startswith("-") or "Scenario" in line:
            continue

        # Parse pipe-separated columns
        # Format: scenario | filter | time | [ospa] | [progress]
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        scenario = parts[0].strip()
        filter_name = parts[1].strip()
        time_str = parts[2].strip()
        ospa_val = None

        if has_ospa and len(parts) >= 4:
            try:
                ospa_val = float(parts[3].strip())
            except (ValueError, IndexError):
                pass

        # Parse time (could be number, TIMEOUT, ERROR, or SKIP)
        if time_str.upper() == "TIMEOUT":
            results.append(
                BenchmarkResult(scenario=scenario, filter=filter_name, impl=impl, status="timeout")
            )
        elif time_str.upper() == "ERROR":
            results.append(
                BenchmarkResult(scenario=scenario, filter=filter_name, impl=impl, status="error")
            )
        elif time_str.upper() == "SKIP":
            results.append(
                BenchmarkResult(scenario=scenario, filter=filter_name, impl=impl, status="skip")
            )
        else:
            try:
                time_ms = float(time_str)
                results.append(
                    BenchmarkResult(
                        scenario=scenario,
                        filter=filter_name,
                        impl=impl,
                        mean_ms=time_ms,
                        ospa=ospa_val,
                    )
                )
            except ValueError:
                print(f"Warning: Could not parse time value: {time_str}")

    return results


def load_all_results(
    criterion_dir: Path | None,
    python_file: Path | None,
    matlab_file: Path | None,
) -> list[BenchmarkResult]:
    """Load results from all three sources."""
    all_results = []

    if criterion_dir and criterion_dir.exists():
        print(f"Loading Rust results from {criterion_dir}...")
        all_results.extend(parse_criterion_results(criterion_dir))
        print(f"  Loaded {sum(1 for r in all_results if r.impl == 'rust')} Rust results")

    if python_file and python_file.exists():
        print(f"Loading Python results from {python_file}...")
        text = python_file.read_text()
        python_results = parse_ascii_table(text, "python", has_ospa=True)
        all_results.extend(python_results)
        print(f"  Loaded {len(python_results)} Python results")

    if matlab_file and matlab_file.exists():
        print(f"Loading MATLAB results from {matlab_file}...")
        text = matlab_file.read_text()
        matlab_results = parse_ascii_table(text, "matlab", has_ospa=False)
        all_results.extend(matlab_results)
        print(f"  Loaded {len(matlab_results)} MATLAB results")

    return all_results


def group_by_scenario_filter(results: list[BenchmarkResult]) -> dict:
    """Group results by (scenario, filter) key."""
    grouped = {}
    for result in results:
        key = (result.scenario, result.filter)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][result.impl] = result
    return grouped


def generate_markdown_table(grouped: dict, output_file: Path):
    """Generate markdown comparison table."""
    lines = [
        "# Benchmark Comparison: MATLAB vs Rust vs Python",
        "",
        (
            "| Scenario | Filter | MATLAB (ms) | Rust (ms) | Python (ms) | "
            "Rust Speedup | Python Speedup |"
        ),
        (
            "|----------|--------|-------------|-----------|-------------|"
            "--------------|----------------|"
        ),
    ]

    for (scenario, filter_name), impls in sorted(grouped.items()):
        matlab = impls.get("matlab")
        rust = impls.get("rust")
        python = impls.get("python")

        # Format cells
        def fmt_time(res: BenchmarkResult | None) -> str:
            if res is None:
                return "N/A"
            if res.status == "timeout":
                return "TIMEOUT"
            if res.status == "error":
                return "ERROR"
            if res.status == "skip":
                return "SKIP"
            if res.mean_ms is None:
                return "N/A"

            # Include confidence interval for Rust
            if res.impl == "rust" and res.std_dev_ms is not None:
                return f"{res.mean_ms:.1f} ±{res.std_dev_ms:.1f}"
            return f"{res.mean_ms:.1f}"

        def fmt_speedup(baseline: BenchmarkResult | None, compare: BenchmarkResult | None) -> str:
            if baseline is None or compare is None:
                return "-"
            if baseline.status != "ok" or compare.status != "ok":
                return "-"
            if baseline.mean_ms is None or compare.mean_ms is None:
                return "-"
            if baseline.mean_ms == 0:
                return "-"
            speedup = baseline.mean_ms / compare.mean_ms
            return f"{speedup:.2f}×"

        matlab_str = fmt_time(matlab)
        rust_str = fmt_time(rust)
        python_str = fmt_time(python)
        rust_speedup = fmt_speedup(matlab, rust)
        python_speedup = fmt_speedup(matlab, python)

        lines.append(
            f"| {scenario} | {filter_name} | {matlab_str} | {rust_str} | {python_str} | "
            f"{rust_speedup} | {python_speedup} |"
        )

    output_file.write_text("\n".join(lines) + "\n")
    print(f"Generated markdown table: {output_file}")


def generate_csv(results: list[BenchmarkResult], output_file: Path):
    """Generate CSV with all raw data."""
    lines = ["scenario,filter,impl,mean_ms,std_dev_ms,ci_lower_ms,ci_upper_ms,ospa,status"]

    for r in sorted(results, key=lambda x: (x.scenario, x.filter, x.impl)):
        lines.append(
            f"{r.scenario},{r.filter},{r.impl},"
            f"{r.mean_ms or ''},"
            f"{r.std_dev_ms or ''},"
            f"{r.ci_lower_ms or ''},"
            f"{r.ci_upper_ms or ''},"
            f"{r.ospa or ''},"
            f"{r.status}"
        )

    output_file.write_text("\n".join(lines) + "\n")
    print(f"Generated CSV: {output_file}")


def generate_json(grouped: dict, output_file: Path):
    """Generate JSON with structured comparison data."""
    output = {}

    for (scenario, filter_name), impls in sorted(grouped.items()):
        key = f"{scenario}/{filter_name}"
        output[key] = {}

        for impl, result in impls.items():
            output[key][impl] = {
                "mean_ms": result.mean_ms,
                "std_dev_ms": result.std_dev_ms,
                "ci_lower_ms": result.ci_lower_ms,
                "ci_upper_ms": result.ci_upper_ms,
                "ospa": result.ospa,
                "status": result.status,
            }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated JSON: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate benchmark results")
    parser.add_argument("--criterion", type=Path, help="Path to Criterion output directory")
    parser.add_argument("--python", type=Path, help="Path to Python benchmark results")
    parser.add_argument("--matlab", type=Path, help="Path to MATLAB benchmark results")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    # Load all results
    results = load_all_results(args.criterion, args.python, args.matlab)

    if not results:
        print("Error: No results found!")
        return 1

    # Group by scenario and filter
    grouped = group_by_scenario_filter(results)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    generate_markdown_table(grouped, args.output / "comparison_summary.md")
    generate_csv(results, args.output / "all_results.csv")
    generate_json(grouped, args.output / "comparison_data.json")

    print(f"\nConsolidation complete! Results in {args.output}/")
    return 0


if __name__ == "__main__":
    exit(main())
