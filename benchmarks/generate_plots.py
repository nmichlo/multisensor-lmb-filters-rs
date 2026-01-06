#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.8",
#     "seaborn>=0.13",
#     "pandas>=2.0",
#     "numpy>=1.26",
# ]
# ///
"""
Generate benchmark visualization plots from cache.csv.

Usage:
    uv run benchmarks/generate_plots.py
    uv run benchmarks/generate_plots.py --cache-file path/to/cache.csv
    uv run benchmarks/generate_plots.py --output-dir path/to/output

Output Structure:
    docs/benchmarks/
        by_filter/       - One plot per filter (comparing languages)
        by_language/     - One plot per language (comparing filters)
        by_sensors/      - One plot per sensor count
        speedup/         - Relative performance plots
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================

# Set seaborn theme
sns.set_theme(style="whitegrid", palette="colorblind")

# Language styling - linestyle differentiates languages (consistent across all plots)
# Markers match what's used in by_sensors for language differentiation
LANG_STYLES = {
    "octave": {"color": "#56B4E9", "marker": "o", "linestyle": "-", "label": "Octave"},  # Solid
    "rust": {"color": "#E69F00", "marker": "o", "linestyle": "--", "label": "Rust"},  # Dashed
    "python": {"color": "#009E73", "marker": "o", "linestyle": ":", "label": "Python"},  # Dotted
}

# Expected X-axis values (object counts) - ensures consistent axes
EXPECTED_OBJECTS = [5, 10, 20, 50]
EXPECTED_SENSORS = [1, 2, 4, 8]

# Y-axis limits for consistent scaling across plots (in ms)
Y_AXIS_MAX_SINGLE = 10000  # 10 seconds for single-sensor
Y_AXIS_MAX_MULTI = 60000  # 60 seconds for multi-sensor

# Semantic encoding for filters:
# - Color = Base filter architecture
# - Marker = Association method (LBP, Gibbs, Murty)
# - Linestyle = Language (defined in LANG_STYLES)

# Base filter colors (7 distinct, colorblind-friendly colors from Tableau 10)
BASE_COLORS = {
    "LMB": "#4e79a7",  # Steel blue
    "LMBM": "#59a14f",  # Green
    "AA-LMB": "#f28e2b",  # Orange
    "IC-LMB": "#e15759",  # Coral red
    "PU-LMB": "#76b7b2",  # Teal
    "GA-LMB": "#b07aa1",  # Mauve/purple
    "MS-LMBM": "#9c755f",  # Brown
}

# Association method markers (3 maximally distinct shapes)
ASSOC_MARKERS = {
    "LBP": "o",  # Circle
    "Gibbs": "^",  # Triangle
    "Murty": "x",  # x (thin)
}


def _get_base_and_assoc(filter_name: str) -> tuple[str, str]:
    """Extract base filter and association method from filter name."""
    if filter_name.endswith("-LBP"):
        return filter_name[:-4], "LBP"
    elif filter_name.endswith("-Gibbs"):
        return filter_name[:-6], "Gibbs"
    elif filter_name.endswith("-Murty"):
        return filter_name[:-6], "Murty"
    return filter_name, "LBP"  # fallback


# Build FILTER_CONFIG from semantic components
FILTER_CONFIG = {}
for filter_name in [
    "LMB-LBP",
    "LMB-Gibbs",
    "LMB-Murty",
    "LMBM-Gibbs",
    "LMBM-Murty",
    "AA-LMB-LBP",
    "IC-LMB-LBP",
    "PU-LMB-LBP",
    "GA-LMB-LBP",
    "MS-LMBM-Gibbs",
]:
    base, assoc = _get_base_and_assoc(filter_name)
    is_multi = base not in ["LMB", "LMBM"]
    FILTER_CONFIG[filter_name] = {
        "multi": is_multi,
        "color": BASE_COLORS[base],
        "marker": ASSOC_MARKERS[assoc],
        "base": base,
        "assoc": assoc,
    }

SINGLE_SENSOR_FILTERS = [f for f, c in FILTER_CONFIG.items() if not c["multi"]]
MULTI_SENSOR_FILTERS = [f for f, c in FILTER_CONFIG.items() if c["multi"]]


# =============================================================================
# Data Loading
# =============================================================================


def load_cache(cache_file: Path) -> pd.DataFrame:
    """Load cache.csv with all results (not just OK)."""
    df = pd.read_csv(cache_file)

    # Convert time_ms to numeric (non-numeric becomes NaN)
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")

    return df


def prepare_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot data for plotting, keeping status information."""
    # Create separate columns for time and status per language
    result = df.pivot_table(
        index=["objects", "sensors", "filter"],
        columns="lang",
        values=["time_ms", "status"],
        aggfunc="first",
    ).reset_index()

    # Flatten column names
    result.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in result.columns]

    return result


# =============================================================================
# Plot Helpers
# =============================================================================


def get_legend_label(lang, has_timeout=False, has_skip=False):
    """Generate legend label with status indicators."""
    style = LANG_STYLES[lang]
    label = style["label"]

    suffixes = []
    if has_timeout:
        suffixes.append("TIMEOUT")
    if has_skip:
        suffixes.append("SKIP")

    if suffixes:
        label += f" ({', '.join(suffixes)})"

    return label


def plot_filter_data(ax, data, x_col, languages=None):
    """
    Plot data for multiple languages on a single axis.
    Always shows all languages in legend with status indicators.
    """
    if languages is None:
        languages = ["octave", "rust", "python"]

    handles = []
    labels = []

    for lang in languages:
        time_col = f"time_ms_{lang}"
        status_col = f"status_{lang}"
        style = LANG_STYLES[lang]

        # Check what data exists
        has_time_col = time_col in data.columns if not data.empty else False
        has_status_col = status_col in data.columns if not data.empty else False

        # Get successful data points
        ok_mask = data[status_col] == "OK" if has_status_col else pd.Series(dtype=bool)
        ok_data = data[ok_mask].sort_values(x_col) if ok_mask.any() else pd.DataFrame()

        # Check for timeouts/skips
        has_timeout = has_status_col and (data[status_col] == "TIMEOUT").any()
        has_skip = has_status_col and (data[status_col] == "SKIP").any()

        # Build label
        label = get_legend_label(lang, has_timeout and ok_data.empty, has_skip and ok_data.empty)
        if not has_time_col and not has_status_col:
            label = style["label"] + " (NO DATA)"

        if not ok_data.empty and has_time_col:
            (line,) = ax.plot(
                ok_data[x_col],
                ok_data[time_col],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
            handles.append(line)
            labels.append(label)
        else:
            # Always add legend entry even with no data
            (line,) = ax.plot(
                [],
                [],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                alpha=0.5,
            )
            handles.append(line)
            labels.append(label)

    return handles, labels


# =============================================================================
# Plot Generators
# =============================================================================


def plot_filter(df: pd.DataFrame, filter_name: str, is_multi: bool, output_dir: Path):
    """Generate plot for a filter - all data on one axis, colors distinguish languages."""
    if is_multi:
        mask = (df["filter"] == filter_name) & (df["sensors"] > 1)
        y_max = Y_AXIS_MAX_MULTI
    else:
        mask = (df["filter"] == filter_name) & (df["sensors"] == 1)
        y_max = Y_AXIS_MAX_SINGLE

    data = df[mask].sort_values("objects")

    fig, ax = plt.subplots(figsize=(8, 5))

    handles, labels = plot_filter_data(ax, data, "objects")

    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{filter_name} Performance", fontsize=12, fontweight="bold")

    # Linear scale with consistent limits
    ax.set_ylim(0, y_max)

    # Always show expected X values
    ax.set_xticks(EXPECTED_OBJECTS)
    ax.set_xlim(min(EXPECTED_OBJECTS) - 2, max(EXPECTED_OBJECTS) + 5)

    if handles:
        ax.legend(handles, labels, loc="upper left")

    plt.tight_layout()
    output_path = output_dir / "by_filter" / f"{filter_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_by_language(df: pd.DataFrame, lang: str, output_dir: Path):
    """Generate plot showing all filters for one language."""

    time_col = f"time_ms_{lang}"
    status_col = f"status_{lang}"

    # Always generate plot even if no successful data (to show TIMEOUT in legend)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Track filter status for legends
    filter_status = {
        f: {"has_data": False, "timeout": False, "skip": False}
        for f in SINGLE_SENSOR_FILTERS + MULTI_SENSOR_FILTERS
    }

    # Single-sensor filters (left)
    ax = axes[0]
    for filter_name in SINGLE_SENSOR_FILTERS:
        filter_data = df[(df["filter"] == filter_name) & (df["sensors"] == 1)]
        fconfig = FILTER_CONFIG[filter_name]

        # Check if columns exist
        has_time_col = time_col in filter_data.columns if not filter_data.empty else False
        has_status_col = status_col in filter_data.columns if not filter_data.empty else False

        ok_mask = filter_data[status_col] == "OK" if has_status_col else pd.Series(dtype=bool)
        ok_data = filter_data[ok_mask].sort_values("objects") if ok_mask.any() else pd.DataFrame()

        has_timeout = has_status_col and (filter_data[status_col] == "TIMEOUT").any()
        has_skip = has_status_col and (filter_data[status_col] == "SKIP").any()

        # Track status
        if not ok_data.empty:
            filter_status[filter_name]["has_data"] = True
        if has_timeout:
            filter_status[filter_name]["timeout"] = True
        if has_skip:
            filter_status[filter_name]["skip"] = True

        if not ok_data.empty and has_time_col:
            ax.plot(
                ok_data["objects"],
                ok_data[time_col],
                color=fconfig["color"],
                marker=fconfig["marker"],
                linestyle="-",
                linewidth=1.5,
                markersize=6,
            )

    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{lang.capitalize()} - Single Sensor Filters")
    ax.set_ylim(0, Y_AXIS_MAX_SINGLE)
    ax.set_xticks(EXPECTED_OBJECTS)
    ax.set_xlim(min(EXPECTED_OBJECTS) - 2, max(EXPECTED_OBJECTS) + 5)

    # Split legends for single-sensor subplot
    _add_split_legends(ax, SINGLE_SENSOR_FILTERS, filter_status, y_offset=0)

    # Multi-sensor filters (right) - use sensors=2 as representative
    ax = axes[1]
    for filter_name in MULTI_SENSOR_FILTERS:
        filter_data = df[(df["filter"] == filter_name) & (df["sensors"] == 2)]
        fconfig = FILTER_CONFIG[filter_name]

        # Check if columns exist
        has_time_col = time_col in filter_data.columns if not filter_data.empty else False
        has_status_col = status_col in filter_data.columns if not filter_data.empty else False

        ok_mask = filter_data[status_col] == "OK" if has_status_col else pd.Series(dtype=bool)
        ok_data = filter_data[ok_mask].sort_values("objects") if ok_mask.any() else pd.DataFrame()

        has_timeout = has_status_col and (filter_data[status_col] == "TIMEOUT").any()
        has_skip = has_status_col and (filter_data[status_col] == "SKIP").any()

        # Track status
        if not ok_data.empty:
            filter_status[filter_name]["has_data"] = True
        if has_timeout:
            filter_status[filter_name]["timeout"] = True
        if has_skip:
            filter_status[filter_name]["skip"] = True

        if not ok_data.empty and has_time_col:
            ax.plot(
                ok_data["objects"],
                ok_data[time_col],
                color=fconfig["color"],
                marker=fconfig["marker"],
                linestyle="-",
                linewidth=1.5,
                markersize=6,
            )

    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{lang.capitalize()} - Multi-Sensor Filters (2 sensors)")
    ax.set_ylim(0, Y_AXIS_MAX_MULTI)
    ax.set_xticks(EXPECTED_OBJECTS)
    ax.set_xlim(min(EXPECTED_OBJECTS) - 2, max(EXPECTED_OBJECTS) + 5)

    # Split legends for multi-sensor subplot
    _add_split_legends(ax, MULTI_SENSOR_FILTERS, filter_status, y_offset=0)

    plt.tight_layout()
    output_path = output_dir / "by_language" / f"{lang}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _add_split_legends(ax, filters, filter_status, y_offset=0):
    """Add split legends (Filter + Association) to an axis."""
    from matplotlib.lines import Line2D

    # Legend 1: Base filter (color)
    base_handles = []
    base_labels = []
    seen_bases = set()
    for filter_name in filters:
        fconfig = FILTER_CONFIG[filter_name]
        base = fconfig["base"]
        if base in seen_bases:
            continue
        seen_bases.add(base)

        # Check if any filter with this base has data
        base_has_data = any(
            filter_status[f]["has_data"] for f in filters if FILTER_CONFIG[f]["base"] == base
        )
        label = base
        if not base_has_data:
            base_has_timeout = any(
                filter_status[f]["timeout"] for f in filters if FILTER_CONFIG[f]["base"] == base
            )
            if base_has_timeout:
                label += " (T/O)"
            else:
                label += " (N/A)"

        handle = Line2D(
            [0], [0], color=fconfig["color"], marker="o", linestyle="-", linewidth=2, markersize=6
        )
        base_handles.append(handle)
        base_labels.append(label)

    # Legend 2: Association method (marker)
    assoc_handles = []
    assoc_labels = []
    # Only show association methods that exist in these filters
    assocs_in_filters = set(FILTER_CONFIG[f]["assoc"] for f in filters)
    for assoc, marker in ASSOC_MARKERS.items():
        if assoc not in assocs_in_filters:
            continue
        handle = Line2D([0], [0], color="gray", marker=marker, linestyle="", markersize=8)
        assoc_handles.append(handle)
        assoc_labels.append(assoc)

    # Position legends outside plot (stacked vertically)
    legend1 = ax.legend(
        base_handles,
        base_labels,
        title="Filter",
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1 - y_offset),
        borderaxespad=0,
        framealpha=0.9,
    )
    ax.add_artist(legend1)

    ax.legend(
        assoc_handles,
        assoc_labels,
        title="Association",
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.5 - y_offset),
        borderaxespad=0,
        framealpha=0.9,
    )


def plot_by_sensors(df: pd.DataFrame, sensors: int, output_dir: Path):
    """Generate plot showing all applicable filters for one sensor count."""
    from matplotlib.lines import Line2D

    mask = df["sensors"] == sensors
    data = df[mask]

    # Determine applicable filters and Y limit
    if sensors == 1:
        filters = SINGLE_SENSOR_FILTERS
        title = "Single Sensor Filters"
        filename = "single_sensor.png"
        y_max = Y_AXIS_MAX_SINGLE
    else:
        filters = MULTI_SENSOR_FILTERS
        title = f"{sensors}-Sensor Filters"
        sensor_names = {2: "dual", 4: "quad", 8: "octa"}
        filename = f"{sensor_names.get(sensors, str(sensors))}_sensor.png"
        y_max = Y_AXIS_MAX_MULTI

    fig, ax = plt.subplots(figsize=(10, 6))

    # Track which filters/languages have data vs timeout/skip
    filter_status = {f: {"has_data": False, "timeout": False, "skip": False} for f in filters}

    for filter_name in filters:
        filter_data = data[data["filter"] == filter_name]
        fconfig = FILTER_CONFIG[filter_name]

        for lang in ["octave", "rust", "python"]:
            time_col = f"time_ms_{lang}"
            status_col = f"status_{lang}"
            lang_style = LANG_STYLES[lang]

            # Check what data exists
            has_time_col = time_col in filter_data.columns if not filter_data.empty else False
            has_status_col = status_col in filter_data.columns if not filter_data.empty else False

            ok_mask = filter_data[status_col] == "OK" if has_status_col else pd.Series(dtype=bool)
            ok_data = (
                filter_data[ok_mask].sort_values("objects") if ok_mask.any() else pd.DataFrame()
            )

            has_timeout = has_status_col and (filter_data[status_col] == "TIMEOUT").any()
            has_skip = has_status_col and (filter_data[status_col] == "SKIP").any()

            # Track filter status
            if not ok_data.empty:
                filter_status[filter_name]["has_data"] = True
            if has_timeout:
                filter_status[filter_name]["timeout"] = True
            if has_skip:
                filter_status[filter_name]["skip"] = True

            # Use filter's color and marker, language's linestyle
            if not ok_data.empty and has_time_col:
                ax.plot(
                    ok_data["objects"],
                    ok_data[time_col],
                    color=fconfig["color"],
                    marker=fconfig["marker"],
                    linestyle=lang_style["linestyle"],
                    linewidth=1.5,
                    markersize=6,
                    alpha=0.8,
                )

    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, y_max)
    ax.set_xticks(EXPECTED_OBJECTS)
    ax.set_xlim(min(EXPECTED_OBJECTS) - 2, max(EXPECTED_OBJECTS) + 5)

    # Create 3 split legends for semantic encoding
    # Legend 1: Base filter (color)
    base_handles = []
    base_labels = []
    seen_bases = set()
    for filter_name in filters:
        fconfig = FILTER_CONFIG[filter_name]
        base = fconfig["base"]
        if base in seen_bases:
            continue
        seen_bases.add(base)

        # Check if any filter with this base has data
        base_has_data = any(
            filter_status[f]["has_data"] for f in filters if FILTER_CONFIG[f]["base"] == base
        )
        label = base
        if not base_has_data:
            label += " (N/A)"

        handle = Line2D(
            [0], [0], color=fconfig["color"], marker="o", linestyle="-", linewidth=2, markersize=6
        )
        base_handles.append(handle)
        base_labels.append(label)

    # Legend 2: Association method (marker)
    assoc_handles = []
    assoc_labels = []
    for assoc, marker in ASSOC_MARKERS.items():
        handle = Line2D([0], [0], color="gray", marker=marker, linestyle="", markersize=8)
        assoc_handles.append(handle)
        assoc_labels.append(assoc)

    # Legend 3: Languages (linestyle)
    lang_handles = []
    lang_labels = []
    for lang in ["octave", "rust", "python"]:
        style = LANG_STYLES[lang]
        handle = Line2D(
            [0], [0], color="gray", marker="", linestyle=style["linestyle"], linewidth=2
        )
        lang_handles.append(handle)
        lang_labels.append(style["label"])

    # Position legends outside plot (stacked vertically)
    legend1 = ax.legend(
        base_handles,
        base_labels,
        title="Filter",
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        framealpha=0.9,
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        assoc_handles,
        assoc_labels,
        title="Association",
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        borderaxespad=0,
        framealpha=0.9,
    )
    ax.add_artist(legend2)

    legend3 = ax.legend(
        lang_handles,
        lang_labels,
        title="Language",
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.25),
        borderaxespad=0,
        framealpha=0.9,
    )

    plt.tight_layout()
    output_path = output_dir / "by_sensors" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_speedup(df: pd.DataFrame, baseline: str, compare: str, output_dir: Path):
    """Generate speedup plot (baseline / compare)."""
    baseline_time = f"time_ms_{baseline}"
    compare_time = f"time_ms_{compare}"
    baseline_status = f"status_{baseline}"
    compare_status = f"status_{compare}"

    if baseline_time not in df.columns or compare_time not in df.columns:
        print(f"  Missing data for {baseline} or {compare}, skipping")
        return

    # Need both to be OK for speedup calculation
    ok_mask = (
        ((df[baseline_status] == "OK") & (df[compare_status] == "OK"))
        if baseline_status in df.columns and compare_status in df.columns
        else (df[baseline_time].notna() & df[compare_time].notna())
    )

    data = df[ok_mask].copy()
    if data.empty:
        print(f"  No overlapping OK data for {baseline} vs {compare}, skipping")
        return

    # Compute speedup
    data["speedup"] = data[baseline_time] / data[compare_time]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Track filter status for legends
    filter_status = {
        f: {"has_data": False, "timeout": False, "skip": False}
        for f in SINGLE_SENSOR_FILTERS + MULTI_SENSOR_FILTERS
    }

    # Single-sensor
    ax = axes[0]
    single_data = data[data["sensors"] == 1]
    for filter_name in SINGLE_SENSOR_FILTERS:
        filter_data = single_data[single_data["filter"] == filter_name]
        fconfig = FILTER_CONFIG[filter_name]
        if filter_data.empty:
            continue
        filter_status[filter_name]["has_data"] = True
        filter_data = filter_data.sort_values("objects")
        ax.plot(
            filter_data["objects"],
            filter_data["speedup"],
            color=fconfig["color"],
            marker=fconfig["marker"],
            linestyle="-",
            linewidth=1.5,
            markersize=6,
        )

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_xlabel("Number of Objects")
    ax.set_ylabel(f"Speedup ({compare.capitalize()} vs {baseline.capitalize()})")
    ax.set_title("Single Sensor Filters")
    ax.set_ylim(0, None)  # Start at 0, auto-scale max
    ax.set_xticks(EXPECTED_OBJECTS)
    ax.set_xlim(min(EXPECTED_OBJECTS) - 2, max(EXPECTED_OBJECTS) + 5)

    # Split legends for single-sensor subplot
    _add_split_legends(ax, SINGLE_SENSOR_FILTERS, filter_status, y_offset=0)

    # Multi-sensor (use sensors=2)
    ax = axes[1]
    multi_data = data[data["sensors"] == 2]
    for filter_name in MULTI_SENSOR_FILTERS:
        filter_data = multi_data[multi_data["filter"] == filter_name]
        fconfig = FILTER_CONFIG[filter_name]
        if filter_data.empty:
            continue
        filter_status[filter_name]["has_data"] = True
        filter_data = filter_data.sort_values("objects")
        ax.plot(
            filter_data["objects"],
            filter_data["speedup"],
            color=fconfig["color"],
            marker=fconfig["marker"],
            linestyle="-",
            linewidth=1.5,
            markersize=6,
        )

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Speedup")
    ax.set_title("Multi-Sensor Filters (2 sensors)")
    ax.set_ylim(0, None)  # Start at 0, auto-scale max
    ax.set_xticks(EXPECTED_OBJECTS)
    ax.set_xlim(min(EXPECTED_OBJECTS) - 2, max(EXPECTED_OBJECTS) + 5)

    # Split legends for multi-sensor subplot
    _add_split_legends(ax, MULTI_SENSOR_FILTERS, filter_status, y_offset=0)

    fig.suptitle(
        f"Speedup: {compare.capitalize()} vs {baseline.capitalize()}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_path = output_dir / "speedup" / f"{compare}_vs_{baseline}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("benchmarks/results/cache.csv"),
        help="Path to cache.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmarks"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    # Resolve paths relative to script location if not absolute
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    cache_file = args.cache_file
    if not cache_file.is_absolute():
        cache_file = project_root / cache_file

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    if not cache_file.exists():
        print(f"Error: Cache file not found: {cache_file}")
        sys.exit(1)

    print(f"Loading cache from: {cache_file}")
    df = load_cache(cache_file)
    print(f"Loaded {len(df)} benchmark results ({(df['status'] == 'OK').sum()} successful)")

    if df.empty:
        print("No data to plot!")
        sys.exit(0)

    # Pivot data for plotting
    pivoted = prepare_plot_data(df)
    print(f"Pivoted to {len(pivoted)} unique (objects, sensors, filter) combinations")
    print()

    # Create output directories
    for subdir in ["by_filter", "by_language", "by_sensors", "speedup"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Generate per-filter plots
    print("Generating per-filter plots...")
    for filter_name, config in FILTER_CONFIG.items():
        plot_filter(pivoted, filter_name, config["multi"], output_dir)
    print()

    # Generate per-language plots
    print("Generating per-language plots...")
    for lang in ["octave", "rust", "python"]:
        plot_by_language(pivoted, lang, output_dir)
    print()

    # Generate per-sensor plots
    print("Generating per-sensor plots...")
    sensor_counts = pivoted["sensors"].unique()
    for sensors in sorted(sensor_counts):
        plot_by_sensors(pivoted, sensors, output_dir)
    print()

    # Generate speedup plots
    print("Generating speedup plots...")
    plot_speedup(pivoted, "octave", "rust", output_dir)
    plot_speedup(pivoted, "octave", "python", output_dir)
    print()

    print("Done!")


if __name__ == "__main__":
    main()
