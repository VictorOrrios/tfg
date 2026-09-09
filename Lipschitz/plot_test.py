#!/usr/bin/env python3

"""
Compare a simple benchmark file against a particle benchmark CSV.

Input 1 format:
    <test_name> <N> <total_time>

Example:
    boxes 16 5.13658
    boxes 32 10.5499
    ...
    spheres 16 715.715

Input 2 format:
    CSV produced by LipschitzBenchmark.

Only tests/variants that exist in BOTH files are plotted.

For the CSV, the total time used is:
    render_ms_median

Usage:
    python plot_benchmark.py benchmark.txt particle_benchmark.csv

    python plot_benchmark.py benchmark.txt particle_benchmark.csv \
        -o comparison.png

    python plot_benchmark.py benchmark.txt particle_benchmark.csv \
        --logy
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Load simple benchmark
# ---------------------------------------------------------------------------

def load_simple_benchmark(path):
    """
    Load benchmark with format:

        <variant> <num_objects> <time>

    Returns:
        {
            "boxes": [(16, 5.13658), (32, 10.5499), ...],
            "spheres": [(16, 715.715), ...],
            ...
        }
    """

    data = defaultdict(list)

    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid line {line_number} in {path!r}: {line!r}\n"
                    "Expected: <variant> <N> <time>"
                )

            variant = parts[0]

            try:
                n = int(parts[1])
                time = float(parts[2])
            except ValueError:
                raise ValueError(
                    f"Invalid numeric value on line {line_number} "
                    f"in {path!r}: {line!r}"
                )

            data[variant].append((n, time))

    for values in data.values():
        values.sort(key=lambda x: x[0])

    return data


# ---------------------------------------------------------------------------
# Load particle benchmark CSV
# ---------------------------------------------------------------------------

def load_particle_csv(path):
    """
    Load LipschitzBenchmark CSV.

    Uses ONLY:
        variant
        num_particles
        render_ms_median

    Returns:
        {
            "spheres": [(16, 1.213160), (32, 1.497440), ...],
            ...
        }
    """

    data = defaultdict(list)

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "variant",
            "num_particles",
            "render_ms_median",
        }

        if reader.fieldnames is None:
            raise ValueError(f"CSV {path!r} has no header")

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"CSV {path!r} is missing columns: "
                + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, start=2):
            variant = row["variant"]

            try:
                n = int(row["num_particles"])
                total_time = float(row["render_ms_median"])
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid data on line {line_number} in {path!r}"
                )

            data[variant].append((n, total_time))

    for values in data.values():
        values.sort(key=lambda x: x[0])

    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(simple_data, csv_data, output, logy=False):
    """
    Plot only variants present in both datasets.
    """

    common_variants = sorted(
        set(simple_data.keys()) & set(csv_data.keys())
    )

    if not common_variants:
        raise ValueError(
            "No common variants found between the two input files."
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    for variant in common_variants:

        # ---------------------------------------------------------------
        # Data from simple benchmark
        # ---------------------------------------------------------------

        simple_values = simple_data[variant]

        simple_x = [n for n, _ in simple_values]
        simple_y = [time for _, time in simple_values]

        # ---------------------------------------------------------------
        # Data from CSV
        # ---------------------------------------------------------------

        csv_values = csv_data[variant]

        csv_x = [n for n, _ in csv_values]
        csv_y = [time for _, time in csv_values]

        # ---------------------------------------------------------------
        # Plot both
        # ---------------------------------------------------------------

        ax.plot(
            simple_x,
            simple_y,
            marker="o",
            linestyle="-",
            label=f"{variant} - Ours"
        )

        ax.plot(
            csv_x,
            csv_y,
            marker="x",
            linestyle="--",
            label=f"{variant} - Lipschitz"
        )

    ax.set_xscale("log", base=2)

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Number of particles N")
    ax.set_ylabel("Total time [ms]")

    ax.set_title("Benchmark comparison")

    ax.grid(
        True,
        which="both",
        alpha=0.3
    )

    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(output, dpi=150)

    plt.close(fig)

    print(f"Wrote {output}")

    print()
    print("Common variants:")
    for variant in common_variants:
        print(f"  {variant}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare two benchmark files"
    )

    parser.add_argument(
        "benchmark",
        help="Simple benchmark file"
    )

    parser.add_argument(
        "csv",
        help="Particle benchmark CSV"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="benchmark_comparison.png",
        help="Output PNG file "
             "(default: benchmark_comparison.png)"
    )

    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic scale on Y axis"
    )

    args = parser.parse_args()

    simple_data = load_simple_benchmark(args.benchmark)
    csv_data = load_particle_csv(args.csv)

    plot_comparison(
        simple_data,
        csv_data,
        args.output,
        logy=args.logy
    )


if __name__ == "__main__":
    main()