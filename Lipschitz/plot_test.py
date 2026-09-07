import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_data(path):
    """
    Read the benchmark file and group values by variant.

    Returns:
        {
            "boxes": [(16, 5.13658), (32, 10.5499), ...],
            "cylinders": [(16, 91.3088), ...],
            ...
        }
    """

    data = defaultdict(list)

    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid line {line_number}: {line!r}\n"
                    "Expected: <variant> <num_objects> <value>"
                )

            variant = parts[0]

            try:
                num_objects = int(parts[1])
                value = float(parts[2])
            except ValueError:
                raise ValueError(
                    f"Invalid numeric value on line {line_number}: {line!r}"
                )

            data[variant].append((num_objects, value))

    # Sort by number of objects
    for values in data.values():
        values.sort(key=lambda x: x[0])

    return data


def plot_data(data, output, logy=False):
    fig, ax = plt.subplots(figsize=(8, 5))

    for variant, values in sorted(data.items()):
        xs = [x for x, _ in values]
        ys = [y for _, y in values]

        ax.plot(
            xs,
            ys,
            marker="o",
            label=variant
        )

    # Number of objects doubles: 16, 32, 64, ...
    ax.set_xscale("log", base=2)

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Number of objects (N)")
    ax.set_ylabel("Time (ms)")

    ax.set_title("Benchmark")

    ax.grid(
        True,
        which="both",
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(output, dpi=150)

    plt.close(fig)

    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results"
    )

    parser.add_argument(
        "input",
        help="Benchmark input file"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="benchmark.png",
        help="Output PNG file (default: benchmark.png)"
    )

    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic scale on Y axis"
    )

    args = parser.parse_args()

    data = load_data(args.input)

    if not data:
        raise ValueError("Input file contains no data")

    plot_data(
        data,
        args.output,
        logy=args.logy
    )


if __name__ == "__main__":
    main()