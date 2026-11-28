"""Plot force-displacement curves of one or several specimen runs.

Each run of scripts/run_fe.py writes <name>_force_displacement.csv in its results folder.
Copy or rename them to keep several runs, then compare them:

    python scripts/plot_results.py results/specimen_with_holes_tensile/*_force_displacement.csv
    python scripts/plot_results.py a.csv b.csv --labels "FE-LSTM" "EPICP" -o comparison.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("csv", nargs="+", help="force-displacement CSV files")
    parser.add_argument("--labels", nargs="+", help="legend labels (default: file names)")
    parser.add_argument("-o", "--output", default="force_displacement.png")
    args = parser.parse_args()

    labels = args.labels or [Path(f).name.replace("_force_displacement.csv", "") for f in args.csv]
    if len(labels) != len(args.csv):
        parser.error("give one label per CSV file")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for path, label in zip(args.csv, labels):
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        ax.plot(data[:, 1], data[:, 2], marker="o", markersize=2.5, linewidth=1.5, label=label)
    ax.set_xlabel("Imposed displacement $u_x$ (right face)")
    ax.set_ylabel("Reaction force $F_x$ (right face)")
    ax.set_title("Specimen with holes, tension")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
