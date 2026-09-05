#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_archive import archive_plot


TERMS = {
    "G": ("G", r"$G(t)$", "tab:blue"),
    "minus_D": ("minus_D", r"$-D(t)$", "tab:orange"),
    "T": ("T", r"$T(t)$", "tab:green"),
}


def load_budget(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.shape == ():
        data = np.array([data])
    return data


def plot_term(data, term, output_dir, prefix, title):
    column, label, color = TERMS[term]
    output = output_dir / f"{prefix}_{term}.png"

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(data["t"], data[column], linewidth=1.8, color=color, label=label)
    ax.set(
        xlabel="t",
        ylabel=label,
        title=f"{title}: {label}",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    archive_plot(output)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot W-budget terms G, -D, and T as separate graphs.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="w_budget")
    parser.add_argument("--title", default="W budget")
    args = parser.parse_args()

    data = load_budget(Path(args.csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for term in ("G", "minus_D", "T"):
        output = plot_term(data, term, output_dir, args.prefix, args.title)
        print(output)


if __name__ == "__main__":
    main()
