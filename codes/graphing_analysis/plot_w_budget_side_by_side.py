#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_archive import archive_plot


def load_budget(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.shape == ():
        data = np.array([data])
    return data


def plot_side_by_side(csv_path, output, title=None):
    data = load_budget(csv_path)
    t = data["t"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(t, data["W"], linewidth=1.8, color="tab:purple")
    axes[0].set(xlabel="t", ylabel=r"$W(t)$", title=r"Global $W$")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, data["G_minus_D_plus_T"], linewidth=1.8, label=r"$G-D+T$")
    axes[1].plot(
        t,
        data["finite_difference_dWdt"],
        "--",
        linewidth=1.3,
        label=r"finite diff. $\partial_t W$",
    )
    axes[1].set(xlabel="t", ylabel=r"$\partial_t W$", title=r"$W$ budget check")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    archive_plot(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot W(t) beside the W-budget reconstruction check.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title")
    args = parser.parse_args()
    plot_side_by_side(Path(args.csv), Path(args.output), args.title)
    print(args.output)


if __name__ == "__main__":
    main()
