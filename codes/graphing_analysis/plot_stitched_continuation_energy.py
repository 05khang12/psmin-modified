#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import h5py as h5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation_codes"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gamma_max import GammaMax
from plot_archive import archive_plot


def load_energy(path):
    with h5.File(path, "r") as fl:
        t = fl["energies/t"][()]
        e = fl["energies/Etot"][()]
    finite = np.isfinite(t) & np.isfinite(e) & (e > 0)
    return t[finite], e[finite]


def stitch(initial_file, continuation_file, gamma):
    t0, e0 = load_energy(initial_file)
    t1, e1 = load_energy(continuation_file)
    t = np.concatenate([t0, t1])
    e = np.concatenate([e0, e1])
    order = np.argsort(t)
    return gamma * t[order], e[order]


def main():
    parser = argparse.ArgumentParser(description="Plot stitched E(t) from gamma_t=0 to 45 continuation branches.")
    parser.add_argument("--data-dir", default="outputs/C10_k1_D1p5e-4_gamma30_cross_continue_2026-06-10/data")
    parser.add_argument("--output", default="outputs/C10_k1_D1p5e-4_gamma30_cross_continue_2026-06-10/plots/stitched_total_energy_gamma0_45.png")
    parser.add_argument("--kap", type=float, default=1.0)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--D", type=float, default=1.5e-4)
    parser.add_argument("--ymin", type=float, default=1e-10)
    parser.add_argument("--ymax", type=float, default=10.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    gamma = GammaMax(args.kap, args.C, args.D).value

    branches = [
        {
            "label": "HW init -> HW continuation",
            "initial": "hwak_gamma30.h5",
            "continuation": "from_hwak_to_hwak_gamma45.h5",
            "color": "#1f77b4",
        },
        {
            "label": "HM init -> HW continuation",
            "initial": "hm_gamma30.h5",
            "continuation": "from_hm_to_hwak_gamma45.h5",
            "color": "#ff7f0e",
        },
        {
            "label": "HM init -> HM continuation",
            "initial": "hm_gamma30.h5",
            "continuation": "from_hm_to_hm_gamma45.h5",
            "color": "#2ca02c",
        },
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for branch in branches:
        x, e = stitch(data_dir / branch["initial"], data_dir / branch["continuation"], gamma)
        over = e > args.ymax
        ax.plot(x, np.minimum(e, args.ymax), label=branch["label"], linewidth=1.8, color=branch["color"])
        if np.any(over):
            ax.scatter(
                x[over],
                np.full(np.count_nonzero(over), args.ymax),
                marker="^",
                s=36,
                color=branch["color"],
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
            )

    ax.axvline(30.0, color="#4B5563", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.text(30.25, args.ymin * 1.7, "continuation", color="#4B5563", fontsize=9, va="bottom")
    ax.set(
        xlabel=r"$\gamma_{max} t$",
        ylabel="E(t)",
        title="Total energy through gamma_t = 45",
    )
    ax.set_xlim(0.0, 45.0)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.02,
        0.02,
        "Excluded: HW init -> HM continuation, divergent/non-finite branch",
        transform=ax.transAxes,
        fontsize=8,
        color="#667085",
    )
    ax.text(
        0.02,
        0.07,
        "Triangle marker: branch exceeds plotted energy range",
        transform=ax.transAxes,
        fontsize=8,
        color="#667085",
    )
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    archive_plot(args.output)
    plt.close(fig)
    print(os.path.abspath(args.output))


if __name__ == "__main__":
    main()
