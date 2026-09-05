#!/usr/bin/env python3
import argparse

import h5py as h5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation_codes"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gamma_max import GammaMax
from plot_archive import archive_plot

SPECTRAL_MULTIPLICITY = 2.0


def load_energy(path):
    with h5.File(path, "r") as fl:
        return (
            fl["energies/t"][()],
            SPECTRAL_MULTIPLICITY * fl["energies/Etot"][()],
            SPECTRAL_MULTIPLICITY * fl["energies/Ez"][()],
        )


def rescale_time_axis(ax, gamma):
    tau_max = gamma * ax.get_xlim()[1]
    tau_ticks = np.arange(0.0, tau_max + 5.0, 10.0)
    ax.set_xticks(tau_ticks / gamma)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{gamma * value:g}"))


def main():
    parser = argparse.ArgumentParser(
        description="Compare HM and HW total and zonal energies."
    )
    parser.add_argument("hm")
    parser.add_argument("hw")
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--kap", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=1.5e-4)
    parser.add_argument("--output", default="hm_hw_energy_comparison.png")
    args = parser.parse_args()

    gamma = GammaMax(args.kap, args.C, args.D).value
    hm_t, hm_e, hm_ez = load_energy(args.hm)
    hw_t, hw_e, hw_ez = load_energy(args.hw)
    xmax = max(float(hm_t[-1]), float(hw_t[-1]))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharex=True)
    panels = (
        (axes[0], hm_e, hw_e, r"$E(t)$", "Total energy"),
        (axes[1], hm_ez, hw_ez, r"$E_Z(t)$", "Zonal energy"),
    )
    for ax, hm_values, hw_values, ylabel, title in panels:
        ax.semilogy(hm_t, hm_values, linewidth=1.8, label="HM i-delta")
        ax.semilogy(hw_t, hw_values, linewidth=1.8, label="HW")
        ax.set(
            xlabel=r"$\gamma_{max}t$",
            ylabel=ylabel,
            title=title,
            xlim=(0.0, xmax),
        )
        rescale_time_axis(ax, gamma)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    archive_plot(args.output)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
