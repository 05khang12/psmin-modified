#!/usr/bin/env python3
import argparse
import os

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


def rescale_time_axis(ax, gamma):
    tau_min, tau_max = [gamma * value for value in ax.get_xlim()]
    step = 5.0
    tau_ticks = np.arange(np.ceil(tau_min / step) * step, tau_max + 0.5 * step, step)
    ax.set_xticks(tau_ticks / gamma)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{gamma * value:g}"))


def clean_log_ticks(ax):
    if ax.get_xscale() == "log":
        ax.xaxis.set_minor_locator(NullLocator())
    if ax.get_yscale() == "log":
        ax.yaxis.set_minor_locator(NullLocator())


def load_energy(infile, dataset):
    with h5.File(infile, "r") as fl:
        return fl["energies/t"][()], SPECTRAL_MULTIPLICITY * fl[f"energies/{dataset}"][()]


def plot_compare(hw_file, hm_file, outfile, dataset, ylabel, title, gamma):
    thw, ehw = load_energy(hw_file, dataset)
    thm, ehm = load_energy(hm_file, dataset)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(thw, ehw, label="HW", linewidth=1.8)
    ax.plot(thm, ehm, label="i-delta HM", linewidth=1.8)
    ax.set(xlabel=r"$\gamma_{max} t$", ylabel=ylabel, title=title)
    ax.set_xlim(float(min(thw[0], thm[0])), float(max(thw[-1], thm[-1])))
    ax.set_yscale("log")
    rescale_time_axis(ax, gamma)
    clean_log_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    archive_plot(outfile)
    plt.close(fig)


def plot_total_vs_zonal(infile, label, outfile, gamma):
    with h5.File(infile, "r") as fl:
        t = fl["energies/t"][()]
        etot = SPECTRAL_MULTIPLICITY * fl["energies/Etot"][()]
        ez = SPECTRAL_MULTIPLICITY * fl["energies/Ez"][()]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(t, etot, label="E(t)", linewidth=1.8)
    ax.plot(t, ez, label="Ez(t)", linewidth=1.8)
    ax.set(xlabel=r"$\gamma_{max} t$", ylabel="energy", title=f"{label}: total vs zonal energy")
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_yscale("log")
    rescale_time_axis(ax, gamma)
    clean_log_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    archive_plot(outfile)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot HW and i-delta HM energies together.")
    parser.add_argument("--hw", default="out_hwak.h5")
    parser.add_argument("--hm", default="out_hm_idelta.h5")
    parser.add_argument("--kap", type=float, default=1.0)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--D", type=float, default=0.15e-3)
    parser.add_argument("--zonal-output", default="zonal_energy_compare_hw_hm.png")
    parser.add_argument("--hw-total-zonal-output", default="hwak_total_vs_zonal_energy.png")
    parser.add_argument("--hm-total-zonal-output", default="hm_idelta_total_vs_zonal_energy.png")
    args = parser.parse_args()

    gamma = GammaMax(args.kap, args.C, args.D)
    plot_compare(args.hw, args.hm, args.zonal_output, "Ez", "Ez(t)", "Zonal energy comparison", gamma.value)
    print(os.path.abspath(args.zonal_output))
    plot_total_vs_zonal(args.hw, "HW", args.hw_total_zonal_output, gamma.value)
    print(os.path.abspath(args.hw_total_zonal_output))
    plot_total_vs_zonal(args.hm, "i-delta HM", args.hm_total_zonal_output, gamma.value)
    print(os.path.abspath(args.hm_total_zonal_output))


if __name__ == "__main__":
    main()
