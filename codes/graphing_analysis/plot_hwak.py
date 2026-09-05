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
import mlsarray.mlsarray as mls
from plot_archive import archive_plot

SPECTRAL_MULTIPLICITY = 2.0


def load_grid(fl):
    lx = float(fl["data/Lx"][()])
    ly = float(fl["data/Ly"][()])
    if "Nx" in fl["data"] and "Ny" in fl["data"]:
        nx = int(fl["data/Nx"][()])
        ny = int(fl["data/Ny"][()])
    else:
        npx, npy = fl["fields/om"].shape[1:3]
        nx, ny = 2 * int(np.floor(npx / 3)), 2 * int(np.floor(npy / 3))
    sl = mls.slicelist(nx, ny)
    lkx, lky = mls.init_kspace_grid(sl)
    kx = lkx * (2 * np.pi / lx)
    ky = lky * (2 * np.pi / ly)
    ksqr = kx**2 + ky**2
    inv_ksqr = np.divide(1.0, ksqr, out=np.zeros_like(ksqr), where=ksqr != 0)
    return sl, kx, ky, ksqr, inv_ksqr


def nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def time_label(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


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


def padded_log_xlim(kvals, spectrum):
    active = kvals[np.isfinite(spectrum) & (spectrum > np.finfo(float).tiny)]
    if active.size == 0:
        active = kvals
    lo = float(np.min(active))
    hi = float(np.max(active))
    if lo <= 0 or hi <= lo:
        return float(np.min(kvals)), float(np.max(kvals))
    pad = 0.05 * (np.log10(hi) - np.log10(lo))
    return 10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad)


def default_spectrum_times(infile, gamma):
    with h5.File(infile, "r") as fl:
        first_time = float(fl["fields/t"][0])
        diagnostic_time = gamma.default_spectrum_time()
        if first_time > diagnostic_time:
            diagnostic_time = first_time
        return [diagnostic_time, float(fl["fields/t"][-1])]


def plot_energy(infile, dataset, outfile, title, ylabel, gamma):
    with h5.File(infile, "r") as fl:
        t = fl["energies/t"][()]
        e = SPECTRAL_MULTIPLICITY * fl[f"energies/{dataset}"][()]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(t, e, linewidth=1.8)
    ax.set(xlabel=r"$\gamma_{max} t$", ylabel=ylabel, title=title)
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_yscale("log")
    rescale_time_axis(ax, gamma)
    clean_log_ticks(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    archive_plot(outfile)
    plt.close(fig)


def plot_energy_overlay(infile, outfile, gamma):
    with h5.File(infile, "r") as fl:
        t = fl["energies/t"][()]
        energy = SPECTRAL_MULTIPLICITY * fl["energies/Etot"][()]
        zonal = SPECTRAL_MULTIPLICITY * fl["energies/Ez"][()]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogy(t, energy, linewidth=1.8, label=r"$E(t)$")
    ax.semilogy(t, zonal, linewidth=1.8, label=r"$E_Z(t)$")
    ax.set(
        xlabel=r"$\gamma_{max} t$",
        ylabel="Energy",
        title="HW total and zonal energy",
    )
    ax.set_xlim(float(t[0]), float(t[-1]))
    rescale_time_axis(ax, gamma)
    clean_log_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    archive_plot(outfile)
    plt.close(fig)


def energy_spectrum(infile, target_time, axis):
    with h5.File(infile, "r") as fl:
        sl, kx, ky, ksqr, inv_ksqr = load_grid(fl)
        times = fl["fields/t"][()]
        tidx = nearest_index(times, target_time)
        om = fl["fields/om"][tidx]
        n = fl["fields/n"][tidx]
        actual_time = float(times[tidx])

    phik = -mls.rft2(om, sl) * inv_ksqr
    nk = mls.rft2(n, sl)
    mode_energy = SPECTRAL_MULTIPLICITY * (np.abs(phik) ** 2 * ksqr + np.abs(nk) ** 2)
    kvals = kx if axis == "kx" else ky
    unique_k = np.unique(kvals)
    spectrum = np.array([np.sum(mode_energy[kvals == value]) for value in unique_k])
    order = np.argsort(unique_k)
    return unique_k[order], spectrum[order], actual_time


def plot_spectrum(infile, outfile, target_time, axis):
    kvals, spectrum, actual_time = energy_spectrum(infile, target_time, axis)
    mask = kvals > 0
    kvals = kvals[mask]
    spectrum = spectrum[mask]
    finite = np.isfinite(spectrum) & (spectrum > 0)
    kvals = kvals[finite]
    spectrum = np.maximum(spectrum[finite], np.finfo(float).tiny)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(kvals, spectrum, color="tab:green", linewidth=1.8)
    ax.set(xlabel=axis, ylabel=f"E({axis})", title=f"HW E({axis}) at t = {actual_time:g}")
    ax.set_xlim(*padded_log_xlim(kvals, spectrum))
    clean_log_ticks(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    archive_plot(outfile)
    plt.close(fig)
    return actual_time


def main():
    parser = argparse.ArgumentParser(description="Plot diagnostics from original HW output.")
    parser.add_argument("--input", default="out_hwak.h5")
    parser.add_argument("--kap", type=float, default=1.0)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--D", type=float, default=0.15e-3)
    parser.add_argument("--energy-output", default="hwak_total_energy.png")
    parser.add_argument("--zonal-output", default="hwak_zonal_energy.png")
    parser.add_argument("--energy-overlay-output", default="hwak_energy_overlay.png")
    parser.add_argument("--spectrum-kx-prefix", default="hwak_spectrum_kx")
    parser.add_argument("--spectrum-ky-prefix", default="hwak_spectrum_ky")
    parser.add_argument("--times", type=float, nargs="+")
    args = parser.parse_args()

    gamma = GammaMax(args.kap, args.C, args.D)
    plot_energy(args.input, "Etot", args.energy_output, "HW total energy", "E(t)", gamma.value)
    print(os.path.abspath(args.energy_output))
    plot_energy(args.input, "Ez", args.zonal_output, "HW zonal energy", "Ez(t)", gamma.value)
    print(os.path.abspath(args.zonal_output))
    plot_energy_overlay(args.input, args.energy_overlay_output, gamma.value)
    print(os.path.abspath(args.energy_overlay_output))

    for target_time in args.times or default_spectrum_times(args.input, gamma):
        for axis, prefix in (("kx", args.spectrum_kx_prefix), ("ky", args.spectrum_ky_prefix)):
            outfile = f"{prefix}_t{time_label(target_time)}.png"
            actual_time = plot_spectrum(args.input, outfile, target_time, axis)
            print(f"{os.path.abspath(outfile)} t={actual_time:g}")


if __name__ == "__main__":
    main()
