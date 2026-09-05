#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py as h5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np

from plot_hm_idelta import energy_spectrum as hmr_energy_spectrum
from plot_hwak import energy_spectrum as hw_energy_spectrum
from plot_archive import archive_plot
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation_codes"))
from gamma_max import GammaMax

SPECTRAL_MULTIPLICITY = 2.0


def load_energy(path):
    with h5.File(path, "r") as fl:
        gamma_ref = float(fl["data/gamma_ref"][()]) if "gamma_ref" in fl["data"] else None
        if gamma_ref is None:
            gamma_ref = GammaMax(float(fl["data/kap"][()]), float(fl["data/C"][()]), float(fl["data/D"][()])).value
        return {
            "t": fl["energies/t"][()],
            "e": SPECTRAL_MULTIPLICITY * fl["energies/Etot"][()],
            "ez": SPECTRAL_MULTIPLICITY * fl["energies/Ez"][()],
            "gamma_ref": gamma_ref,
        }


def clean(k, e):
    mask = (k > 0.0) & np.isfinite(e) & (e > 0.0)
    return k[mask], e[mask]


def padded_limits(*arrays):
    values = np.concatenate([a[np.isfinite(a) & (a > 0.0)] for a in arrays])
    lo = float(np.min(values))
    hi = float(np.max(values))
    pad = 0.05 * (np.log10(hi) - np.log10(lo))
    return 10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad)


def rescale_time_axis(ax, gamma_ref):
    xmax = ax.get_xlim()[1]
    tau_max = gamma_ref * xmax
    ticks = np.arange(0.0, tau_max + 5.0, 10.0)
    ax.set_xticks(ticks / gamma_ref)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{gamma_ref * value:g}"))


def plot_energy_pair(hmr, hw, label, color, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharex=True)
    panels = (
        (axes[0], "e", r"$E(t)$", "Total energy"),
        (axes[1], "ez", r"$E_Z(t)$", "Zonal energy"),
    )
    xmax = max(float(hmr["t"][-1]), float(hw["t"][-1]))
    for ax, key, ylabel, title in panels:
        ax.semilogy(hmr["t"], hmr[key], color=color, linewidth=1.9, label=f"HMR full {label}")
        ax.semilogy(hw["t"], hw[key], color=color, linestyle="--", linewidth=1.9, label=f"HW {label}")
        ax.set(xlabel=r"$\gamma_{max}^{HMR}t$", ylabel=ylabel, title=title, xlim=(0.0, xmax))
        rescale_time_axis(ax, hmr["gamma_ref"])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    archive_plot(output)
    plt.close(fig)
    print(output)


def plot_spectrum_pair(hmr_path, hw_path, time, label, color, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    for ax, axis in zip(axes, ("kx", "ky")):
        hmr_k, hmr_e, hmr_t = hmr_energy_spectrum(hmr_path, time, axis)
        hw_k, hw_e, hw_t = hw_energy_spectrum(hw_path, time, axis)
        hmr_k, hmr_e = clean(hmr_k, hmr_e)
        hw_k, hw_e = clean(hw_k, hw_e)
        ax.loglog(hmr_k, hmr_e, color=color, linewidth=1.9, label=f"HMR full {label}")
        ax.loglog(hw_k, hw_e, color=color, linestyle="--", linewidth=1.9, label=f"HW {label}")
        ax.set(
            xlabel=rf"${axis}$",
            ylabel=rf"$E({axis})$",
            title=rf"$E({axis})$ at final time",
            xlim=padded_limits(hmr_k, hw_k),
        )
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    archive_plot(output)
    plt.close(fig)
    print(output)


def plot_energy_zonal_shift(cases, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    panels = (
        (axes[0], "HMR full", "hmr", "-"),
        (axes[1], "HW", "hw", "--"),
    )
    for ax, method_label, key, linestyle in panels:
        xmax = 0.0
        for case in cases:
            data = case[key]
            xmax = max(xmax, float(data["t"][-1]))
            ax.semilogy(
                data["t"],
                data["e"],
                color=case["color"],
                linestyle=linestyle,
                linewidth=1.9,
                label=rf"$E$, {case['label']}",
            )
            ax.semilogy(
                data["t"],
                data["ez"],
                color=case["color"],
                linestyle=":",
                linewidth=2.0,
                label=rf"$E_Z$, {case['label']}",
            )
        ax.set(
            xlabel=r"$\gamma_{max}^{HMR}t$",
            ylabel=r"$E, E_Z$",
            title=method_label,
            xlim=(0.0, xmax),
        )
        rescale_time_axis(ax, cases[0]["hmr"]["gamma_ref"])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Total and zonal energy transition")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    archive_plot(output)
    plt.close(fig)
    print(output)


def plot_case_energy_zonal(case, output):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for key, method, style in (("hmr", case["hmr_label"], "-"), ("hw", "HW", "--")):
        data = case[key]
        ax.semilogy(
            data["t"],
            data["e"],
            color="tab:blue",
            linestyle=style,
            linewidth=2.0,
            alpha=1.0,
            label=rf"$E$, {method}",
        )
        ax.semilogy(
            data["t"],
            data["ez"],
            color="tab:red",
            linestyle=style,
            linewidth=1.7,
            alpha=0.95,
            label=rf"$E_Z$, {method}",
        )
    ax.set(
        xlabel=r"$\gamma_{max}t$",
        ylabel=r"$E, E_Z$",
        title=rf"Energy and zonal energy, {case['label']}",
        xlim=(0.0, max(float(case["hmr"]["t"][-1]), float(case["hw"]["t"][-1]))),
    )
    rescale_time_axis(ax, case["hmr"]["gamma_ref"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    archive_plot(output)
    plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser(description="Overlay HMR-full and HW benchmark diagnostics.")
    parser.add_argument("--hmr-c1", required=True)
    parser.add_argument("--hw-c1", required=True)
    parser.add_argument("--hmr-c0p1", required=True)
    parser.add_argument("--hw-c0p1", required=True)
    parser.add_argument("--hmr-c10")
    parser.add_argument("--hw-c10")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = [
        {"label": "C=1", "color": "tab:blue", "hmr_label": "HMR full", "hmr_path": args.hmr_c1, "hw_path": args.hw_c1, "stem": "C1"},
        {"label": "C=0.1", "color": "tab:orange", "hmr_label": "HMR full", "hmr_path": args.hmr_c0p1, "hw_path": args.hw_c0p1, "stem": "C0p1"},
    ]
    if args.hmr_c10 and args.hw_c10:
        cases.append({"label": "C=10", "color": "tab:green", "hmr_label": "HMR", "hmr_path": args.hmr_c10, "hw_path": args.hw_c10, "stem": "C10"})
    for case in cases:
        case["hmr"] = load_energy(case["hmr_path"])
        case["hw"] = load_energy(case["hw_path"])
        plot_energy_pair(case["hmr"], case["hw"], case["label"], case["color"], outdir / f"hmr_full_hw_{case['stem']}_E_Ez_overlay.png")
        if case["stem"] != "C10":
            plot_spectrum_pair(case["hmr_path"], case["hw_path"], max(case["hmr"]["t"][-1], case["hw"]["t"][-1]), case["label"], case["color"], outdir / f"hmr_full_hw_{case['stem']}_final_spectra.png")
        plot_case_energy_zonal(case, outdir / f"hmr_hw_{case['stem']}_energy_zonal_same_axes.png")
    plot_energy_zonal_shift(cases, outdir / "hmr_full_hw_energy_zonal_shift.png")


if __name__ == "__main__":
    main()
