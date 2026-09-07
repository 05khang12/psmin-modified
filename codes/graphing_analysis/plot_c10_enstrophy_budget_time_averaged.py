from pathlib import Path
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


CSV = Path(os.environ.get(
    "PSMIN_DIAGNOSTIC_CSV",
    "report_khang/figures/results/C10_gamma100_HMR_enstrophy_budget.csv",
))
H5 = Path(os.environ.get("PSMIN_DIAGNOSTIC_INPUT", str(Path(
    "outputs/HMR_full_gamma100_D50pct_2026-09-04/data/"
    "hmr_full_C10_gamma100_D50pct.h5"
))))
CASE_C = float(os.environ.get("PSMIN_DIAGNOSTIC_C", "10"))
FINAL_GAMMA_TIME = float(os.environ.get("PSMIN_DIAGNOSTIC_FINAL_GAMMA", "100"))
OUTPUT = Path(os.environ.get("PSMIN_DIAGNOSTIC_OUTPUT", str(Path(
    "report_khang/figures/results/"
    "C10_gamma100_HMR_enstrophy_budget_time_averaged_semilog.png"
))))
BIN_WIDTH = 1.0


def bin_average(tau, values):
    edges = np.arange(0.0, np.floor(tau[-1]) + BIN_WIDTH + 1.0e-12, BIN_WIDTH)
    centers = 0.5 * (edges[:-1] + edges[1:])
    result = np.full(centers.shape, np.nan)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if index == centers.size - 1:
            selected = (tau >= left) & (tau <= right)
        else:
            selected = (tau >= left) & (tau < right)
        if np.any(selected):
            result[index] = np.mean(values[selected])
    return centers, result


def main():
    data = np.genfromtxt(CSV, delimiter=",", names=True)
    with h5py.File(H5, "r") as handle:
        gamma = float(handle["data/gamma_ref"][()])
    tau = gamma * data["t"]

    terms = (
        ("G", r"$\langle G\rangle_t$", "#2ca02c", "-"),
        ("minus_D_normal", r"$|\langle-D_{\mathrm{normal}}\rangle_t|$", "#d62728", "-."),
        ("minus_D_hypo", r"$|\langle-D_{\mathrm{hypo}}\rangle_t|$", "#ff7f0e", "--"),
        ("T", r"$|\langle T\rangle_t|$", "#9467bd", ":"),
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for field, label, color, style in terms:
        centers, averaged = bin_average(tau, data[field])
        magnitude = np.abs(averaged)
        magnitude[magnitude == 0.0] = np.nan
        ax.semilogy(centers, magnitude, label=label, color=color,
                    linestyle=style, linewidth=1.9)
    ax.set(
        xlabel=r"$\gamma_{\max}t$",
        ylabel=r"magnitude of contribution to $\partial_t Z$",
        title=rf"Time-averaged enstrophy-budget terms, $C={CASE_C:g}$",
        xlim=(0.0, FINAL_GAMMA_TIME),
    )
    ax.grid(True, which="major", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
