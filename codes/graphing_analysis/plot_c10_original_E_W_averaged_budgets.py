from pathlib import Path
import sys
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlsarray.mlsarray as mls
from codes.graphing_analysis.analyze_w_budget import (
    SPECTRAL_MULTIPLICITY,
    load_grid,
    nonlinear_phi_tendency,
    phi_spectrum,
    response_from_file,
)


INPUT = Path(os.environ.get("PSMIN_DIAGNOSTIC_INPUT", str(Path(
    "outputs/HMR_full_gamma100_D50pct_2026-09-04/data/"
    "hmr_full_C10_gamma100_D50pct.h5"
))))
CASE_C = float(os.environ.get("PSMIN_DIAGNOSTIC_C", "10"))
FINAL_GAMMA_TIME = float(os.environ.get("PSMIN_DIAGNOSTIC_FINAL_GAMMA", "100"))
OUTPUT = Path(os.environ.get("PSMIN_DIAGNOSTIC_OUTPUT", str(Path(
    "report_khang/figures/results/"
    "C10_gamma100_HMR_original_E_W_time_averaged_budgets.png"
))))
CSV = Path(os.environ.get("PSMIN_DIAGNOSTIC_CSV", str(Path(
    "report_khang/figures/results/"
    "C10_gamma100_HMR_original_E_W_budgets.csv"
))))
STRIDE = int(os.environ.get("PSMIN_DIAGNOSTIC_STRIDE", "10"))
BIN_WIDTH = 1.0


def reduced_nonlinear_tendency(phi_k, sl, kx, ky, p_k, nonzero):
    q_k = p_k * phi_k
    dx_phi = mls.irft2(1j * kx * phi_k, sl)
    dy_phi = mls.irft2(1j * ky * phi_k, sl)
    dx_q = mls.irft2(1j * kx * q_k, sl)
    dy_q = mls.irft2(1j * ky * q_k, sl)
    bracket_k = mls.rft2(dx_phi * dy_q - dy_phi * dx_q, sl)
    tendency = np.zeros_like(phi_k)
    tendency[nonzero] = -bracket_k[nonzero] / p_k[nonzero]
    return tendency


def bin_average(tau, values):
    edges = np.arange(0.0, np.floor(tau[-1]) + BIN_WIDTH + 1.0e-12, BIN_WIDTH)
    centers = 0.5 * (edges[:-1] + edges[1:])
    result = np.full_like(centers, np.nan)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (tau >= left) & ((tau <= right) if index == centers.size - 1 else (tau < right))
        if np.any(mask):
            result[index] = np.mean(values[mask])
    return centers, result


def main():
    with h5py.File(INPUT, "r") as handle:
        gamma = float(handle["data/gamma_ref"][()])
        kap = float(handle["data/kap"][()])
        nu = float(handle["data/nu"][()])
        diffusion = float(handle["data/D"][()])
        nu_l = float(handle["data/nu_l"][()])
        hypo_power = float(handle["data/hypo_power"][()])
        times = np.asarray(handle["fields/t"])
        indices = np.arange(0, times.size, STRIDE, dtype=int)
        if indices[-1] != times.size - 1:
            indices = np.r_[indices, times.size - 1]

        sl, kx, ky, k2, inv_k2 = load_grid(handle)
        response = response_from_file(handle, ky, k2)
        p_k = k2 + response
        nonzero = k2 > 0.0
        active = nonzero & (ky > 0.0)

        drive = np.zeros_like(response, dtype=complex)
        normal = np.zeros_like(response, dtype=complex)
        hypo = np.zeros_like(k2)
        drive[active] = -1j * kap * ky[active] / p_k[active]
        normal[active] = (
            -nu * k2[active] ** 2 - diffusion * k2[active] * response[active]
        ) / p_k[active]
        hypo[nonzero] = -nu_l * inv_k2[nonzero] ** (0.5 * hypo_power)

        weights = {
            "E": k2 + np.real(response),
            "W": np.abs(p_k) ** 2,
        }
        rows = []
        for index in indices:
            phi_k = phi_spectrum(handle, int(index), sl, k2, inv_k2)
            phi2 = np.abs(phi_k) ** 2
            nonlinear = reduced_nonlinear_tendency(
                phi_k, sl, kx, ky, p_k, nonzero
            )
            row = [times[index]]
            for weight in weights.values():
                quantity = SPECTRAL_MULTIPLICITY * np.sum(weight * phi2)
                g = SPECTRAL_MULTIPLICITY * np.sum(2.0 * weight * np.real(drive) * phi2)
                d_normal = SPECTRAL_MULTIPLICITY * np.sum(2.0 * weight * np.real(normal) * phi2)
                d_hypo = SPECTRAL_MULTIPLICITY * np.sum(2.0 * weight * hypo * phi2)
                transfer = SPECTRAL_MULTIPLICITY * 2.0 * np.real(
                    np.sum(weight * np.conj(phi_k) * nonlinear)
                )
                row.extend((quantity, g, d_normal, d_hypo, transfer))
            rows.append(row)

    values = np.asarray(rows, dtype=float)
    header = (
        "t,E,E_G,E_minus_D_normal,E_minus_D_hypo,E_T,"
        "W,W_G,W_minus_D_normal,W_minus_D_hypo,W_T"
    )
    CSV.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(CSV, values, delimiter=",", header=header, comments="")
    tau = gamma * values[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.2), sharex=True)
    definitions = (
        (axes[0], "E", 2, r"Physical-energy budget, $E$"),
        (axes[1], "W", 7, "Generalized-vorticity budget"),
    )
    styles = (
        (0, r"$|\langle G\rangle_t|$", "#2ca02c", "-"),
        (1, r"$|\langle-D_{\mathrm{normal}}\rangle_t|$", "#d62728", "-."),
        (2, r"$|\langle-D_{\mathrm{hypo}}\rangle_t|$", "#ff7f0e", "--"),
        (3, r"$|\langle T\rangle_t|$", "#9467bd", ":"),
    )
    for ax, name, first_column, title in definitions:
        for offset, label, color, style in styles:
            centers, averaged = bin_average(tau, values[:, first_column + offset])
            magnitude = np.abs(averaged)
            magnitude[magnitude == 0.0] = np.nan
            ax.semilogy(centers, magnitude, label=label, color=color,
                        linestyle=style, linewidth=1.9)
        ax.set_ylabel(rf"contribution to $\partial_t {name}$")
        ax.set_title(title)
        ax.grid(True, which="major", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9)
    axes[-1].set(xlabel=r"$\gamma_{\max}t$", xlim=(0.0, FINAL_GAMMA_TIME))
    fig.suptitle(
        rf"Time-averaged energy and generalized-vorticity budget terms, $C={CASE_C:g}$"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.resolve())
    print(CSV.resolve())
    for name, first_column in (("E", 2), ("W", 7)):
        scale = np.max(np.abs(values[:, first_column:first_column + 3]), axis=1)
        relative = np.divide(
            np.abs(values[:, first_column + 3]), scale,
            out=np.zeros_like(scale), where=scale > 0.0,
        )
        print(f"max |T_{name}| / max linear term = {np.max(relative):.6e}")


if __name__ == "__main__":
    main()
