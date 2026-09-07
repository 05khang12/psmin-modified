from pathlib import Path
import sys
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
    "C10_gamma100_HMR_enstrophy_budget.png"
))))
CSV = Path(os.environ.get("PSMIN_DIAGNOSTIC_CSV", str(Path(
    "report_khang/figures/results/"
    "C10_gamma100_HMR_enstrophy_budget.csv"
))))
STRIDE = int(os.environ.get("PSMIN_DIAGNOSTIC_STRIDE", "10"))


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

        drive_operator = np.zeros_like(response, dtype=complex)
        normal_operator = np.zeros_like(response, dtype=complex)
        hypo_operator = np.zeros_like(k2)
        drive_operator[active] = -1j * kap * ky[active] / p_k[active]
        normal_operator[active] = (
            -nu * k2[active] ** 2
            - diffusion * k2[active] * response[active]
        ) / p_k[active]
        hypo_operator[nonzero] = (
            -nu_l * inv_k2[nonzero] ** (0.5 * hypo_power)
        )

        rows = []
        weight = k2**2
        for index in indices:
            phi_k = phi_spectrum(handle, int(index), sl, k2, inv_k2)
            phi2 = np.abs(phi_k) ** 2
            nonlinear = nonlinear_phi_tendency(phi_k, sl, kx, ky, k2, inv_k2)
            z = SPECTRAL_MULTIPLICITY * np.sum(weight * phi2)
            g = SPECTRAL_MULTIPLICITY * np.sum(
                2.0 * weight * np.real(drive_operator) * phi2
            )
            normal = SPECTRAL_MULTIPLICITY * np.sum(
                2.0 * weight * np.real(normal_operator) * phi2
            )
            hypo = SPECTRAL_MULTIPLICITY * np.sum(
                2.0 * weight * hypo_operator * phi2
            )
            transfer = SPECTRAL_MULTIPLICITY * 2.0 * np.real(
                np.sum(weight * np.conj(phi_k) * nonlinear)
            )
            rows.append((times[index], z, g, normal, hypo, transfer))

    values = np.asarray(rows, dtype=float)
    header = "t,Z,G,minus_D_normal,minus_D_hypo,T"
    CSV.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(CSV, values, delimiter=",", header=header, comments="")

    tau = gamma * values[:, 0]
    terms = (
        (values[:, 2], r"$G(t)$", "#2ca02c", "-"),
        (values[:, 3], r"$-D_{\mathrm{normal}}(t)$", "#d62728", "-."),
        (values[:, 4], r"$-D_{\mathrm{hypo}}(t)$", "#ff7f0e", "--"),
        (values[:, 5], r"$T(t)$", "#9467bd", ":"),
    )
    magnitudes = np.concatenate([np.abs(series[np.isfinite(series)]) for series, *_ in terms])
    positive = magnitudes[magnitudes > 0.0]
    linthresh = max(float(np.max(positive)) * 1.0e-6, float(np.min(positive)))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.2), sharex=True)
    axes[0].semilogy(tau, values[:, 1], color="#1f77b4", linewidth=2.0)
    axes[0].set(ylabel=r"$Z(t)$", title=rf"Enstrophy evolution and budget, $C={CASE_C:g}$")
    axes[0].grid(True, which="major", linestyle=":", alpha=0.5)

    for series, label, color, style in terms:
        axes[1].plot(tau, series, label=label, color=color,
                     linestyle=style, linewidth=1.8)
    axes[1].set_yscale("symlog", linthresh=linthresh)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel=r"$\gamma_{\max}t$",
        ylabel=r"contribution to $\partial_t Z$",
        xlim=(0.0, FINAL_GAMMA_TIME),
    )
    axes[1].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    scale = np.max(np.abs(values[:, 2:5]), axis=1)
    relative_t = np.divide(
        np.abs(values[:, 5]), scale,
        out=np.zeros_like(scale), where=scale > 0.0,
    )
    print(OUTPUT.resolve())
    print(CSV.resolve())
    print(f"max |T| / max(|G|,|D_normal|,|D_hypo|) = {np.max(relative_t):.6e}")


if __name__ == "__main__":
    main()
