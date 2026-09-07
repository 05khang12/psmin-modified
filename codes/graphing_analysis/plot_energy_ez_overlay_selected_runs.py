from pathlib import Path
import importlib.util

import h5py
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "reviewer_spectra", HERE / "plot_reviewer_spectral_diagnostics.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


ROOT = Path(".")
OUT = ROOT / "report_khang/figures/results"
CASES = (
    {
        "C": 0.1,
        "duration": 45,
        "hw": ROOT / "outputs/HMR_full_gamma45_D50pct_2026-09-04/data/hwak_C0p1_t324_D50pctHMR.h5",
        "hmr": ROOT / "outputs/HMR_full_gamma45_D50pct_2026-09-04/data/hmr_full_C0p1_gamma45.h5",
    },
    {
        "C": 1.0,
        "duration": 100,
        "hw": ROOT / "outputs/HW_C1_gamma100_Dhalf_at_kstar_256_2026-09-07/data/hw_C1_gamma100_Dhalf_at_kstar_256.h5",
        "hmr": ROOT / "outputs/HMR_full_FIXED_NL_C1_gamma100_Dhalf_at_kstar_256_2026-09-07/data/hmr_full_FIXED_NL_C1_gamma100_Dhalf_at_kstar_256.h5",
    },
    {
        "C": 10.0,
        "duration": 60,
        "hw": ROOT / "outputs/HMR_full_gamma60_D10pct_2026-09-04/data/hwak_C10_t4088_D10pctHMR.h5",
        "hmr": ROOT / "outputs/HMR_full_FIXED_NL_C10_gamma60_D10pct_256_2026-09-07/data/hmr_full_FIXED_NL_C10_gamma60_D10pct_256.h5",
    },
)


def load(path, gamma=None):
    with h5py.File(path, "r") as handle:
        if gamma is None:
            gamma = float(handle["data/gamma_ref"][()])
        return {
            "gamma": gamma,
            "tau": gamma * np.asarray(handle["energies/t"]),
            "E": 2.0 * np.asarray(handle["energies/Etot"]),
            "Ez": 2.0 * np.asarray(handle["energies/Ez"]),
        }


def load_hmr(path):
    with h5py.File(path, "r") as handle:
        gamma = float(handle["data/gamma_ref"][()])
        c = float(handle["data/C"][()])
        kappa = float(handle["data/kap"][()])
        diffusion = float(handle["data/D"][()])
        nu = float(handle["data/nu"][()])
        lx = float(handle["data/Lx"][()])
        ly = float(handle["data/Ly"][()])
        nx = int(handle["data/Nx"][()])
        ny = int(handle["data/Ny"][()])
        fields = handle["fields/phi"]
        times = np.asarray(handle["fields/t"])
        _, ky, k2, retained, multiplicity, _ = review.grid(
            fields.shape[1], fields.shape[2], lx, ly, nx, ny
        )
        response = review.hw_response(k2, ky, c, kappa, diffusion, nu)
        weight = k2 + np.real(response)
        total = np.empty(times.size)
        zonal = np.empty(times.size)
        for index in range(times.size):
            phi_k = np.fft.rfft2(np.asarray(fields[index]), norm="forward")
            modes = multiplicity * weight * np.abs(phi_k) ** 2 * retained
            total[index] = np.sum(modes)
            zonal[index] = np.sum(modes * (ky == 0.0))
        return {"gamma": gamma, "tau": gamma * times, "E": total, "Ez": zonal}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    axes = axes.ravel()
    for ax, case in zip(axes, CASES):
        hmr = load_hmr(case["hmr"])
        hw = load(case["hw"], hmr["gamma"])
        for data, linestyle, model in ((hmr, "-", "HMR"), (hw, "--", "HW")):
            ax.semilogy(data["tau"], data["E"], color="#1f77b4",
                        linestyle=linestyle, linewidth=2.0,
                        label=rf"$E$, {model}")
            ax.semilogy(data["tau"], data["Ez"], color="#d62728",
                        linestyle=linestyle, linewidth=2.0,
                        label=rf"$E_Z$, {model}")
        ax.set_title(
            rf"$C={case['C']:g}$, $\gamma_{{\max}}t_{{\mathrm{{final}}}}={case['duration']}$"
        )
        ax.set_xlabel(r"$\gamma_{\max}t$")
        ax.set_ylabel(r"$E(t),\ E_Z(t)$")
        ax.set_xlim(0.0, case["duration"])
        ax.grid(True, which="major", linestyle=":", alpha=0.55)
        ax.legend(fontsize=9)
    fig.delaxes(axes[-1])
    fig.suptitle("Energy evolution comparison", fontsize=16)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "energy_evolution_comparison_selected_durations.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
