from pathlib import Path
import importlib.util

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "reviewer_spectra", HERE / "plot_reviewer_spectral_diagnostics.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

HW = Path(
    "outputs/HW_C1_gamma100_Dhalf_at_kstar_256_2026-09-07/data/"
    "hw_C1_gamma100_Dhalf_at_kstar_256.h5"
)
HMR = Path(
    "outputs/HMR_full_FIXED_NL_C1_gamma100_Dhalf_at_kstar_256_2026-09-07/data/"
    "hmr_full_FIXED_NL_C1_gamma100_Dhalf_at_kstar_256.h5"
)
OUT = Path("report_khang/figures/results")


def load_energy(path, gamma):
    with h5py.File(path, "r") as handle:
        return {
            "tau": gamma * np.asarray(handle["energies/t"]),
            "E": 2.0 * np.asarray(handle["energies/Etot"]),
            "Ez": 2.0 * np.asarray(handle["energies/Ez"]),
        }


def load_hmr_energy(path, gamma):
    with h5py.File(path, "r") as handle:
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
        kx, ky, k2, retained, multiplicity, _ = review.grid(
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
        return {"tau": gamma * times, "E": total, "Ez": zonal}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(HMR, "r") as handle:
        gamma = float(handle["data/gamma_ref"][()])

    energy = {"HW": load_energy(HW, gamma), "HMR": load_hmr_energy(HMR, gamma)}
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for model, linestyle in (("HMR", "-"), ("HW", "--")):
        ax.semilogy(energy[model]["tau"], energy[model]["E"],
                    color="#1f77b4", linestyle=linestyle, linewidth=2.0,
                    label=rf"$E$, {model}")
        ax.semilogy(energy[model]["tau"], energy[model]["Ez"],
                    color="#d62728", linestyle=linestyle, linewidth=2.0,
                    label=rf"$E_Z$, {model}")
    ax.set(
        xlabel=r"$\gamma_{\max}t$", ylabel=r"$E(t),\ E_Z(t)$",
        title=r"Energy evolution comparison, $C=1$", xlim=(0.0, 100.0),
    )
    ax.grid(True, which="major", linestyle=":", alpha=0.55)
    ax.legend()
    fig.tight_layout()
    energy_output = OUT / "C1_gamma100_FIXED_NL_HW_HMR_E_Ez_comparison.png"
    fig.savefig(energy_output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    spectra = {
        "HW": review.average_hw(HW, average_gamma_window=20.0, gamma_ref=gamma),
        "HMR": review.average_hmr(HMR, average_gamma_window=20.0, gamma_ref=gamma),
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for model, color, linestyle in (
        ("HW", "#1f77b4", "--"), ("HMR", "#d62728", "-"),
    ):
        ky, spectrum = spectra[model]["energy"]["ky"]
        ax.loglog(ky, review.positive_for_log(spectrum), color=color,
                  linestyle=linestyle, linewidth=2.0, label=model)
    ax.set(
        xlabel=r"$k_y$", ylabel=r"$E(k_y)$",
        title=r"$E(k_y)$ spectra comparison (log--log), $C=1$"
              "\n" r"average over $80\leq\gamma_{\max}t\leq100$",
    )
    ax.set_ylim(bottom=1.0e-10)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend()
    fig.tight_layout()
    spectrum_output = OUT / "C1_gamma100_FIXED_NL_HW_HMR_Eky_last20_loglog.png"
    fig.savefig(spectrum_output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(energy_output.resolve())
    print(spectrum_output.resolve())


if __name__ == "__main__":
    main()
