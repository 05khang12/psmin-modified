from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("outputs/HMR_full_gamma45_D50pct_2026-09-04/data")
OUT = Path("report_khang/figures/results")
CASES = (
    (0.1, ROOT / "hwak_C0p1_t324_D50pctHMR.h5", ROOT / "hmr_full_C0p1_gamma45.h5"),
    (1.0, ROOT / "hwak_C1_t421_D50pctHMR.h5", ROOT / "hmr_full_C1_gamma45.h5"),
    (10.0, ROOT / "hwak_C10_t3066_D50pctHMR.h5", ROOT / "hmr_full_C10_gamma45_D50pct.h5"),
)
SPECTRAL_MULTIPLICITY = 2.0


def load(path, gamma=None):
    with h5py.File(path, "r") as handle:
        if gamma is None:
            gamma = float(handle["data/gamma_ref"][()])
        return {
            "gamma": gamma,
            "tau": gamma * np.asarray(handle["energies/t"]),
            "E": SPECTRAL_MULTIPLICITY * np.asarray(handle["energies/Etot"]),
            "Ez": SPECTRAL_MULTIPLICITY * np.asarray(handle["energies/Ez"]),
        }


def main():
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.0), sharex=True)
    for row, (c_value, hw_path, hmr_path) in enumerate(CASES):
        hmr = load(hmr_path)
        data = {"HW": load(hw_path, hmr["gamma"]), "HMR": hmr}
        for column, (quantity, ylabel, title) in enumerate((
            ("E", r"$E(t)$", "Total energy"),
            ("Ez", r"$E_Z(t)$", "Zonal energy"),
        )):
            ax = axes[row, column]
            for model, color, style in (
                ("HW", "#1f77b4", "--"),
                ("HMR", "#d62728", "-"),
            ):
                ax.semilogy(data[model]["tau"], data[model][quantity],
                            color=color, linestyle=style, linewidth=2.0,
                            label=model)
            ax.set_title(rf"{title}, $C={c_value:g}$")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.0, 45.0)
            ax.grid(True, which="major", linestyle=":", alpha=0.55)
            ax.legend()
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\gamma_{\max}t$")
    fig.suptitle(r"HW--HMR energy evolution, $D=\nu=0.5\,\gamma_{\max}$")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "C0p1_C1_C10_D50pct_E_Ez_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
