from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from codes.graphing_analysis.analyze_w_budget import (
    SPECTRAL_MULTIPLICITY,
    load_grid,
    phi_spectrum,
    response_from_file,
)


INPUT = Path(
    "outputs/HMR_full_gamma100_D50pct_2026-09-04/data/"
    "hmr_full_C10_gamma100_D50pct.h5"
)
OUTPUT = Path(
    "report_khang/figures/results/"
    "C10_gamma100_HMR_nonlinear_invariants_semilog.png"
)


def main():
    with h5py.File(INPUT, "r") as handle:
        gamma = float(handle["data/gamma_ref"][()])
        energy_time = np.asarray(handle["energies/t"])
        energy = SPECTRAL_MULTIPLICITY * np.asarray(handle["energies/Etot"])

        field_time = np.asarray(handle["fields/t"])
        indices = np.searchsorted(field_time, energy_time, side="left")
        indices = np.clip(indices, 0, field_time.size - 1)
        previous = np.maximum(indices - 1, 0)
        use_previous = np.abs(field_time[previous] - energy_time) < np.abs(
            field_time[indices] - energy_time
        )
        indices[use_previous] = previous[use_previous]

        sl, kx, ky, k2, inv_k2 = load_grid(handle)
        response = response_from_file(handle, ky, k2)
        p_k = k2 + response
        generalized_vorticity = []
        for index in indices:
            phi_k = phi_spectrum(handle, int(index), sl, k2, inv_k2)
            generalized_vorticity.append(
                SPECTRAL_MULTIPLICITY * np.sum(np.abs(p_k * phi_k) ** 2)
            )

    tau = gamma * energy_time
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.semilogy(tau, energy, color="#1f77b4", linewidth=2.0, label=r"$E(t)$")
    ax.semilogy(
        tau, generalized_vorticity, color="#d62728", linewidth=2.0,
        label=r"$W(t)=\sum_{\mathbf{k}}|q_{\mathbf{k}}|^2$",
    )
    ax.set(
        xlabel=r"$\gamma_{\max}t$",
        ylabel="quadratic quantity",
        title=r"Non-linearly conserved quantities, $C=10$",
        xlim=(0.0, 100.0),
    )
    ax.grid(True, which="major", linestyle=":", alpha=0.55)
    ax.legend()
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
