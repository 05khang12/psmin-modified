from pathlib import Path
import importlib.util
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import mlsarray.mlsarray as mls

spec = importlib.util.spec_from_file_location(
    "budget_tools", HERE / "plot_c10_original_E_W_averaged_budgets.py"
)
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)

OUT = ROOT / "report_khang/figures/results"
CASES = (
    {
        "C": 1.0,
        "duration": 100.0,
        "path": ROOT / (
            "outputs/HMR_full_FIXED_NL_C1_gamma100_Dhalf_at_kstar_256_2026-09-07/"
            "data/hmr_full_FIXED_NL_C1_gamma100_Dhalf_at_kstar_256.h5"
        ),
    },
    {
        "C": 10.0,
        "duration": 60.0,
        "path": ROOT / (
            "outputs/HMR_full_FIXED_NL_C10_gamma60_D10pct_256_2026-09-07/"
            "data/hmr_full_FIXED_NL_C10_gamma60_D10pct_256.h5"
        ),
    },
)


def calculate(case):
    with h5py.File(case["path"], "r") as handle:
        gamma = float(handle["data/gamma_ref"][()])
        kappa = float(handle["data/kap"][()])
        nu = float(handle["data/nu"][()])
        diffusion = float(handle["data/D"][()])
        nu_l = float(handle["data/nu_l"][()])
        hypo_power = float(handle["data/hypo_power"][()])
        times = np.asarray(handle["fields/t"])
        indices = np.arange(times.size, dtype=int)

        sl, kx, ky, k2, inv_k2 = tools.load_grid(handle)
        response = tools.response_from_file(handle, ky, k2)
        p_k = k2 + response
        nonzero = k2 > 0.0
        active = nonzero & (ky > 0.0)

        drive = np.zeros_like(response, dtype=complex)
        normal = np.zeros_like(response, dtype=complex)
        hypo = np.zeros_like(k2)
        drive[active] = -1j * kappa * ky[active] / p_k[active]
        normal[active] = (
            -nu * k2[active] ** 2
            - diffusion * k2[active] * response[active]
        ) / p_k[active]
        hypo[nonzero] = -nu_l * inv_k2[nonzero] ** (0.5 * hypo_power)

        weights = {
            "E": k2 + np.real(response),
            "W": np.abs(p_k) ** 2,
        }
        rows = []
        for index in indices:
            phi_k = tools.phi_spectrum(handle, int(index), sl, k2, inv_k2)
            phi2 = np.abs(phi_k) ** 2
            nonlinear = tools.reduced_nonlinear_tendency(
                phi_k, sl, kx, ky, p_k, nonzero
            )
            row = [times[index]]
            for weight in weights.values():
                quantity = tools.SPECTRAL_MULTIPLICITY * np.sum(weight * phi2)
                source = tools.SPECTRAL_MULTIPLICITY * np.sum(
                    2.0 * weight * np.real(drive) * phi2
                )
                normal_damping = tools.SPECTRAL_MULTIPLICITY * np.sum(
                    2.0 * weight * np.real(normal) * phi2
                )
                hypo_damping = tools.SPECTRAL_MULTIPLICITY * np.sum(
                    2.0 * weight * hypo * phi2
                )
                transfer = tools.SPECTRAL_MULTIPLICITY * 2.0 * np.real(
                    np.sum(weight * np.conj(phi_k) * nonlinear)
                )
                row.extend((quantity, source, normal_damping, hypo_damping, transfer))
            rows.append(row)
    return gamma * np.asarray(rows, dtype=float)[:, 0], np.asarray(rows, dtype=float)


STYLES = (
    (0, r"$|\langle G\rangle_t|$", "#2ca02c", "-"),
    (1, r"$|\langle-D_{\mathrm{normal}}\rangle_t|$", "#d62728", "-."),
    (2, r"$|\langle-D_{\mathrm{hypo}}\rangle_t|$", "#ff7f0e", "--"),
    (3, r"$|\langle T\rangle_t|$", "#9467bd", ":"),
)


def make_figure(results, quantity, first_column, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for ax, case, (tau, values) in zip(axes, CASES, results):
        for offset, label, color, linestyle in STYLES:
            centers, averaged = tools.bin_average(
                tau, values[:, first_column + offset]
            )
            magnitude = np.abs(averaged)
            magnitude[magnitude == 0.0] = np.nan
            ax.semilogy(
                centers, magnitude, label=label, color=color,
                linestyle=linestyle, linewidth=1.9,
            )
        ax.set_title(rf"$C={case['C']:g}$")
        ax.set_xlabel(r"$\gamma_{\max}t$")
        ax.set_xlim(0.0, case["duration"])
        ax.grid(True, which="major", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9)
    axes[0].set_ylabel(rf"contribution to $\partial_t {quantity}$")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    output = OUT / filename
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output.resolve())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = [calculate(case) for case in CASES]
    make_figure(
        results, "E", 2, "Energy-budget decomposition",
        "C1_C10_HMR_energy_budgets_side_by_side.png",
    )
    make_figure(
        results, "W", 7, "Generalized-vorticity-budget decomposition",
        "C1_C10_HMR_generalized_vorticity_budgets_side_by_side.png",
    )
    for case, (_, values) in zip(CASES, results):
        for name, first_column in (("E", 2), ("W", 7)):
            scale = np.max(np.abs(values[:, first_column:first_column + 3]), axis=1)
            relative = np.divide(
                np.abs(values[:, first_column + 3]), scale,
                out=np.zeros_like(scale), where=scale > 0.0,
            )
            print(
                f"C={case['C']:g}: max |T_{name}| / max linear term "
                f"= {np.max(relative):.6e}"
            )


if __name__ == "__main__":
    main()
