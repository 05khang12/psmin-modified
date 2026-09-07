from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "reviewer_spectra", HERE / "plot_reviewer_spectral_diagnostics.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

ROOT = Path("outputs/HMR_full_gamma45_D50pct_2026-09-04/data")
OUT = Path("report_khang/figures/results")
CASES = (
    {
        "tag": "C0p1",
        "C": 0.1,
        "hw": ROOT / "hwak_C0p1_t324_D50pctHMR.h5",
        "hmr": ROOT / "hmr_full_C0p1_gamma45.h5",
    },
    {
        "tag": "C10",
        "C": 10.0,
        "hw": ROOT / "hwak_C10_t3066_D50pctHMR.h5",
        "hmr": ROOT / "hmr_full_C10_gamma45_D50pct.h5",
    },
)


def plot_energy(case, result):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax, direction, xlabel, ylabel in (
        (axes[0], "kx", r"$|k_x|$", r"$E(|k_x|)$"),
        (axes[1], "ky", r"$k_y$", r"$E(k_y)$"),
    ):
        for model, color in (("HW", "#1f77b4"), ("HMR", "#d62728")):
            coordinate, values = result[model]["energy"][direction]
            ax.loglog(coordinate, review.positive_for_log(values), label=model,
                      color=color, linewidth=2.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.legend()
    fig.suptitle(
        rf"Late-time directional energy spectra (log--log), $C={case['C']:g}$, "
        rf"$D=\nu=0.5\,\gamma_{{\max}}$, final 20\% average"
    )
    fig.tight_layout()
    path = OUT / f"{case['tag']}_gamma45_D50pct_energy_spectra_kx_ky_loglog.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rates(case, result, model):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    if model == "HW":
        terms = (
            ("drive", "gradient injection", "#2ca02c", "-"),
            ("coupling", "resistive coupling", "#9467bd", "--"),
            ("normal", "normal dissipation", "#d62728", "-."),
        )
    else:
        terms = (
            ("drive", "gradient injection", "#2ca02c", "-"),
            ("normal", "normal dissipation", "#d62728", "-."),
            ("hypo", "hypodiffusion", "#ff7f0e", "--"),
        )
    for ax, direction, xlabel in (
        (axes[0], "kx", r"$|k_x|$"),
        (axes[1], "ky", r"$k_y$"),
    ):
        spectra = result[model]
        series = [spectra[name][direction][1] for name, _, _, _ in terms]
        for name, label, color, style in terms:
            coordinate, values = spectra[name][direction]
            ax.plot(coordinate, values, label=label, color=color,
                    linestyle=style, linewidth=1.8)
        ax.set_yscale("symlog", linthresh=review.symlog_threshold(series))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"contribution to $\partial_t E$")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.legend(fontsize=8)
    fig.suptitle(
        rf"{model} directional injection and dissipation spectra, "
        rf"$C={case['C']:g}$, $D=\nu=0.5\,\gamma_{{\max}}$, final 20\% average"
    )
    fig.tight_layout()
    path = OUT / (
        f"{case['tag']}_gamma45_D50pct_{model}_"
        "injection_dissipation_kx_ky_symlog.png"
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        print(f"Processing C={case['C']:g}")
        result = {
            "HW": review.average_hw(case["hw"]),
            "HMR": review.average_hmr(case["hmr"]),
        }
        for path in (plot_energy(case, result), plot_rates(case, result, "HW"),
                     plot_rates(case, result, "HMR")):
            print(path.resolve())


if __name__ == "__main__":
    main()
