from pathlib import Path
import importlib.util

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


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
        "C": 0.1, "duration": 45,
        "hw": ROOT / "outputs/HMR_full_gamma45_D50pct_2026-09-04/data/hwak_C0p1_t324_D50pctHMR.h5",
        "hmr": ROOT / "outputs/HMR_full_gamma45_D50pct_2026-09-04/data/hmr_full_C0p1_gamma45.h5",
    },
    {
        "C": 1.0, "duration": 100,
        "hw": ROOT / "outputs/HW_C1_gamma100_Dhalf_at_kstar_256_2026-09-07/data/hw_C1_gamma100_Dhalf_at_kstar_256.h5",
        "hmr": ROOT / "outputs/HMR_full_FIXED_NL_C1_gamma100_Dhalf_at_kstar_256_2026-09-07/data/hmr_full_FIXED_NL_C1_gamma100_Dhalf_at_kstar_256.h5",
    },
    {
        "C": 10.0, "duration": 60,
        "hw": ROOT / "outputs/HMR_full_gamma60_D10pct_2026-09-04/data/hwak_C10_t4088_D10pctHMR.h5",
        "hmr": ROOT / "outputs/HMR_full_FIXED_NL_C10_gamma60_D10pct_256_2026-09-07/data/hmr_full_FIXED_NL_C10_gamma60_D10pct_256.h5",
    },
)


def gamma_ref(path):
    with h5py.File(path, "r") as handle:
        return float(handle["data/gamma_ref"][()])


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    axes = axes.ravel()
    for ax, case in zip(axes, CASES):
        gamma = gamma_ref(case["hmr"])
        spectra = {
            "HW": review.average_hw(case["hw"], 5.0, gamma),
            "HMR": review.average_hmr(case["hmr"], 5.0, gamma),
        }
        for model, color, style in (
            ("HW", "#1f77b4", "--"),
            ("HMR", "#d62728", "-"),
        ):
            ky, energy = spectra[model]["energy"]["ky"]
            ax.loglog(ky, review.positive_for_log(energy), color=color,
                      linestyle=style, linewidth=2.0, label=model)
        ax.set_title(rf"$C={case['C']:g}$, average over final $5/\gamma_{{\max}}$")
        ax.set_xlabel(r"$k_y$")
        ax.set_ylabel(r"$E(k_y)$")
        ax.set_ylim(bottom=1.0e-10)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.legend()
    fig.delaxes(axes[-1])
    fig.suptitle(r"$E(k_y)$ spectra comparisons", fontsize=16)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "Eky_selected_runs_last_five_gamma_loglog.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
