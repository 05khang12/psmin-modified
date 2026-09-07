from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


CSV = Path("report_khang/figures/results/C10_gamma100_HMR_W_budget.csv")
H5 = Path(
    "outputs/HMR_full_gamma100_D50pct_2026-09-04/data/"
    "hmr_full_C10_gamma100_D50pct.h5"
)
OUTPUT = Path(
    "report_khang/figures/results/"
    "C10_gamma100_HMR_W_budget_terms_symlog.png"
)


def main():
    data = np.genfromtxt(CSV, delimiter=",", names=True)
    with h5py.File(H5, "r") as handle:
        gamma = float(handle["data/gamma_ref"][()])
    tau = gamma * data["t"]
    series = (
        (data["G"], r"$G(t)$", "#2ca02c", "-"),
        (-data["D_normal"], r"$-D_{\mathrm{normal}}(t)$", "#d62728", "-."),
        (-data["D_hypo"], r"$-D_{\mathrm{hypo}}(t)$", "#ff7f0e", "--"),
        (data["T"], r"$T(t)$", "#9467bd", ":"),
    )
    nonzero = np.concatenate([np.abs(values[values != 0.0]) for values, *_ in series])
    linthresh = max(float(np.max(nonzero)) * 1.0e-5, float(np.min(nonzero)))

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for values, label, color, style in series:
        ax.plot(tau, values, label=label, color=color, linestyle=style,
                linewidth=1.9)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(
        xlabel=r"$\gamma_{\max}t$",
        ylabel=r"contribution to $\partial_t W$",
        title=r"Contributions to generalized-vorticity evolution, $C=10$",
        xlim=(0.0, 100.0),
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
