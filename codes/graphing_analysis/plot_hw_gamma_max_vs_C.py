from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KAPPA = 1.0


def stationary_s(y):
    """Positive root of the analytic stationarity condition."""
    return (3.0 - y + np.sqrt(y**2 + 2.0 * y + 17.0)) / (2.0 * (1.0 + y))


def c_at_stationary_y(y, kappa=KAPPA):
    s = stationary_s(y)
    return 2.0 * kappa * y**1.5 / ((1.0 + y) ** 2 * s * np.sqrt(s**2 - 1.0))


def analytic_gamma_max(c, kappa=KAPPA):
    # The maximizing y=k_y^2 lies in 0<y<2.  C(y) is monotone on this interval.
    low = 1.0e-14
    high = 2.0 - 1.0e-14
    for _ in range(90):
        middle = 0.5 * (low + high)
        if c_at_stationary_y(middle, kappa) < c:
            low = middle
        else:
            high = middle

    y = 0.5 * (low + high)
    s = stationary_s(y)
    a = c * (1.0 + y) / (2.0 * y)
    return a * (s - 1.0), np.sqrt(y)


def main():
    output = Path("report_khang/figures/results/hw_gamma_max_vs_C.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    c_values = np.logspace(-1.0, 1.0, 300)
    solutions = np.array([analytic_gamma_max(c) for c in c_values])
    gamma_values = solutions[:, 0]

    reference_c = np.array([0.1, 1.0, 10.0])
    reference_gamma = np.array([analytic_gamma_max(c)[0] for c in reference_c])

    peak_index = int(np.argmax(gamma_values))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(c_values, gamma_values, color="#1f77b4", linewidth=2.2)
    ax.scatter(reference_c, reference_gamma, color="#c62828", s=42, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$C$")
    ax.set_ylabel(r"$\gamma_{\max}^{+}$")
    ax.set_title(r"Maximum HW linear growth rate, $\kappa=1$, $D=\nu=0$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"interval peak: C={c_values[peak_index]:.8f}, "
        f"gamma_max={gamma_values[peak_index]:.8f}"
    )
    for c in reference_c:
        gamma, ky = analytic_gamma_max(c)
        print(f"C={c:g}: gamma_max={gamma:.8f}, ky_max={ky:.8f}")
    print(output.resolve())


if __name__ == "__main__":
    main()
