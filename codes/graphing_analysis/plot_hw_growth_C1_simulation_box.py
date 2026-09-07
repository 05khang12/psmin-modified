import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


KAPPA = 1.0
NX = 84
NY = 84
LX = 12.0 * np.pi
LY = 12.0 * np.pi


def gamma_plus(kx, ky, c, diffusion=0.0, nu=0.0):
    k2 = kx**2 + ky**2
    gamma = np.zeros_like(k2, dtype=float)
    active = k2 > 0.0
    a_k = 0.5 * (
        diffusion * k2[active] + c + c / k2[active] + nu * k2[active]
    )
    b_k = 0.5 * (
        diffusion * k2[active] + c - c / k2[active] - nu * k2[active]
    )
    g_k = b_k**2 + c**2 / k2[active]
    h_k = np.sqrt(g_k**2 + c**2 * KAPPA**2 * ky[active] ** 2 / k2[active] ** 2)
    gamma[active] = np.sqrt((h_k + g_k) / 2.0) - a_k
    gamma[np.isclose(ky, 0.0)] = 0.0
    return gamma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()
    c = args.C

    dx = LX / NX
    dy = LY / NY
    kx_values = np.sort(2.0 * np.pi * np.fft.fftfreq(NX, d=dx))
    ky_values = 2.0 * np.pi * np.fft.rfftfreq(NY, d=dy)
    kx, ky = np.meshgrid(kx_values, ky_values)
    growth = gamma_plus(kx, ky, c)
    gamma_max = float(np.max(growth))

    damping_ratios = np.array([0.01, 0.1, 0.5])
    colors = ("#f5f5f5", "#ff9800", "#d32f2f")
    styles = (":", "--", "-")

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    image = ax.pcolormesh(
        kx,
        ky,
        growth,
        shading="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=gamma_max,
        rasterized=True,
    )
    for ratio, color, style in zip(damping_ratios, colors, styles):
        damping = ratio * gamma_max
        damped_growth = gamma_plus(
            kx,
            ky,
            c,
            diffusion=damping,
            nu=damping,
        )
        ax.contour(
            kx,
            ky,
            damped_growth,
            levels=[0.0],
            colors=[color],
            linestyles=[style],
            linewidths=1.8,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=style,
            linewidth=1.8,
            label=rf"$D=\nu={ratio:g}\,\gamma_{{\max}}^+$",
        )
        for ratio, color, style in zip(damping_ratios, colors, styles)
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.92)
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title(
        rf"Inviscid HW growth rate and dissipative neutral curves, $C={c:g}$, "
        rf"$L_x=L_y=12\pi$, $N_x=N_y=84$"
    )
    ax.set_xlim(kx_values[0], kx_values[-1])
    ax.set_ylim(ky_values[0], ky_values[-1])
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label(r"$\gamma_{\boldsymbol{k}}^{+}$")
    fig.tight_layout()

    c_label = f"{c:g}".replace(".", "p")
    output = Path(
        f"report_khang/figures/results/"
        f"hw_gamma_plus_C{c_label}_simulation_box_neutral_curves.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    index = np.unravel_index(np.argmax(growth), growth.shape)
    print(f"dkx={2*np.pi/LX:.8f}, dky={2*np.pi/LY:.8f}")
    print(
        f"gamma_max={gamma_max:.8f}, "
        f"kx={kx[index]:.8f}, ky={ky[index]:.8f}"
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
