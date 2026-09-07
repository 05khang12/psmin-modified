from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


KAPPA = 1.0
NU = 0.0
DENSITY_DIFFUSIVITY = 0.0
C_VALUES = (0.1, 1.0, 10.0)
K_LIMIT = 3.0
N_POINTS = 601


def gamma_plus(kx, ky, c, kappa=KAPPA, nu=NU, diffusion=DENSITY_DIFFUSIVITY):
    k2 = kx**2 + ky**2
    gamma = np.zeros_like(k2, dtype=float)
    active = k2 > 0.0

    k2_active = k2[active]
    ky_active = ky[active]
    a_k = 0.5 * (
        diffusion * k2_active + c + c / k2_active + nu * k2_active
    )
    b_k = 0.5 * (
        diffusion * k2_active + c - c / k2_active - nu * k2_active
    )
    g_k = b_k**2 + c**2 / k2_active
    h_k = np.sqrt(g_k**2 + c**2 * kappa**2 * ky_active**2 / k2_active**2)
    gamma[active] = np.sqrt((h_k + g_k) / 2.0) - a_k

    # In the modified HW formulation, zonal modes do not evolve linearly.
    gamma[np.isclose(ky, 0.0)] = 0.0
    return gamma


def configure_axis(ax, c):
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title(rf"$C={c:g}$")
    ax.set_xlim(-K_LIMIT, K_LIMIT)
    ax.set_ylim(0.0, K_LIMIT)


def main():
    output_dir = Path("report_khang/figures/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    kx_coordinates = np.linspace(-K_LIMIT, K_LIMIT, N_POINTS)
    ky_coordinates = np.linspace(0.0, K_LIMIT, N_POINTS // 2 + 1)
    kx, ky = np.meshgrid(kx_coordinates, ky_coordinates)
    maps = {c: gamma_plus(kx, ky, c) for c in C_VALUES}

    for c, values in maps.items():
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        local_max = float(np.nanmax(values))
        image = ax.pcolormesh(
            kx,
            ky,
            values,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=local_max,
            rasterized=True,
        )
        if np.isclose(c, 1.0) or np.isclose(c, 10.0):
            fractions = np.array([0.01, 0.1, 0.5])
            contour_colors = ("#f5f5f5", "#ff9800", "#d32f2f")
            contour_styles = (":", "--", "-")
            ax.contour(
                kx,
                ky,
                values,
                levels=fractions * local_max,
                colors=contour_colors,
                linestyles=contour_styles,
                linewidths=1.7,
            )
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=style,
                    linewidth=1.7,
                    label=rf"${fraction:g}\,\gamma_{{\max}}^+$",
                )
                for fraction, color, style in zip(
                    fractions, contour_colors, contour_styles
                )
            ]
            ax.legend(handles=handles, loc="upper right", framealpha=0.92)
        configure_axis(ax, c)
        colorbar = fig.colorbar(image, ax=ax, pad=0.03)
        colorbar.set_label(r"$\gamma_{\boldsymbol{k}}^{+}$")
        fig.suptitle(
            rf"HW linear growth rate, $\kappa={KAPPA:g}$, $D=\nu=0$",
            y=0.995,
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"hw_gamma_plus_C{str(c).replace('.', 'p')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    fig = plt.figure(figsize=(12.5, 9.0))
    grid = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.72)
    axes = (
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    )
    for ax, c in zip(axes, C_VALUES):
        values = maps[c]
        image = ax.pcolormesh(
            kx,
            ky,
            values,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=float(np.nanmax(values)),
            rasterized=True,
        )
        configure_axis(ax, c)
        colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.047)
        colorbar.set_label(r"$\gamma_{\boldsymbol{k}}^{+}$")

    fig.suptitle(
        rf"HW linear growth rate, $\kappa={KAPPA:g}$, $D=\nu=0$, $k_y>0$",
        y=0.99,
    )
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.08, top=0.91)
    fig.savefig(
        output_dir / "hw_gamma_plus_C_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    for c, values in maps.items():
        index = np.unravel_index(np.nanargmax(values), values.shape)
        print(
            f"C={c:g}: gamma_max={values[index]:.8f}, "
            f"kx={kx[index]:.4f}, ky={ky[index]:.4f}"
        )
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
