#!/usr/bin/env python3
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

from plot_archive import archive_plot


def growth_rate(kx, ky, c, kap, d, nu):
    k2 = kx**2 + ky**2
    gamma = np.full_like(k2, np.nan, dtype=float)
    nonzero = k2 > 0
    q = k2[nonzero]
    kyq = ky[nonzero]

    g = c**2 / q + (((q - 1.0) * c + (d - nu) * q) ** 2) / (4.0 * q**2)
    h = np.sqrt(g**2 + c**2 * kyq**2 * kap**2 / q**2)
    gamma[nonzero] = (
        np.sqrt((h + g) / 2.0)
        - ((1.0 + q) * c + q**2 * (d + nu)) / (2.0 * q)
    )
    return gamma


def main():
    parser = argparse.ArgumentParser(
        description="Plot zonal, unstable, and damped regions from the HW dispersion relation."
    )
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--kap", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=1.5e-4)
    parser.add_argument("--nu", type=float, default=1.5e-4)
    parser.add_argument("--kmax", type=float, default=7.0)
    parser.add_argument("--points", type=int, default=801)
    parser.add_argument("--output", default="linear_stability_regions.png")
    args = parser.parse_args()

    kvals = np.linspace(-args.kmax, args.kmax, args.points)
    kx, ky = np.meshgrid(kvals, kvals, indexing="xy")
    gamma = growth_rate(kx, ky, args.C, args.kap, args.D, args.nu)

    regions = np.where(gamma > 0.0, 1, 2)
    dk = 2.0 * args.kmax / (args.points - 1)
    zonal = np.abs(ky) < 0.5 * dk
    regions[zonal] = 0

    colors = ["#333333", "#d95f02", "#3973ac"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.pcolormesh(kx, ky, regions, cmap=cmap, norm=norm, shading="nearest")
    ax.contour(kx, ky, gamma, levels=[0.0], colors="white", linewidths=1.0)
    ax.axhline(0.0, color=colors[0], linewidth=2.5)
    ax.set(
        xlabel=r"$k_x$",
        ylabel=r"$k_y$",
        xlim=(-args.kmax, args.kmax),
        ylim=(-args.kmax, args.kmax),
        aspect="equal",
        title=(
            rf"Linear stability regions: $C={args.C:g}$, "
            rf"$\kappa={args.kap:g}$, $D={args.D:g}$, $\nu={args.nu:g}$"
        ),
    )
    ax.legend(
        handles=[
            Patch(color=colors[0], label=r"Zonal: $k_y=0$"),
            Patch(color=colors[1], label=r"Unstable: $\gamma_+>0$"),
            Patch(color=colors[2], label=r"Damped: $\gamma_+<0$"),
        ],
        loc="upper right",
        framealpha=0.95,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    archive_plot(args.output)
    plt.close(fig)

    finite = np.isfinite(gamma)
    index = np.nanargmax(gamma)
    iy, ix = np.unravel_index(index, gamma.shape)
    print(args.output)
    print(
        f"gamma_max={gamma[iy, ix]:.9g} at "
        f"kx={kx[iy, ix]:.6g}, ky={ky[iy, ix]:.6g}"
    )
    print(f"finite grid points={np.count_nonzero(finite)}")


if __name__ == "__main__":
    main()
