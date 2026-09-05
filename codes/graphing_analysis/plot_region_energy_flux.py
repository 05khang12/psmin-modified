#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py as h5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, Patch
import mlsarray.mlsarray as mls
import numpy as np

from plot_linear_stability_regions import growth_rate
from plot_archive import archive_plot

SPECTRAL_MULTIPLICITY = 2.0


REGION_NAMES = {
    1: r"$R_1$: zonal",
    2: r"$R_2$: unstable",
    3: r"$R_3$: damped",
}


def nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def load_grid(fl):
    nx = int(fl["data/Nx"][()])
    ny = int(fl["data/Ny"][()])
    lx = float(fl["data/Lx"][()])
    ly = float(fl["data/Ly"][()])
    sl = mls.slicelist(nx, ny)
    lkx, lky = mls.init_kspace_grid(sl)
    kx = lkx * (2.0 * np.pi / lx)
    ky = lky * (2.0 * np.pi / ly)
    return sl, kx, ky, kx**2 + ky**2


def phi_spectrum_from_fields(fl, tidx, sl, k2):
    if "phi" in fl["fields"]:
        return mls.rft2(np.asarray(fl["fields/phi"][tidx]), sl)
    if "om" in fl["fields"]:
        inv_k2 = np.divide(1.0, k2, out=np.zeros_like(k2), where=k2 != 0.0)
        phik = -mls.rft2(np.asarray(fl["fields/om"][tidx]), sl) * inv_k2
        phik[k2 == 0.0] = 0.0
        return phik
    raise ValueError("Expected fields/phi or fields/om in the HDF5 file")


def classify_regions(kx, ky, gamma):
    regions = np.where(gamma > 0.0, 2, 3)
    regions[np.isclose(ky, 0.0, atol=1e-12)] = 1
    return regions


def regional_nonlinear_terms(phik, sl, kx, ky, k2, regions):
    nonzero = k2 != 0.0
    inv_k2 = np.zeros_like(k2)
    inv_k2[nonzero] = 1.0 / k2[nonzero]

    irft = lambda values: mls.irft2(values, sl)
    rft = lambda values: mls.rft2(values, sl)
    dxphi = irft(1j * kx * phik)
    dyphi = irft(1j * ky * phik)

    terms = {}
    for region in (1, 2, 3):
        donor_omega = -k2 * phik * (regions == region)
        omega = irft(donor_omega)
        term = (
            -1j * kx * rft(dyphi * omega)
            + 1j * ky * rft(dxphi * omega)
        ) * inv_k2
        term[~nonzero] = 0.0
        terms[region] = term
    return terms


def response_from_file(fl, kx, ky, k2):
    if "spectral" in fl and "r_k" in fl["spectral"]:
        return np.asarray(fl["spectral/r_k"])
    if "spectral" in fl and "delta_k" in fl["spectral"]:
        delta_k = np.asarray(fl["spectral/delta_k"])
        return (ky > 0.0) * (1.0 - 1j * delta_k)
    if "data" in fl and "C" in fl["data"] and "kap" in fl["data"]:
        c = float(fl["data/C"][()])
        kap = float(fl["data/kap"][()])
        delta_k = (kap / c) * ky * k2 / (1.0 + k2)
        return (ky > 0.0) * (1.0 - 1j * delta_k)
    return np.zeros_like(k2, dtype=complex)


def infer_method(fl):
    if "spectral" in fl and "omega_k" in fl["spectral"]:
        return "HMR"
    if "spectral" in fl and "delta_k" in fl["spectral"]:
        return "HM"
    return "HW"


def transfer_matrices(phik, energy_weight, regions, nonlinear_terms):
    transfer = np.zeros((3, 3))
    for recipient in (1, 2, 3):
        recipient_mask = regions == recipient
        for donor in (1, 2, 3):
            transfer[recipient - 1, donor - 1] = 2.0 * np.sum(
                SPECTRAL_MULTIPLICITY
                * energy_weight[recipient_mask]
                * np.real(
                    np.conj(phik[recipient_mask])
                    * nonlinear_terms[donor][recipient_mask]
                )
            )

    flux = np.zeros_like(transfer)
    for recipient in (1, 2, 3):
        for donor in (1, 2, 3):
            flux[recipient - 1, donor - 1] = 0.5 * (
                transfer[recipient - 1, donor - 1]
                - transfer[donor - 1, recipient - 1]
            )
    return transfer, flux


def invariant_value(phik, weight):
    return SPECTRAL_MULTIPLICITY * np.sum(weight * np.abs(phik) ** 2)


def relative_budget(global_balance, value):
    if not np.isfinite(value) or value == 0.0:
        return np.nan
    return global_balance / value


def draw_flux_arrow(ax, start, end, value, max_flux, color):
    if value <= 0.0 or max_flux <= 0.0:
        return
    width = 1.5 + 6.0 * value / max_flux
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14.0 + 8.0 * value / max_flux,
        linewidth=width,
        color=color,
        connectionstyle="arc3,rad=0.12",
        shrinkA=18,
        shrinkB=18,
        alpha=0.9,
        zorder=8,
    )
    ax.add_patch(arrow)
    midpoint = 0.5 * (np.asarray(start) + np.asarray(end))
    ax.text(
        midpoint[0],
        midpoint[1],
        f"{value:.2e}",
        ha="center",
        va="center",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.9, "pad": 1.5},
        zorder=9,
    )


def draw_region_background(ax, plot_kx, plot_ky, plot_gamma, plot_regions, colors, cmap, norm, kmax):
    ax.pcolormesh(
        plot_kx,
        plot_ky,
        plot_regions,
        cmap=cmap,
        norm=norm,
        shading="nearest",
    )
    ax.contour(plot_kx, plot_ky, plot_gamma, levels=[0.0], colors="white", linewidths=1.0)
    ax.axhline(0.0, color=colors[0], linewidth=2.5)
    ax.set(
        xlabel=r"$k_x$",
        ylabel=r"$k_y$",
        xlim=(-kmax, kmax),
        ylim=(0.0, kmax),
        aspect="equal",
    )


def draw_flux_panel(ax, flux, global_balance, global_value, relative, anchors, kmax, title, invariant):
    positive_fluxes = [
        flux[recipient - 1, donor - 1]
        for recipient in (1, 2, 3)
        for donor in (1, 2, 3)
        if recipient != donor and flux[recipient - 1, donor - 1] > 0.0
    ]
    max_flux = max(positive_fluxes, default=0.0)
    arrow_colors = ["#f2c14e", "#2a9d8f", "#e63946"]
    for recipient in (1, 2, 3):
        for donor in (1, 2, 3):
            value = flux[recipient - 1, donor - 1]
            if recipient != donor and value > 0.0:
                draw_flux_arrow(
                    ax,
                    anchors[donor],
                    anchors[recipient],
                    value,
                    max_flux,
                    arrow_colors[donor - 1],
                )
    ax.text(
        0.98,
        0.98,
        "\n".join(
            [
                rf"${invariant}={global_value:.2e}$",
                rf"$\partial_t^{{NL}}{invariant}/{invariant}={relative:.2e}$",
                rf"$\partial_t^{{NL}}{invariant}={global_balance:.2e}$",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9, "pad": 2.5},
        zorder=11,
    )
    ax.set_title(title)


def plot_flux_map(
    outfile,
    c,
    kap,
    d,
    nu,
    kmax,
    physical_time,
    physical_energy_transfer,
    physical_energy_flux,
    physical_energy_value,
    hm_energy_transfer,
    hm_energy_flux,
    hm_energy_value,
    enstrophy_transfer,
    enstrophy_flux,
    enstrophy_value,
    method,
    condition,
):
    points = 801
    kx_values = np.linspace(-kmax, kmax, points)
    ky_values = np.linspace(0.0, kmax, points // 2 + 1)
    plot_kx, plot_ky = np.meshgrid(kx_values, ky_values, indexing="xy")
    plot_gamma = growth_rate(plot_kx, plot_ky, c, kap, d, nu)
    plot_regions = classify_regions(plot_kx, plot_ky, plot_gamma)

    colors = ["#333333", "#d95f02", "#3973ac"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(18.4, 5.9), sharex=True, sharey=True)

    anchors = {
        1: (-0.65 * kmax, 0.0),
        2: (0.0, 0.42 * kmax),
        3: (0.62 * kmax, 0.78 * kmax),
    }

    for ax in axes:
        draw_region_background(ax, plot_kx, plot_ky, plot_gamma, plot_regions, colors, cmap, norm, kmax)
        for region, point in anchors.items():
            ax.text(
                point[0],
                point[1],
                REGION_NAMES[region],
                ha="center",
                va="center",
                color="white",
                weight="bold",
                bbox={"facecolor": colors[region - 1], "edgecolor": "white", "pad": 3},
                zorder=10,
            )

    physical_energy_global_balance = np.sum(physical_energy_transfer)
    hm_energy_global_balance = np.sum(hm_energy_transfer)
    enstrophy_global_balance = np.sum(enstrophy_transfer)
    physical_energy_relative = relative_budget(physical_energy_global_balance, physical_energy_value)
    hm_energy_relative = relative_budget(hm_energy_global_balance, hm_energy_value)
    enstrophy_relative = relative_budget(enstrophy_global_balance, enstrophy_value)
    draw_flux_panel(
        axes[0],
        physical_energy_flux,
        physical_energy_global_balance,
        physical_energy_value,
        physical_energy_relative,
        anchors,
        kmax,
        r"$E_{\rm phys}$ flux",
        r"E_{\rm phys}",
    )
    draw_flux_panel(
        axes[1],
        hm_energy_flux,
        hm_energy_global_balance,
        hm_energy_value,
        hm_energy_relative,
        anchors,
        kmax,
        r"$E_k=(k^2+\mathrm{Re}\,R_k)|\phi_k|^2$ flux",
        "E_k",
    )
    draw_flux_panel(
        axes[2],
        enstrophy_flux,
        enstrophy_global_balance,
        enstrophy_value,
        enstrophy_relative,
        anchors,
        kmax,
        r"$W$ flux",
        "W",
    )
    axes[0].legend(
        handles=[
            Patch(color=colors[0], label=r"$R_1$: $k_y=0$"),
            Patch(color=colors[1], label=r"$R_2$: $\gamma_+>0$"),
            Patch(color=colors[2], label=r"$R_3$: $\gamma_+<0$"),
        ],
        loc="lower left",
        framealpha=0.95,
    )
    title = rf"{method}: nonlinear quadratic fluxes, $k_y \geq 0$"
    if condition:
        title += f", {condition}"
    title += rf", $C={c:g}$, $\kappa={kap:g}$, $D={d:g}$, $\nu={nu:g}$, $t={physical_time:g}$"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    archive_plot(outfile)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and plot nonlinear energy fluxes between HM stability regions."
    )
    parser.add_argument("input")
    parser.add_argument("--time", type=float)
    parser.add_argument("--kmax", type=float, default=7.0)
    parser.add_argument("--output", default="hm_region_energy_flux.png")
    parser.add_argument(
        "--method",
        choices=["HM", "HMR", "HW"],
        help="Simulation method shown in the title. If omitted, infer from the HDF5 file.",
    )
    parser.add_argument(
        "--label",
        help="Run condition shown in the plot title, e.g. 'with hypoviscosity'.",
    )
    args = parser.parse_args()

    with h5.File(args.input, "r") as fl:
        sl, kx, ky, k2 = load_grid(fl)
        c = float(fl["data/C"][()])
        kap = float(fl["data/kap"][()])
        d = float(fl["data/D"][()])
        nu = float(fl["data/nu"][()])
        times = fl["fields/t"][()]
        tidx = len(times) - 1 if args.time is None else nearest_index(times, args.time)
        physical_time = float(times[tidx])
        phik = phi_spectrum_from_fields(fl, tidx, sl, k2)
        response_k = response_from_file(fl, kx, ky, k2)
        method = args.method or infer_method(fl)

    gamma = growth_rate(kx, ky, c, kap, d, nu)
    regions = classify_regions(kx, ky, gamma)
    nonlinear_terms = regional_nonlinear_terms(phik, sl, kx, ky, k2, regions)
    physical_energy_weight = k2 + np.abs(response_k) ** 2
    hm_energy_weight = k2 + np.real(response_k)
    enstrophy_weight = np.abs(k2 + response_k) ** 2
    physical_energy_value = invariant_value(phik, physical_energy_weight)
    hm_energy_value = invariant_value(phik, hm_energy_weight)
    enstrophy_value = invariant_value(phik, enstrophy_weight)
    physical_energy_transfer, physical_energy_flux = transfer_matrices(
        phik,
        physical_energy_weight,
        regions,
        nonlinear_terms,
    )
    hm_energy_transfer, hm_energy_flux = transfer_matrices(
        phik,
        hm_energy_weight,
        regions,
        nonlinear_terms,
    )
    enstrophy_transfer, enstrophy_flux = transfer_matrices(
        phik,
        enstrophy_weight,
        regions,
        nonlinear_terms,
    )
    condition = args.label or Path(args.input).stem
    plot_flux_map(
        args.output,
        c,
        kap,
        d,
        nu,
        args.kmax,
        physical_time,
        physical_energy_transfer,
        physical_energy_flux,
        physical_energy_value,
        hm_energy_transfer,
        hm_energy_flux,
        hm_energy_value,
        enstrophy_transfer,
        enstrophy_flux,
        enstrophy_value,
        method,
        condition,
    )

    np.set_printoptions(precision=7, suppress=False)
    print(f"method={method}")
    print(f"condition={condition}")
    print(f"time={physical_time:g}")
    print("E_phys T[recipient, donor] =")
    print(physical_energy_transfer)
    print(f"E_phys = {physical_energy_value:.16e}")
    print(f"E_phys global sum(T) = {np.sum(physical_energy_transfer):.16e}")
    print(f"E_phys relative global sum(T)/E_phys = {relative_budget(np.sum(physical_energy_transfer), physical_energy_value):.16e}")
    print("E_phys Pi[recipient, donor], positive means donor -> recipient =")
    print(physical_energy_flux)
    print(f"E_phys regional antisymmetric sum(Pi) = {np.sum(physical_energy_flux):.16e}")
    print("E_k T[recipient, donor] =")
    print(hm_energy_transfer)
    print(f"E_k = {hm_energy_value:.16e}")
    print(f"E_k global sum(T) = {np.sum(hm_energy_transfer):.16e}")
    print(f"E_k relative global sum(T)/E_k = {relative_budget(np.sum(hm_energy_transfer), hm_energy_value):.16e}")
    print("E_k Pi[recipient, donor], positive means donor -> recipient =")
    print(hm_energy_flux)
    print(f"E_k regional antisymmetric sum(Pi) = {np.sum(hm_energy_flux):.16e}")
    print("W T[recipient, donor] =")
    print(enstrophy_transfer)
    print(f"W = {enstrophy_value:.16e}")
    print(f"W global sum(T) = {np.sum(enstrophy_transfer):.16e}")
    print(f"W relative global sum(T)/W = {relative_budget(np.sum(enstrophy_transfer), enstrophy_value):.16e}")
    print("W Pi[recipient, donor], positive means donor -> recipient =")
    print(enstrophy_flux)
    print(f"W regional antisymmetric sum(Pi) = {np.sum(enstrophy_flux):.16e}")
    print(args.output)


if __name__ == "__main__":
    main()
