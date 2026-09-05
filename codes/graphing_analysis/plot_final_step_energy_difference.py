#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py as h5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mlsarray.mlsarray as mls
import numpy as np
from plot_archive import archive_plot

SPECTRAL_MULTIPLICITY = 2.0


def load_grid(fl):
    nx = int(fl["data/Nx"][()])
    ny = int(fl["data/Ny"][()])
    lx = float(fl["data/Lx"][()])
    ly = float(fl["data/Ly"][()])
    sl = mls.slicelist(nx, ny)
    lkx, lky = mls.init_kspace_grid(sl)
    kx = lkx * (2.0 * np.pi / lx)
    ky = lky * (2.0 * np.pi / ly)
    k2 = kx**2 + ky**2
    inv_k2 = np.divide(1.0, k2, out=np.zeros_like(k2), where=k2 != 0.0)
    return sl, kx, ky, k2, inv_k2


def phi_spectrum(fl, index, sl, k2, inv_k2):
    if "phi" in fl["fields"]:
        return mls.rft2(np.asarray(fl["fields/phi"][index]), sl)

    omk = mls.rft2(np.asarray(fl["fields/om"][index]), sl)
    phik = -omk * inv_k2
    phik[k2 == 0.0] = 0.0
    return phik


def density_spectrum(fl, index, sl, phik):
    if "n" in fl["fields"]:
        return mls.rft2(np.asarray(fl["fields/n"][index]), sl)
    return np.zeros_like(phik)


def spectral_map(values, kx, ky):
    unique_kx = np.sort(np.unique(kx))
    unique_ky = np.sort(np.unique(ky))
    heatmap = np.zeros((unique_ky.size, unique_kx.size))
    x_index = {value: index for index, value in enumerate(unique_kx)}
    y_index = {value: index for index, value in enumerate(unique_ky)}
    for value, x, y in zip(values, kx, ky):
        heatmap[y_index[y], x_index[x]] += value
    return unique_kx, unique_ky, heatmap


def final_step_difference(path):
    with h5.File(path, "r") as fl:
        times = np.asarray(fl["fields/t"])
        if times.size < 2:
            raise ValueError(f"{path} has fewer than two field snapshots")
        sl, kx, ky, k2, inv_k2 = load_grid(fl)
        phi0 = phi_spectrum(fl, -2, sl, k2, inv_k2)
        phi1 = phi_spectrum(fl, -1, sl, k2, inv_k2)
        n0 = density_spectrum(fl, -2, sl, phi0)
        n1 = density_spectrum(fl, -1, sl, phi1)
        t0 = float(times[-2])
        t1 = float(times[-1])

    e0 = SPECTRAL_MULTIPLICITY * (k2 * np.abs(phi0) ** 2 + np.abs(n0) ** 2)
    e1 = SPECTRAL_MULTIPLICITY * (k2 * np.abs(phi1) ** 2 + np.abs(n1) ** 2)
    delta = e1 - e0
    map_kx, map_ky, heatmap = spectral_map(delta, kx, ky)
    return map_kx, map_ky, heatmap, delta, e0, e1, kx, ky, t0, t1


def extrema(delta, e0, e1, kx, ky):
    gain = int(np.nanargmax(delta))
    loss = int(np.nanargmin(delta))
    return {
        "gain": (kx[gain], ky[gain], delta[gain], e0[gain], e1[gain]),
        "loss": (kx[loss], ky[loss], delta[loss], e0[loss], e1[loss]),
        "sum": float(np.nansum(delta)),
    }


def signed_log_visibility(heatmap, floor):
    visible = np.zeros_like(heatmap)
    mask = np.isfinite(heatmap) & (np.abs(heatmap) >= floor)
    visible[mask] = (
        np.sign(heatmap[mask])
        * np.log10(np.abs(heatmap[mask]) / floor)
    )
    return visible


def main():
    parser = argparse.ArgumentParser(
        description="Compare signed modal energy changes over the final saved timestep."
    )
    parser.add_argument("hm")
    parser.add_argument("hw")
    parser.add_argument("--white-below", type=float, default=1e-6)
    parser.add_argument("--output", default="hm_hw_final_step_delta_energy.png")
    args = parser.parse_args()

    hm = final_step_difference(args.hm)
    hw = final_step_difference(args.hw)
    hm_visible = signed_log_visibility(hm[2], args.white_below)
    hw_visible = signed_log_visibility(hw[2], args.white_below)
    vmax = max(
        float(np.nanmax(np.abs(hm_visible))),
        float(np.nanmax(np.abs(hw_visible))),
    )
    if vmax == 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True)
    image = None
    for ax, data, visible, name in (
        (axes[0], hm, hm_visible, "HM i-delta"),
        (axes[1], hw, hw_visible, "HW"),
    ):
        map_kx, map_ky, heatmap = data[:3]
        image = ax.imshow(
            visible,
            origin="lower",
            aspect="auto",
            extent=[
                float(map_kx.min()),
                float(map_kx.max()),
                float(map_ky.min()),
                float(map_ky.max()),
            ],
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        ax.set(
            xlabel=r"$k_x$",
            title=rf"{name}: $\Delta E_{{\mathbf{{k}}}}$",
        )
    axes[0].set_ylabel(r"$k_y$")
    fig.suptitle(rf"Final saved timestep: $t={hm[-2]:g}$ to $t={hm[-1]:g}$")
    cbar = fig.colorbar(image, ax=axes, pad=0.02, shrink=0.92)
    cbar.set_label(
        r"$\mathrm{sign}(\Delta E_{\mathbf{k}})"
        r"\log_{10}(|\Delta E_{\mathbf{k}}|/10^{-6})$"
    )
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.88, wspace=0.08)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    archive_plot(args.output)
    plt.close(fig)

    print(args.output)
    for name, data in (("HM", hm), ("HW", hw)):
        result = extrema(data[3], data[4], data[5], data[6], data[7])
        print(f"{name}: t={data[-2]:g} to {data[-1]:g}, sum DeltaE={result['sum']:.12g}")
        for kind in ("gain", "loss"):
            x, y, change, before, after = result[kind]
            print(
                f"  {kind}: kx={x:g}, ky={y:g}, DeltaE={change:.12g}, "
                f"E0={before:.12g}, E1={after:.12g}"
            )


if __name__ == "__main__":
    main()
