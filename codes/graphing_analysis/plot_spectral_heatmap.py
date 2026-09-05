#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py as h5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import mlsarray.mlsarray as mls
from plot_archive import archive_plot

SPECTRAL_MULTIPLICITY = 2.0


def load_grid(fl):
    if "Nx" in fl["data"] and "Ny" in fl["data"]:
        nx = int(fl["data/Nx"][()])
        ny = int(fl["data/Ny"][()])
    else:
        npx, npy = fl["fields/om"].shape[1:3]
        nx, ny = 2 * int(np.floor(npx / 3)), 2 * int(np.floor(npy / 3))

    lx = float(fl["data/Lx"][()])
    ly = float(fl["data/Ly"][()])
    sl = mls.slicelist(nx, ny)
    lkx, lky = mls.init_kspace_grid(sl)
    kx = lkx * (2 * np.pi / lx)
    ky = lky * (2 * np.pi / ly)
    ksqr = kx**2 + ky**2
    inv_ksqr = np.divide(1.0, ksqr, out=np.zeros_like(ksqr), where=ksqr != 0)
    return sl, kx, ky, ksqr, inv_ksqr


def nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def spectra_at_time(fl, target_time=None):
    sl, kx, ky, ksqr, inv_ksqr = load_grid(fl)
    times = fl["fields/t"][()]
    tidx = -1 if target_time is None else nearest_index(times, target_time)
    t = float(times[tidx])

    if "phi" in fl["fields"]:
        phik = mls.rft2(np.array(fl["fields/phi"][tidx]), sl)
    else:
        omk = mls.rft2(np.array(fl["fields/om"][tidx]), sl)
        phik = -omk * inv_ksqr
        phik[ksqr == 0] = 0.0

    nk = mls.rft2(np.array(fl["fields/n"][tidx]), sl) if "n" in fl["fields"] else np.zeros_like(phik)
    return phik, nk, kx, ky, ksqr, t


def spectral_map(phik, nk, kx, ky, ksqr):
    energy = SPECTRAL_MULTIPLICITY * (np.abs(phik) ** 2 * ksqr + np.abs(nk) ** 2)
    unique_kx = np.sort(np.unique(kx))
    unique_ky = np.sort(np.unique(ky))
    heatmap = np.zeros((unique_ky.size, unique_kx.size))

    kx_index = {value: i for i, value in enumerate(unique_kx)}
    ky_index = {value: i for i, value in enumerate(unique_ky)}
    for value, x, y in zip(energy, kx, ky):
        heatmap[ky_index[y], kx_index[x]] += value

    return unique_kx, unique_ky, heatmap


def crop_active_range(unique_kx, unique_ky, heatmap, relative_floor=1e-6):
    finite_positive = heatmap[np.isfinite(heatmap) & (heatmap > 0)]
    if finite_positive.size == 0:
        return unique_kx, unique_ky, heatmap

    threshold = float(np.max(finite_positive)) * relative_floor
    active = np.isfinite(heatmap) & (heatmap > threshold)
    rows = np.where(np.any(active, axis=1))[0]
    cols = np.where(np.any(active, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return unique_kx, unique_ky, heatmap

    row_pad = max(1, int(np.ceil(0.05 * rows.size)))
    col_pad = max(1, int(np.ceil(0.05 * cols.size)))
    r0 = max(0, int(rows.min()) - row_pad)
    r1 = min(unique_ky.size, int(rows.max()) + row_pad + 1)
    c0 = max(0, int(cols.min()) - col_pad)
    c1 = min(unique_kx.size, int(cols.max()) + col_pad + 1)
    return unique_kx[c0:c1], unique_ky[r0:r1], heatmap[r0:r1, c0:c1]


def plot_heatmap(infile, outfile, target_time=None, title=None):
    with h5.File(infile, "r") as fl:
        phik, nk, kx, ky, ksqr, t = spectra_at_time(fl, target_time)

    unique_kx, unique_ky, heatmap = spectral_map(phik, nk, kx, ky, ksqr)
    unique_kx, unique_ky, heatmap = crop_active_range(unique_kx, unique_ky, heatmap)
    finite = np.isfinite(heatmap)
    positive = heatmap[finite & (heatmap > 0)]
    floor = positive.min() if positive.size else np.finfo(float).tiny
    ceiling = positive.max() if positive.size else 1.0
    heatmap = np.nan_to_num(heatmap, nan=floor, posinf=ceiling, neginf=floor)
    heatmap = np.maximum(heatmap, floor)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    image = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=[
            float(unique_kx.min()),
            float(unique_kx.max()),
            float(unique_ky.min()),
            float(unique_ky.max()),
        ],
        cmap="viridis",
        norm=LogNorm(vmin=floor, vmax=float(ceiling)),
    )
    ax.set(
        xlabel="kx",
        ylabel="ky",
        title=title or f"E(kx, ky) at t = {t:g}",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"$E(k_x,k_y)$")
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    archive_plot(outfile)
    plt.close(fig)
    return t


def main():
    parser = argparse.ArgumentParser(description="Plot spectral heatmap E(kx, ky) with log-scaled energy.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time", type=float, help="Target time; nearest saved field is used. Defaults to final time.")
    parser.add_argument("--title")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    actual_time = plot_heatmap(args.input, args.output, args.time, args.title)
    print(f"{os.path.abspath(args.output)} t={actual_time:g}")


if __name__ == "__main__":
    main()
