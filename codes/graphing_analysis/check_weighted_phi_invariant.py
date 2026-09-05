#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py as h5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def response_from_file(fl, ky, k2):
    if "spectral" in fl and "r_k" in fl["spectral"]:
        return np.asarray(fl["spectral/r_k"])
    if "spectral" in fl and "delta_k" in fl["spectral"]:
        delta_k = np.asarray(fl["spectral/delta_k"])
        return (ky > 0.0) * (1.0 - 1j * delta_k)
    c = float(fl["data/C"][()])
    kap = float(fl["data/kap"][()])
    delta_k = (kap / c) * ky * k2 / (1.0 + k2)
    return (ky > 0.0) * (1.0 - 1j * delta_k)


def phi_spectrum(fl, tidx, sl, k2, inv_k2):
    if "phi" in fl["fields"]:
        return mls.rft2(np.asarray(fl["fields/phi"][tidx]), sl)
    omk = mls.rft2(np.asarray(fl["fields/om"][tidx]), sl)
    phik = -omk * inv_k2
    phik[k2 == 0.0] = 0.0
    return phik


def nonlinear_phi_tendency(phik, sl, kx, ky, k2, inv_k2):
    dxphi = mls.irft2(1j * kx * phik, sl)
    dyphi = mls.irft2(1j * ky * phik, sl)
    omega = mls.irft2(-k2 * phik, sl)
    tendency = (
        -1j * kx * mls.rft2(dyphi * omega, sl)
        + 1j * ky * mls.rft2(dxphi * omega, sl)
    ) * inv_k2
    tendency[k2 == 0.0] = 0.0
    return tendency


def weighted_quantity_and_transfer(phik, nphik, weight):
    quantity = SPECTRAL_MULTIPLICITY * np.sum(weight * np.abs(phik) ** 2)
    transfer = (
        2.0
        * SPECTRAL_MULTIPLICITY
        * np.real(np.sum(weight * np.conj(phik) * nphik))
    )
    return float(quantity), float(transfer)


def rows_from_file(input_path, stride):
    with h5.File(input_path, "r") as fl:
        sl, kx, ky, k2, inv_k2 = load_grid(fl)
        response_k = response_from_file(fl, ky, k2)
        weights = {
            "K_k2": k2,
            "Omega_k4": k2**2,
            "A_k2_plus_ReR": k2 + np.real(response_k),
        }
        times = np.asarray(fl["fields/t"])
        indices = np.arange(0, times.size, stride, dtype=int)
        if indices[-1] != times.size - 1:
            indices = np.r_[indices, times.size - 1]

        rows = []
        for tidx in indices:
            phik = phi_spectrum(fl, int(tidx), sl, k2, inv_k2)
            nphik = nonlinear_phi_tendency(phik, sl, kx, ky, k2, inv_k2)
            row = {"t": float(times[tidx])}
            for name, weight in weights.items():
                quantity, transfer = weighted_quantity_and_transfer(phik, nphik, weight)
                row[name] = quantity
                row[f"T_{name}"] = transfer
                row[f"rel_T_{name}"] = transfer / quantity if quantity else np.nan
            rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(path, rows, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([row["t"] for row in rows])
    labels = {
        "K_k2": r"$K=\sum k^2|\phi_k|^2$",
        "Omega_k4": r"$\Omega=\sum k^4|\phi_k|^2$",
        "A_k2_plus_ReR": r"$A=\sum (k^2+\mathrm{Re}\,R_k)|\phi_k|^2$",
    }

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    for name, label in labels.items():
        axes[0].plot(t, [row[f"T_{name}"] for row in rows], label=label, linewidth=1.5)
        axes[1].plot(t, [row[f"rel_T_{name}"] for row in rows], label=label, linewidth=1.5)

    axes[0].set(ylabel=r"$\partial_t^{NL} A$", title=title)
    axes[1].set(xlabel="t", ylabel=r"$\partial_t^{NL} A/A$")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    archive_plot(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Check nonlinear conservation of quadratic phi-space weights."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--title")
    args = parser.parse_args()

    rows = rows_from_file(Path(args.input), max(1, args.stride))
    write_csv(Path(args.csv), rows)
    plot_rows(Path(args.plot), rows, args.title or Path(args.input).stem)

    last = rows[-1]
    print(args.csv)
    print(args.plot)
    print(
        "final "
        f"t={last['t']:g}, "
        f"T_K={last['T_K_k2']:.12g}, rel={last['rel_T_K_k2']:.12g}, "
        f"T_Omega={last['T_Omega_k4']:.12g}, rel={last['rel_T_Omega_k4']:.12g}, "
        f"T_A={last['T_A_k2_plus_ReR']:.12g}, rel={last['rel_T_A_k2_plus_ReR']:.12g}"
    )


if __name__ == "__main__":
    main()
