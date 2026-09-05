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
    if "data" in fl and "C" in fl["data"] and "kap" in fl["data"]:
        c = float(fl["data/C"][()])
        kap = float(fl["data/kap"][()])
        delta_k = (kap / c) * ky * k2 / (1.0 + k2)
        return (ky > 0.0) * (1.0 - 1j * delta_k)
    raise ValueError("Could not determine R_k from the HDF5 file")


def phi_spectrum(fl, tidx, sl, k2, inv_k2):
    if "phi" in fl["fields"]:
        return mls.rft2(np.asarray(fl["fields/phi"][tidx]), sl)
    if "om" in fl["fields"]:
        omk = mls.rft2(np.asarray(fl["fields/om"][tidx]), sl)
        phik = -omk * inv_k2
        phik[k2 == 0.0] = 0.0
        return phik
    raise ValueError("Expected fields/phi or fields/om in the HDF5 file")


def nonlinear_phi_tendency(phik, sl, kx, ky, k2, inv_k2):
    irft = lambda values: mls.irft2(values, sl)
    rft = lambda values: mls.rft2(values, sl)
    dxphi = irft(1j * kx * phik)
    dyphi = irft(1j * ky * phik)
    omega = irft(-k2 * phik)
    tendency = (-1j * kx * rft(dyphi * omega) + 1j * ky * rft(dxphi * omega)) * inv_k2
    tendency[k2 == 0.0] = 0.0
    return tendency


def budget_at_time(fl, tidx, sl, kx, ky, k2, inv_k2, response_k, kap, nu, nu_h, hypo_power):
    phik = phi_spectrum(fl, tidx, sl, k2, inv_k2)
    p_k = k2 + response_k
    p2 = np.abs(p_k) ** 2
    phi2 = np.abs(phik) ** 2

    w_modes = p2 * phi2
    growth_modes = -2.0 * ky * kap * phi2 * np.imag(response_k)
    damping_normal_modes = nu * k2 * p2 * phi2
    damping_hypo_modes = nu_h * inv_k2 ** (0.5 * hypo_power) * p2 * phi2
    damping_modes = damping_normal_modes + damping_hypo_modes

    nphi = nonlinear_phi_tendency(phik, sl, kx, ky, k2, inv_k2)
    qk = p_k * phik
    dqdt_nl = p_k * nphi
    transfer = 2.0 * SPECTRAL_MULTIPLICITY * np.real(np.sum(np.conj(qk) * dqdt_nl))

    return {
        "W": float(SPECTRAL_MULTIPLICITY * np.sum(w_modes)),
        "G": float(SPECTRAL_MULTIPLICITY * np.sum(growth_modes)),
        "D_normal": float(SPECTRAL_MULTIPLICITY * np.sum(damping_normal_modes)),
        "D_hypo": float(SPECTRAL_MULTIPLICITY * np.sum(damping_hypo_modes)),
        "D": float(SPECTRAL_MULTIPLICITY * np.sum(damping_modes)),
        "T": float(transfer),
    }


def finite_difference(values, times):
    values = np.asarray(values)
    times = np.asarray(times)
    if values.size < 2:
        return np.full_like(values, np.nan, dtype=float)
    return np.gradient(values, times)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "t",
                "W",
                "G",
                "D_normal",
                "D_hypo",
                "D",
                "minus_D",
                "T",
                "G_minus_D_plus_T",
                "finite_difference_dWdt",
                "relative_T_over_W",
                "relative_budget_over_W",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_budget(path, rows, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([row["t"] for row in rows])
    g = np.array([row["G"] for row in rows])
    minus_d = np.array([row["minus_D"] for row in rows])
    transfer = np.array([row["T"] for row in rows])
    budget = np.array([row["G_minus_D_plus_T"] for row in rows])
    fd = np.array([row["finite_difference_dWdt"] for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    axes[0].plot(t, g, label=r"$G(t)$", linewidth=1.7)
    axes[0].plot(t, minus_d, label=r"$-D(t)$", linewidth=1.7)
    axes[0].plot(t, transfer, label=r"$T(t)$", linewidth=1.7)
    axes[0].set(ylabel=r"terms in $\partial_t W$", title=title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, budget, label=r"$G-D+T$", linewidth=1.7)
    axes[1].plot(t, fd, "--", label=r"finite diff. $\partial_t W$", linewidth=1.3)
    axes[1].set(xlabel="t", ylabel=r"$\partial_t W$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    archive_plot(path)
    plt.close(fig)


def plot_damping_split(path, rows, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([row["t"] for row in rows])
    d_normal = np.array([row["D_normal"] for row in rows])
    d_hypo = np.array([row["D_hypo"] for row in rows])
    d_total = np.array([row["D"] for row in rows])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(t, d_total, label=r"$D(t)$", linewidth=1.9)
    ax.plot(t, d_normal, label=r"$D_{\rm normal}(t)$", linewidth=1.6)
    ax.plot(t, d_hypo, label=r"$D_{\rm hypo}(t)$", linewidth=1.6)
    ax.set(xlabel="t", ylabel=r"damping contribution to $\partial_t W$", title=title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    archive_plot(path)
    plt.close(fig)


def analyze(input_path, output_csv, output_plot, damping_plot, stride, title):
    with h5.File(input_path, "r") as fl:
        sl, kx, ky, k2, inv_k2 = load_grid(fl)
        response_k = response_from_file(fl, ky, k2)
        kap = float(fl["data/kap"][()])
        nu = float(fl["data/nu"][()])
        nu_h = float(fl["data/nu_l"][()]) if "nu_l" in fl["data"] else 0.0
        hypo_power = float(fl["data/hypo_power"][()]) if "hypo_power" in fl["data"] else 6.0
        times = np.asarray(fl["fields/t"])
        indices = np.arange(0, times.size, stride, dtype=int)
        if indices[-1] != times.size - 1:
            indices = np.r_[indices, times.size - 1]

        rows = []
        for tidx in indices:
            row = budget_at_time(
                fl,
                int(tidx),
                sl,
                kx,
                ky,
                k2,
                inv_k2,
                response_k,
                kap,
                nu,
                nu_h,
                hypo_power,
            )
            row["t"] = float(times[tidx])
            rows.append(row)

    dwdt = finite_difference([row["W"] for row in rows], [row["t"] for row in rows])
    for row, derivative in zip(rows, dwdt):
        row["minus_D"] = -row["D"]
        row["G_minus_D_plus_T"] = row["G"] - row["D"] + row["T"]
        row["finite_difference_dWdt"] = float(derivative)
        row["relative_T_over_W"] = row["T"] / row["W"] if row["W"] else np.nan
        row["relative_budget_over_W"] = row["G_minus_D_plus_T"] / row["W"] if row["W"] else np.nan

    write_csv(output_csv, rows)
    plot_budget(output_plot, rows, title or f"W budget: {Path(input_path).stem}")
    if damping_plot:
        plot_damping_split(
            damping_plot,
            rows,
            title or f"W damping split: {Path(input_path).stem}",
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compute global W-budget terms G(t), D(t), and T(t) from HM/HMR output.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--damping-plot")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth saved field snapshot.")
    parser.add_argument("--title")
    args = parser.parse_args()

    rows = analyze(
        Path(args.input),
        Path(args.csv),
        Path(args.plot),
        Path(args.damping_plot) if args.damping_plot else None,
        max(1, args.stride),
        args.title,
    )
    print(args.csv)
    print(args.plot)
    if args.damping_plot:
        print(args.damping_plot)
    if rows:
        last = rows[-1]
        print(
            f"final t={last['t']:g}, W={last['W']:.12g}, "
            f"G={last['G']:.12g}, D={last['D']:.12g}, "
            f"D_normal={last['D_normal']:.12g}, D_hypo={last['D_hypo']:.12g}, "
            f"T={last['T']:.12g}, "
            f"G-D+T={last['G_minus_D_plus_T']:.12g}"
        )


if __name__ == "__main__":
    main()
