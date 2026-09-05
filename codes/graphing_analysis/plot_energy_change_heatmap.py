#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_archive import archive_plot

SPECTRAL_MULTIPLICITY = 2.0


def model_name(data):
    value = data["model"]
    return str(value.item() if value.shape == () else value)


def spectral_states(data):
    state0 = data["state0"]
    state1 = data["state1"]
    model = model_name(data)

    if model == "hwak":
        state0 = state0.reshape(2, -1)[0]
        state1 = state1.reshape(2, -1)[0]
        n0 = data["state0"].reshape(2, -1)[1]
        n1 = data["state1"].reshape(2, -1)[1]
    else:
        n0 = data["nstate0"] if "nstate0" in data else np.zeros_like(state0)
        n1 = data["nstate1"] if "nstate1" in data else np.zeros_like(state1)

    return state0, state1, n0, n1, model


def spectral_map(values, kx, ky):
    unique_kx = np.sort(np.unique(kx))
    unique_ky = np.sort(np.unique(ky))
    heatmap = np.zeros((unique_ky.size, unique_kx.size))
    kx_index = {value: i for i, value in enumerate(unique_kx)}
    ky_index = {value: i for i, value in enumerate(unique_ky)}

    for value, x, y in zip(values, kx, ky):
        heatmap[ky_index[y], kx_index[x]] += value

    return unique_kx, unique_ky, heatmap


def draw_heatmap(kx, ky, heatmap, outfile, title, colorbar_label):
    vmax = float(np.nanmax(np.abs(heatmap)))
    if vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    image = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=[float(kx.min()), float(kx.max()), float(ky.min()), float(ky.max())],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set(xlabel="kx", ylabel="ky", title=title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    archive_plot(outfile)
    plt.close(fig)


def fastest_growth(delta_e, e0, e1, kx, ky):
    idx = int(np.nanargmax(delta_e))
    return {
        "kx": float(kx[idx]),
        "ky": float(ky[idx]),
        "delta_e": float(delta_e[idx]),
        "e0": float(e0[idx]),
        "e1": float(e1[idx]),
    }


def plot_changes(npz_file, output):
    data = np.load(npz_file)
    state0, state1, n0, n1, model = spectral_states(data)
    kx = data["kx"]
    ky = data["ky"]
    ksqr = data["ksqr"]
    t0 = float(data["t0"])
    t1 = float(data["t1"])

    e0 = SPECTRAL_MULTIPLICITY * (np.abs(state0) ** 2 * ksqr + np.abs(n0) ** 2)
    e1 = SPECTRAL_MULTIPLICITY * (np.abs(state1) ** 2 * ksqr + np.abs(n1) ** 2)
    delta_e = e1 - e0
    mode = fastest_growth(delta_e, e0, e1, kx, ky)
    unique_kx, unique_ky, delta_map = spectral_map(delta_e, kx, ky)

    draw_heatmap(
        unique_kx,
        unique_ky,
        delta_map,
        output,
        f"{model} Delta E(kx,ky), t={t0:g} to {t1:g}",
        "Delta E",
    )
    return model, t0, t1, mode


def main():
    parser = argparse.ArgumentParser(description="Plot one-step spectral energy changes.")
    parser.add_argument("--input", default="one_timestep_tendency.npz")
    parser.add_argument("--output", default="energy_change_heatmap.png")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model, t0, t1, mode = plot_changes(args.input, args.output)
    print(os.path.abspath(args.output))
    print(
        f"{model} fastest growing mode from t={t0:g} to {t1:g}: "
        f"kx={mode['kx']:g}, ky={mode['ky']:g}, "
        f"DeltaE={mode['delta_e']:.12g}, E0={mode['e0']:.12g}, E1={mode['e1']:.12g}"
    )


if __name__ == "__main__":
    main()
