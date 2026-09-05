#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from plot_hm_idelta import energy_spectrum as hm_energy_spectrum
from plot_hwak import energy_spectrum as hw_energy_spectrum
from plot_archive import archive_plot


def clean_spectrum(kvals, spectrum):
    mask = (kvals > 0.0) & np.isfinite(spectrum) & (spectrum > 0.0)
    return kvals[mask], spectrum[mask]


def padded_limits(*arrays):
    values = np.concatenate(arrays)
    lo = float(np.min(values))
    hi = float(np.max(values))
    pad = 0.05 * (np.log10(hi) - np.log10(lo))
    return 10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad)


def main():
    parser = argparse.ArgumentParser(
        description="Compare final HM and HW one-dimensional energy spectra."
    )
    parser.add_argument("hm")
    parser.add_argument("hw")
    parser.add_argument("--time", type=float, default=4132.0)
    parser.add_argument("--output", default="hm_hw_spectrum_comparison.png")
    args = parser.parse_args()

    spectra = {}
    actual_times = {}
    for axis in ("kx", "ky"):
        hm_k, hm_e, hm_t = hm_energy_spectrum(args.hm, args.time, axis)
        hw_k, hw_e, hw_t = hw_energy_spectrum(args.hw, args.time, axis)
        spectra[axis] = (
            *clean_spectrum(hm_k, hm_e),
            *clean_spectrum(hw_k, hw_e),
        )
        actual_times[axis] = (hm_t, hw_t)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    for ax, axis in zip(axes, ("kx", "ky")):
        hm_k, hm_e, hw_k, hw_e = spectra[axis]
        ax.loglog(hm_k, hm_e, linewidth=1.8, label="HM i-delta")
        ax.loglog(hw_k, hw_e, linewidth=1.8, label="HW")
        ax.set(
            xlabel=rf"${axis}$",
            ylabel=rf"$E({axis})$",
            title=rf"$E({axis})$ at $t={actual_times[axis][0]:g}$",
            xlim=padded_limits(hm_k, hw_k),
        )
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    archive_plot(args.output)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
