#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys

PLOT_DIR = Path(__file__).resolve().parent


def run(cmd):
    if cmd and cmd[0].endswith(".py"):
        cmd = [str(PLOT_DIR / cmd[0])] + cmd[1:]
    subprocess.run([sys.executable] + cmd, check=True)


def plot_hwak(infile, prefix, args):
    run([
        "plot_hwak.py",
        "--input", infile,
        "--kap", str(args.kap),
        "--C", str(args.C),
        "--D", str(args.D),
        "--energy-output", f"{prefix}_total_energy.png",
        "--zonal-output", f"{prefix}_zonal_energy.png",
        "--spectrum-kx-prefix", f"{prefix}_spectrum_kx",
        "--spectrum-ky-prefix", f"{prefix}_spectrum_ky",
    ])


def plot_hm(infile, prefix, args):
    run([
        "plot_hm_idelta.py",
        "--input", infile,
        "--kap", str(args.kap),
        "--C", str(args.C),
        "--D", str(args.D),
        "--energy-output", f"{prefix}_total_energy.png",
        "--zonal-output", f"{prefix}_zonal_energy.png",
        "--spectrum-kx-prefix", f"{prefix}_spectrum_kx",
        "--spectrum-ky-prefix", f"{prefix}_spectrum_ky",
    ])


def compare(hw, hm, prefix, args):
    run([
        "plot_compare_energy.py",
        "--hw", hw,
        "--hm", hm,
        "--kap", str(args.kap),
        "--C", str(args.C),
        "--D", str(args.D),
        "--zonal-output", f"{prefix}_zonal_energy_compare.png",
        "--hw-total-zonal-output", f"{prefix}_hwak_total_vs_zonal.png",
        "--hm-total-zonal-output", f"{prefix}_hm_total_vs_zonal.png",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--plot-dir", required=True)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--kap", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=1.5e-4)
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    p = lambda name: os.path.join(args.data_dir, name)
    q = lambda name: os.path.join(args.plot_dir, name)

    hw30 = p("hwak_gamma30.h5")
    hm30 = p("hm_gamma30.h5")
    plot_hwak(hw30, q("gamma30_hwak"), args)
    plot_hm(hm30, q("gamma30_hm"), args)
    compare(hw30, hm30, q("gamma30"), args)

    hw_hw = p("from_hwak_to_hwak_gamma45.h5")
    hw_hm = p("from_hwak_to_hm_gamma45.h5")
    hm_hw = p("from_hm_to_hwak_gamma45.h5")
    hm_hm = p("from_hm_to_hm_gamma45.h5")
    plot_hwak(hw_hw, q("gamma45_from_hwak_to_hwak"), args)
    plot_hm(hw_hm, q("gamma45_from_hwak_to_hm"), args)
    compare(hw_hw, hw_hm, q("gamma45_from_hwak"), args)
    plot_hwak(hm_hw, q("gamma45_from_hm_to_hwak"), args)
    plot_hm(hm_hm, q("gamma45_from_hm_to_hm"), args)
    compare(hm_hw, hm_hm, q("gamma45_from_hm"), args)


if __name__ == "__main__":
    main()
