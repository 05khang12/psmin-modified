#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys

PLOT_DIR = Path(__file__).resolve().parents[1] / "graphing_analysis"


RUNS = [
    ("seed1", "out_hwak_seed1.h5", "out_hm_seed1.h5"),
    ("seed2", "out_hwak_seed2.h5", "out_hm_seed2.h5"),
]


def run(cmd):
    if cmd and cmd[0].endswith(".py"):
        cmd = [str(PLOT_DIR / cmd[0])] + cmd[1:]
    subprocess.run([sys.executable] + cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--kap", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=1.5e-4)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    for seed, hw, hm in RUNS:
        hw_path = os.path.join(args.data_dir, hw)
        hm_path = os.path.join(args.data_dir, hm)

        hw_prefix = os.path.join(args.dest, f"hwak_{seed}")
        hm_prefix = os.path.join(args.dest, f"hm_{seed}")
        run([
            "plot_hwak.py",
            "--input", hw_path,
            "--kap", str(args.kap),
            "--C", str(args.C),
            "--D", str(args.D),
            "--energy-output", f"{hw_prefix}_total_energy.png",
            "--zonal-output", f"{hw_prefix}_zonal_energy.png",
            "--spectrum-ky-prefix", f"{hw_prefix}_spectrum_ky",
        ])
        run([
            "plot_hm_idelta.py",
            "--input", hm_path,
            "--kap", str(args.kap),
            "--C", str(args.C),
            "--D", str(args.D),
            "--energy-output", f"{hm_prefix}_total_energy.png",
            "--zonal-output", f"{hm_prefix}_zonal_energy.png",
            "--spectrum-ky-prefix", f"{hm_prefix}_spectrum_ky",
        ])
        run([
            "plot_compare_energy.py",
            "--hw", hw_path,
            "--hm", hm_path,
            "--kap", str(args.kap),
            "--C", str(args.C),
            "--D", str(args.D),
            "--zonal-output", os.path.join(args.dest, f"{seed}_zonal_energy_compare_hw_hm.png"),
            "--hw-total-zonal-output", os.path.join(args.dest, f"{seed}_hwak_total_vs_zonal_energy.png"),
            "--hm-total-zonal-output", os.path.join(args.dest, f"{seed}_hm_total_vs_zonal_energy.png"),
        ])


if __name__ == "__main__":
    main()
