#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys


PY = sys.executable
SIM_DIR = Path(__file__).resolve().parents[1] / "simulation_codes"


def run(script, output, args, log_name, restart_file=None):
    env = os.environ.copy()
    env["MLSARRAY_BACKEND"] = "numpy"
    env["PSMIN_C"] = str(args.C)
    env["PSMIN_KAP"] = str(args.kap)
    env["PSMIN_NU"] = str(args.nu)
    env["PSMIN_D"] = str(args.D)
    env["PSMIN_FLNAME"] = output
    env["PSMIN_SEED"] = str(args.seed)
    env["PSMIN_GAMMA_T1"] = str(args.gamma_t1)
    env["PSMIN_CONTINUE_GAMMA_T1"] = str(args.continue_gamma_t1)
    env["PSMIN_WECONTINUE"] = "0"
    if restart_file is not None:
        env["PSMIN_RESTART_FILE"] = restart_file

    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, log_name), "w", encoding="utf-8") as log:
        subprocess.run([PY, str(SIM_DIR / script)], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--kap", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=1.5e-4)
    parser.add_argument("--D", type=float, default=1.5e-4)
    parser.add_argument("--gamma-t1", type=float, default=30.0)
    parser.add_argument("--continue-gamma-t1", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/current/data")
    parser.add_argument("--log-dir", default="outputs/current/logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hw30 = os.path.join(args.output_dir, "hwak_gamma30.h5")
    hm30 = os.path.join(args.output_dir, "hm_gamma30.h5")
    outputs45 = [
        ("hwak_min.py", os.path.join(args.output_dir, "from_hwak_to_hwak_gamma45.h5"), hw30, "continue_from_hwak_to_hwak.log"),
        ("hm_idelta_min.py", os.path.join(args.output_dir, "from_hwak_to_hm_gamma45.h5"), hw30, "continue_from_hwak_to_hm.log"),
        ("hwak_min.py", os.path.join(args.output_dir, "from_hm_to_hwak_gamma45.h5"), hm30, "continue_from_hm_to_hwak.log"),
        ("hm_idelta_min.py", os.path.join(args.output_dir, "from_hm_to_hm_gamma45.h5"), hm30, "continue_from_hm_to_hm.log"),
    ]

    print("initial HWAK ->", hw30, flush=True)
    run("hwak_min.py", hw30, args, "initial_hwak_gamma30.log")
    print("initial HM ->", hm30, flush=True)
    run("hm_idelta_min.py", hm30, args, "initial_hm_gamma30.log")

    for script, output, source, log_name in outputs45:
        print(f"continue {source} with {script} -> {output}", flush=True)
        run(script, output, args, log_name, restart_file=source)


if __name__ == "__main__":
    main()
