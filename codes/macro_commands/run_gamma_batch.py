#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys

SIM_DIR = Path(__file__).resolve().parents[1] / "simulation_codes"


RUNS = [
    ("hwak_seed1", "hwak", "hwak_min.py", "out_hwak_seed1.h5", 1),
    ("hm_seed1", "hm", "hm_idelta_min.py", "out_hm_seed1.h5", 1),
    ("hwak_seed2", "hwak", "hwak_min.py", "out_hwak_seed2.h5", 2),
    ("hm_seed2", "hm", "hm_idelta_min.py", "out_hm_seed2.h5", 2),
]


def run_case(name, script, output, seed, args, continue_run=False):
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output)
    env = os.environ.copy()
    env["MLSARRAY_BACKEND"] = "numpy"
    env["PSMIN_C"] = str(args.C)
    env["PSMIN_KAP"] = str(args.kap)
    env["PSMIN_NU"] = str(args.nu)
    env["PSMIN_D"] = str(args.D)
    env["PSMIN_FLNAME"] = output_path
    env["PSMIN_SEED"] = str(seed)
    env["PSMIN_WECONTINUE"] = "1" if continue_run else "0"
    if continue_run:
        env["PSMIN_CONTINUE_GAMMA_T1"] = str(args.continue_gamma_t1)
    else:
        env["PSMIN_GAMMA_T1"] = str(args.gamma_t1)
    os.makedirs(args.log_dir, exist_ok=True)
    stage = "continue" if continue_run else "initial"
    log_path = os.path.join(args.log_dir, f"{stage}_{name}.log")
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(SIM_DIR / script)], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--kap", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=1.5e-4)
    parser.add_argument("--D", type=float, default=1.5e-4)
    parser.add_argument("--gamma-t1", type=float, default=30.0)
    parser.add_argument("--continue-gamma-t1", type=float, default=15.0)
    parser.add_argument("--stage", choices=["initial", "continue", "both"], default="both")
    parser.add_argument("--case", default="all")
    parser.add_argument("--log-dir", default="run_logs")
    parser.add_argument("--output-dir", default="outputs/current")
    args = parser.parse_args()

    selected_runs = RUNS
    if args.case != "all":
        selected_runs = [run for run in RUNS if run[0] == args.case]
        if not selected_runs:
            raise ValueError(f"unknown case {args.case}")

    if args.stage in ("initial", "both"):
        for case, name, script, output, seed in selected_runs:
            print(f"initial {name} seed={seed} -> {output}", flush=True)
            run_case(case, script, output, seed, args, continue_run=False)

    if args.stage in ("continue", "both"):
        for case, name, script, output, seed in selected_runs:
            print(f"continue {name} seed={seed} -> {output}", flush=True)
            run_case(case, script, output, seed, args, continue_run=True)


if __name__ == "__main__":
    main()
