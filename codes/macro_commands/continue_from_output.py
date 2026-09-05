#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "simulation_codes"
sys.path.insert(0, str(SIM_DIR))

from gamma_max import GammaMax


# Edit these defaults for quick local use, or override them from the command line.
INPUT_FILE = "out.h5"
OUTPUT_FILE = "continued.h5"
METHOD = "hm"  # "hm" or "hwak"

C = 10.0
KAP = 1.0
NU = 1.5e-4
D = 1.5e-4

# Physical continuation time. The new run starts from the last saved field time
# in INPUT_FILE and ends at last_time + CONTINUE_T1.
CONTINUE_T1 = 100.0

# If this is not None, it overrides CONTINUE_T1 and uses gamma_max * t instead.
CONTINUE_GAMMA_T1 = None

SEED = 1
LOG_FILE = None


SCRIPTS = {
    "hm": SIM_DIR / "hm_idelta_min.py",
    "hwak": SIM_DIR / "hwak_min.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue any saved psmin HDF5 output with HM i-delta or HWAK."
    )
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--method", choices=sorted(SCRIPTS), default=METHOD)
    parser.add_argument("--C", type=float, default=C)
    parser.add_argument("--kap", type=float, default=KAP)
    parser.add_argument("--nu", type=float, default=NU)
    parser.add_argument("--D", type=float, default=D)
    parser.add_argument("--t1", type=float, default=CONTINUE_T1)
    parser.add_argument("--gamma-t1", type=float, default=CONTINUE_GAMMA_T1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log", default=LOG_FILE)
    return parser.parse_args()


def continuation_gamma(args):
    if args.gamma_t1 is not None:
        return args.gamma_t1
    return args.t1 * GammaMax(args.kap, args.C, args.D).value


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    script = SCRIPTS[args.method]

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MLSARRAY_BACKEND"] = "numpy"
    env["PSMIN_RESTART_FILE"] = str(input_path)
    env["PSMIN_FLNAME"] = str(output_path)
    env["PSMIN_C"] = str(args.C)
    env["PSMIN_KAP"] = str(args.kap)
    env["PSMIN_NU"] = str(args.nu)
    env["PSMIN_D"] = str(args.D)
    env["PSMIN_SEED"] = str(args.seed)
    env["PSMIN_WECONTINUE"] = "0"
    env["PSMIN_CONTINUE_GAMMA_T1"] = str(continuation_gamma(args))

    command = [sys.executable, str(script)]
    print(f"continuing {input_path} with {script}")
    print(f"writing {output_path}")

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    else:
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
