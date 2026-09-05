#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py as h5
import mlsarray.mlsarray as mls
import numpy as np


def nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def main():
    parser = argparse.ArgumentParser(
        description="Convert an HM potential snapshot into an HW om,n seed."
    )
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--time", type=float, default=1.0)
    args = parser.parse_args()

    with h5.File(args.input, "r") as src:
        nx = int(src["data/Nx"][()])
        ny = int(src["data/Ny"][()])
        lx = float(src["data/Lx"][()])
        ly = float(src["data/Ly"][()])
        kap = float(src["data/kap"][()])
        c = float(src["data/C"][()])
        nu = float(src["data/nu"][()])
        d = float(src["data/D"][()])
        times = src["fields/t"][()]
        tidx = nearest_index(times, args.time)
        source_time = float(times[tidx])
        phi = np.asarray(src["fields/phi"][tidx])

    sl = mls.slicelist(nx, ny)
    lkx, lky = mls.init_kspace_grid(sl)
    kx = lkx * (2.0 * np.pi / lx)
    ky = lky * (2.0 * np.pi / ly)
    ksqr = kx**2 + ky**2
    sigk = ky > 0

    phik = mls.rft2(phi, sl)
    delta_k = (kap / c) * ky * ksqr / (1.0 + ksqr)
    nk = sigk * (1.0 - 1j * delta_k) * phik
    om = mls.irft2(-ksqr * phik, sl)
    n = mls.irft2(nk, sl)

    with h5.File(args.output, "w", libver="latest") as dst:
        data = dst.create_group("data")
        for name, value in (
            ("kap", kap),
            ("C", c),
            ("nu", nu),
            ("D", d),
            ("Lx", lx),
            ("Ly", ly),
            ("Nx", nx),
            ("Ny", ny),
            ("source_t", source_time),
        ):
            data.create_dataset(name, data=value)

        fields = dst.create_group("fields")
        fields.create_dataset("t", data=np.array([0.0]))
        fields.create_dataset("om", data=om[np.newaxis, ...])
        fields.create_dataset("n", data=n[np.newaxis, ...])

    print(f"{args.output}: HM t={source_time:g} stored as HW seed t=0")


if __name__ == "__main__":
    main()
