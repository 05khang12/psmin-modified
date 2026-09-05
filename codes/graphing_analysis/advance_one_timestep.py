#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py as h5
import numpy as np

import mlsarray.mlsarray as mls
from etdrk4cp.etdrk4cp import etdrk4cp
from etdrk4cp.gsol import gsol, dot


class NullCallbacks:
    def act(self, t, y):
        pass


def nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def hwak_state(fl, sl, kx, ky, ksqr, target_time):
    times = fl["fields/t"][()]
    tidx = nearest_index(times, target_time)
    om = fl["fields/om"][tidx]
    n = fl["fields/n"][tidx]
    omk = mls.rft2(om, sl)
    nk = mls.rft2(n, sl)
    phik = np.zeros_like(omk)
    mask = ksqr != 0
    phik[mask] = -omk[mask] / ksqr[mask]
    return float(times[tidx]), np.vstack((phik, nk)).ravel()


def hm_state(fl, sl, ksqr, target_time):
    times = fl["fields/t"][()]
    tidx = nearest_index(times, target_time)
    if "phi" in fl["fields"]:
        phik = mls.rft2(fl["fields/phi"][tidx], sl)
    else:
        omk = mls.rft2(fl["fields/om"][tidx], sl)
        phik = np.zeros_like(omk)
        mask = ksqr != 0
        phik[mask] = -omk[mask] / ksqr[mask]
    return float(times[tidx]), phik


def grid_from_file(fl):
    nx = int(fl["data/Nx"][()]) if "Nx" in fl["data"] else fl["fields/om"].shape[1]
    ny = int(fl["data/Ny"][()]) if "Ny" in fl["data"] else fl["fields/om"].shape[2]
    lx = float(fl["data/Lx"][()])
    ly = float(fl["data/Ly"][()])
    sl = mls.slicelist(nx, ny)
    lkx, lky = mls.init_kspace_grid(sl)
    kx = lkx * (2.0 * np.pi / lx)
    ky = lky * (2.0 * np.pi / ly)
    return sl, kx, ky, kx**2 + ky**2


def hm_response_from_file(fl, kx, ky, ksqr):
    if "spectral" in fl and "r_k" in fl["spectral"]:
        return np.asarray(fl["spectral/r_k"])
    if "spectral" in fl and "delta_k" in fl["spectral"]:
        delta_k = np.asarray(fl["spectral/delta_k"])
        return (ky > 0) * (1.0 - 1j * delta_k)

    kap = float(fl["data/kap"][()])
    c = float(fl["data/C"][()])
    delta_k = (kap / c) * ky * ksqr / (1.0 + ksqr)
    return (ky > 0) * (1.0 - 1j * delta_k)


def hwak_step(fl, target_time, dt, tol, m):
    c = float(fl["data/C"][()])
    kap = float(fl["data/kap"][()])
    nu = float(fl["data/nu"][()])
    d = float(fl["data/D"][()])
    sl, kx, ky, ksqr = grid_from_file(fl)
    sigk = ky > 0

    irft = lambda x: mls.irft2(x, sl)
    rft = lambda x: mls.rft2(x, sl)

    t0, y0 = hwak_state(fl, sl, kx, ky, ksqr, target_time)

    lk = np.zeros(kx.shape + (2, 2), dtype=complex)
    inv_ksqr = np.zeros_like(ksqr)
    inv_ksqr[ksqr != 0] = 1.0 / ksqr[ksqr != 0]
    lk[:, 0, 0] = -c * sigk * inv_ksqr - nu * sigk * ksqr
    lk[:, 0, 1] = c * sigk * inv_ksqr
    lk[:, 1, 0] = c * sigk - 1j * kap * ky
    lk[:, 1, 1] = -c * sigk - d * ksqr * sigk
    lk[ksqr == 0] = 0.0

    def rhsnl(t, zk):
        dzkdt = np.zeros_like(zk)
        phik, nk = zk[0, :], zk[1, :]
        dxphi = irft(1j * kx * phik)
        dyphi = irft(1j * ky * phik)
        om = irft(-ksqr * phik)
        n = irft(nk)
        dzkdt[0, :] = (-1j * kx * rft(dyphi * om) + 1j * ky * rft(dxphi * om)) * inv_ksqr
        dzkdt[1, :] = 1j * kx * rft(dyphi * n) - 1j * ky * rft(dxphi * n)
        dzkdt[:, ksqr == 0] = 0.0
        return dzkdt

    solver = gsol(rhsnl, t0, y0, t0 + dt, lk, dt, callbacks=NullCallbacks(), tol=tol, M=m, maxstep=dt)
    solver.run()
    if hasattr(solver, "Tk"):
        y1 = dot(solver.Tk, solver.y.reshape(solver.yshp)).ravel()
    else:
        y1 = solver.y
    return t0, solver.t, y0, y1, kx, ky, ksqr


def hm_step(fl, target_time, dt, tol, m):
    kap = float(fl["data/kap"][()])
    nu = float(fl["data/nu"][()])
    d = float(fl["data/D"][()])
    sl, kx, ky, ksqr = grid_from_file(fl)
    nonzero = ksqr != 0
    sigk = ky > 0
    inv_ksqr = np.zeros_like(ksqr)
    inv_ksqr[nonzero] = 1.0 / ksqr[nonzero]

    irft = lambda x: mls.irft2(x, sl)
    rft = lambda x: mls.rft2(x, sl)

    t0, phik0 = hm_state(fl, sl, ksqr, target_time)
    response_k = hm_response_from_file(fl, kx, ky, ksqr)
    polarization_k = ksqr + response_k
    lk = np.zeros_like(phik0)
    lk[nonzero] = (
        -1j * kap * ky[nonzero]
        - nu * ksqr[nonzero] ** 2
        - d * ksqr[nonzero] * response_k[nonzero]
    ) / polarization_k[nonzero]
    if "nu_l" in fl["data"]:
        lk[nonzero] -= float(fl["data/nu_l"][()]) * inv_ksqr[nonzero] ** 3

    def rhsnl(t, phik):
        om = irft(-ksqr * phik)
        dxphi = irft(1j * kx * phik)
        dyphi = irft(1j * ky * phik)
        dphikdt = (-1j * kx * rft(dyphi * om) + 1j * ky * rft(dxphi * om)) * inv_ksqr
        dphikdt[~nonzero] = 0.0
        return dphikdt

    solver = etdrk4cp(rhsnl, lk, phik0.copy(), dt, M=m, maxstep=dt)
    phik1, err = solver.step(t0, phik0)
    return t0, t0 + solver.h, phik0, phik1, response_k * phik0, response_k * phik1, err, kx, ky, ksqr


def main():
    parser = argparse.ArgumentParser(
        description="Advance one saved state by one timestep in memory and save the tendency."
    )
    parser.add_argument("--model", choices=["hwak", "hm", "hmr"], required=True)
    parser.add_argument("--input")
    parser.add_argument("--time", type=float)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--output", default="one_timestep_tendency.npz")
    args = parser.parse_args()

    infile = args.input or {
        "hwak": "out_hwak.h5",
        "hm": "out_hm_idelta.h5",
        "hmr": "out_hmr.h5",
    }[args.model]
    with h5.File(infile, "r") as fl:
        target_time = float(fl["fields/t"][-1]) if args.time is None else args.time
        if args.model == "hwak":
            t0, t1, y0, y1, kx, ky, ksqr = hwak_step(fl, target_time, args.dt, args.tol, args.M)
            np.savez(
                args.output,
                model=args.model,
                t0=t0,
                t1=t1,
                state0=y0,
                state1=y1,
                tendency=y1 - y0,
                kx=kx,
                ky=ky,
                ksqr=ksqr,
            )
        else:
            t0, t1, y0, y1, n0, n1, err, kx, ky, ksqr = hm_step(fl, target_time, args.dt, args.tol, args.M)
            np.savez(
                args.output,
                model=args.model,
                t0=t0,
                t1=t1,
                state0=y0,
                state1=y1,
                nstate0=n0,
                nstate1=n1,
                tendency=y1 - y0,
                kx=kx,
                ky=ky,
                ksqr=ksqr,
                err=err,
            )

    print(args.output)


if __name__ == "__main__":
    main()
