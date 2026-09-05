import os
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py as h5
import numpy as np

import mlsarray.mlsarray as mls
from etdrk4cp.gsol import gsol, callbacks
from etdrk4cp.h5tools import save_data
from gamma_max import GammaMax


# Physics parameters
C = 10.0
kap = 1.0
nu = 0.15e-3
D = 0.15e-3
hypo_factor = 1.0
C = float(os.environ.get("PSMIN_C", C))
kap = float(os.environ.get("PSMIN_KAP", kap))
nu = float(os.environ.get("PSMIN_NU", nu))
D = float(os.environ.get("PSMIN_D", D))
hypo_factor = float(os.environ.get("PSMIN_HYPO_FACTOR", hypo_factor))

# Simulation parameters
flname = "out_hm_idelta.h5"
wecontinue = False
Npx, Npy = 128, 128
gamma_t1 = 30.0
flname = os.environ.get("PSMIN_FLNAME", flname)
wecontinue = os.environ.get("PSMIN_WECONTINUE", "0") == "1"
gamma_t1 = float(os.environ.get("PSMIN_GAMMA_T1", gamma_t1))
restart_file = os.environ.get("PSMIN_RESTART_FILE")
t0 = 0.0
t1 = float(int(gamma_t1 / GammaMax(kap, C, D).value))
tol = 1e-8
Nx, Ny = 2 * int(np.floor(Npx / 3)), 2 * int(np.floor(Npy / 3))
Lx, Ly = 12 * np.pi, 12 * np.pi
dkx, dky = 2 * np.pi / Lx, 2 * np.pi / Ly

# Set up the pseudospectral grid using mlsarray.
sl = mls.slicelist(Nx, Ny)
lkx, lky = mls.init_kspace_grid(sl)
kx, ky = lkx * dkx, lky * dky
ksqr = kx**2 + ky**2
nonzero = ksqr != 0
sigk = ky > 0
inv_ksqr = np.zeros_like(ksqr)
inv_ksqr[nonzero] = 1.0 / ksqr[nonzero]
kmin = float(np.sqrt(np.min(ksqr[nonzero])))
kmax = float(np.sqrt(np.max(ksqr[nonzero])))
hypo_power = 6
nu_l = hypo_factor * nu * kmax**2 * kmin**hypo_power

fft_count = {"fft": 0, "ifft": 0}


def rft(x):
    fft_count["fft"] += 1
    return mls.rft2(x, sl)


def irft(x):
    fft_count["ifft"] += 1
    return mls.irft2(x, sl)


def fft_count_total():
    nfft = fft_count["fft"]
    nifft = fft_count["ifft"]
    return nfft, nifft, nfft + nifft


get = mls.get

#   delta_k = (kap / C) ky k^2 / (1 + k^2)
#   R_k = sigk * (1 - i delta_k)
#   n_k = R_k phi_k
delta_k = (kap / C) * ky * ksqr / (1.0 + ksqr)
r_k = sigk * (1.0 - 1j * delta_k)


def n_phi(phik):
    return r_k * phik


# Initial condition in phi_k.
seed = os.environ.get("PSMIN_SEED")
if seed is not None:
    np.random.seed(int(seed))
Ak, wk = 1e-4, 2.0
phik0 = np.zeros(kx.size, dtype=complex)
phik0[:] = (
    Ak
    * np.exp(-(lkx**2) / (2 * wk**2) - (lky**2) / (wk**2))
    * np.exp(1j * 2 * np.pi * np.random.rand(kx.size).reshape(kx.shape))
)
phik0[~nonzero] = 0.0


#   d phi_k / dt = L_k phi_k + NL(phi)_k
#   L_k = (-i kap ky - nu k^4 - D k^2 R_k) / (k^2 + R_k) - nu_l / k^6
#   R_k = r_k
#   p_k = k^2 + R_k
# The nu_l coefficient is calibrated so nu_l/kmin^6 = nu*kmax^2.

p_k = ksqr + r_k
Lk = np.zeros_like(phik0)
Lk[nonzero] = (
    -1j * kap * ky[nonzero]
    - nu * ksqr[nonzero] ** 2
    - D * ksqr[nonzero] * r_k[nonzero]
) / p_k[nonzero]
Lk[nonzero] -= nu_l * inv_ksqr[nonzero] ** 3


def rhsnl(t, phik):
    om = irft(-ksqr * phik)
    dxphi = irft(1j * kx * phik)
    dyphi = irft(1j * ky * phik)

    # Same pseudospectral Jacobian convention as hwak_min.py.
    dphikdt = (-1j * kx * rft(dyphi * om) + 1j * ky * rft(dxphi * om)) * inv_ksqr
    dphikdt[~nonzero] = 0.0
    return dphikdt


def save_callback(fl, t, phik, flag):
    nk = n_phi(phik)
    save_data(fl, "last", ext_flag=False, phik=get(phik), t=get(t))

    if flag == "fields":
        print("saving fields")
        phi = irft(phik)
        om = irft(-ksqr * phik)
        n = irft(nk)
        save_data(
            fl,
            "fields",
            ext_flag=True,
            phi=get(phi),
            om=get(om),
            n=get(n),
            t=get(t),
        )

    if flag == "energies":
        print("saving energies")
        density_energy = np.abs(nk) ** 2
        mode_energy = np.abs(phik) ** 2 * ksqr + density_energy
        Etot = np.sum(mode_energy)
        Ez = np.sum(mode_energy * (ky == 0))
        Ftot = np.sum(density_energy)
        Fz = np.sum(density_energy * (ky == 0))
        save_data(
            fl,
            "energies",
            ext_flag=True,
            Etot=get(Etot),
            Ez=get(Ez),
            Ftot=get(Ftot),
            Fz=get(Fz),
            t=get(t),
        )


if restart_file is not None:
    with h5.File(restart_file, "r", libver="latest") as src:
        if "phi" in src["fields"]:
            phik0 = rft(np.array(src["fields/phi"][-1,]))
        else:
            omk = rft(np.array(src["fields/om"][-1,]))
            phik0 = -omk * inv_ksqr
            phik0[~nonzero] = 0.0
        t0 = src["fields/t"][-1]
    t1 = float(t0) + float(int(float(os.environ.get("PSMIN_CONTINUE_GAMMA_T1", gamma_t1)) / GammaMax(kap, C, D).value))
    if os.path.exists(flname):
        os.remove(flname)
    fl = h5.File(flname, "w", libver="latest")
    fl.swmr_mode = True
    save_data(
        fl,
        "data",
        ext_flag=False,
        kap=kap,
        C=C,
        nu=nu,
        D=D,
        nu_l=nu_l,
        hypo_factor=hypo_factor,
        hypo_power=hypo_power,
        hypo_kmin=kmin,
        hypo_kmax=kmax,
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
    )
    save_data(fl, "spectral", ext_flag=False, kx=get(kx), ky=get(ky), delta_k=get(delta_k))
elif wecontinue:
    fl = h5.File(flname, "r+", libver="latest")
    fl.swmr_mode = True
    phik0 = np.array(fl["last/phik"])
    t0 = fl["last/t"][()]
    continue_gamma_t1 = os.environ.get("PSMIN_CONTINUE_GAMMA_T1")
    if continue_gamma_t1 is not None:
        t1 = float(t0) + float(int(float(continue_gamma_t1) / GammaMax(kap, C, D).value))
else:
    if os.path.exists(flname):
        os.remove(flname)
    fl = h5.File(flname, "w", libver="latest")
    fl.swmr_mode = True
    save_data(
        fl,
        "data",
        ext_flag=False,
        kap=kap,
        C=C,
        nu=nu,
        D=D,
        nu_l=nu_l,
        hypo_factor=hypo_factor,
        hypo_power=hypo_power,
        hypo_kmin=kmin,
        hypo_kmax=kmax,
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
    )
    save_data(fl, "spectral", ext_flag=False, kx=get(kx), ky=get(ky), delta_k=get(delta_k))


ct = time()
fcbs = [
    lambda t, y: print("t=", t, ", ", time() - ct, " secs elapsed"),
    lambda t, y: save_callback(fl, t, y, flag="fields"),
    lambda t, y: save_callback(fl, t, y, flag="energies"),
]
dtstep = 1.0
dtcbs = [1.0, 1.0, 10.0]
cbs = callbacks(dtcbs, fcbs)

r = gsol(rhsnl, t0, phik0, t1, Lk, dtstep, callbacks=cbs, tol=tol, M=64, maxstep=dtstep)
r.run()

print("fft_count, ifft_count, total_fft_count =", fft_count_total())
fl.close()
