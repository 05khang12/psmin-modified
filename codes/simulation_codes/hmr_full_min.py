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
from gamma_max import GammaMaxHWFullGrid


# Physics parameters
C = float(os.environ.get("PSMIN_C", 1.0))
kap = float(os.environ.get("PSMIN_KAP", 1.0))
damping_ratio = float(os.environ.get("PSMIN_DAMPING_RATIO", 0.01))
hypo_factor = float(os.environ.get("PSMIN_HYPO_FACTOR", 1.0))
hypo_method = os.environ.get("PSMIN_HYPO_METHOD", "H6").upper()
if hypo_method not in ("H4", "H6"):
    raise ValueError("PSMIN_HYPO_METHOD must be H4 or H6")

nu_env = os.environ.get("PSMIN_NU")
D_env = os.environ.get("PSMIN_D")

# Simulation parameters
flname = os.environ.get("PSMIN_FLNAME", "out_hmr_full.h5")
wecontinue = os.environ.get("PSMIN_WECONTINUE", "0") == "1"
gamma_t1 = float(os.environ.get("PSMIN_GAMMA_T1", 30.0))
restart_file = os.environ.get("PSMIN_RESTART_FILE")
t0 = 0.0
tol = 1e-8
Npx, Npy = 128, 128
Nx, Ny = 2 * int(np.floor(Npx / 3)), 2 * int(np.floor(Npy / 3))
Lx, Ly = 12 * np.pi, 12 * np.pi
dkx, dky = 2 * np.pi / Lx, 2 * np.pi / Ly

sl = mls.slicelist(Nx, Ny)
lkx, lky = mls.init_kspace_grid(sl)
kx, ky = lkx * dkx, lky * dky
ksqr = kx**2 + ky**2
nonzero = ksqr != 0
sigk = ky > 0
inv_ksqr = np.zeros_like(ksqr)
inv_ksqr[nonzero] = 1.0 / ksqr[nonzero]
gamma_ref = GammaMaxHWFullGrid(ksqr, ky, kap=kap, C=C).value
nu = float(nu_env) if nu_env is not None else damping_ratio * gamma_ref
D = float(D_env) if D_env is not None else damping_ratio * gamma_ref
t1 = float(int(gamma_t1 / gamma_ref))
kmin = float(np.sqrt(np.min(ksqr[nonzero])))
kmax = float(np.sqrt(np.max(ksqr[nonzero])))
hypo_power = int(hypo_method[1:])
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


def hw_full_omega_plus(k2, ky_values):
    omega = np.zeros_like(k2, dtype=complex)
    active = nonzero & sigk
    a_k = np.zeros_like(k2)
    b_k = np.zeros_like(k2)
    g_k = np.zeros_like(k2)
    h_k = np.zeros_like(k2)
    w_k = np.zeros_like(k2, dtype=complex)

    k2a = k2[active]
    kya = ky_values[active]
    a_k[active] = 0.5 * ((D * k2a + C) + (C / k2a + nu * k2a))
    b_k[active] = 0.5 * ((D * k2a + C) - (C / k2a + nu * k2a))
    g_k[active] = b_k[active] ** 2 + C**2 / k2a
    h_k[active] = np.sqrt(g_k[active] ** 2 + C**2 * kap**2 * kya**2 / k2a**2)
    w_k[active] = (
        np.sqrt((h_k[active] - g_k[active]) / 2.0)
        + 1j * np.sqrt((h_k[active] + g_k[active]) / 2.0)
    )
    omega[active] = w_k[active] - 1j * a_k[active]
    return omega, a_k, b_k, g_k, h_k, w_k


omega_k, A_k, B_k, G_k, H_k, W_k = hw_full_omega_plus(ksqr, ky)
r_k = np.zeros_like(omega_k)
r_k[sigk] = (C - 1j * kap * ky[sigk]) / (C - 1j * omega_k[sigk])


def n_phi(phik):
    return r_k * phik


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
#   L_k = (-i kap ky - nu k^4 - D k^2 R_k) / (k^2 + R_k) - nu_l / k^hypo_power
#   R_k uses the full non-asymptotic HW + eigenvalue.
p_k = ksqr + r_k
Lk = np.zeros_like(phik0)
active = nonzero & sigk
Lk[active] = (
    -1j * kap * ky[active]
    - nu * ksqr[active] ** 2
    - D * ksqr[active] * r_k[active]
) / p_k[active]
Lk[nonzero] -= nu_l * inv_ksqr[nonzero] ** (0.5 * hypo_power)


def rhsnl(t, phik):
    om = irft(-ksqr * phik)
    dxphi = irft(1j * kx * phik)
    dyphi = irft(1j * ky * phik)
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
        save_data(fl, "fields", ext_flag=True, phi=get(phi), om=get(om), n=get(n), t=get(t))

    if flag == "energies":
        print("saving energies")
        density_energy = np.abs(nk) ** 2
        mode_energy = np.abs(phik) ** 2 * ksqr + density_energy
        save_data(
            fl,
            "energies",
            ext_flag=True,
            Etot=get(np.sum(mode_energy)),
            Ez=get(np.sum(mode_energy * (ky == 0))),
            Ftot=get(np.sum(density_energy)),
            Fz=get(np.sum(density_energy * (ky == 0))),
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
    t1 = float(t0) + float(int(float(os.environ.get("PSMIN_CONTINUE_GAMMA_T1", gamma_t1)) / gamma_ref))
    if os.path.exists(flname):
        os.remove(flname)
    fl = h5.File(flname, "w", libver="latest")
elif wecontinue:
    fl = h5.File(flname, "r+", libver="latest")
    fl.swmr_mode = True
    phik0 = np.array(fl["last/phik"])
    t0 = fl["last/t"][()]
    continue_gamma_t1 = os.environ.get("PSMIN_CONTINUE_GAMMA_T1")
    if continue_gamma_t1 is not None:
        t1 = float(t0) + float(int(float(continue_gamma_t1) / gamma_ref))
else:
    if os.path.exists(flname):
        os.remove(flname)
    fl = h5.File(flname, "w", libver="latest")

if not wecontinue:
    fl.swmr_mode = True
    save_data(
        fl,
        "data",
        ext_flag=False,
        kap=kap,
        C=C,
        nu=nu,
        D=D,
        damping_ratio=damping_ratio,
        gamma_ref=gamma_ref,
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
    save_data(
        fl,
        "spectral",
        ext_flag=False,
        kx=get(kx),
        ky=get(ky),
        omega_k=get(omega_k),
        A_k=get(A_k),
        B_k=get(B_k),
        G_k=get(G_k),
        H_k=get(H_k),
        W_k=get(W_k),
        r_k=get(r_k),
    )

ct = time()
fcbs = [
    lambda t, y: print("t=", t, ", ", time() - ct, " secs elapsed"),
    lambda t, y: save_callback(fl, t, y, flag="fields"),
    lambda t, y: save_callback(fl, t, y, flag="energies"),
]
dtstep = 1.0
dtcbs = [1.0, 1.0, 10.0]
cbs = callbacks(dtcbs, fcbs)

print(
    "parameters:",
    "C=", C,
    "kap=", kap,
    "gamma_ref=", gamma_ref,
    "nu=", nu,
    "D=", D,
    "hypo_power=", hypo_power,
    "nu_l=", nu_l,
    "t1=", t1,
)
r = gsol(rhsnl, t0, phik0, t1, Lk, dtstep, callbacks=cbs, tol=tol, M=64, maxstep=dtstep)
r.run()

print("fft_count, ifft_count, total_fft_count =", fft_count_total())
fl.close()
