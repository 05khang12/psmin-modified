import numpy as np
#import cupy as xp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as xp
import mlsarray.mlsarray as mls
import os
import h5py as h5
from time import time
#from mlsarray.mlsarray import init_kspace_grid,slicelist,irft,rft
from etdrk4cp.gsol import gsol,callbacks
from etdrk4cp.etdrk4cp import etdrk4cp
from etdrk4cp.h5tools import save_data
from gamma_max import GammaMax

# Physics Paramteres
C=10.0
kap=1.0
nu=0.15e-3
D=0.15e-3
C=float(os.environ.get("PSMIN_C", C))
kap=float(os.environ.get("PSMIN_KAP", kap))
nu=float(os.environ.get("PSMIN_NU", nu))
D=float(os.environ.get("PSMIN_D", D))

# Simulation Parameters
flname="out_hwak.h5"
wecontinue=False
Npx,Npy=128,128
gamma_t1=30.0
flname=os.environ.get("PSMIN_FLNAME", flname)
wecontinue=os.environ.get("PSMIN_WECONTINUE", "0") == "1"
Npx=int(os.environ.get("PSMIN_NPX", Npx))
Npy=int(os.environ.get("PSMIN_NPY", Npy))
gamma_t1=float(os.environ.get("PSMIN_GAMMA_T1", gamma_t1))
restart_file=os.environ.get("PSMIN_RESTART_FILE")
t0=0
t1=float(int(gamma_t1/GammaMax(kap,C,D).value))
t1=float(os.environ.get("PSMIN_T1", t1))
tol=1e-8
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))
Lx,Ly=12*np.pi,12*np.pi
dkx,dky=2*np.pi/Lx,2*np.pi/Ly

#setting up the grid
sl=mls.slicelist(Nx,Ny)
lkx,lky=mls.init_kspace_grid(sl)
Nk=lkx.size
kx,ky=lkx*dkx,lky*dky
ksqr=kx**2+ky**2
inv_ksqr=np.divide(1.0,ksqr,out=np.zeros_like(ksqr),where=ksqr!=0)
sigk=(ky>0)

# some shortcuts
fft_count = {"fft": 0, "ifft": 0}

def rft(x):
    fft_count["fft"] += 1
    return mls.rft2(x,sl)

def irft(x):
    fft_count["ifft"] += 1
    return mls.irft2(x,sl)

def fft_count_total():
    nfft = fft_count["fft"]
    nifft = fft_count["ifft"]
    return nfft, nifft, nfft + nifft

get = mls.get

#Initial conditions
seed=os.environ.get("PSMIN_SEED")
if seed is not None:
    xp.random.seed(int(seed))
Ak,wk=1e-4,2.0
zk0=xp.zeros((2,kx.size),dtype=complex)
zk0[0,:]=Ak*xp.exp(-lkx**2/2/wk**2-lky**2/wk**2)*xp.exp(1j*2*xp.pi*xp.random.rand(lkx.size).reshape(lkx.shape));
zk0[1,:]=Ak*xp.exp(-lkx**2/wk**2-lky**2/wk**2)*xp.exp(1j*2*xp.pi*xp.random.rand(lkx.size).reshape(lkx.shape));

#Linear Matrix
Lk=np.zeros(kx.shape+(2,2),dtype=complex)
Lk[:,0,0]=get(-C*sigk/ksqr-nu*sigk*ksqr)
Lk[:,0,1]=get(C*sigk/ksqr)
Lk[:,1,0]=get(C*sigk-1j*kap*ky)
Lk[:,1,1]=get(-C*sigk-D*ksqr*sigk)

#The nonlinear terms
def rhsnl(t,zk):
    dzkdt=xp.zeros_like(zk)
    phik,nk=zk[0,:],zk[1,:]
    dphikdt,dnkdt=dzkdt[0,:],dzkdt[1,:]
    dxphi=irft(1j*kx*phik)
    dyphi=irft(1j*ky*phik)
    om=irft(-ksqr*phik)
    n=irft(nk)
    dphikdt[:]=(-1j*kx*rft(dyphi*om)+1j*ky*rft(dxphi*om))/ksqr
    dnkdt[:]=1j*kx*rft(dyphi*n)-1j*ky*rft(dxphi*n)
    return dzkdt


#Save Stuff
def save_callback(fl,t,zk,flag):
    phink=zk.reshape((2,kx.size))
    phik,nk=phik,nk=phink[0,:],phink[1,:]
    save_data(fl,'last',ext_flag=False,zk=get(zk),t=get(t))
    if flag=='fields':
        print('saving fields')
        om=irft(-phik*(kx**2+ky**2))
        n=irft(nk)
        save_data(fl,'fields',ext_flag=True,om=get(om),n=get(n),t=get(t))
    if flag=='energies':
        print('saving energies')
        density_energy=xp.abs(nk)**2
        mode_energy=xp.abs(phik)**2*ksqr+density_energy
        Etot=xp.sum(mode_energy)
        Ez=xp.sum(mode_energy*(ky==0))
        Ftot=xp.sum(density_energy)
        Fz=xp.sum(density_energy*(ky==0))
        save_data(fl,'energies',ext_flag=True,Etot=get(Etot),Ez=get(Ez),Ftot=get(Ftot),Fz=get(Fz),t=get(t))

# initialize the hdf5 file
if restart_file is not None:
    with h5.File(restart_file, 'r', libver='latest') as src:
        omk=rft(xp.array(src['fields/om'][-1,]))
        nk=rft(xp.array(src['fields/n'][-1,]))
        phik=-omk*inv_ksqr
        t0=src['fields/t'][-1]
    t1=float(t0)+float(int(float(os.environ.get("PSMIN_CONTINUE_GAMMA_T1", gamma_t1))/GammaMax(kap,C,D).value))
    if os.path.exists(flname):
        os.remove(flname)
    fl=h5.File(flname,'w',libver='latest')
    fl.swmr_mode = True
    save_data(fl,'data',ext_flag=False,kap=kap,C=C,nu=nu,D=D,Lx=Lx,Ly=Ly,Nx=Nx,Ny=Ny)
    zk0=xp.hstack((phik,nk))
elif(wecontinue):
    fl=h5.File(flname,'r+',libver='latest')
    fl.swmr_mode = True
    omk,nk=rft(xp.array(fl['fields/om'][-1,])),rft(xp.array(fl['fields/n'][-1,]))
    phik=-omk*inv_ksqr
    t0=fl['fields/t'][-1]
    continue_gamma_t1=os.environ.get("PSMIN_CONTINUE_GAMMA_T1")
    if continue_gamma_t1 is not None:
        t1=float(t0)+float(int(float(continue_gamma_t1)/GammaMax(kap,C,D).value))
    zk0=xp.hstack((phik,nk))
else:
    if os.path.exists(flname):
        os.remove(flname)
    fl=h5.File(flname,'w',libver='latest')
    fl.swmr_mode = True
    save_data(fl,'data',ext_flag=False,kap=kap,C=C,nu=nu,D=D,Lx=Lx,Ly=Ly,Nx=Nx,Ny=Ny)

# define callbacks
ct=time()
fcbs = [(lambda t,y : print('t=',t,', ',time()-ct,' secs elapsed')),
        (lambda t,y : save_callback(fl,t,y,flag='fields')),
         (lambda t,y : save_callback(fl,t,y,flag='energies'))]
dtstep=1.0
dtcbs=[1.0,1.0,10.0]
cbs=callbacks(dtcbs,fcbs)

# initiate and run the solver
r=gsol(rhsnl,t0,zk0.ravel(),t1,Lk,dtstep,callbacks=cbs,tol=tol,M=64,maxstep=dtstep)
r.run()

print("fft_count, ifft_count, total_fft_count =", fft_count_total())
fl.close()
