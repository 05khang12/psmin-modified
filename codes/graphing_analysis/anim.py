#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 23:33:00 2018

@author: ogurcan
"""
import sys
import os
import shutil
import numpy as np
import h5py as h5
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pylab as plt
from mpi4py import MPI
plt.rcParams['text.usetex'] = True
comm=MPI.COMM_WORLD
infl="out.h5"
outfl="out.mp4"

vminom=-10
vmaxom=10
vminn=-10
vmaxn=10

fl=h5.File(infl,"r",libver='latest',swmr=True)
n,om=fl['fields/n'],fl['fields/om']
C,kap,nu,D,Lx,Ly=[fl['data'][l][()] for l in ['C','kap','nu','D','Lx','Ly']]

Nx=n.shape[1]
Ny=n.shape[2]
w, h = 9.6,5.4
fig,ax=plt.subplots(1,2,sharey=True,figsize=(w,h))
qd=[]
qd.append(ax[0].imshow(om[0,].T,cmap='seismic',rasterized=True,vmin=vminom,vmax=vmaxom,origin='lower'))
qd.append(ax[1].imshow(n[0,].T,cmap='seismic',rasterized=True,vmin=vminn,vmax=vmaxn,origin='lower'))
ax[0].set_title('$\\Omega$',pad=-1)
ax[1].set_title('$n$',pad=-1)
ax[0].tick_params('y', labelleft=False)
ax[0].tick_params('x', labelbottom=False)
ax[1].tick_params('y', labelleft=False)
ax[1].tick_params('x', labelbottom=False)
plt.tight_layout()
t=fl['fields/t'][()]
plt.subplots_adjust(wspace=0.01, hspace=0.01)


nt0=50
Nt=t.shape[0]
for l in range(1):
    fig.colorbar(qd[l],ax=ax,format="%.2g", aspect=50,shrink=0.8,pad=0.02,fraction=0.03,location='bottom')

ax[0].axis('off')
ax[1].axis('off')

tx=fig.text(0.85, 0.1, "t=0")
if (comm.rank==0):
    lt=np.arange(Nt)
#    lt=np.hstack((np.arange(0,800,10),np.arange(800,Nt)))
    lt_loc=np.array_split(lt,comm.size)
    if not os.path.exists('_tmpimg_folder'):
        os.makedirs('_tmpimg_folder')
else:
    lt_loc=None
lt_loc=comm.scatter(lt_loc,root=0)

for j in lt_loc:
    print(j)
    qd[0].set_data(om[j,].T)
    qd[1].set_data(n[j,].T)
    tx.set_text('t='+str(int(t[j])*1.0))
    fig.savefig("_tmpimg_folder/tmpout%04i"%(j+nt0)+".png",dpi=200)#,bbox_inches='tight')
comm.Barrier()

if comm.rank==0:
    qd[0].set_data(om[0,].T)
    qd[1].set_data(n[0,].T)
    tx.set_text('')

    fig.text(0.5, 0.85, f"{Nx}x{Ny} Simulation",fontsize=24,ha='center')
    fig.text(0.8, 0.80, "Ö. D.Gürcan",fontsize=12,ha='right')
    fig.text(0.1, 0.75, f"Box size : $[L_x,L_y]=[{Lx/np.pi}\\pi,{Ly/np.pi}\\pi]$",fontsize=16,ha='left')
    fig.text(0.1, 0.65, f"Viscosity : $\\nu = {nu:.2e}$",fontsize=16,ha='left')
    fig.text(0.1, 0.55, f"$C = {C}$, $\\kappa={kap}$",fontsize=16,ha='left')

    fig.savefig("_tmpimg_folder/tmpout%04i"%(0)+".png",dpi=200)#,bbox_inches='tight')
    for j in range(1,nt0):
        os.system("cp _tmpimg_folder/tmpout%04i"%(0)+".png _tmpimg_folder/tmpout%04i"%(j)+".png")
    
    os.system("ffmpeg -framerate 25 -y -pattern_type glob -i \"_tmpimg_folder/tmpout*.png\" -c:v libx264 -pix_fmt yuv420p -vf fps=25 "+outfl)
    shutil.rmtree("_tmpimg_folder")
