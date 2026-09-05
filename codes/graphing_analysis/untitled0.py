# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:14:16 2026

@author: GMC
"""

import sys
import os
import shutil
import numpy as np
import h5py as h5

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

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
