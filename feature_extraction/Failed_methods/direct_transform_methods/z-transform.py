# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:29:15 2020

@author: rbnwy
"""

import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
t=np.linspace(0,30,101)
f=np.exp(-t)*np.sin(t*10)
#plt.plot(t,f)

ROC_start=0
#ROC_start=.7
Z_r=np.linspace(ROC_start,1,101)
Z_theta=np.linspace(0,2*np.pi,101)
F=np.zeros((len(Z_r),len(Z_theta)),dtype=complex)
for i,z_r in tqdm(enumerate(Z_r)):
    for j,z_theta in enumerate(Z_theta):
        z=z_r*np.exp(1j*z_theta)
        ans=0
        for n,fn in enumerate(f):
            ans+=fn*np.power(z,-n)
        F[i,j]=np.log(np.abs(ans))
ZZ_R,ZZ_theta=np.meshgrid(Z_r,Z_theta,indexing='ij')
X, Y = ZZ_R*np.cos(ZZ_theta), ZZ_R*np.sin(ZZ_theta)
def plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, np.abs(F))
    
    ax.set_xlabel(r'$\phi_\mathrm{real}$')
    ax.set_ylabel(r'$\phi_\mathrm{im}$')
    ax.set_zlabel(r'$V(\phi)$')
    
    plt.show()
plot()
