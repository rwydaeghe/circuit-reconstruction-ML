# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:44:41 2020

@author: rbnwy
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')
Z_r=np.linspace(0,1,51)
Z_theta=np.linspace(0,2*np.pi,51)
ZZ_R,ZZ_theta=np.meshgrid(Z_r,Z_theta,indexing='ij')
X, Y = ZZ_R*np.cos(ZZ_theta), ZZ_R*np.sin(ZZ_theta)
T=1/101
F=np.zeros((len(Z_r),len(Z_theta)),dtype=complex)
for i,z_r in tqdm(enumerate(Z_r)):
    for j,z_theta in enumerate(Z_theta):
        z=z_r*np.exp(1j*z_theta)
        F[i,j]=np.log(z)/T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.real(F))
ax.plot_surface(X, Y, np.imag(F))

# Tweak the limits and add latex math labels.
#ax.set_zlim(0, zmax)
ax.set_xlabel(r'$\phi_\mathrm{real}$')
ax.set_ylabel(r'$\phi_\mathrm{im}$')
ax.set_zlabel(r'$V(\phi)$')

plt.show()