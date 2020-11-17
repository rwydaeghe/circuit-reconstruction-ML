# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:20:45 2020

@author: rbnwy
"""

import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.simplefilter("ignore")
# from inspect import getsourcefile
# import os.path as path, sys
# current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
# sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
# from generate_circuits import *
# sys.path.pop(0)

from prony import *

plt.close('all')
start=0
stop=100
fs=101
T=1/fs
t=np.linspace(start,stop,int((stop-start)/T))
x=np.exp(-t)*(np.sin(t))
#x=np.exp(-t)*(np.cos(t)*t+np.sin(t)*(1+2*t))/2
plt.plot(t,x)

(b, a, err) = prony(x, 2, 0)
print('prony',b,a,err)

import harold
from scipy import signal

G = harold.Transfer(b,a,dt=T)
H_tus = harold.undiscretize(G, method='tustin')
num=H_tus.polynomials[0][0]
den=H_tus.polynomials[1][0]
(zero,pole,gain)=signal.tf2zpk(num,den)
print(zero,pole,gain)

from scipy import signal

(z,p,k)=signal.tf2zpk(b,a)
print('zpk',z,p,k)
def bilinear(z,p,k,T):
    eps=np.finfo(complex).eps*100
    for i,z_i in enumerate(z):
        if type(z[i]) is np.float64:
            if np.abs(z_i)<eps:
                z[i]=0.0
        else:
            if np.abs(np.real(z_i))<eps:
                z[i]=0.0+1j*np.imag(z[i])
            if np.abs(np.imag(z_i))<eps:
                z[i]=np.real(z[i])+1j*0.0
        if np.imag(z[i])==0.0:
            z[i]=np.real(z[i])
    for i,p_i in enumerate(p):
        if type(p[i]) is np.float64:
            if np.abs(p_i)<eps:
                p[i]=0.0
        else:
            if np.abs(np.real(p_i))<eps:
                p[i]=0.0+1j*np.imag(p[i])
            if np.abs(np.imag(p_i))<eps:
                p[i]=np.real(p[i])+1j*0.0
        if np.imag(p[i])==0.0:
            p[i]=np.real(p[i])
    print('temper',z,p,k)
    (z,p,k)=(1/T*np.log(z),1/T*np.log(p),k/T)#/T**2/np.sqrt(T)*1.3)
    print('converted',z,p,k)
    z=list(z)
    p=list(p)
    for i,z_i in enumerate(z):
        if not np.isfinite(np.abs(z_i)):
            z.pop(i)
    for i,p_i in enumerate(p):
        if not np.isfinite(np.abs(p_i)):
            z.pop(i)
    z=np.array(z)
    p=np.array(p)
    return z,p,k
system = bilinear(z,p,k,T)
print('final',system)
tt, y = signal.impulse(system)
plt.plot(tt, y)
