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
zpk=signal.tf2zpk(num,den)
zpk2=(np.array([]),zpk[1],zpk[2])
print(zpk2)
tt, y = signal.impulse(zpk2)
plt.plot(tt, y*fs)