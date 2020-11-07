# -*- coding: utf-8 -*-
"""
Created on Nov 7 00:06:45

@author: rbnwy
"""

# import SpicePy modules
from spicepy import netlist as ntl
from spicepy.netsolve import net_solve
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

plt.ion()
plt.close('all')
net = ntl.Network('RLC.net')
net_solve(net)
#net.plot()
"""
def get_all_voltages(components):
    v=[]
    for component in components:
        v.append(net.get_voltage(component))        
    return np.array(v)
plt.plot(net.t,np.transpose(get_all_voltages(['R1','C1','L1'])))
"""

time=net.t[net.t<0.005]
signal=net.get_voltage('R1')
signal=signal[net.t<0.005]

#"""
plt.plot(time,signal)
pad_reps=10
freq=np.fft.rfftfreq(time.size*pad_reps,d=time[1]-time[0])
BW=freq<20000
ROC=np.linspace(10000,-10000,1001)
ans=np.zeros((ROC.size, np.fft.rfft(signal,n=signal.size*pad_reps).size))
for i,theta in enumerate(ROC):
    #plt.plot(time,np.multiply(signal,np.exp(-theta*time)))
    spectrum=np.abs(np.fft.rfft(np.multiply(signal,np.exp(-theta*time)),n=signal.size*pad_reps))
    ans[i,:]=spectrum
    #plt.plot(freq,spectrum)
    #plt.show()
    #plt.pause(5)
ffreq, RROC = np.meshgrid(freq[BW],ROC)
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(ffreq,RROC,ans[:,BW])
#"""

print('hi')
global f_interp
f_interp=interp1d(time,signal)
def f(t):
    if t>time[0] and t<time[-1]*0.9:
        return f_interp(t)
    else:
        return 0.0

def laplace(sre,sim):
    s=sre+1j*sim
    N=101
    du=0.5/N
    y=0
    for u in np.linspace(0,1,N):
        y+=2*np.power(u+du,s-1)*f(-np.log(u+du))+np.power(u+2*du,s-1)*f(-np.log(u+2*du))
    return y

"""
PZ_box_size=2500
PZ_box_N=101
s_re=np.linspace(-PZ_box_size,PZ_box_size,PZ_box_N)
s_im=np.linspace(-PZ_box_size,PZ_box_size,PZ_box_N)
S_re, S_im = np.meshgrid(s_re,s_im)
Laplace=np.zeros(S_re.shape,dtype=complex)

#laplace_vect=np.vectorize(laplace)
for i_re, sre in enumerate(s_re):
    for i_im, sim in enumerate(s_im):
        Laplace[i_re,i_im]=laplace(sre,sim)
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_re,S_im,np.abs(Laplace))
"""

        
        
