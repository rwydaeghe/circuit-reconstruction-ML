# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:25:05 2020

@author: rbnwy
"""


from spicepy import netlist as ntl
from spicepy.netsolve import net_solve
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
plt.close('all')

# =============================================================================
# Changing dt to see the impact on fit 
# Conclusion: it's second order accurate space derivatives (see pdf also) 
# =============================================================================

analytic_I=lambda t: 25.64102564*np.exp(-5000.*t)*(39.*np.cos(31224.98999*t) - 6.244997998*np.sin(31224.98999*t))
mse=lambda x,y: np.mean(np.power(x-y,2))
netm = ntl.Network('RLC10u.net')
net_solve(netm)
simulated_Im=netm.get_current('R1')
new_tm=netm.t[1:]
dtm=netm.t[2]-netm.t[1]
new_tm-=dtm/2
mse_m=mse(simulated_Im[1:],analytic_I(new_tm))

netu = ntl.Network('RLCu.net')
net_solve(netu)
simulated_Iu=netu.get_current('R1')
new_tu=netu.t[1:]
dtu=netu.t[2]-netu.t[1]
new_tu-=dtu/2
mse_u=mse(simulated_Iu[1:],analytic_I(new_tu))

netn = ntl.Network('RLC100n.net')
net_solve(netn)
simulated_In=netn.get_current('R1')
new_tn=netn.t[1:]
dtn=netn.t[2]-netn.t[1]
new_tn-=dtn/2
mse_n=mse(simulated_In[1:],analytic_I(new_tn))

plt.loglog(np.array([0.1,1,10]),np.array([mse_n,mse_u,mse_m]))
plt.xlabel('dt')
plt.ylabel('mse')
# it is 2nd order accurate (see pdf for why)