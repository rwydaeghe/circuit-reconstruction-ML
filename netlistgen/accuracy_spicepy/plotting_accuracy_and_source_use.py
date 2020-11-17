# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:31:23 2020

@author: rbnwy
"""

from spicepy import netlist as ntl
from spicepy.netsolve import net_solve
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
plt.close('all')

# =============================================================================
# Testing the accuracy of spicepy using plots and the use of dirac source.
# Conclusion: needs time shift and the amplitude is -2/dt
# =============================================================================

#net = ntl.Network('RLC10u.net')
net = ntl.Network('RLCu.net')
#net = ntl.Network('RLC100n.net')
net_solve(net)

analytic_I=lambda t: 25.64102564*np.exp(-5000.*t)*(39.*np.cos(31224.98999*t) - 6.244997998*np.sin(31224.98999*t))
analytic_V_C=lambda t: 32025.63076*np.exp(-5000.*t)*np.sin(31224.98999*t)
simulated_I=net.get_current('R1')
simulated_V_C=net.get_voltage('C1')
new_t=net.t[1:]
dt=net.t[2]-net.t[1]
new_t-=dt/2
plt.figure('I')
plt.plot(new_t, simulated_I[1:],label='sim I')
plt.plot(new_t, analytic_I(new_t),label='ana I') #cuts off the last point
plt.legend()
plt.figure('V over C')
plt.plot(new_t, simulated_V_C[1:],label='sim V_C')
plt.plot(new_t, analytic_V_C(new_t),label='ana V_C') #cuts off the last point
plt.legend()

