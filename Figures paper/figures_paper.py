# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:49:13 2020

@author: rbnwy
"""


import numpy as np
import matplotlib.pyplot as plt
from spicepy import netlist as ntl
from spicepy.netsolve import net_solve
import matplotlib.pyplot as plt
plt.close('all')

net = ntl.Network('RLC_figure.net')
net_solve(net)
#net.plot()

def plot(net):
    plt.xlabel('Time [$s$]', fontsize=16)
    plt.ylabel('Voltage [$V$]', fontsize=16)
    plt.xlim([0,1e-3])
    plt.ylim([-3e4,3e4])
    #plt.grid()
    plt.plot(net.t, net.get_voltage('R1'),label='v(R1)')
    plt.plot(net.t, net.get_voltage('L1'),label='v(L1)')
    plt.plot(net.t, net.get_voltage('C1'),label='v(C1)')    
    plt.ticklabel_format(style='sci', scilimits=(-2,2))
    plt.legend(fontsize=16)
    plt.tight_layout()
    
plot(net)