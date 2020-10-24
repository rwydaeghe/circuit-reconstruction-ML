# import SpicePy modules
from spicepy import netlist as ntl
from spicepy.netsolve import net_solve
import matplotlib.pyplot as plt
plt.ion()

# read netlist
net = ntl.Network('tran_network6.net')

# compute the circuit solution
net_solve(net)

# plot results
net.plot()