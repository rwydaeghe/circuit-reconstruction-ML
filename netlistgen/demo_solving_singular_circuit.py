# import SpicePy modules
from spicepy import netlist as ntl
from spicepy.netsolve import net_solve
import matplotlib.pyplot as plt
plt.ion()

plt.close('all')

net = ntl.Network('Parallel_sources_test/C_parallel_voltage_source.net')
net_solve(net)
net.plot()

# Singular RLC circuit due to three isolating capacitors failing to create a DC path to ground
# net = ntl.Network('Parallel_capacitors_test/parallel_capacitors.net')
# net_solve(net)
# net.plot()

# net = ntl.Network('RLC.net')
# net_solve(net)
# net.plot()

"""
https://help.simetrix.co.uk/8.0/simetrix/mergedProjects/user_manual/topics/um_gettingstarted_circuitrules.htm
https://www.multisim.com/help/simulation/singular-matrix-errors/
https://help.simetrix.co.uk/8.0/simetrix/mergedProjects/simulator_reference/topics/simref_convergence_accuracyandperformance_singularmatrixerrors.htm
"""


