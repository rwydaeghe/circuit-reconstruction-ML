# import SpicePy modules
import netlist as ntl
from netsolve import net_solve
import matplotlib.pyplot as plt
plt.ion()

plt.close('all')

# Non singular RLC circuit
net = ntl.Network('my network.net')
net_solve(net)
net.plot()

# Singular RLC circuit due to three isolating capacitors failing to create a DC path to ground
# net = ntl.Network('singular isolating capacitors.net')
# net_solve(net)
# net.plot()

"""
https://help.simetrix.co.uk/8.0/simetrix/mergedProjects/user_manual/topics/um_gettingstarted_circuitrules.htm
https://www.multisim.com/help/simulation/singular-matrix-errors/
https://help.simetrix.co.uk/8.0/simetrix/mergedProjects/simulator_reference/topics/simref_convergence_accuracyandperformance_singularmatrixerrors.htm
"""