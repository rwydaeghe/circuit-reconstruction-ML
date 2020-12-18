# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:59:36 2020

@author: rbnwy
"""

from generate_circuits import Data_set, Read_and_write
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# CREATE

#you still have to create the target directory

"""
data_set_1=Data_set('RLC.net', target_dir='data/varied_values_data/', seed=1)
data_set_1.delete_files() #be careful
data_set_1.varied_values_data_set(size=100)
"""

#"""
data_set_2=Data_set('RLC.net', target_dir='data/varied_topology_data/', seed=2)
#data_set_2.delete_files() #be careful
#data_set_2.varied_topology_data_set(size=10, allow_C=False)
#"""

"""
data_set_3=Data_set('RLC.net', target_dir='data/paper_figure_dataset/', seed=3)
data_set_3.delete_files()
data_set_3.varied_topology_data_set(data_set_size=10,
                                    network_size=8,
                                    allow_C=True, 
                                    write_to_pkl=False,
                                    verbose=False)
"""

# READ

#"""
#data_set_2=Data_set('RLC.net', target_dir='data/varied_topology_data/', seed=2)
#data_set_1=Data_set('RLC.net', target_dir='data/varied_values_data/', seed=1)
data_set_2.read_dir()
data_set_2.plot_paper_ready([0,1])
#"""

#data_set_1.plot([1,2,3])
#data, t = data_set_1.get_all_transients('i','R1')
#import scipy.io
#scipy.io.savemat('varied_values_data_set.mat', {"data": data,"t": t})


