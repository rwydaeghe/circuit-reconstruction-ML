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

#"""
data_set_1=Data_set('RLC.net', target_dir='data/varied_values_data/', seed=1)
#data_set_1.delete_files() #be careful
data_set_1.varied_values_data_set(size=100)
#"""

"""
data_set_2=Data_set('RLC.net', target_dir='data/varied_topology_data/', seed=2)
data_set_2.delete_files() #be careful
data_set_2.varied_topology_data_set(size=10, allow_C=False)
"""

# READ

#"""
#data_set_2=Data_set('RLC.net', target_dir='data/varied_topology_data/', seed=2)
#data_set_1=Data_set('RLC.net', target_dir='data/varied_values_data/', seed=1)
#data_set_2.read_dir()
#data_set_2.plot('all',content_list='all_I')
#"""

data_set_1.plot([1,2,3])
data, t = data_set_1.get_all_transients('i','R1')
import scipy.io
scipy.io.savemat('varied_values_data_set.mat', {"data": data,"t": t})


