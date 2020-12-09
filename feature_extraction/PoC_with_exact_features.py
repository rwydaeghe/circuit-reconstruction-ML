# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:57:37 2020

@author: rbnwy
"""

import numpy as np
#from generate_circuits import *
import cmath
import numpy as np
import random
from spicepy import netlist
from spicepy.netsolve import net_solve
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import circuitgen
import scipy
import scipy.io


# def values_to_features(values):
#     R=values[0]
#     L=values[1]
#     C=values[2]
#     K=C/(L*C+R*C+1)
#     p1=-(C*R-cmath.sqrt(C**2*R**2-4*C*L))/(2*C*L)
#     p_re=np.real(p1)
#     p_im=np.imag(p1)
#     return np.array([K,p_re,p_im])
#
# data_set_1=Generate_data('RLC.net', target_dir='../data/varied_values_data/', seed=1)
# data_set_1.read_dir()
# data_point=data_set_1.read_and_write_objects[0]
# data_point.compute_transients()
# #plt.plot(data_point.net.t,data_point.net.get_voltage('R1'))
#
# values=np.zeros((3,len(data_set_1.read_and_write_objects)))
# features=np.zeros((3,len(data_set_1.read_and_write_objects)))
# times=np.zeros((data_point.net.t.size,len(data_set_1.read_and_write_objects)))
# signals=np.zeros((data_point.net.t.size,len(data_set_1.read_and_write_objects)))
# from tqdm import tqdm
# for i,rw_obj in tqdm(enumerate(data_set_1.read_and_write_objects)):
#     rw_obj.compute_transients()
#     if not rw_obj.is_singular:
#         values[:,i]=rw_obj.net.values[1:]
#         features[:,i]=values_to_features(values[:,i])
#         times[:,i]=rw_obj.net.t
#         signals[:,i]=rw_obj.net.get_current('R1')
#


class Training_data_set(object):
    def __init__(self, values, features, times, signal):
        self.values = values
        self.features = features
        self.times = times
        self.signals = signal
# training_data_set=Training_data_set(values,features,times,signals)
# import pickle as pkl
# with open('training_data_set.pkl','wb') as f:
#     pickle.dump(training_data_set,f)

mat = scipy.io.loadmat('features.mat')
f = open("training_data_set.pkl", "rb")
features = mat.get("features")
values = scipy.io.loadmat("varied_values_data_set.mat").get("data")
data = pickle.load(f)
# f.close()
circuitgen.train.train_features_to_value(features,data)