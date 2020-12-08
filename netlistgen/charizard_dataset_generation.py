# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:16:14 2020

@author: rbnwy

The charizard dataset is the largest dataset we have as of 8/12.
It consists of multiple known circuits in the data/charizard_data/charizard_circuits directory
These networks are created with varied_topology
For each of these circuits, they are copied 100 times and the values are randomized for each one
those 100 circuits are then simulated to yield transient data, which are sent to the Matlab charizard_prony script
Matlab returns a list of features for each of those 100 circuits, which are related in a simple and unique way to the values
The python script also adds the expected values to be returned by ML in the dataset.
This is repeated for all circuits in /charizard_circuits

->Manually make the target directory data/charizard_data/charizard_circuits
"""

#import circuitgen
from generate_circuits import Data_set, Read_and_write
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import pickle
plt.close('all')

general_seed='charizard'
generate_transients_and_values=True
process_features_mat_file=True
if generate_transients_and_values:
    values={}    
    charizard_dataset=Data_set('RLC.net', 
                                target_dir='data/charizard_data/charizard_circuits',
                                seed=general_seed)
    charizard_dataset.delete_files()
    charizard_dataset.varied_topology_data_set(data_set_size=5,
                                               network_size=8,
                                               allow_C=True, 
                                               write_to_pkl=False)
    transient_data={}
    for i,rw_obj in enumerate(charizard_dataset.read_and_write_objects):
        circuit_file_name='RLC_'+str(i+1)+'.net'
        current_ds=Data_set(circuit_file_name, 
                            path_to_basic_file='data/charizard_data/charizard_circuits/',
                            target_dir='data/charizard_data', 
                            seed=general_seed+str(i))
        current_ds.delete_files()
        current_ds.varied_values_data_set(data_set_size=100,
                                          #compute_transients=False,
                                          write_to_pkl=False)
        last_added_node=max(charizard_dataset.read_and_write_objects[i].nodes)
        measured_node=last_added_node
        #transient_data[circuit_file_name[:-len('.net')]]=current_ds.get_all_transients('v','(1,'+str(measured_node)+')')
        
        # add values already to the values dictionary (for supervised learning)
        list_of_dict=[]
        for j,rw_obj_j in enumerate(current_ds.read_and_write_objects):
            list_of_dict.append(rw_obj_j.read_network(return_values_dictionary=True))
        tot_var_dict={key: np.zeros(len(current_ds.read_and_write_objects)) for key in list_of_dict[0].keys()}
        for j,d in enumerate(list_of_dict):
            for k,v in d.items():
                tot_var_dict[k][j]=v
        values['circuit_'+str(i+1)]=tot_var_dict
    scipy.io.savemat('charizard_transient_data_set.mat', transient_data)
    with open('data/charizard_data/charizard_values.pkl','wb') as f:
        pickle.dump(values,f)

"""
MATLAB PART
->Now that you have this file, copy it into the feature_extraction directory and run the feature_extraction_charizard.m file
The output of this script is charizard_features.mat, a struct containing fields for each circuit containing arrays of the zeros/poles/gain
Every time there's a NaN value, that means that amount and/or the ''structure' of poles/zeros does not fit into the list of amounts/structure of the previous features, which should not depend on the varied
->Copy the charizard_features.mat file back to the data/charizard_data/ directory
"""

if process_features_mat_file:
    features={}
    #start by converting the matlab struct of struct's file to a python dict of dict 
    D=scipy.io.loadmat('data/charizard_data/charizard_features.mat')
    vals = D['features'][0,0]
    keys = D['features'][0,0].dtype.descr
    features={}
    for i in range(len(keys)):
        circuit_feat={}
        key = keys[i][0]
        (zeros_re, zeros_im, poles_re, poles_im, gains) = np.squeeze(vals[key][0][0])
        circuit_feat['zeros_re']=zeros_re
        circuit_feat['zeros_im']=zeros_im
        circuit_feat['poles_re']=poles_re
        circuit_feat['poles_im']=poles_im
        circuit_feat['gains']=gains        
        features['circuit_'+str(i+1)]=circuit_feat #matlab indexing!
    with open('data/charizard_data/charizard_features.pkl','wb') as f:
        pickle.dump(features,f)