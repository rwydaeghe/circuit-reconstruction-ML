# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 19:46:04 2020

@author: rbnwy

The bulbasaur dataset consists of 100 different networks, each with a different topology.
Additionally, all the components have the same values, namely:
R=50 Ohm
L=1e-3 Henry
C=1e-6 Farad
This way, it's ideal data to study the GNN algorithm without needing to do regression for each of the topologies.
For a GNN algorithm to work, all features of all nodes need to be known.
Therefore, for each circuit, all possible nodes to ground are taken into account and their features get processed

->Manually make the target directory data/bulbasaur_data
"""

#import circuitgen
from generate_circuits import Data_set, Read_and_write
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import pickle
plt.close('all')

general_seed='bulbasaur'
generate_transients_and_values=False
process_features_mat_file=True
analyse_data=True
if analyse_data:
    if process_features_mat_file==False:
        print('You need to process the data to analyse it')
    print("A '-' means that the signal is zero everywhere")

## To get an idea of the values and timescales
# from spicepy import netlist as ntl
# from spicepy.netsolve import net_solve
# net = ntl.Network('RLC.net')
# net_solve(net)
# net.plot()

if generate_transients_and_values:
    values={}    
    bulbasaur_dataset=Data_set('RLC.net', 
                                target_dir='data/bulbasaur_data',
                                seed=general_seed)
    bulbasaur_dataset.delete_files()
    bulbasaur_dataset.varied_topology_data_set(data_set_size=100,
                                                network_size=8,
                                                allow_C=True,
                                                one_value_component_space=True,
                                                write_to_pkl=False,
                                                verbose=False)
    def topology_ifu_nodes(topology_ifu_component):
        d=topology_ifu_component
        new_d={}
        for comp, nodes in d.items():
            if str(nodes[0]) not in new_d.keys():
                new_d[str(nodes[0])]=[]
            new_d[str(nodes[0])].append((comp,nodes[1]))
            if str(nodes[1]) not in new_d.keys():
                new_d[str(nodes[1])]=[]
            new_d[str(nodes[1])].append((comp,nodes[0]))
        return new_d
    transient_data={}    
    topology_data={}
    for i,rw_obj in enumerate(bulbasaur_dataset.read_and_write_objects):
        circuit_file_name='RLC_'+str(i+1)+'.net'
        node_tran_dict={}
        t=rw_obj.net.t[1:]
        dt=t[1]-t[0]
        t-=dt/2
        node_tran_dict['t']=t
        for node in np.delete(rw_obj.nodes,[0,1]): # nodes 0 & 1 do not qualify to take features (zpk) from as they have tf-function identic 0 and 1 for all s.
            node_tran_dict['node_'+str(node)]=rw_obj.net.get_voltage('(0,'+str(node)+')')[1:]
        transient_data[circuit_file_name[:-len('.net')]]=node_tran_dict

        topology_data['circuit_'+str(i+1)]=topology_ifu_nodes(rw_obj.read_network(return_topology_dictionary=True))
    scipy.io.savemat('bulbasaur_transient_data_set.mat', transient_data)
    with open('data/bulbasaur_data/bulbasaur_topology.pkl','wb') as f:
        pickle.dump(topology_data,f)

"""
MATLAB PART
->Now that you have this file, copy it into the feature_extraction directory and run the feature_extraction_bulbasaur.m file
The output of this script is bulbasaur_features.mat, a struct containing fields for each circuit containing arrays of the zeros/poles/gain
Every time there's a NaN value, that means that amount and/or the 'structure' of poles/zeros does not fit into the list of amounts/structure of the previous features, which should not depend on the varied
->Copy the bulbasaur_features.mat file back to the data/bulbasaur_data/ directory
"""

if process_features_mat_file:
    features={}
    #start by converting the matlab struct of struct's file to a python dict of dict 
    D=scipy.io.loadmat('data/bulbasaur_data/bulbasaur_features.mat')
    circuit_vals = D['datastruct'][0,0]
    circuit_keys = D['datastruct'][0,0].dtype.descr
    for circuit_i in range(len(circuit_keys)):
        circuit_nodes={}
        circuit_key = circuit_keys[circuit_i][0]        
        node_vals = circuit_vals[circuit_key][0,0]
        node_keys = circuit_vals[circuit_key][0,0].dtype.descr
        for node_i in range(len(node_keys)):
            node_feat={}
            node_key = node_keys[node_i][0]
            zeros,poles,gain,err_plot,nq_plot,np_plot=np.squeeze(node_vals[node_key][0][0])                         
            if analyse_data:
                if err_plot.size==0:
                    err_plot='-'
                else:
                    err_plot=np.log10(err_plot[0,-1])
                if nq_plot.size==0:
                    nq_plot='-'
                else:
                    nq_plot=nq_plot[0,-1]
                if np_plot.size==0:
                    np_plot='-'
                else:
                    np_plot=np_plot[0,-1]
                print(circuit_i+1, nq_plot, np_plot, poles.size, err_plot, max(poles))
        if nq_plot=='-' and np_plot=='-':
            node_feat['zeros']='no signal'
            node_feat['poles']='no signal'
            node_feat['error_evolution']='no signal'
            node_feat['nq_evolution']='no signal'
            node_feat['np_evolution']='no signal'
        else:
            node_feat['zeros']=zeros
            node_feat['poles']=poles
            node_feat['error_evolution']=err_plot
            node_feat['nq_evolution']=nq_plot
            node_feat['np_evolution']=np_plot
            circuit_nodes['node_'+str(node_i+2)]=node_feat     
        features['circuit_'+str(circuit_i+1)]=circuit_nodes #matlab indexing!
    with open('data/bulbasaur_data/bulbasaur_features.pkl','wb') as f:
        pickle.dump(features,f)


