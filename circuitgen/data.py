import os
import glob
import spicepy.netlist as ntl
from spicepy.netsolve import net_solve
import numpy as np
import pickle
import circuitgen
def get_regression_data():

    f_features = open("data/input/regression_data/charizard_features.pkl", "rb")
    f_values = open("data/input/regression_data/charizard_values.pkl", "rb")
    features = pickle.load(f_features)
    values = pickle.load(f_values)
    return features, values

def read_netlist(dirpath):
    comp_to_id = {"R":1, "L": 2, "C": 3 , "V": 4}
    input = []
    output = []
    i = 0
    for file in glob.glob(os.path.join(dirpath, "*")):
        print("Parsing file {}: ".format(i) + file)
        i += 1
        current_file = open(file, "r")
        for line in current_file.readline():
            if line[0] != "*" and line[0] != ".":
                values = line.split(" ")
                values[0] = comp_to_id.get(values[0][0])
                input.append(values[:-1])
                output.append(values[-1])
    return input, output

def get_gnn_data():
    f_features = open("data/input/bulbasaur_data/bulbasaur_features.pkl", "rb")
    f_topology = open("data/input/bulbasaur_data/bulbasaur_topology.pkl", "rb")
    features = pickle.load(f_features)
    topology = pickle.load(f_topology)
    input_graphs = circuitgen.gnn.create_training_graphs(features,topology)
    target_graphs = circuitgen.gnn.convert_to_graph_data(features,topology)
    return input_graphs,target_graphs

