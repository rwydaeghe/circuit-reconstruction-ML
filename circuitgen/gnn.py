import tensorflow as tf

from graph_nets import blocks

from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf
import networkx as nx
import matplotlib.pyplot as plt


def convert_to_graph_data(features, topology):
    comp_to_id = {'R': 1, 'L': 2, 'C': 3,'V': 4}
    graphs = []
    for circuit in range(len(features.keys())):
        circuit_features = features["circuit_{}".format(circuit+1)]
        circuit_topology = topology["circuit_{}".format(circuit+1)]
        nodes = []
        edges = []
        senders = []
        receivers = []
        for sender, receiver in circuit_topology.items():
            for i in range(len(receiver)):
                senders.append(sender)
                receivers.append(receiver[i][1])
                edges.append(comp_to_id[receiver[i][0][0]])
        for i in range(len(circuit_topology.keys())):
            if 'node_{}'.format(i) not in circuit_features:
                nodes.append([0])
            else:
                poles_re = circuit_features['node_{}'.format(i)]['poles_re']
                zeros_re = circuit_features['node_{}'.format(i)]['zeros_re']
                feat = []
                if len(poles_re)>0:
                    feat.append(poles_re)
                if len(zeros_re)>0:
                    feat.append(zeros_re)
                nodes.append(feat)
        graph = {
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers
        }
        graphs.append(graph)
    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(graphs)
    return graphs_tuple






