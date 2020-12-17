import tensorflow as tf
import tree

from graph_nets import blocks

from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
import sonnet as snt

def convert_to_graph_data(features, topology):
    comp_to_id = {'R': 1, 'L': 2, 'C': 3,'V': 4}
    graphs_list = []
    for circuit in range(len(features.keys())):
        print(circuit)
        circuit_features = features["circuit_{}".format(circuit+1)]
        circuit_topology = topology["circuit_{}".format(circuit+1)]
        nodes = []
        edges = []
        senders = []
        receivers = []
        for sender, receiver in circuit_topology.items():
            for i in range(len(receiver)):
                senders.append(float(sender))
                receivers.append(float(receiver[i][1]))
                edges.append([float(comp_to_id[receiver[i][0][0]])])
        maximum = 0
        for i in range(len(circuit_topology.keys())):
            if 'node_{}'.format(i) not in circuit_features:
                nodes.append([0.0])
            else:
                poles_re = circuit_features['node_{}'.format(i)]['poles_re']
                zeros_re = circuit_features['node_{}'.format(i)]['zeros_re']
                feat = []
                if len(poles_re)>0:
                    feat.append(poles_re)
                if len(zeros_re)>0:
                    feat.append(zeros_re)
                feat = [item for sublist in feat for item in sublist]
                if len(feat) > maximum:
                    maximum = len(feat)
                nodes.append(feat)

        for i in range(len(nodes)):
            while len(nodes[i]) < 7:
                nodes[i].append(0.0)

        graph = {
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": [0.0,0.0,0.0]
        }
        print(edges)
        graphs_list.append(graph)

    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(graphs_list)
    graphs_tuple = tree.map_structure(lambda x: tf.constant(x) if x is not None else None, graphs_tuple)

    return graphs_tuple


def train_gnn(input_graphs):
    graph_network = modules.GraphNetwork(
        edge_model_fn=lambda: snt.nets.MLP(output_sizes=[32,32]),
        node_model_fn=lambda: snt.nets.MLP(output_sizes=[32,32]),
        global_model_fn=lambda: snt.nets.MLP(output_sizes=[32,32]))
    output_graphs = graph_network(input_graphs)
    print(f"Output edges size: {output_graphs.edges.shape[-1]}")
    print(f"Output nodes size: {output_graphs.nodes.shape[-1]}")
    print(f"Output globals size: {output_graphs.globals.shape[-1]}")
    print_graphs_tuple(output_graphs)


def print_graphs_tuple(graphs_tuple):
  print("Shapes of GraphsTuple's fields:")
  print(graphs_tuple.map(lambda x: x if x is None else x.shape, fields=graphs.ALL_FIELDS))
  print("\nData contained in GraphsTuple's fields:")
  print(f"globals:\n{graphs_tuple.globals}")
  print(f"nodes:\n{graphs_tuple.nodes}")
  print(f"edges:\n{graphs_tuple.edges}")
  print(f"senders:\n{graphs_tuple.senders}")
  print(f"receivers:\n{graphs_tuple.receivers}")
  print(f"n_node:\n{graphs_tuple.n_node}")
  print(f"n_edge:\n{graphs_tuple.n_edge}")



