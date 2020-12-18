import tensorflow as tf
import tree

from graph_nets import blocks

from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_tf
from graph_nets import utils_np
import sonnet as snt
from graph_nets.demos_tf2 import models


def create_no_nodefeatures_graphs(features, topology):
    comp_to_id = {'R': 1, 'L': 2, 'C': 3, 'V': 4}
    comp_to_value = {'R': 50, 'L': 1e-3, 'C': 1e-6, 'V': 0}
    graphs_list = []
    for circuit in range(len(features.keys())):
        circuit_features = features["circuit_{}".format(circuit + 1)]
        circuit_topology = topology["circuit_{}".format(circuit + 1)]
        nodes = []
        edges = []
        senders = []
        receivers = []
        for sender, receiver in circuit_topology.items():
            for i in range(len(receiver)):
                senders.append(float(sender))
                receivers.append(float(receiver[i][1]))
                edges.append([float(comp_to_id[receiver[i][0][0]]), float(comp_to_value[receiver[i][0][0]])])
        maximum = 0
        for i in range(len(circuit_topology.keys())):
            nodes.append([0.0])


        for i in range(len(nodes)):
            while len(nodes[i]) < 7:
                nodes[i].append(0.0)

        graph = {
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": [0.0, 0.0, 0.0]
        }
        graphs_list.append(graph)

    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(graphs_list)
    graphs_tuple = tree.map_structure(lambda x: tf.constant(x) if x is not None else None, graphs_tuple)
    return graphs_tuple
def create_no_edgefeatures_graphs(features, topology):
    graphs_list = []
    for circuit in range(len(features.keys())):
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
        edges.append([0.0,0.0])
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
        graphs_list.append(graph)

    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(graphs_list)
    graphs_tuple = tree.map_structure(lambda x: tf.constant(x) if x is not None else None, graphs_tuple)

    return graphs_tuple

def convert_to_graph_data(features, topology):
    comp_to_id = {'R': 1, 'L': 2, 'C': 3,'V': 4}
    comp_to_value = {'R': 50, 'L': 1e-3, 'C': 1e-6, 'V':0}
    graphs_list = []
    for circuit in range(len(features.keys())):
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
                edges.append([float(comp_to_id[receiver[i][0][0]]),float(comp_to_value[receiver[i][0][0]])])
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
        graphs_list.append(graph)

    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(graphs_list)
    graphs_tuple = tree.map_structure(lambda x: tf.constant(x) if x is not None else None, graphs_tuple)

    return graphs_tuple


def train_gnn(input_graphs,target_graphs):
    iterations = 1000
    learning_rate = 1e-3
    optimizer = snt.optimizers.Adam(learning_rate)

    model = models.EncodeProcessDecode(node_output_size=7)

    def create_loss(target, outputs):
        loss = [
            tf.compat.v1.losses.mean_squared_error(target.nodes, output.nodes)
            for output in outputs
        ]
        return tf.stack(loss)

    def update_step(inputs_tr, targets_tr):
        with tf.GradientTape() as tape:
            outputs_tr = model(inputs_tr,10)
            loss_tr = create_loss(targets_tr, outputs_tr)
            loss_tr = tf.math.reduce_sum(loss_tr) / 10
        gradients = tape.gradient(loss_tr, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)
        return outputs_tr, loss_tr

    for iteration in range(iterations):

        inputs_tr = utils_tf.get_graph(input_graphs, slice(0, 80))
        targets_tr = utils_tf.get_graph(target_graphs, slice(0, 80))

        outputs_tr, loss_tr = update_step(inputs_tr, targets_tr)
        print(loss_tr)
    print(outputs_tr)

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



