import torch
from torch_geometric.data import Data,Dataset
import glob
import os
import numpy as np
#
# class NetDataset(Dataset):
#     def __init__(self, dirpath):
#         super().__init__()
#         self.dirpath = dirpath,
#         self.data = self.process()
#
#     @property
#     def raw_file_names(self):
#         return []
#
#     @property
#     def processed_file_names(self):
#         return []
#
#     def download(self):
#         pass
#
#     def process(self):
#         comp_to_id = {"R": 1, "L": 2, "C": 3, "V": 4}
#         input = []
#         output = []
#         i = 0
#         graph_data = []
#         for file in glob.glob(os.path.join(self.dirpath, "*")):
#             print("Parsing file {}: ".format(i) + file)
#             i += 1
#             current_file = open(file, "r")
#             x = []
#             y = []
#             edge_index = [[], []]
#             for line in current_file.readline():
#                 if line[0] != "*" and line[0] != ".":
#                     values = line.split(" ")
#                     values[0] = comp_to_id.get(values[0][0])
#                     x.append(values[:-1])
#                     y.append(values[0])
#                     edge_index[0].append(values[1])
#                     edge_index[1].append(values[2])
#                     input.append(values[:-1])
#                     output.append(values[-1])
#             tensor_x = torch.from_numpy(np.array(x))
#             tensor_y = torch.from_numpy(np.array(y))
#             tensor_edges = torch.from_numpy(np.array(edge_index))
#             data = Data(x=tensor_x, y=tensor_y, edge_index=tensor_edges)
#             graph_data.append(data)
#
#         return graph_data
#
#
#


# def convert_data(dirpath):
#     comp_to_id = {"R": 1, "L": 2, "C": 3, "V": 4}
#     i = 0
#     graph_data = []
#     for file in glob.glob(os.path.join(dirpath, "*")):
#         print("Parsing file {}: ".format(i) + file)
#         i += 1
#         current_file = open(file, "r")
#         x = []
#         y = []
#         edge_index = [[], []]
#         for line in current_file.readline():
#             if line[0] != "*" and line[0] != ".":
#                 values = line.split(" ")
#                 values[0] = comp_to_id.get(values[0][0])
#                 x.append(values[:-1])
#                 y.append(values[0])
#                 edge_index[0].append(values[1])
#                 edge_index[1].append(values[2])
#         tensor_x = torch.from_numpy(np.array(x))
#         tensor_y = torch.from_numpy(np.array(y))
#         tensor_edges = torch.from_numpy(np.array(edge_index))
#         data = Data(x=tensor_x, y=tensor_y, edge_index=tensor_edges)
#         graph_data.append(data)
#
#     return graph_data
