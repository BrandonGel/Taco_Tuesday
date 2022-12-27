""" Tutorial on torch geometric hetero data
https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
"""

import torch
# import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.data.batch import Batch



# data['paper'].x = torch.zeros((1, 2))

# Node types are identified by a single string
# Edge types are identified by using a triplet of (source_node_type, edge_type, destination_node_type) strsings

# model = HeteroGNN(...)
# output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

# data_list = []

# data=HeteroData()
# data['agent'].x = torch.zeros((5, 2))
# data['fugitive'].x = torch.tensor([[15, 15]])
# data['agent', 'interacts', 'agent'].edge_index = torch.tensor([[0], [2]])

# data_1=HeteroData()
# data_1['agent'].x = torch.ones((5, 2))
# data_1['fugitive'].x = torch.tensor([[20, 20]])
# data_1['agent', 'interacts', 'agent'].edge_index = torch.tensor([[1], [2]])

data=HeteroData()
# data['paper'].x = torch.ones((5, 2))
# data['institution'].x = torch.ones((10, 1))
# data['author'].x = torch.tensor([[20., 20.]])
data['author', 'writes', 'paper'].edge_index = torch.tensor([[0], [0]])


data_1=HeteroData()
# data_1['paper'].x = torch.zeros((5, 2))
# data_1['author'].x = torch.tensor([[20., 20.]])
data_1['author', 'writes', 'paper'].edge_index = torch.tensor([[0], [0]])

data_list = [data, data_1]

# batch = Batch()
# batched_data = batch.from_data_list(data_list)
# print(batched_data.x_dict)
batched_data = Batch.from_data_list(data_list)
print(batched_data.x_dict, batched_data.edge_index_dict)

# from torch_geometric.nn import HeteroConv, SAGEConv
# print(data.x_dict, data.edge_index_dict)
# conv = SAGEConv(in_channels=(-1, -1), out_channels=16, bias=True)

# dst='paper'
# src='author'
# out = conv((data.x_dict[src], data.x_dict[dst]), data.edge_index_dict[('author', 'writes', 'paper')])
# print(out.shape)

# from torch_geometric.nn.conv.transformer_conv import TransformerConv
# t_conv = TransformerConv(in_channels=(-1, -1), out_channels=16)

# out = t_conv((data.x_dict[src], data.x_dict[dst]), data.edge_index_dict[('author', 'writes', 'paper')])

# print(out.shape)