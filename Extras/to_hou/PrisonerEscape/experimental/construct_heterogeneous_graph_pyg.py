import torch
import dgl
from torch_geometric.data import HeteroData
from torch_geometric_temporal.nn.hetero.heterogclstm import HeteroGCLSTM
from torch_geometric.nn.conv.hetero_conv import HeteroConv
from torch_geometric.nn import GCNConv, SAGEConv
from torch.nn import Parameter
import torch_geometric.transforms as T

# pytorch geometric - you feed the model an x_dict of {'agent': (num_agents, num_features), 'hideouts': (num_hideouts, num_features)}
# and an edge_dict of {('agent', 'interacts', 'agent'): tensor([[0],

# dgl - in zheyuan's code - he passes in the graph and node features and edge features
# in dgl, you can batch the graphs together and grab the node features 

def construct_het_graph(num_agents, num_hideouts):
    """ Script to construct heterogeneous graph from dataset 
    
    We have a graph with node types: agent, hideout, hideout summary node, agent summary node, timestep node
    
    
    agent -> agent summary node
    hideout -> hideout summary node
    
    timestep, hideout summary, agent summary -> state summary

    """

    # from agent to agent summary node
    agent_indices = torch.arange(0, num_agents)
    agent_summary_index = torch.tensor([0] * num_agents) # torch.zeros didn't work - error with dgl

    hideout_indices = torch.arange(0, num_hideouts)
    hideout_summary_index = torch.tensor([0] * num_hideouts)
    # dgl datadict is from (from_node_index_tensor, to_node_index_tensor) where different node types are indexed differently

    data_dict = {
        ('agent', 'to', 'agent_summ'): (agent_indices, agent_summary_index),
        # ('agent_summ', 'contains', 'agent'): (agent_summary_index, agent_indices),
        ('hideout', 'to', 'hideout_summ'): (hideout_indices, hideout_summary_index),
        # ('hideout_summ', 'contains', 'hideout'): (hideout_summary_index, hideout_indices),
        ('hideout_summ', 'to', 'state_summ'): (torch.tensor([0]), torch.tensor([0])),
        ('agent_summ', 'to', 'state_summ') : (torch.tensor([0]), torch.tensor([0])),
        ('timestep', 'to', 'state_summ'): (torch.tensor([0]), torch.tensor([0]))
    }

    return data_dict

def convert_batched_dgl_graph_to_pyg(bg):
    """ Converts a batch of dgl graphs to single pytorch geometric heterodata graph """
    data=HeteroData()
    edge_list = [('agent', 'to', 'agent_summ'), 
            ('hideout', 'to', 'hideout_summ'), 
            ('hideout_summ', 'to', 'state_summ'), 
            ('agent_summ', 'to', 'state_summ'), 
            ('timestep', 'to', 'state_summ')]
            # ('agent_summ', 'contains', 'agent'),
            # ('hideout_summ', 'contains', 'hideout')]

    for edge in edge_list:
        data[edge[0], edge[1], edge[2]].edge_index = torch.stack(bg.edges(etype=edge))

    return data

dd = construct_het_graph(5, 2)
# print(dd)
dd2 = construct_het_graph(10, 3)

g = dgl.heterograph(dd)
g2 = dgl.heterograph(dd2)

bg = dgl.batch([g, g2])
# print(bg.edges('agent', 'to', 'agent_summ'))


# bg = bg.to("cuda")

# print(bg.nodes['agent'])
# print(bg.edges(('agent', 'to', 'agent_summ')))
# print(bg.edges(etype=('agent', 'to', 'agent_summ')))
# print(bg.edges(etype=('hideout', 'to', 'hideout_summ')))

pyg = convert_batched_dgl_graph_to_pyg(bg)
pyg = T.ToUndirected()(pyg)
pyg['agent'].x = torch.zeros((15, 2))
pyg['hideout'].x = torch.ones((5, 2))
pyg['timestep'].x = torch.ones((2, 1))
pyg['agent_summ'].x = torch.zeros((2, 1))
pyg['hideout_summ'].x = torch.zeros((2, 1))
pyg['state_summ'].x = torch.zeros((2, 1))


print(pyg)

# pyg['hideout_summ'].x = torch.zeros(())
# pyg['']

# print(pyg.x_dict)
# print(pyg.metadata())

in_channels_dict = {'agent': 2, 'hideout': 2, 'timestep': 1, 'agent_summ': 1, 'hideout_summ': 1, 'state_summ': 1}

metadata = pyg.metadata()
# metadata = (['agent_summ', 'hideout_summ', 'state_summ'], [('agent', 'to', 'agent_summ'), ('hideout', 'to', 'hideout_summ'), ('hideout_summ', 'to', 'state_summ'), ('agent_summ', 'to', 'state_summ'), ('timestep', 'to', 'state_summ'), ('agent_summ', 'contains', 'agent')])

out_channels = 16
bias = True
conv_i = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                                out_channels=out_channels,
                                                bias=bias) for edge_type in metadata[1]})

# print(pyg.metadata())

def _set_hidden_state(x_dict, h_dict):
    if h_dict is None:
        h_dict = {node_type: torch.zeros(X.shape[0], out_channels) for node_type, X in x_dict.items()}
    return h_dict

h_dict = None
x_dict = pyg.x_dict
edge_index_dict =  pyg.edge_index_dict
h_dict = _set_hidden_state(x_dict, h_dict)
# print(h_dict)

W_i = {node_type: Parameter(torch.Tensor(in_channels, out_channels))
            for node_type, in_channels in in_channels_dict.items()}

i_dict = {node_type: torch.matmul(X, W_i[node_type]) for node_type, X in x_dict.items()}
# for node_type, I in i_dict.items():
#     conv_val = self.conv_i(h_dict, edge_index_dict)[node_type]
#     i_dict[node_type] = I + conv_val
# i_dict = {node_type: I + conv_i(h_dict, edge_index_dict)[node_type] for node_type, I in i_dict.items()}

    # print(val)

# hetero_conv = HeteroConv({
#     ('paper', 'cites', 'paper'): GCNConv(-1, 64),
#     ('author', 'writes', 'paper'): GCNConv((-1, -1), 64),
#     ('paper', 'written_by', 'author'): GCNConv((-1, -1), 64),
# }, aggr='sum')

# out_dict = hetero_conv(x_dict, edge_index_dict)

# print(list(out_dict.keys()))

# metadata = (['agent_summ', 'hideout_summ', 'state_summ'], [('agent', 'to', 'agent_summ'), ('hideout', 'to', 'hideout_summ'), ('hideout_summ', 'to', 'state_summ'), ('agent_summ', 'to', 'state_summ'), ('timestep', 'to', 'state_summ')])

layer = HeteroGCLSTM(in_channels_dict = in_channels_dict, out_channels = 16, metadata = pyg.metadata())

h_dict, c_dict = layer(pyg.x_dict, pyg.edge_index_dict)
print(pyg.metadata())
print(h_dict)

