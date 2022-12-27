import torch
import dgl
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
        ('hideout', 'to', 'hideout_summ'): (hideout_indices, hideout_summary_index),
        ('hideout_summ', 'to', 'state_summ'): (torch.tensor([0]), torch.tensor([0])),
        ('agent_summ', 'to', 'state_summ') : (torch.tensor([0]), torch.tensor([0])),
        ('timestep', 'to', 'state_summ'): (torch.tensor([0]), torch.tensor([0]))
    }

    return data_dict

dd = construct_het_graph(5, 2)
# print(dd)
dd2 = construct_het_graph(10, 3)

g = dgl.heterograph(dd)
g2 = dgl.heterograph(dd2)

bg = dgl.batch([g, g2])
print(bg)
