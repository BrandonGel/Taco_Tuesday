"""
Temporal GCN for batched, static homogenous graphs (fixed cameras)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric
# from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import TGCN2, A3TGCN2
from torch_geometric_temporal.nn.attention import STConv, ASTGCN
from torch_geometric_temporal.nn.attention.tsagcn import AAGCN
from torch_geometric_temporal.nn.attention.dnntsp import DNNTSP
from torch_geometric.nn import global_mean_pool
from itertools import combinations


def fully_connected(num_nodes):
    # create fully connected graph
    test_list = range(num_nodes)
    edges = list(combinations(test_list, 2))
    start_nodes = [i[0] for i in edges]
    end_nodes = [i[1] for i in edges]
    return torch.Tensor(start_nodes), torch.Tensor(end_nodes)


DEVICE = torch.device('cpu')  # cpu
shuffle = True
batch_size = 32


# Making the model
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, periods, batch_size):
        super(TemporalGNN, self).__init__()
        self.batch_size = batch_size
        # Temporal Graph Convolutional Cell
        self.tgnn = TGCN2(in_channels=node_features, out_channels=hidden_dim,
                          batch_size=batch_size)
        # self.tgnn = DCRNN(in_channels=node_features, out_channels=hidden_dim, K=5)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(hidden_dim, periods)
        self.device = torch.device('cpu')

        # Loading the graph once because it's a static graph
        test_list = range(83) # Num agents for fixed camera setting
        edges = list(combinations(test_list, 2))
        start_nodes = [i[0] for i in edges]
        end_nodes = [i[1] for i in edges]
        self.edge_index = torch.LongTensor((start_nodes, end_nodes)).to(self.device)

    def forward(self, x, edge_index=None):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        agent_obs, hideout_obs, timestep_obs, num_agents, last_k_fugitive_detections = x  # Extract input features from train loader
        agent_obs = agent_obs.to(self.device).float()

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()

        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        num_agents = agent_obs.shape[2]
        features = agent_obs.shape[3]

        # Permute data to make it compatible for A3TGCN2 (B x N x F x T)
        agent_obs = agent_obs.permute(0, 2, 3, 1)  # (batch_size, num_agents, features, seq_len)

        if edge_index is None:
            edge_index = self.edge_index

        h = None
        for time in range(seq_len):
            h = self.tgnn(X=agent_obs[:, :, :, time], edge_index=edge_index, H=h, edge_weight=None)  # agent_obs [batch, num_agents, feats]  returns h [batch, num_agents, out_channels]

        # Use the hidden layer from the final timestep
        # Optional??
        h = F.relu(h)
        h = self.linear(h)

        # TODO: Fix this hack for computing the Global Average Pool
        # num_agents = h.shape[1]
        # Reshape to perform global mean pool
        h = h.reshape(batch_size * num_agents, -1)
        batch = np.repeat(np.arange(batch_size), num_agents)
        batch = torch.from_numpy(batch).to(self.device)
        h = global_mean_pool(h, batch=batch)

        # Concatenate the hideout and timestep obs
        res = torch.cat((h, hideout_obs, timestep_obs, last_k_fugitive_detections), dim=-1)
        return res


class STCGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, periods, batch_size):
        super(STCGNN, self).__init__()
        self.batch_size = batch_size
        # Temporal Graph Convolutional Cell
        self.tgnn = STConv(in_channels=node_features,
                           out_channels=hidden_dim,
                           hidden_channels=16,
                           kernel_size=3,  # Temporal kernel size
                           K=3,  # Graph Conv Kernel size. Both were set to 3 in the paper
                           num_nodes=83)  # For fixed cameras setting
        # self.tgnn = DCRNN(in_channels=node_features, out_channels=hidden_dim, K=5)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(hidden_dim, periods)
        self.device = torch.device('cpu')

        # Loading the graph once because it's a static graph
        test_list = range(83) # Num agents for fixed camera setting
        edges = list(combinations(test_list, 2))
        start_nodes = [i[0] for i in edges]
        end_nodes = [i[1] for i in edges]
        self.edge_index = torch.LongTensor((start_nodes, end_nodes)).to(self.device)

    def forward(self, x, edge_index=None):
        """
        X (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
        edge_index (PyTorch LongTensor) - Graph edge indices.
        edge_weight (PyTorch LongTensor, optional)- Edge weight vector.
        """
        agent_obs, hideout_obs, timestep_obs, num_agents = x  # Extract input features from train loader
        agent_obs = agent_obs.to(self.device).float()

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()

        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        num_agents = agent_obs.shape[2]
        features = agent_obs.shape[3]

        if edge_index is None:
            edge_index = self.edge_index  # Load Fully Connected Graph


        h = self.tgnn(X=agent_obs, edge_index=edge_index, edge_weight=None)  # agent_obs [batch, seq len, num_agents, feats]  returns h [batch, m-2*(k-1), num_agents, out_channels]

        # Take the last hidden output
        h = h[:, -1, :, :]
        # Use the hidden layer from the final timestep
        # Optional??
        h = F.relu(h)
        h = self.linear(h)

        # TODO: Fix this hack for computing the Global Average Pool
        # num_agents = h.shape[1]
        # Reshape to perform global mean pool
        h = h.reshape(batch_size * num_agents, -1)
        batch = np.repeat(np.arange(batch_size), num_agents)
        batch = torch.from_numpy(batch).to(self.device)
        h = global_mean_pool(h, batch=batch)

        # Concatenate the hideout and timestep obs
        res = torch.cat((h, hideout_obs, timestep_obs), dim=-1)
        return res


class ASTGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, periods, batch_size):
        super(ASTGNN, self).__init__()
        self.batch_size = batch_size
        # Temporal Graph Convolutional Cell
        self.tgnn = ASTGCN(nb_block=2,
                           in_channels=node_features,
                           K=3,
                           nb_chev_filter=1,
                           nb_time_filter=2,
                           time_strides=1,
                           len_input=periods,
                           num_of_vertices=83,
                           num_for_predict=1
                           )
        # self.tgnn = DCRNN(in_channels=node_features, out_channels=hidden_dim, K=5)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(hidden_dim, periods)
        self.device = torch.device('cpu')

        # Loading the graph once because it's a static graph
        test_list = range(83) # Num agents for fixed camera setting
        edges = list(combinations(test_list, 2))
        start_nodes = [i[0] for i in edges]
        end_nodes = [i[1] for i in edges]
        self.edge_index = torch.LongTensor((start_nodes, end_nodes)).to(self.device)

    def forward(self, x, edge_index=None):
        """
        X (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
        edge_index (PyTorch LongTensor) - Graph edge indices.
        edge_weight (PyTorch LongTensor, optional)- Edge weight vector.
        """
        agent_obs, hideout_obs, timestep_obs, num_agents = x  # Extract input features from train loader
        agent_obs = agent_obs.to(self.device).float()

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()

        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        num_agents = agent_obs.shape[2]
        features = agent_obs.shape[3]

        if edge_index is None:
            edge_index = self.edge_index  # Load Fully Connected Graph

        h = self.tgnn(X=agent_obs, edge_index=edge_index, edge_weight=None)  # agent_obs [batch, seq len, num_agents, feats]  returns h [batch, num_agents, out_channels]

        # Use the hidden layer from the final timestep
        # Optional??
        h = F.relu(h)
        h = self.linear(h)

        # TODO: Fix this hack for computing the Global Average Pool
        # num_agents = h.shape[1]
        # Reshape to perform global mean pool
        h = h.reshape(batch_size * num_agents, -1)
        batch = np.repeat(np.arange(batch_size), num_agents)
        batch = torch.from_numpy(batch).to(self.device)
        h = global_mean_pool(h, batch=batch)

        # Concatenate the hideout and timestep obs
        res = torch.cat((h, hideout_obs, timestep_obs), dim=-1)
        return res


class AttnTemporalGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, periods, batch_size):
        super(AttnTemporalGNN, self).__init__()
        self.batch_size = batch_size
        # Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=hidden_dim, periods=periods,
                            batch_size=batch_size)

        # Equals single-shot prediction
        self.linear = torch.nn.Linear(hidden_dim, periods)
        self.device = torch.device('cpu')

        # Loading the graph once because it's a static graph
        test_list = range(83) # Num agents for fixed camera setting
        edges = list(combinations(test_list, 2))
        start_nodes = [i[0] for i in edges]
        end_nodes = [i[1] for i in edges]
        self.edge_index = torch.LongTensor((start_nodes, end_nodes)).to(self.device)

    def forward(self, x, edge_index=None):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        agent_obs, hideout_obs, timestep_obs, num_agents = x  # Extract input features from train loader
        agent_obs = agent_obs.to(self.device).float()

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()

        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        num_agents = agent_obs.shape[2]
        features = agent_obs.shape[3]

        # Permute data to make it compatible for A3TGCN2 (B x N x F x T)
        agent_obs = agent_obs.permute(0, 2, 3, 1)  # (batch_size, num_agents, features, seq_len)

        if edge_index is None:
            edge_index = self.edge_index

        h = self.tgnn(agent_obs, edge_index)  # agent_obs [batch, num_agents, feats, t]  returns h [batch, num_agents, out_channels]
        # Use the hidden layer from the final timestep
        # Optional??
        h = F.relu(h)
        h = self.linear(h)

        # TODO: Fix this hack for computing the Global Average Pool
        # num_agents = h.shape[1]
        # Reshape to perform global mean pool
        h = h.reshape(batch_size * num_agents, -1)
        batch = np.repeat(np.arange(batch_size), num_agents)
        batch = torch.from_numpy(batch).to(self.device)
        h = global_mean_pool(h, batch=batch)

        # Concatenate the hideout and timestep obs
        res = torch.cat((h, hideout_obs, timestep_obs), dim=-1)
        return res


class AAGCNGraph(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, periods, batch_size):
        super(AAGCNGraph, self).__init__()
        self.batch_size = batch_size

        # Equals single-shot prediction
        self.linear = torch.nn.Linear(hidden_dim, periods)
        self.device = torch.device('cpu')

        # Loading the graph once because it's a static graph
        test_list = range(83)  # Num agents for fixed camera setting
        edges = list(combinations(test_list, 2))
        start_nodes = [i[0] for i in edges]
        end_nodes = [i[1] for i in edges]
        self.edge_index = torch.LongTensor((start_nodes, end_nodes)).to(self.device)

        # Temporal Graph Convolutional Cell
        self.tgnn = AAGCN(in_channels=node_features,
                          out_channels=hidden_dim,
                          num_nodes=83,
                          edge_index=self.edge_index,
                          stride=1)

    def forward(self, x):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        agent_obs, hideout_obs, timestep_obs, num_agents = x  # Extract input features from train loader
        agent_obs = agent_obs.to(self.device).float()

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()

        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        num_agents = agent_obs.shape[2]
        features = agent_obs.shape[3]

        # Permute data to make it compatible for AAGCN (B x F x T x N)
        agent_obs = agent_obs.permute(0, 3, 1, 2)  # (batch_size, features, seq_len, num_agents)

        h = self.tgnn(agent_obs.contiguous())  # agent_obs [batch, feats, t, num_agents]  returns h [batch, out_channels, T//stride, num_agents]
        # Use the hidden layer from the final timestep
        h = h[:, :, -1, :]

        h = h.permute(0, 2, 1)  # (batch_size, num_agents, out_channels)

        # Optional??
        h = F.relu(h)
        h = self.linear(h)

        # TODO: Fix this hack for computing the Global Average Pool
        # num_agents = h.shape[1]
        # Reshape to perform global mean pool
        h = h.reshape(batch_size * num_agents, -1)
        batch = np.repeat(np.arange(batch_size), num_agents)
        batch = torch.from_numpy(batch).to(self.device)
        h = global_mean_pool(h, batch=batch)

        # Concatenate the hideout and timestep obs
        res = torch.cat((h, hideout_obs, timestep_obs), dim=-1)
        return res
