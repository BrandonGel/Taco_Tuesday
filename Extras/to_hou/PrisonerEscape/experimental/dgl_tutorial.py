import dgl
import torch
import matplotlib.pyplot as plt
import networkx as nx

# tutorial located here: https://docs.dgl.ai/guide/graph-graphs-nodes-edges.html

# edges 0->1, 0->2, 0->3, 1->3

# start nodes, end nodes
u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])

# bidirectional graph can be created with dgl.to_bidirected()

g = dgl.graph((u, v))
# print(g)
# print(g.nodes())
# print(g.edges())

## storing node and edges features
# g = dgl.graph([0, 0, 1, 5], [1, 2, 2, 0])
# g.ndata['x'] = torch.ones(g.num_nodes(), 3)
# g.edata['x'] = torch.ones(g.num_edges(), dtype=torch.int32)
# print(g.ndata['x'][1])

data_dict = {
    ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ('user', 'follows', 'topic'): (torch.tensor([1, 1]), torch.tensor([1, 2])),
    ('user', 'plays', 'game'): (torch.tensor([0, 3]), torch.tensor([3, 4]))
}
g = dgl.heterograph(data_dict)

options = {
    'node_color': 'black',
    'node_size': 20,
    'width': 1,
}

G = dgl.to_networkx(dgl.to_homogeneous(g))
plt.figure(figsize=[15,7])
nx.draw(G, **options)
plt.savefig("experimental/test.png")

print(g.num_nodes())

g2 = dgl.heterograph(data_dict)

bg = dgl.batch([g, g2])
print(bg.batch_num_nodes('user'))

bg.nodes['user'].data['x'] = torch.ones(8, 3)

print(bg.nodes['user'])
print(dgl.unbatch(bg)[0].nodes['user'])

# print(g.ndata.keys())
# print(g.edata)
# print(g.num_nodes())
# print(g.nodes('game'))