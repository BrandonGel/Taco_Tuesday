# def func(a):
#     print("ran")
#     return a

# b = {1: 2, 3: 4, 5: 6}

# c = {key: func(key) for key, ret in b.items()}

from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
import torch

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

edge_index_2 = torch.tensor([[1, 1], [0, 1]], dtype=torch.long)

data = Data(edge_index=edge_index)
data_2 = Data(edge_index=edge_index_2)

features = torch.tensor([[1], [2], [3], [4], [5]])
b = Batch.from_data_list([data, data_2])
b.x = features

# print(b)

print(b.edge_index)
print(b.x)