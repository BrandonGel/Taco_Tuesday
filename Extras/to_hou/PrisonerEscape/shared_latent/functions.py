from torch import Tensor, Size
import numpy as np

from typing import Callable, Dict, List, Tuple, TypeVar, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor, Size

def logavgexp(x: Tensor, dim: int) -> Tensor:
    if x.size(dim) > 1:
        # TODO: cast to float32 here for IWAE?
        return x.logsumexp(dim=dim) - np.log(x.size(dim))
    else:
        return x.squeeze(dim)

def flatten_batch(x: Tensor, nonbatch_dims=1) -> Tuple[Tensor, Size]:
    # (b1,b2,..., X) => (B, X)
    if nonbatch_dims > 0:
        batch_dim = x.shape[:-nonbatch_dims]
        x = torch.reshape(x, (-1,) + x.shape[-nonbatch_dims:])
        return x, batch_dim
    else:
        batch_dim = x.shape
        x = torch.reshape(x, (-1,))
        return x, batch_dim


def unflatten_batch(x: Tensor, batch_dim: Union[Size, Tuple]) -> Tensor:
    # (B, X) => (b1,b2,..., X)
    x = torch.reshape(x, batch_dim + x.shape[1:])
    return x


def insert_dim(x: Tensor, dim: int, size: int) -> Tensor:
    """Inserts dimension and expands it to size."""
    x = x.unsqueeze(dim)
    x = x.expand(*x.shape[:dim], size, *x.shape[dim + 1:])
    return x

def diag_normal(x: Tensor, min_std=0.0001, max_std=20.0):
    # DreamerV2:
    # std = {
    #     'softplus': lambda: tf.nn.softplus(std),
    #     'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
    # }[self.std_act]()
    # std = std + self.min_std
    mean, std = x.chunk(2, -1)
    # std = F.softplus(std) + min_std
    std = max_std * torch.sigmoid(std) + min_std
    return D.independent.Independent(D.normal.Normal(mean, std), 1)

class NoNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x