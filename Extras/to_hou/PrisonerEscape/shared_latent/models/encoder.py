# from filtering.model_consolidated.filtering_model import FilteringModel
import sys, os

sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from filtering.infrastructure.lstm import EncoderRNN
# from filtering.utils.train_utils import sample_sequence_from_buffer, save_policy
import pickle
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import math
import random

from torch import Tensor
from typing import Callable, Dict, List, Tuple, TypeVar, Union
import torch.distributions as D
from shared_latent.functions import logavgexp, flatten_batch, unflatten_batch, insert_dim, NoNorm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class GRUStepEncoder(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        """ This GRU Encoder is used to encode a smaller number of timesteps in the RSSM model
        """
        super(GRUStepEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first = False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, h=None):
        """ 
        x: (T,B,D)
        h: (B,D)
        """

        if h is None:
            h = torch.zeros(1, x.shape[1], self.hidden_size, device=self.device)
        
        o, h = self.gru(x, h)

        # h is of shape (1,B,D)
        # we want to return h of shape (B,D)
        return h.view(h.shape[1], h.shape[2])

    