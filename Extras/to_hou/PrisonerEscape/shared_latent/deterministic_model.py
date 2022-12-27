""" Instead of an RSSM, let's just use a simple recurrent model to see if this works """

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functions import diag_normal, NoNorm
import shared_latent.models.rnn as rnn

class RecurrentModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, gru_layers, gru_type, gru_implementation=True):
        super().__init__()

        self.hid_dim = hidden_dim
        self.gru_layers = gru_layers
        
        # embed our input to a smaller dimension
        self.linear = nn.Linear(input_dim, hidden_dim)

        self.gru_implementation = gru_implementation
        if gru_implementation:
            self.gru = rnn.GRUCellStack(hidden_dim, hidden_dim, gru_layers, gru_type)
        else:
            self.gru = nn.GRU(hidden_dim, hidden_dim)
        

    def forward(self, obs, reset, h, do_open_loop = False):
        T, B = obs.shape[:2]

        # def expand(x):
        #     # (T,B,X) -> (T,BI,X)
        #     return x.unsqueeze(2).expand(T, B, -1).reshape(T, B, -1).to(torch.float32)

        embeds = obs.unbind(0)     # (T,B,...) => List[(BI,...)]
        embeds = [self.linear(e.to(torch.float32)) for e in embeds]
        # reset_masks = ~reset.unsqueeze(1).unbind(0)
        reset_masks = (~reset.unsqueeze(1)).unbind(0)
        reset_masks = [i.permute(1, 0).to(torch.float32) for i in reset_masks]

        # obs = obs.to(torch.float64)
        # embeds = self.linear(obs)

        states_h = []

        if self.gru_implementation:
            for i in range(T):
                if not do_open_loop:
                    h = h * reset_masks[i]
                    h = self.gru(embeds[i], h)
                else:
                    pass
                    # post, (h, z) = self.cell.forward_prior(reset_masks[i], (h, z))  # open loop: post=prior
                # posts.append(post)
                states_h.append(h)
            states_h = torch.stack(states_h)    # (T,BI,D)
            return states_h
        else:
            # gru returns (output, hn), where output is hidden lyaer outputs of network for each timestep (but only final layer)
            # hn is the hidden layer output for last step only, but for all layers
            x = self.linear(obs.to(torch.float32))
            output, hn = self.gru(x, h)
            return output

    def init_hidden(self, batch_size, device):
        if self.gru_implementation:
            return torch.zeros(batch_size, self.hid_dim, device=device).requires_grad_()
        else:
            return torch.zeros(1, batch_size, self.hid_dim, device=device).requires_grad_()
