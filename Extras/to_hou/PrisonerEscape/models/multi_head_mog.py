import sys, os

sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch.distributions import Normal
import math
import torch.distributions as D
# from shared_latent.functions import logavgexp, flatten_batch, unflatten_batch, insert_dim, NoNorm


import math
import torch
from torch.distributions import Categorical

from models.utils import log_prob


class MixureDecoderMultiHead(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim=2, num_gaussians=2, num_heads=1, log_std_init=0.0):
        super(MixureDecoderMultiHead, self).__init__()

        self.fc = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(0.2)

        input_dim = 32

        # self.batch_size = batch_size
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim

        # Predict Mixture of gaussians from encoded embedding
        self.pi =nn.Linear(input_dim, num_gaussians*num_heads)
        nn.init.xavier_uniform_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)
        self.softmax = nn.Softmax(dim=2)

        self.sigma = nn.Linear(input_dim, output_dim * num_gaussians * num_heads)
        nn.init.xavier_normal_(self.sigma.weight)
        nn.init.zeros_(self.sigma.bias)

        self.mu = nn.Linear(input_dim, output_dim * num_gaussians * num_heads)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        self.num_heads = num_heads

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        x = self.fc(x)
        x = self.dropout(x)

        batch_size = x.size(0)
        # Predict the mixture of gaussians around the fugitive
        pi = self.pi(x)
        pi = pi.view(batch_size, self.num_heads, self.num_gaussians)
        pi = self.softmax(pi)

        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(batch_size, self.num_heads,  self.num_gaussians, self.output_dim)

        mu = self.mu(x)
        mu = mu.view(batch_size, self.num_heads, self.num_gaussians, self.output_dim)

        sigma = nn.ELU()(sigma) + 1e-15
        # sigma = torch.clamp(mu, min=0.00001)
        # sigma = self.relu(sigma)
        return pi, mu, sigma

    def compute_loss(self, nn_output, red_locs):
        # nn_output = self.forward(features)
        red_locs = red_locs.view(-1, self.num_heads, self.output_dim)

        losses = self.mdn_negative_log_likelihood_loss(nn_output, red_locs)
        loss = torch.sum(losses, dim=1).mean()
        return loss

    def get_stats(self, nn_output, red_locs):
        red_locs = red_locs.view(-1, self.num_heads, self.output_dim)
        return -self.mdn_negative_log_likelihood_loss(nn_output, red_locs)
        

    def mdn_negative_log_likelihood(self, pi, mu, sigma, target):
        """ Use torch.logsumexp for more stable training 
        
        This is equivalent to the mdn_loss but computed in a numerically stable way

        """
        target = target.unsqueeze(2).expand_as(sigma)
        neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
            ((target - mu) / sigma)**2 / 2
        
        # (B, num_heads, num_gaussians)
        inner = torch.log(pi) + torch.sum(neg_logprob, 3) # Sum the log probabilities of (x, y) for each 2D Gaussian
        return -torch.logsumexp(inner, dim=2)

    def mdn_negative_log_likelihood_loss(self, nn_output, target):
        """
        Compute the negative log likelihood loss for a MoG model.
        """
        pi, mu, sigma = nn_output
        return self.mdn_negative_log_likelihood(pi, mu, sigma, target)