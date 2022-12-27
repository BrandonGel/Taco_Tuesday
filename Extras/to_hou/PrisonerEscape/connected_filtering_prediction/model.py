"""
Copyright (2022)
Georgia Tech Research Corporation (Sean Ye, Manisha Natarajan, Rohan Paleja, Letian Chen, Matthew Gombolay)
All rights reserved.
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
from models.encoders import EncoderRNN
from models.decoder import MixtureDensityDecoder

from torch.distributions import Categorical
from torch.distributions.gumbel import Gumbel

class MixtureInMiddle(nn.Module):
    def __init__(self, encoder, mixture_decoder: MixtureDensityDecoder, location_decoder, hidden_connected_bool: bool):
        super(MixtureInMiddle, self).__init__()
        self.encoder = encoder
        self.mixture_decoder = mixture_decoder
        self.location_decoder = location_decoder
        self.hidden_connected_bool = hidden_connected_bool

    def forward(self, blue_obs):
        z = self.encoder(blue_obs)
        mixture_output = self.mixture_decoder(z)
        pi, mu, sigma = mixture_output
        samples = sample_mixture_gumbel(pi, mu, sigma, temp=0.1)

        if self.hidden_connected_bool:
            inputs = torch.cat((samples, z), dim=1)
            predicted_locations = self.location_decoder(inputs)
        else:
            predicted_locations = self.location_decoder(samples)
        return mixture_output, predicted_locations

    def compute_loss(self, blue_obs, y, writer=None):
        blue_obs = blue_obs.to(self.device).float()
        current_red_loc, future_red_locs = y
        current_red_loc = current_red_loc.to(self.device).float()
        future_red_locs = future_red_locs.to(self.device).float()
        
        z = self.encoder(blue_obs)
        nn_output = self.mixture_decoder(z)
        pi, mu, sigma = nn_output
        l1 = self.mixture_decoder.compute_loss(nn_output, current_red_loc)
        
        samples = sample_mixture_gumbel(pi, mu, sigma)

        if self.hidden_connected_bool:
            inputs = torch.cat((samples, z), dim=1)
            predicted_locations = self.location_decoder(inputs)
        else:
            predicted_locations = self.location_decoder(samples)
        l2 = self.location_decoder.compute_loss(predicted_locations, future_red_locs)

        loss = l1 + l2
        if writer is not None:
            pass
        # self.location_decoder.compute_loss(future_red_obs)
        return loss

    def get_stats(self, obs, target):
        obs = obs.to(self.device).float()
        # current_red_loc, future_red_locs = target
        future_red_locs = target.view(target.shape[0], -1)
        future_red_locs = future_red_locs.to(self.device).float()
        mixture_output, predicted_locations = self.forward(obs)
        return self.location_decoder.get_stats(predicted_locations, future_red_locs)

    @property
    def device(self):
        return next(self.parameters()).device

def sample_mixture_gumbel(pi, mu, sigma, temp=0.1):
    """ 
    
    Given a mixture of gaussians, sample from the mixture in a way that we can backpropagate through

    pi: (B, G)
    mu: (B, G, D)
    sigma: (B, G, D)

    First, sample categorically from the mixture pi with gumbel softmax.
    Then, sample from the corresponding gaussian by multiplying and adding with mean and std.

    Returns shape of (B, D) where we have batch size and dimension of gaussian

    """
    # ensure all the dimensions are correct
    assert pi.size(0) == mu.size(0) == sigma.size(0)
    assert pi.size(1) == mu.size(1) == sigma.size(1)
    assert mu.size(2) == sigma.size(2)

    # sample from gumbel softmax
    m = Gumbel(torch.zeros_like(pi), torch.ones_like(pi))
    g = m.sample()
    gumbel_softmax = torch.softmax((torch.log(pi) + g)/temp, dim=-1) # (B, num_gaussians)

    # reparamaterize the gaussians
    eps = torch.randn_like(sigma)
    samples = mu + (eps * sigma)

    gumbel_weighted = torch.einsum('bgd,bg->bd', [samples, gumbel_softmax])
    return gumbel_weighted
    