"""
Copyright (2022)
Georgia Tech Research Corporation (Sean Ye, Manisha Natarajan, Rohan Paleja, Letian Chen, Matthew Gombolay)
All rights reserved.
"""
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs):
        # obs = obs.to(self.device).float()
        z = self.encoder(obs)
        y = self.decoder(z)
        return y

    def compute_loss(self, obs, target, i=None):
        # obs = obs.to(self.device).float()
        target = target.to(self.device).float()
        out = self.forward(obs)
        return self.decoder.compute_loss(out, target)

    def eval_loss(self, obs, target, i=None):
        return self.compute_loss(obs, target, i)

    def get_stats(self, obs, target):
        target = target.to(self.device).float()
        out = self.forward(obs)
        return self.decoder.get_stats(out, target)

    def sample(self, obs):
        out = self.forward(obs)
        pi = out[0][0,0,:]
        sigma = out[1][0,0,:,:]
        mu = out[2][0,0,:,:]
        i = pi.multinomial(num_samples=1, replacement=True)
        selected_sigma = sigma[i,:]
        selected_mu = mu[i,:]
        loc = selected_mu + torch.randn_like(selected_mu) * selected_sigma
        return loc

    def sample_highest_pi(self, obs):
        out = self.forward(obs)
        pi = out[0][0,0,:]
        sigma = out[1][0,0,:,:]
        mu = out[2][0,0,:,:]
        i = torch.argmax(pi)   
        selected_sigma = sigma[i,:]
        selected_mu = mu[i,:]
        loc = selected_mu + torch.randn_like(mu) * selected_sigma
        return loc    

    @property
    def device(self):
        return next(self.parameters()).device