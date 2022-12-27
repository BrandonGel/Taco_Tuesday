from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from shared_latent.functions import diag_normal, NoNorm
import shared_latent.models.rnn as rnn

from shared_latent.models.encoder import GRUStepEncoder 

class RSSMCore(nn.Module):

    def __init__(self, input_dim, embed_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.cell = RSSMCell(input_dim, embed_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm)

    # T - time dimension
    # B - batch dimension
    # A - action dimension
    # S - stochastic dimension
    # D - deterministic dimension
    def forward(self,
                embed: Tensor,       # tensor(T, B, E)
                reset: Tensor,       # tensor(T, B)
                in_state: Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                iwae_samples: int = 1,
                do_open_loop=False,
                ):

        #embed shape is (O, N, B, D)
        # T, B = embed.shape[:2]
        O, N, B, D = embed.shape
        I = iwae_samples

        # Multiply batch dimension by I samples
        def expand(x):
            # (T,B,X) -> (T,BI,X)
            # (O,N,B,X) -> (O,N,BI,X)
            return x.unsqueeze(3).expand(O, N, B, I, -1).reshape(O,N, B * I, -1).to(torch.float32)

        # (O,N,B,D) => List[(N,BI...)]
        embeds = expand(embed).unbind(0)     # (T,B,...) => List[(BI,...)]

        reset = torch.all(~reset, dim=1) # We set a block to true if at least one of the samples is reset
        reset_masks = reset.unsqueeze(2).unbind(0)

        priors = []
        posts = []
        states_h = []
        samples = []
        (h, z) = in_state

        for i in range(O):
            if not do_open_loop:
                post, (h, z) = self.cell.forward(embeds[i], reset_masks[i], (h, z))
            else:
                post, (h, z) = self.cell.forward_prior(reset_masks[i], (h, z))  # open loop: post=prior
            posts.append(post)
            states_h.append(h)
            samples.append(z)

        posts = torch.stack(posts)          # (T,BI,2S)
        states_h = torch.stack(states_h)    # (T,BI,D)
        samples = torch.stack(samples)      # (T,BI,S)
        priors = self.cell.batch_prior(states_h)  # (T,BI,2S)
        features = self.to_feature(states_h, samples)   # (T,BI,D+S)

        posts = posts.reshape(O, B, I, -1)  # (T,BI,X) => (T,B,I,X)
        states_h = states_h.reshape(O, B, I, -1)
        samples = samples.reshape(O, B, I, -1)
        priors = priors.reshape(O, B, I, -1)
        states = (states_h, samples)
        features = features.reshape(O, B, I, -1)

        return (
            priors,                      # tensor(T,B,I,2S)
            posts,                       # tensor(T,B,I,2S)
            samples,                     # tensor(T,B,I,S)
            features,                    # tensor(T,B,I,D+S)
            states,
            (h.detach(), z.detach()),
        )

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def to_feature(self, h: Tensor, z: Tensor) -> Tensor:
        return torch.cat((h, z), -1)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        h, _ = features.split([self.cell.deter_dim, z.shape[-1]], -1)
        return self.to_feature(h, z)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)


class RSSMCell(nn.Module):

    def __init__(self, input_dim, embed_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.deter_dim = deter_dim
        # self.embed_layer = nn.Linear(input_dim, embed_dim)
        self.embed_layer = GRUStepEncoder(input_dim, embed_dim)

        norm = nn.LayerNorm if layer_norm else NoNorm

        self.z_mlp = nn.Linear(stoch_dim * (stoch_discrete or 1), hidden_dim)
        self.in_norm = norm(hidden_dim, eps=1e-3)

        self.gru = rnn.GRUCellStack(hidden_dim, deter_dim, gru_layers, gru_type)

        self.prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_norm = norm(hidden_dim, eps=1e-3)
        self.prior_mlp = nn.Linear(hidden_dim, stoch_dim * (stoch_discrete or 2))

        self.post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.post_mlp_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_norm = norm(hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(hidden_dim, stoch_dim * (stoch_discrete or 2))

    def init_state(self, batch_size):
        device = next(self.gru.parameters()).device
        return (
            torch.zeros((batch_size, self.deter_dim), device=device),
            torch.zeros((batch_size, self.stoch_dim * (self.stoch_discrete or 1)), device=device),
        )

    def forward(self,
                embed: Tensor,                    # tensor(B,E)
                reset_mask: Tensor,               # tensor(B,1)
                in_state: Tuple[Tensor, Tensor],
                ) -> Tuple[Tensor,
                           Tuple[Tensor, Tensor]]:

        # embed shape is (N,B,E)
        embed = self.embed_layer(embed)
        in_h, in_z = in_state
        in_h = in_h * reset_mask
        in_z = in_z * reset_mask
        # B = action.shape[0]
        B = embed.shape[0]

        x = self.z_mlp(in_z) # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h) # (B, D)

        # embed = embed.to(torch.float32)
        # a = self.post_mlp_h(h)
        # b = self.post_mlp_e(embed)

        x = self.post_mlp_h(h) + self.post_mlp_e(embed)
        # x = a + b

        # x = torch.concat((a, b), dim=-1)
        x = self.post_norm(x)
        post_in = F.elu(x)
        post = self.post_mlp(post_in)                                    # (B, S*S)
        post_distr = self.zdistr(post)
        sample = post_distr.rsample().reshape(B, -1)

        return (
            post,                         # tensor(B, 2*S)
            (h, sample),                  # tensor(B, D+S+G)
        )

    def forward_prior(self,
                      reset_mask: Optional[Tensor],               # tensor(B,1)
                      in_state: Tuple[Tensor, Tensor],  # tensor(B,D+S)
                      ) -> Tuple[Tensor,
                                 Tuple[Tensor, Tensor]]:

        in_h, in_z = in_state
        if reset_mask is not None:
            in_h = in_h * reset_mask
            in_z = in_z * reset_mask

        B = in_state[0].shape[0]

        x = self.z_mlp(in_z) # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)  # (B, D)

        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)          # (B,2S)
        prior_distr = self.zdistr(prior)
        sample = prior_distr.rsample().reshape(B, -1)

        return (
            prior,                        # (B,2S)
            (h, sample),                  # (B,D+S)
        )

    def batch_prior(self,
                    h: Tensor,     # tensor(T, B, D)
                    ) -> Tensor:
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)  # tensor(B,2S)
        return prior

    def zdistr(self, pp: Tensor) -> D.Distribution:
        # pp = post or prior
        if self.stoch_discrete:
            logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_discrete))
            distr = D.OneHotCategoricalStraightThrough(logits=logits.float())  # NOTE: .float() needed to force float32 on AMP
            distr = D.independent.Independent(distr, 1)  # This makes d.entropy() and d.kl() sum over stoch_dim
            return distr
        else:
            return diag_normal(pp)