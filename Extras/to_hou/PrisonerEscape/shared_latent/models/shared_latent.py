""" This model takes two RSSM models with two different latent spaces and combines them. """
from shared_latent.models.dreamer import RSSMCore
import torch
import torch.nn as nn
import torch.distributions as D

from shared_latent.utils import get_configs
from shared_latent.models.decoder import SingleGaussianDecoder
from shared_latent.models.rssm import RSSMCore
from shared_latent.functions import logavgexp, flatten_batch, unflatten_batch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class SharedLatent(nn.Module):
    def __init__(self, red_rssm, blue_rssm, decoder):
        super().__init__()
        self.red_rssm = red_rssm
        self.blue_rssm = blue_rssm
        self.decoder = decoder
        self.kl_balancing = 0.8

    def calculate_kl_balancing(self, priors, posts, rssm):
        d = rssm.zdistr
        dprior = d(priors)
        dpost = d(posts)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)# (T,B,I)
        if self.kl_balancing == 0:
            loss_kl = loss_kl_exact
        else:
            loss_kl_postgrad = D.kl.kl_divergence(dpost, d(priors.detach()))
            loss_kl_priograd = D.kl.kl_divergence(d(posts.detach()), dprior)
            loss_kl = (1 - self.kl_balancing) * loss_kl_postgrad + self.kl_balancing * loss_kl_priograd
        return loss_kl
        
    def compute_loss(self, red_obs, blue_obs, reset, red_locs, in_state, this_batch_size, writer, do_open_loop=False):
        """ The RSSM model should have the same decoder for both the red RSSM and blue RSSM
        
        An additional loss should be included to bring the kl between the two RSSMs down
        """    

        if in_state is None:
            in_state = self.red_rssm.cell.init_state(this_batch_size)

        # print(red_obs.shape)
        red_priors, red_posts, _, red_features, red_states, red_hidden_states = self.red_rssm.forward(
                    red_obs,       # tensor(T, B, E)
                    reset,       # tensor(T, B)
                    in_state,# Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                    iwae_samples =1,#: int = 1,
                    do_open_loop=do_open_loop)

        blue_priors, blue_posts, _, blue_features, blue_states, blue_hidden_states = self.blue_rssm.forward(
            blue_obs,       # tensor(T, B, E)
            reset,       # tensor(T, B)
            in_state,# Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
            iwae_samples =1,#: int = 1,
            do_open_loop=do_open_loop)

        # T x B x 1 x features_dim -> T*B x 1 x features_dim
        red_features_flattened, bd = flatten_batch(red_features)
        red_locs_flattened, bd_r = flatten_batch(red_locs)
        red_locs_flattened = red_locs_flattened.to(device)
        # decoder_output = self.decoder.forward(features)

        blue_features_flattened, bd = flatten_batch(blue_features)

        red_decoder_loss = self.decoder.compute_loss(red_features_flattened, red_locs_flattened)
        blue_decoder_loss = self.decoder.compute_loss(blue_features_flattened, red_locs_flattened)

        red_kl_loss = self.calculate_kl_balancing(red_priors, red_posts, self.red_rssm)
        blue_kl_loss = self.calculate_kl_balancing(blue_priors, blue_posts, self.blue_rssm)

        # bring priors close together
        red_dprior = self.red_rssm.zdistr(red_priors)
        blue_dprior = self.blue_rssm.zdistr(blue_priors)
        # shared = D.kl.kl_divergence(red_dprior, blue_dprior)

        # print(red_priors.shape)
        # print(type(red_dprior))
        # print(blue_dprior.shape)
        cos = nn.CosineSimilarity(dim=3, eps=1e-6)
        shared = cos(red_priors, blue_priors)

        return shared.mean(), red_kl_loss.mean(), blue_kl_loss.mean(), red_decoder_loss.mean(), blue_decoder_loss.mean()