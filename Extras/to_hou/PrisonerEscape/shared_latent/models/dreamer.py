""" Combine the RSSM and Decoder into single pytorch module """
import torch
import torch.nn as nn
import torch.distributions as D

from shared_latent.utils import get_configs
from shared_latent.models.decoder import SingleGaussianDecoder, MixtureDensityDecoder, DenseNormalDecoder, SingleGaussianDecoderStd
from shared_latent.models.rssm import RSSMCore
from shared_latent.functions import logavgexp, flatten_batch, unflatten_batch


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Dreamer(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.rssm = RSSMCore(input_dim=conf["in_dim"],
                    embed_dim=conf["embed_dim"],
                    deter_dim=conf["deter_dim"],
                    stoch_dim=conf["stoch_dim"],
                    stoch_discrete=conf["stoch_discrete"],
                    hidden_dim=conf["hidden_dim"],
                    gru_layers=conf["gru_layers"],
                    gru_type=conf["gru_type"],
                    layer_norm=conf["layer_norm"]).to(device)

        features_dim = conf["deter_dim"] + conf["stoch_dim"] * (conf["stoch_discrete"] or 1)
        if conf["decoder"] == "SingleParam":
            self.decoder = SingleGaussianDecoder(features_dim, conf["output_dim"]).to(device)
        elif conf["decoder"] == "SingleStd":
            self.decoder = SingleGaussianDecoderStd(features_dim, conf["output_dim"]).to(device)
        elif conf["decoder"] == "Mixture":
            self.decoder = MixtureDensityDecoder(features_dim, output_dim=conf["output_dim"]).to(device)
        else:
            raise NotImplementedError

        self.reconstructor_str = conf["reconstruction"]
        if self.reconstructor_str == None:
            pass
        elif self.reconstructor_str == "Full":
            # self.reconstructor = DenseNormalDecoder(in_dim=features_dim, out_dim=conf["in_dim"]).to(device)
            self.reconstructor = SingleGaussianDecoder(input_dim=features_dim, hidden_dim=64, output_dim = conf["in_dim"]).to(device)
        else:
            raise NotImplementedError
    
    def rollout_prior(self, in_state, reset, this_batch_size, timesteps):
        """ Given a hidden state, return decoder output for n sequences into the future """
        if in_state is None:
            in_state = self.rssm.cell.init_state(this_batch_size)
        
        T = timesteps
        B = this_batch_size
        I = 1

        def expand(x):
            # (T,B,X) -> (T,BI,X)
            return x.unsqueeze(2).expand(T, B, I, -1).reshape(T, B * I, -1).to(torch.float32)

        reset_masks = expand(~reset.unsqueeze(2)).unbind(0)

        posts = []
        states_h = []
        samples = []
        (h, z) = in_state

        for i in range(T):
            post, (h, z) = self.rssm.cell.forward_prior(reset_masks[i], (h, z))  # open loop: post=prior
            posts.append(post)
            states_h.append(h)
            samples.append(z)

        states_h = torch.stack(states_h)    # (T,BI,D)
        samples = torch.stack(samples)      # (T,BI,S)
        features = self.rssm.to_feature(states_h, samples)   # (T,BI,D+S)

        features = features.reshape(T, B, I, -1)
        features_flattened, bd = flatten_batch(features)
        # decoder_output = self.decoder.forward(features_flattened)
        mean, log_std = self.decoder.get_mean_variance(features_flattened)

        return mean, log_std

    def warm_start_rssm(self, obs, reset, in_state, this_batch_size):
        """ Given a sequence of observations and resets, compute the hidden state """
        if in_state is None:
            in_state = self.rssm.cell.init_state(this_batch_size)
        
        _, _, _, _, _, hidden_states = self.rssm.forward(
                obs,       # tensor(T, B, E)
                reset,       # tensor(T, B)
                in_state,# Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                do_open_loop=False)
        return hidden_states

    def get_log_probs(self, obs, reset, red_locs, in_state, this_batch_size, do_open_loop=False):
        """ Compute the loss per batch using the rssm with the decoder head """
        if in_state is None:
            in_state = self.rssm.cell.init_state(this_batch_size)

        # print(red_obs.shape)
        priors, posts, samples, features, states, hidden_states = self.rssm.forward(
                    obs,       # tensor(T, B, E)
                    reset,       # tensor(T, B)
                    in_state,# Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                    iwae_samples =1,#: int = 1,
                    do_open_loop=do_open_loop)

        # T x B x 1 x features_dim -> T*B x 1 x features_dim
        features_flattened, bd = flatten_batch(features)
        red_locs_flattened, bd_r = flatten_batch(red_locs)
        red_locs_flattened = red_locs_flattened.to(device)
        # decoder_output = self.decoder.forward(features)

        if self.conf["decoder"] is not None:
            log_prob = self.decoder.compute_log_prob(features_flattened, red_locs_flattened)
            log_loss = unflatten_batch(log_prob, bd).squeeze()
        return log_loss

    def get_mse(self, obs, reset, red_locs, in_state, this_batch_size, do_open_loop=False):
        """ Compute the mse between the mean and the red locs """
        if in_state is None:
            in_state = self.rssm.cell.init_state(this_batch_size)

        # print(red_obs.shape)
        priors, posts, samples, features, states, hidden_states = self.rssm.forward(
                    obs,       # tensor(T, B, E)
                    reset,       # tensor(T, B)
                    in_state,# Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                    iwae_samples =1,#: int = 1,
                    do_open_loop=do_open_loop)

        # T x B x 1 x features_dim -> T*B x 1 x features_dim
        features_flattened, bd = flatten_batch(features)
        red_locs_flattened, bd_r = flatten_batch(red_locs)
        red_locs_flattened = red_locs_flattened.to(device)
        # decoder_output = self.decoder.forward(features)

        if self.conf["decoder"] is not None:
            mean, std = self.decoder.forward(features_flattened)
            mse = (2428*mean - 2428*red_locs_flattened)**2
            mse = unflatten_batch(mse, bd).squeeze()
        return mse


    def compute_loss(self, obs, reset, red_locs, in_state, this_batch_size, do_open_loop=False):
        """ Compute the loss per batch using the rssm with the decoder head """
        if in_state is None:
            in_state = self.rssm.cell.init_state(this_batch_size)

        # print(red_obs.shape)
        priors, posts, samples, features, states, hidden_states = self.rssm.forward(
                    obs,       # tensor(T, B, E)
                    reset,       # tensor(T, B)
                    in_state,# Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                    iwae_samples =1,#: int = 1,
                    do_open_loop=do_open_loop)

        # T x B x 1 x features_dim -> T*B x 1 x features_dim
        features_flattened, bd = flatten_batch(features)
        red_locs_flattened, bd_r = flatten_batch(red_locs)
        red_locs_flattened = red_locs_flattened.to(device)
        # decoder_output = self.decoder.forward(features)

        if self.conf["decoder"] is not None:
            decoder_loss = self.decoder.compute_loss(features_flattened, red_locs_flattened)
        # log_loss = unflatten_batch(logprob, bd).squeeze()

        d = self.rssm.zdistr
        dprior = d(priors)
        dpost = d(posts)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)# (T,B,I)
        if self.conf["kl_balancing"] == 0:
            loss_kl = loss_kl_exact
        else:
            loss_kl_postgrad = D.kl.kl_divergence(dpost, d(priors.detach()))
            loss_kl_priograd = D.kl.kl_divergence(d(posts.detach()), dprior)
            loss_kl = (1 - self.conf["kl_balancing"]) * loss_kl_postgrad + self.conf["kl_balancing"] * loss_kl_priograd
        
        if self.reconstructor_str is not None:
            obs_last = obs[:, -1, :, :] 
            obs_flattened, bd = flatten_batch(torch.tensor(obs_last).float().to(device))
            # outer_dim = int(obs_flattened.shape[0] / features_flattened.shape[0])
            # print(outer_dim)
            # obs_flattened = obs_flattened[::outer_dim, :]
            reconstruction_loss = self.reconstructor.compute_loss(features_flattened, obs_flattened)
        else:
            reconstruction_loss = torch.zeros_like(loss_kl)

        # total_loss = loss_kl.squeeze() - log_loss
        return loss_kl, decoder_loss, reconstruction_loss
        
if __name__ == "__main__":
    conf, config_path = get_configs()
    d = Dreamer(conf)