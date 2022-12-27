from dataset import RedBlueSequence, RedBlueSequenceSkip
from shared_latent.models.rssm import RSSMCore
from torch.utils.data import DataLoader
from torch import Tensor
import torch
import torch.distributions as D
from functions import logavgexp, flatten_batch, unflatten_batch
from shared_latent.models.decoder import MixureDecoder, DenseNormalDecoder, SingleGaussianDecoder
from torch.cuda.amp import GradScaler

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os
from datetime import datetime
from torch.distributions import Normal

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.
    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

def log_prob(distribution, actions: torch.Tensor) -> torch.Tensor:
    """
    Get the log probabilities of actions according to the distribution.
    :param distribution: (Torch.distributions type) Calculate prob w.r.t distribution
    :param actions: Actions whose probability is computed
    :return:
    """
    logprob = distribution.log_prob(actions)
    return sum_independent_dims(logprob)

def compute_loss(rssm, decoder, red_obs, reset, red_locs, this_batch_size, kl_balancing, iwae_samples = 1):
 # B x deter_dim, B x stoch_dim*stoch_discrete
    in_state = rssm.cell.init_state(this_batch_size)

    # print(red_obs.shape)
    priors, posts, samples, features, states, hidden_states = rssm.forward(
                red_obs,       # tensor(T, B, E)
                reset,       # tensor(T, B)
                in_state,# Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                iwae_samples,#: int = 1,
                do_open_loop=False)

    # T x B x 1 x features_dim -> T*B x 1 x features_dim
    features_flattened, bd = flatten_batch(features)
    decoder_output = decoder(features_flattened)
    red_locs_flattened, bd_r = flatten_batch(red_locs)
    red_locs_flattened = red_locs_flattened.to(device)

    std = torch.ones_like(decoder_output) * decoder.log_std.exp()
    # print(decoder_output, red_locs_flattened)
    distribution = Normal(decoder_output, std)
    logprob = log_prob(distribution, red_locs_flattened)

    log_loss = unflatten_batch(logprob, bd).squeeze()
    # prob_true_act = torch.exp(logprob).mean()
    logprob = logprob.mean()
    decoder_loss = -logprob

    # decoder_output_fix = [unflatten_batch(element, bd) for element in decoder_output]
    # print(bd)
    # y = unflatten_batch(y, bd)
    # decoder_loss = decoder.mdn_negative_log_likelihood_loss(decoder_output, red_locs_flattened)
    # decoder_loss = torch.reshape(decoder_loss, (seq_len, this_batch_size))

    d = rssm.zdistr
    dprior = d(priors)
    dpost = d(posts)
    loss_kl_exact = D.kl.kl_divergence(dpost, dprior)# (T,B,I)
    if kl_balancing == 0:
        loss_kl = loss_kl_exact
    else:
        loss_kl_postgrad = D.kl.kl_divergence(dpost, d(priors.detach()))
        loss_kl_priograd = D.kl.kl_divergence(d(posts.detach()), dprior)
        loss_kl = (1 - kl_balancing) * loss_kl_postgrad + kl_balancing * loss_kl_priograd
    
    total_loss = loss_kl.squeeze() - log_loss
    # total_loss = -logavgexp(-loss_model, dim=2)

    # total_loss = loss_model

    return total_loss, loss_kl.mean(), decoder_loss

def test_stochastic_rollouts(rssm, decoder, obs, reset, red_locs, this_batch_size):
    """ Given the rssm and a decoder model, check our prediction a few steps into the future """
    in_state = rssm.cell.init_state(this_batch_size)

    start_obs = obs[:4]
    start_reset = reset[:4]

    # give the model a few steps from the past
    _, _, _, _, _, hidden_state = rssm.forward(
                start_obs,       # tensor(T, B, E)
                start_reset,       # tensor(T, B)
                in_state,# Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                iwae_samples=1,#: int = 1,
                do_open_loop=False)

    predict_obs = obs[4:]
    predict_locs = red_locs[4:]
    predict_resets = reset[4:]
    # open loop the rest of the features
    priors, posts, samples, features, states, hidden_state = rssm.forward(
                predict_obs,       # tensor(T, B, E)
                predict_resets,       # tensor(T, B)
                hidden_state, # Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                iwae_samples = 1,#: int = 1,
                do_open_loop=True)

    # T x B x 1 x features_dim -> T*B x 1 x features_dim
    features_flattened, bd = flatten_batch(features)
    decoder_output = decoder(features_flattened)
    red_locs_flattened, bd_r = flatten_batch(predict_locs)
    red_locs_flattened = red_locs_flattened.to(device)

    std = torch.ones_like(decoder_output) * decoder.log_std.exp()
    distribution = Normal(decoder_output, std)
    logprob = log_prob(distribution, red_locs_flattened)

    # log_loss = unflatten_batch(logprob, bd).squeeze()
    logprob = logprob.mean()
    decoder_loss = -logprob

    return decoder_loss
    

def train():
    np_file = np.load("/nethome/sye40/PrisonerEscape/shared_latent/dataset/map_0_run_300_eps_0.npz", allow_pickle=True)
    test_file = np.load("/nethome/sye40/PrisonerEscape/shared_latent/dataset/map_0_run_100_eps_0.npz", allow_pickle=True)

    total_resets = np.sum(np_file['dones'])
    print(total_resets)

    batch_size = 64 # B

    seq_len = 16 # T

    # train_length = 124
    # red_blue_dataset = RedBlueSequence(np_file['red_observations'][:train_length], np_file['blue_observations'][:train_length], np_file['red_locations'][:train_length], np_file['dones'][:train_length], seq_len)
    red_blue_dataset = RedBlueSequence(np_file['red_observations'], np_file['blue_observations'], np_file['red_locations'], np_file['dones'], seq_len)
    # skip_step = 5
    # red_blue_dataset = RedBlueSequenceSkip(np_file['red_observations'], np_file['blue_observations'], np_file['red_locations'], np_file['dones'], skip_step, seq_len)
    train_dataloader = DataLoader(red_blue_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = RedBlueSequence(test_file['red_observations'], test_file['blue_observations'], test_file['red_locations'], test_file['dones'], seq_len)
    # test_dataset = RedBlueSequenceSkip(test_file['red_observations'], test_file['blue_observations'], test_file['red_locations'], test_file['dones'], skip_step, seq_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # action_dim = 2
    embed_dim = 32
    stoch_dim = 32
    deter_dim = 64
    loss_kl_weight = 1
    stoch_discrete = 0
    # stoch_discrete = 32
    hidden_dim = 1000       
    gru_layers = 1
    gru_type = 'gru'
    layer_norm = False
    num_gaussians = 1
    kl_balancing = 0.8
    obs_type = "red"

    summary_dir = f"logs/shared_latent/{obs_type}"

    # Initialize for writing on tensorboard
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(summary_dir, str(time))

    summary_dir = os.path.join(log_dir, 'summary')
    writer = SummaryWriter(log_dir=summary_dir)

    if obs_type == "red":
        in_dim =  red_blue_dataset[0][0].shape[-1]
    else:
        in_dim =  red_blue_dataset[0][1].shape[-1]

    rssm = RSSMCore(input_dim=in_dim,
                    embed_dim=embed_dim,
                    deter_dim=deter_dim,
                    stoch_dim=stoch_dim,
                    stoch_discrete=stoch_discrete,
                    hidden_dim=hidden_dim,
                    gru_layers=gru_layers,
                    gru_type=gru_type,
                    layer_norm=layer_norm).to(device)

    features_dim = deter_dim + stoch_dim * (stoch_discrete or 1)
    # decoder = DenseNormalDecoder(features_dim, output_dim=2).to(device)
    decoder = SingleGaussianDecoder(features_dim, output_dim=2).to(device)
    decoder_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220427-1508/summary/decoder_best.pth'
    decoder.load_state_dict(torch.load(decoder_path))
    for param in decoder.parameters():
        param.requires_grad = False

    # opt_1 = torch.optim.Adam(rssm.parameters(), lr=0.001)
    # opt_2 = torch.optim.Adam(decoder.parameters(), lr=0.001)

    opt = torch.optim.Adam([{'params': rssm.parameters()}, {'params': decoder.parameters()}], lr=3e-4)

    # optimizers = [opt_1, opt_2]
    scaler = GradScaler(enabled=True)

    iwae_samples = 1
    i = 0
    best_test_loss = float('inf')
    for epoch in tqdm(range(250)):
        epoch_loss = 0
        for data in train_dataloader:
            i += 1
            red_obs, blue_obs, red_locs, reset = data
            this_batch_size = red_obs.shape[0]

            blue_obs = blue_obs.permute(1, 0, 2).to(device)
            red_obs = red_obs.permute(1, 0, 2).to(device)
            reset = reset.permute(1, 0).to(device)
            red_locs = red_locs.permute(1, 0, 2).to(device)

            if obs_type == "red":
                total_loss, loss_model, decoder_loss = compute_loss(rssm, decoder, red_obs, reset, red_locs, this_batch_size, kl_balancing)
            else:
                total_loss, loss_model, decoder_loss = compute_loss(rssm, decoder, blue_obs, reset, red_locs, this_batch_size, kl_balancing)

            writer.add_scalar('loss/train/decoder', decoder_loss.mean(), i)
            writer.add_scalar('loss/train/model', loss_model.mean(), i)

            # for opt in optimizers:
            opt.zero_grad()
        
            # loss = total_loss.mean(dim=0).mean()
            loss = loss_kl_weight * loss_model.mean() + torch.clamp(decoder_loss.mean(), min=-3)
            loss.backward()
            # scaler.scale(loss).backward()
            
            # for opt in optimizers:
            opt.step()
            # scaler.unscale_(opt)
            # scaler.step(opt)
            # scaler.update()

            # print(loss)
            epoch_loss += loss

        ##### Test losses ##### 
        with torch.no_grad():
            test_decoder_losses = 0
            test_loss_model = 0
            test_total_loss = 0
            test_prior_decoder_loss = 0
            j = 0
            for data in test_dataloader:
                j += 1
                red_obs, blue_obs, red_locs, reset = data
                this_batch_size = red_obs.shape[0]

                blue_obs = blue_obs.permute(1, 0, 2).to(device)
                red_obs = red_obs.permute(1, 0, 2).to(device)
                reset = reset.permute(1, 0).to(device)
                red_locs = red_locs.permute(1, 0, 2).to(device)

                if obs_type == "red":
                    total_loss, loss_model, decoder_loss = compute_loss(rssm, decoder, red_obs, reset, red_locs, this_batch_size, kl_balancing)
                    prior_stochastic_decoder_loss = test_stochastic_rollouts(rssm, decoder, red_obs, reset, red_locs, this_batch_size)
                else:
                    total_loss, loss_model, decoder_loss = compute_loss(rssm, decoder, blue_obs, reset, red_locs, this_batch_size, kl_balancing)
                    prior_stochastic_decoder_loss = test_stochastic_rollouts(rssm, decoder, blue_obs, reset, red_locs, this_batch_size)
                
                test_decoder_losses += decoder_loss.mean()
                test_loss_model += loss_model.mean()
                test_total_loss += total_loss.mean()
                test_prior_decoder_loss += prior_stochastic_decoder_loss.mean()
            
                writer.add_scalar('loss/test/prior_decoder', prior_stochastic_decoder_loss.mean(), j)
                writer.add_scalar('loss/test/decoder', decoder_loss.mean(), j)
                writer.add_scalar('loss/test/model', loss_model.mean(), j)
            
            overall_test_loss = test_total_loss / j
            writer.add_scalar('loss/test/overall', overall_test_loss, epoch)
            writer.add_scalar('loss/test/prior_decoder_epoch', test_prior_decoder_loss / j, epoch)
            if overall_test_loss < best_test_loss:
                best_test_loss = overall_test_loss
                torch.save(rssm.state_dict(), summary_dir + '/rssm_best.pth')
                torch.save(decoder.state_dict(), summary_dir + '/decoder_best.pth')


        writer.add_scalar('loss/train/overall', epoch_loss/batch_size, epoch)

if __name__ == "__main__":
    train()