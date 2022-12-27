from dataset import RedBlueSequence
from shared_latent.models.dreamer import RSSMCore
from torch.utils.data import DataLoader
from torch import Tensor
import torch
import torch.distributions as D
from functions import logavgexp, flatten_batch, unflatten_batch
from shared_latent.models.decoder import MixureDecoder
from torch.cuda.amp import GradScaler

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os
from datetime import datetime

from simulator.prisoner_env_variations import initialize_prisoner_environment
import numpy as np
import torch
from filtering.models.blue_state_nll_model_mixture import NLLModel
from simulator.prisoner_batch_wrapper import PrisonerBatchEnv
from fugitive_policies.heuristic import HeuristicPolicy
from simulator.prisoner_env_variations import initialize_prisoner_environment
from heatmap import generate_heatmap_img
import os
from filtering.utils.render_utils import combine_game_heatmap, save_video, plot_mog_heatmap
from filtering.model_consolidated.lstm_mixture_model import MixureLSTM
from filtering.model_consolidated.filtering_model import LSTMMixtureModel

# def compute_loss(rssm, decoder, red_obs, reset, red_locs, this_batch_size, iwae_samples = 1):
#  # B x deter_dim, B x stoch_dim*stoch_discrete
#     in_state = rssm.cell.init_state(this_batch_size)

#     # print(red_obs.shape)
#     priors, posts, samples, features, states, hidden_states = rssm.forward(
#                 red_obs,       # tensor(T, B, E)
#                 reset,       # tensor(T, B)
#                 in_state,# Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
#                 iwae_samples,#: int = 1,
#                 do_open_loop=False)

#     # T x B x 1 x features_dim -> T*B x 1 x features_dim
#     features_flattened, bd = flatten_batch(features)
#     decoder_output = decoder(features_flattened)
#     red_locs_flattened, bd_r = flatten_batch(red_locs)
#     red_locs_flattened = red_locs_flattened.to(device)
#     # decoder_output_fix = [unflatten_batch(element, bd) for element in decoder_output]
#     # print(bd)
#     # y = unflatten_batch(y, bd)
#     decoder_loss = decoder.mdn_negative_log_likelihood_loss(decoder_output, red_locs_flattened)
#     decoder_loss = torch.reshape(decoder_loss, (seq_len, this_batch_size))

#     d = rssm.zdistr
#     dprior = d(priors)
#     dpost = d(posts)
#     loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (T,B,I)
#     loss_model = -logavgexp(-loss_kl_exact, dim=2)

#     total_loss = loss_model + decoder_loss

#     return total_loss, loss_model, decoder_loss

def get_probability_grid(nn_output, true_location=None):
    pi, sigma, mu = nn_output
    pi = pi.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()
    mu = mu.detach().cpu().numpy()
    grid = plot_mog_heatmap(mu[0], sigma[0], pi[0])
    return grid

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

path = "shared_latent/videos/"

batch_size = 64 # B
seq_len = 25 # T

action_dim = 2
stoch_dim = 32
deter_dim = 64
stoch_discrete = 32
hidden_dim = 1000
gru_layers = 1
gru_type = 'gru'
layer_norm = True
num_gaussians = 4
summary_dir = "logs/shared_latent/"
render = True
seed = 1
env = initialize_prisoner_environment(variation=0,
                                    observation_step_type="Blue",
                                    seed=seed)
policy = HeuristicPolicy(env)

in_dim = env.observation_space.shape[0]
rssm = RSSMCore(embed_dim=in_dim,
                deter_dim=deter_dim,
                stoch_dim=stoch_dim,
                stoch_discrete=stoch_discrete,
                hidden_dim=hidden_dim,
                gru_layers=gru_layers,
                gru_type=gru_type,
                layer_norm=layer_norm).to(device)

decoder_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/blue/20220419-0400/summary/decoder_best.pth'
rssm_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/blue/20220419-0400/summary/rssm_best.pth'
# load the model
rssm.load_state_dict(torch.load(rssm_path))

features_dim = deter_dim + stoch_dim * (stoch_discrete or 1)
decoder = MixureDecoder(features_dim, output_dim=2, num_gaussians=num_gaussians).to(device)

decoder.load_state_dict(torch.load(decoder_path))

imgs = []

episode_return = 0.0
done = False

# Initialize empty observations
observation = env.reset()

i = 0
timestep = 0
in_state = rssm.cell.init_state(1)

while not done:
    i += 1
    fugitive_observation = env.get_fugitive_observation()
    action = policy.predict(fugitive_observation)
    observation, reward, done, _ = env.step(action[0])
    episode_return += reward

    demonstration_tensor = torch.from_numpy(observation).to(device).float().view(1, 1, -1)
    # print(demonstration_tensor.shape)
    reset = torch.full((1, 1), False).to(device)

    post, in_state = rssm.cell.forward(demonstration_tensor, reset ,in_state)
    features = rssm.to_feature(in_state[0], in_state[1])   # (T,BI,D+S)

    features_flattened, bd = flatten_batch(features)
    decoder_output = decoder(features_flattened)
    # print(decoder_output)
    print(i)

    # output = predict(demonstration_tensor)
    true_location = np.array(env.prisoner.location)
    # stats = module.get_statistics(output, env.get_prisoner_location())
    # print(stats['probability'])

    if render:
        # heatmap_img = module.render(output, true_location)
        grid = get_probability_grid(decoder_output, true_location)
        heatmap_img = generate_heatmap_img(grid, true_location=env.get_prisoner_location())
        game_img = env.render('Policy', show=False, fast=True)
        img = combine_game_heatmap(game_img, heatmap_img)
        imgs.append(img)
    timestep += 1
    if done:
        break

if render:
    save_video(imgs, os.path.dirname(path) + f"/heatmap_{seed}.mp4", fps=5)
    # save_video(imgs, f'temp/mog/test_{seed}.mp4', fps=5)