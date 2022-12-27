import torch
from models.shared_latent import SharedLatent
# from stats_nll import collect_stats
from stats_nll import test_stochastic_rollouts

import numpy as np
import torch
from fugitive_policies.heuristic import HeuristicPolicy
from heatmap import generate_heatmap_img
# from utils import plot_gaussian_heatmap, save_video
import os
from shared_latent.models.dreamer import Dreamer
from shared_latent.dataset import RedBlueSequence, RedBlueSequenceSkip, RedBlueSequenceEncoded
from torch.utils.data import DataLoader
from shared_latent.filter_dataset import filter_dataset
import yaml

def collect_stats(conf, model, obs_type, prediction_length, device):
    batch_size = 256
    num_warmup = 16
    inner_seq = conf["inner_seq"] # inner sequence length
    num_outer_warmups = max(num_warmup // inner_seq, 1)
    num_outer_predict = max(prediction_length // inner_seq, 1)
    total_outer = num_outer_warmups + num_outer_predict

    test_file = np.load(conf["test_dataset_path"], allow_pickle=True)

    # test_dataset = RedBlueSequence(test_file['red_observations'], test_file['blue_observations'], test_file['red_locations'], test_file['dones'], seq_len)
    # test_dataset = RedBlueSequenceSkip(test_file['red_observations'], test_file['blue_observations'], test_file['red_locations'], test_file['dones'], skip_step, seq_len)
    # test_dataset = RedBlueSequenceEncoded(test_file['red_observations'], 
    #                                     test_file['blue_observations'], 
    #                                     test_file['red_locations'], 
    #                                     test_file['dones'], 
    #                     total_outer, inner_seq)

    # feature_names = ['time', 'prisoner_loc', 'search_party_detect', 'helicopter_detect', 'prev_action', 'hideout_loc']
    # prediction_obs_dict = test_file['prediction_dict'].item()
    # red_file = filter_dataset(test_file['red_observations'], prediction_obs_dict, feature_names, split_categorical=False)
    # test_file = np.load(conf["test_dataset_path"], allow_pickle=True)
    test_dataset = RedBlueSequenceEncoded(test_file['red_observations'], test_file['blue_observations'], test_file['red_locations'], test_file['dones'], 
                                total_outer, inner_seq)

    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    log_probs = []
    mses = []

    j = 0
    for data in test_dataloader:
        j += 1
        red_obs, blue_obs, red_locs, reset = data
        this_batch_size = red_obs.shape[0]

        blue_obs = blue_obs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
        red_obs = red_obs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
        reset = reset.permute(1, 2, 0).to(device) #BxOxN -> OxNxB
        red_locs = red_locs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
        red_locs = red_locs[:, -1, :, :] # grab the last location in the inner dimension

        if obs_type == "red":
            log_prob, mse = test_stochastic_rollouts(model, red_obs, reset, red_locs, num_outer_warmups, this_batch_size)
        else:
            log_prob, mse = test_stochastic_rollouts(model, blue_obs, reset, red_locs, num_outer_warmups, this_batch_size)

        log_probs.append(log_prob.detach())
        mses.append(mse.detach())

    mses = torch.cat(mses, dim=1)
    log_probs = torch.concat(log_probs, dim=1)
    # print(log_probs.shape)

    mses = torch.mean(mses, dim=1)
    rmses = (torch.sum(mses, dim=1)/2) ** 0.5
    print(rmses)

    log_likelihood = torch.mean(log_probs, dim=1)
    std = torch.std(log_probs, dim=1)
    # print(torch.max(log_probs))
    
    return log_likelihood, std



if __name__ == "__main__":
    conf = "/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220512-1348/config.yaml"
    path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/combined/20220523-1329/0_whole.pth'
    model_shared = torch.load(path)
    
    # load config
    with open(conf, 'r') as stream:
        config = yaml.safe_load(stream)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Dreamer(config)
    model.rssm = model_shared.red_rssm
    model.decoder = model_shared.decoder
    ll, std = collect_stats(config, model, "red", 60, device)
    print(ll)

    model.rssm = model_shared.blue_rssm
    model.decoder = model_shared.decoder
    ll, std = collect_stats(config, model, "blue", 60, device)
    print(ll, std)
    
    # print(model.red_rssm.state_dict())
