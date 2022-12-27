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

def test_stochastic_rollouts(model, obs, reset, red_locs, num_warmstart_timesteps, this_batch_size):
    """ 
    :param start_size number of steps to warmstart the model with
    
    
    """

    # start_obs = obs[:start_size]
    # start_reset = reset[:start_size]

    # hidden_state = model.warm_start_rssm(start_obs, start_reset, None, this_batch_size)

    # predict_obs = obs[start_size:]
    # predict_locs = red_locs[start_size:]
    # predict_resets = reset[start_size:]
    start_obs = obs[:num_warmstart_timesteps]
    start_reset = reset[:num_warmstart_timesteps]

    hidden_state = model.warm_start_rssm(start_obs, start_reset, None, this_batch_size)

    predict_obs = obs[num_warmstart_timesteps:]
    predict_locs = red_locs[num_warmstart_timesteps:]
    predict_resets = reset[num_warmstart_timesteps:]

    # _, decoder_loss = model.compute_loss(predict_obs, predict_resets, predict_locs, hidden_state, this_batch_size, do_open_loop=True)

    # _, _, _, log_prob = model.compute_loss(predict_obs, predict_resets, predict_locs, hidden_state, this_batch_size, do_open_loop=True)
    log_prob = model.get_log_probs(predict_obs, predict_resets, predict_locs, hidden_state, this_batch_size, do_open_loop=True)
    mses = model.get_mse(predict_obs, predict_resets, predict_locs, hidden_state, this_batch_size, do_open_loop=True)
    # print(mses.shape)
    return log_prob, mses

def collect_stats(model, conf, obs_type, prediction_length, device):
    # observation = torch.from_numpy(observation).to(device).view(1, 8, -1)
    # obs = observation.permute(1, 0, 2)
    # r = torch.full((8, 1), False).to(device)

    # in_state = model.warm_start_rssm(obs, r, None, 1)
    # reset = torch.full((10, 1), False).to(device)
    # mean, log_std = model.rollout_prior(in_state, reset, 1, 10)
    # start_size = 4
    # seq_len = prediction_length + start_size
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

    feature_names = ['time', 'prisoner_loc', 'search_party_detect', 'helicopter_detect', 'prev_action', 'hideout_loc']
    prediction_obs_dict = test_file['prediction_dict'].item()
    red_file = filter_dataset(test_file['red_observations'], prediction_obs_dict, feature_names, split_categorical=False)
    test_dataset = RedBlueSequenceEncoded(red_file, test_file['blue_observations'], test_file['red_locations'], test_file['dones'], 
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

def plot_stats(mean, std):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(mean.shape[0])
    fig, ax = plt.subplots()

    # print(mean)

    ax.errorbar(x, mean,
                # yerr=std/2,
                fmt='-o')

    ax.set_xlabel('Timesteps into the future')
    ax.set_ylabel('Log-likelihood')
    ax.set_title('Prediction Error with RSSM')
    # save the figure
    plt.savefig("error_blue.png")

def main(path, model_pth, prediction_length, render=False):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(path, 'config.yaml')
    import yaml
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    seq_len = config['seq_len']
    obs_type = config['obs_type']

    # if obs_type == "red":
    #     observation_step_type = "Prediction"
    # else:
    #     observation_step_type = "Blue"

    model = Dreamer(config)
    # model.load_state_dict(torch.load(os.path.join(path, '32.pth')))
    model.load_state_dict(torch.load(os.path.join(path, model_pth)))
    model.eval()
    nll_means, nll_std = collect_stats(model, config, obs_type, prediction_length, device)
    nll_means = nll_means.detach().cpu().numpy()
    nll_std = nll_std.detach().cpu().numpy()
    plot_stats(nll_means, nll_std)
    return nll_means, nll_std
 
if __name__ == "__main__":
    # red_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220501-1425/summary/'

    # blue_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/blue/20220502-1108/summary/'

    # new_red_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220511-1453/summary/'
    
    # new_red_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220515-2333/'
    # blue_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/blue/20220516-0831/'

    # nll_means, nll_stds = main(blue_path, '50.pth', prediction_length=60, render=True)
    
    blue_non_hierarchical = '/nethome/sye40/PrisonerEscape/logs/shared_latent/blue/20220516-1005'
    blue_single, blue_stds = main(blue_non_hierarchical, "8.pth", prediction_length=60, render=True)
    print(blue_single)