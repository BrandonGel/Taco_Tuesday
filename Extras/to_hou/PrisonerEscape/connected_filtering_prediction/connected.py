import torch
import torch.nn as nn
import os
import time
import yaml
from models.configure_model import configure_model
from datasets.load_datasets import load_datasets
from models.decoder import mdn_negative_log_likelihood
import math
import numpy as np
from datasets.dataset_old import RedBlueDataset, RedBlueSequence, RedBlueSequenceOriginal
from datasets.multi_head_dataset_old import MultiHeadDataset
from torch.utils.data import DataLoader

from utils import sample_n_times

### We create a module where the filtering and prediction pipelines are connected
# We assume Filtering module has a gaussian head - we then sample from this gaussian and use it as the input 
# to the prediction module.

# We sample from this module n times and combine the gaussians of the output together

class ConnectedFilteringPrediction(nn.Module):
    def __init__(self, filtering_module, prediction_module, prediction_dict):
        super(ConnectedFilteringPrediction, self).__init__()
        self.filtering_module = filtering_module
        self.prediction_module = prediction_module
        self.prediction_dict = prediction_dict
    
    def forward(self, blue_obs, prediction_obs):
        pi, mu, sigma = self.filtering_module(blue_obs) # each are (B, n_gaussians, 2)
        
        test_mu_one = mu[:, 0, :]
        # print(test_mu_one.shape)
        prediction_obs = self.transform_filter_output_to_prediction_input(prediction_obs, test_mu_one)
        # print(prediction_obs.shape)
        # print(prediction_obs.shape)
        predictions = self.prediction_module(prediction_obs)
        return predictions

    def get_stats(self, blue_obs, prediction_obs, true_locations):
        B = prediction_obs.shape[0]
        n = 5000

        # Get Mixture of Gaussians from filtering module
        pi, mu, sigma = self.filtering_module(blue_obs)

        # Sample n samples from the mixture
        locs = sample_n_times(pi, mu, sigma, n, device=device)
        prediction_obs = prediction_obs.repeat(1, n, 1)
        
        # Insert the sampled prisoner locations into real observations
        loc = self.prediction_dict['prisoner_loc']
        prediction_obs[:,:,loc[0]:loc[1]] = locs
        prediction_obs = prediction_obs.view(B*n, prediction_obs.size(2)).unsqueeze(1)

        # Get the gaussian from the prediction output
        mean, std = self.prediction_module(prediction_obs)
        mean = mean.view(B, n, -1)
        std = std.view(B, n, -1)

        t_l = true_locations.unsqueeze(1).expand_as(std)
        neg_logprob = -torch.log(std) - (math.log(2 * math.pi) / 2) - \
            ((t_l - mean) / std)**2 / 2 # (B, n, 24)

        neg_logprob = neg_logprob.view(B, n, 12, 2).permute(0, 2, 1, 3) # (B, 12, n, 2)
        test = torch.sum(neg_logprob, 3)
        test = torch.log(torch.ones_like(test) * 1/n) + test # (B, 12, n)
        nll = -torch.logsumexp(test, dim=2)
        # print(nll.mean(dim=0))
        return nll.mean(dim=0)

        # final try - let's just split original and location tensor
        # This is equivalent to the above code, used to check that implementation was correct
        # means = torch.tensor_split(mean, 12, dim=-1)
        # stds = torch.tensor_split(std, 12, dim=-1)
        # true_locs = torch.tensor_split(true_locations, 12, dim=-1)
        # losses = []
        # for m, s, t in zip(means, stds, true_locs):
        #     pi = torch.ones((B, n), device=device) * (1/n) # (B, n)
        #     loss = mdn_negative_log_likelihood(pi, m, s, t)
        #     losses.append(loss.mean().item())
        # print(losses)

    def transform_filter_output_to_prediction_input(self, base_prediction_observation, locations):
        """ The filtering output gives us a location in which we have to produce a prediction input 
        
        base_prediction_observation (torch.tensor) We assume that this is of shape (B, 1, n)
        locations (torch.tensor)

        """
        loc = self.prediction_dict['prisoner_loc']
        base_prediction_observation = base_prediction_observation.squeeze()
        # print(base_prediction_observation.shape)
        base_prediction_observation[:,loc[0]:loc[1]] = locations
        return base_prediction_observation.unsqueeze(1)

def initiate_connected_dataset(config, batch_size):
    """ May be a little hacky, but we replace the true position from the red observations by the predicted location 
    Everything in the prediction input can be computed from the blue observation
    """
    seq_len = config["seq_len"]
    test_path = "/nethome/sye40/PrisonerEscape/shared_latent/map_0_run_1000_eps_0_norm.npz"
    test_np = np.load(test_path, allow_pickle=True)
    test_dataset = MultiHeadDataset(test_np['red_observations'], 
                test_np['blue_observations'], 
                test_np['red_locations'], 
                test_np['dones'], 
                seq_len, 
                config["num_heads"],
                config["step_length"], 
                "both")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_dataloader, test_np["blue_dict"].item(), test_np["prediction_dict"].item()

if __name__ == "__main__":
    filtering_path = '/nethome/sye40/PrisonerEscape/logs/filtering/normal/mean_mixture_4/20220521-1334/epoch_65.pth'
    prediction_path = '/nethome/sye40/PrisonerClean/logs/prediction/20220521-1141'

    config_path = "configs/filtering.yaml"
    with open(config_path, 'r') as stream:
        blue_config = yaml.safe_load(stream)
    batch_size = blue_config["batch_size"]

    prediction_config_path = os.path.join(prediction_path, "config.yaml")
    with open(prediction_config_path, 'r') as stream:
        red_config = yaml.safe_load(stream)

    red_model = configure_model(red_config["model"], red_config["datasets"]["num_heads"])
    red_model.load_state_dict(torch.load(os.path.join(prediction_path, "epoch_100.pth")))

    # Configure dataloaders
    test_dataloader, blue_dict, predict_dict = initiate_connected_dataset(red_config["datasets"], batch_size)
    device = blue_config["device"]
    # Load model
    blue_model = configure_model(blue_config["model"], blue_config["datasets"]["num_heads"]).to(device)
    blue_model.load_state_dict(torch.load(filtering_path))

    connected_model = ConnectedFilteringPrediction(blue_model, red_model, predict_dict).to(device)

    logprob_stack = []
    i = 0
    for blue_obs, red_obs, y_test in test_dataloader:
        i += 1
        blue_obs = blue_obs.to(device).float()
        red_obs = red_obs.to(device).float()
        y_test = y_test.to(device).float()
        with torch.no_grad():
            logprob = connected_model.get_stats(blue_obs, red_obs, y_test)
        # time.sleep(100)
        logprob_stack.append(logprob)
        if i>100:
            break
    logprob_stack = torch.stack(logprob_stack, dim=0)
    print(logprob_stack.shape)
    print(-logprob_stack.mean(dim=0))
    # print(logprob_stack.shape)
    # Mean of the logprobability ordered by log probability
    # print(logprob_stack.mean(dim=0))