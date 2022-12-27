""" Dataset for the shared latent model """
import torch
import numpy as np
import math
import random

# from utils import load_environment
from simulator import PrisonerEnv
import os

# input to network is prediction observation ( fugitive observation without unknown hideout locations)
# create single dataset where we can query for how many timesteps (rather than creating multiple datasets)
class RedBlueDataset(torch.utils.data.Dataset):
    def __init__(self, reb_obs, blue_obs, red_locs, dones, max_env_timesteps):
        self.red_obs_np = reb_obs
        self.red_locs_np = red_locs
        self.blue_obs_np = blue_obs
        self.max_env_timesteps = max_env_timesteps

        # ensure that we have the same number of timesteps 
        assert len(self.red_obs_np) == len(self.red_locs_np)
        assert len(self.red_obs_np) == len(self.red_locations_np)
    
    def __len__(self):
        return len(self.red_obs_np)

    def __getitem__(self, idx):
        # Generates one sample of data
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.red_obs_np[idx], self.blue_obs_np[idx], self.red_obs_np

class RedBlueSequence(torch.utils.data.Dataset):
    
    def __init__(self, reb_obs, blue_obs, red_locs, dones, sequence_length):
        """ Dataset for both filtering and prediction
        
        This dataset stacks a sequence of observations

        :param future_step: Number of timesteps to predict into the future, 0 is used for filtering
            For example, future_step = 10 indicates returning the prisoner location from 10 steps into the future
        :param view: "red" or "blue"
        """
        
        self.red_obs_np = reb_obs
        self.red_locs_np = red_locs
        self.blue_obs_np = blue_obs
        self.dones = dones
        self.sequence_length = sequence_length

        self.red_shape = self.red_obs_np[0].shape
        self.blue_shape = self.blue_obs_np[0].shape
        self.red_locs_shape = self.red_locs_np[0].shape
        self.dones_shape = self.dones[0].shape

        # ensure that we have the same number of timesteps 
        assert len(self.red_obs_np) == len(self.red_locs_np)
        assert len(self.blue_obs_np) == len(self.red_locs_np)

        # These mark the end of each episode
        self.done_locations = np.where(self.dones == True)[0]
    
    def __len__(self):
        return len(self.red_obs_np)

    def __getitem__(self, idx):
        
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx
        
        if idx - episode_start_idx >= self.sequence_length:
            red_sequence = self.red_obs_np[idx - self.sequence_length:idx]
            blue_sequence = self.blue_obs_np[idx - self.sequence_length:idx]
        else:
            last_red_observation = self.red_obs_np[idx]
            shape = (self.sequence_length - (idx - episode_start_idx + 1),) + last_red_observation.shape
            empty_sequences = np.zeros(shape)
            red_sequence = self.red_obs_np[episode_start_idx:idx+1]
            red_sequence = np.concatenate((empty_sequences, red_sequence), axis=0)

            last_blue_observation = self.blue_obs_np[idx]
            shape = (self.sequence_length - (idx - episode_start_idx + 1),) + last_blue_observation.shape
            empty_sequences = np.zeros(shape)
            blue_sequence = self.blue_obs_np[episode_start_idx:idx+1]
            blue_sequence = np.concatenate((empty_sequences, blue_sequence), axis=0)

            # Get the red locations
            last_red_locations = self.red_locs_np[idx]
            shape = (self.sequence_length - (idx - episode_start_idx + 1),) + last_red_locations.shape
            empty_sequences = np.zeros(shape)
            red_locations = self.red_locs_np[episode_start_idx:idx+1]
            red_locations = np.concatenate((empty_sequences, red_locations), axis=0)
        return red_sequence, blue_sequence, red_locations


if __name__ == "__main__":
    np_file = np.load("/nethome/sye40/PrisonerClean/datasets/map_0_run_300_eps_0_normalized.npz", allow_pickle=True)
    seq_len = 4
    future_step = 10
    red_blue_dataset = RedBlueSequence(
        np_file["red_observations"], 
        np_file["blue_observations"], 
        np_file["red_locations"], 
        np_file["dones"], 
        seq_len, future_step, "blue")

    # print(red_blue_dataset[0][0].shape)
    done_locations = (red_blue_dataset.done_locations)
    print(done_locations)
    # print(np.where(done_locations > 88550)[0])

    idx = 326
    # print(np_file["red_locations"][idx])
    print(red_blue_dataset[idx][0])
    print(red_blue_dataset[idx][0].shape)

    print(np_file["red_locations"][325:328])

    # print(np_file["red_locations"][idx+future_step])