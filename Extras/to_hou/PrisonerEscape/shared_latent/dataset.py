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
    """ For loading the dataset into an LSTM"""
    def __init__(self, reb_obs, blue_obs, red_locs, dones, sequence_length):
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
    
    def __len__(self):
        return len(self.red_obs_np)

    def __getitem__(self, idx):
        while idx < self.sequence_length:
            idx = random.randint(self.sequence_length, len(self.red_obs_np))
        red_sequence = self.red_obs_np[idx - self.sequence_length:idx]
        blue_sequence = self.blue_obs_np[idx - self.sequence_length:idx]
        red_locs_sequence = self.red_locs_np[idx - self.sequence_length:idx]
        dones_sequence = self.dones[idx - self.sequence_length:idx]
        return red_sequence, blue_sequence, red_locs_sequence/2428, dones_sequence

class RedBlueSequenceEncoded(RedBlueSequence):
    """ Format the data into (O x N x E) 
    O is the outer dimension of sequence length
    N is number of observations within the stack
    E is number of features in the observation
    The total number of observations is O*N

    We sample a sequence of O*N observations and reshape it into (O x N x E)
    So [1, 2, 3, 4] becomes [[1, 2], [3, 4]] with O = 2 and N = 2
    """

    def __init__(self, reb_obs, blue_obs, red_locs, dones, outer_sequence_length, inner_sequence_length):
        total_sequence_length = outer_sequence_length * inner_sequence_length
        super().__init__(reb_obs, blue_obs, red_locs, dones, total_sequence_length)
        self.inner_dim = inner_sequence_length
        self.outer_dim = outer_sequence_length

    def reshape_np(self, np_array):
        # reshape the data into (T x N x E)
        # print(self.outer_dim, self.inner_dim, np_array.shape[-1])
        return np_array.reshape(self.outer_dim, self.inner_dim, np_array.shape[-1])

    def __getitem__(self, idx):
        red_sequence, blue_sequence, red_locs_sequence, dones_sequence = super().__getitem__(idx)
        
        red_sequence = self.reshape_np(red_sequence)
        blue_sequence = self.reshape_np(blue_sequence)
        red_locs_sequence = self.reshape_np(red_locs_sequence)
        dones_sequence = dones_sequence.reshape(self.outer_dim, self.inner_dim)

        return red_sequence, blue_sequence, red_locs_sequence, dones_sequence

class RedBlueSequenceSkip(torch.utils.data.Dataset):
    """ This dataset skips timesteps so each sequence is i, i+n, i+2n, i+3n, ..., i+n*(sequence_length-1) """
    def __init__(self, reb_obs, blue_obs, red_locs, dones, skip_step, sequence_length):
        self.red_obs_np = reb_obs
        self.red_locs_np = red_locs
        self.blue_obs_np = blue_obs
        self.dones = dones
        self.sequence_length = sequence_length
        self.skip_step = skip_step

        self.red_shape = self.red_obs_np[0].shape
        self.blue_shape = self.blue_obs_np[0].shape
        self.red_locs_shape = self.red_locs_np[0].shape
        self.dones_shape = self.dones[0].shape

        # ensure that we have the same number of timesteps 
        assert len(self.red_obs_np) == len(self.red_locs_np)
        assert len(self.blue_obs_np) == len(self.red_locs_np)
    
    def __len__(self):
        return len(self.red_obs_np)

    def __getitem__(self, idx):

        # Ensure we have enough data at the end
        if idx > len(self.blue_obs_np) - self.sequence_length * self.skip_step:
            idx = random.randint(0, len(self.blue_obs_np) - self.sequence_length * self.skip_step)

        # Starting at idx, sample [idx, idx+skip_step, idx+2*skip_step, ..., idx+skip_step*(sequence_length-1)]
        # If there is a reset within this range, we return the last step
        indices = np.arange(idx, idx + self.sequence_length * self.skip_step, self.skip_step)
        start = indices[0]
        for index, value in enumerate(indices[1:]):
            if True in self.dones[start:value]:
                indices[index:] = np.full(indices[index:].shape, indices[index])
                break
            start = value
        # print(indices)

        red_sequence = self.red_obs_np[indices]
        blue_sequence = self.blue_obs_np[indices]
        red_locs_sequence = self.red_locs_np[indices]
        dones_sequence = self.dones[indices]

        return red_sequence, blue_sequence, red_locs_sequence/2428, dones_sequence

if __name__ == "__main__":
    # np_file = np.load("/nethome/sye40/PrisonerEscape/shared_latent/map_0_run_100_eps_0.npz", allow_pickle=True)
    np_file = np.load("/nethome/sye40/PrisonerEscape/shared_latent/dataset/map_0_run_300_eps_0.npz", allow_pickle=True)
    # red_blue_dataset = RedBlueSequence(np_file['red_observations'][:20], np_file['blue_observations'][:20], np_file['red_locations'][:20], 4320, 4)
    # seq_len = 4
    # train_length = 500
    # red_blue_dataset = RedBlueSequenceSkip(np_file['red_observations'][:train_length], np_file['blue_observations'][:train_length], np_file['red_locations'][:train_length], np_file['dones'][:train_length], 5, seq_len)
    
    outer_sequence_length = 5
    inner_sequence_length = 5
    rb_dataset = RedBlueSequenceEncoded(np_file['red_observations'], np_file['blue_observations'], np_file['red_locations'], np_file['dones'], outer_sequence_length, inner_sequence_length)

    # find first value of true in numpy array
    # print(np.where(np_file['dones']== True))

    print(rb_dataset[0][0].shape)
    print(rb_dataset[0][2]*2428)
    # red_blue_dataset[326]
    
    # print(red_blue_dataset[0][2].shape)