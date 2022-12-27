""" Dataset for the gnn models """
import numpy as np
import os
from torch.utils.data import DataLoader
import torch

class GNNDataset(torch.utils.data.Dataset):
    def __init__(self, agent_obs, hideouts, timesteps, red_locs, dones, max_env_timesteps):
        self.agent_obs = agent_obs
        self.red_locs_np = red_locs
        self.timesteps = timesteps
        self.hideouts = hideouts
        self.max_env_timesteps = max_env_timesteps

        # ensure that we have the same number of timesteps 
        assert len(self.agent_obs) == len(self.red_locs_np)
    
    def __len__(self):
        return len(self.agent_obs)

    def __getitem__(self, idx):
        # Generates one sample of data
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.agent_obs[idx], self.timesteps[idx], self.hideouts[idx], self.red_locs_np[idx]

class BasePrisonerDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head):
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.step_length = step_length
        self.include_current = include_current
        self.multi_head = multi_head
        if self.multi_head:
            self.future_step = step_length * num_heads
        else:
            self.future_step = self.step_length

        self.dones = []
        self.red_locs = []
        self._load_data(folder_path)
        self.dones_shape = self.dones[0].shape

        # These mark the end of each episode
        self.done_locations = np.where(self.dones == True)[0]

    def _load_data(self, folder_path):
        pass

    def _produce_input(self, idx):
        pass

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        x = self._produce_input(idx)
        y = self._produce_output(idx)
        # GNN obs [B x A x 3], Hideouts [B x 2], Timestep [B], Num Agents [B]
        return x, y

    def _produce_output(self, idx):
        if self.multi_head:
            return self._produce_multi_step_output(idx)
        else:
            return self._produce_single_step_output(idx)

    def _produce_single_step_output(self, idx):
        next_done = self.done_locations[np.where(self.done_locations >= idx)[0][0]]
        target_red_loc_idx = idx + self.future_step
        if target_red_loc_idx > next_done:
            red_loc = self.red_locs[next_done]
        else:
            red_loc = self.red_locs[target_red_loc_idx]
        return red_loc

    def _produce_multi_step_output(self, idx):
        next_done = self.done_locations[np.where(self.done_locations >= idx)[0][0]]
        target_red_loc_idx = idx + self.future_step
        if target_red_loc_idx > next_done:
            # if we are at the end of the episode, just use the last location
            if idx + self.step_length > next_done:
                end_loc = np.expand_dims(self.red_locs[next_done], axis=0)
                red_loc = np.repeat(end_loc, self.num_heads, axis=0)
            else:
                # if there are steps between the current and end of the episode
                begin_locs = self.red_locs[idx+self.step_length:next_done:self.step_length]
                # unsqueeze a dimension in numpy
                end_loc = np.expand_dims(self.red_locs[next_done], axis=0)
                end_loc = np.repeat(end_loc, self.num_heads - len(begin_locs), axis=0)
                red_loc = np.concatenate((begin_locs, end_loc))
        else:
            red_loc = self.red_locs[idx+self.step_length:target_red_loc_idx+self.step_length:self.step_length]
        
        if self.include_current:
            # add the current location to the prediction
            red_loc = np.concatenate((np.expand_dims(self.red_locs[idx], 0), red_loc), axis=0)
        return red_loc

    def _process_start_observations(self, np_array, idx, episode_start_idx):
        """ If we're indexing at the start of an episode, need to pad the start with zeros"""
        last_obs = np_array[idx]
        shape = (self.sequence_length - (idx - episode_start_idx + 1),) + last_obs.shape
        empty_sequences = np.zeros(shape)
        sequence = np_array[episode_start_idx:idx+1]
        sequence = np.concatenate((empty_sequences, sequence), axis=0)
        return sequence

class GNNPrisonerDataset(BasePrisonerDataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head,
                    one_hot=False, timestep=False, detected_location=False, 
                    get_last_k_fugitive_detections=False, get_start_location=False):
        self.graphs = []
        self.max_agent_size = 0
        self.process_first_graph = True

        self.one_hot_bool = one_hot
        self.timestep_bool = timestep
        self.detected_location_bool = detected_location
        self.last_k_fugitive_detection_bool = get_last_k_fugitive_detections
        self.get_start_location_bool = get_start_location

        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head)

    def _load_data(self, folder_path):
        np_files = []
        for file_name in os.listdir(folder_path):
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
            np_files.append(np_file)

        for np_file in np_files:
            self._load_graph(np_file)

    def _load_graph(self, file):
        agent_obs = file["agent_observations"]
        agent_obs = np.squeeze(agent_obs)
        # [timesteps, agents, 3]
        # print(agent_obs.shape)
        num_agents = agent_obs.shape[1]

        detected_location = file["detected_locations"]
        if self.detected_location_bool:
            detected_bools = agent_obs[:, :, 0]
            detected_agent_locs = np.einsum("ij,ik->ijk", detected_bools, detected_location) # timesteps, agents, 2
            agent_obs = np.concatenate((agent_obs, detected_agent_locs), axis=2)

        timesteps = file["timestep_observations"]
        if self.timestep_bool:
            t = np.expand_dims(timesteps, axis=1)
            t = np.repeat(t, num_agents, axis=1)
            # print(t.shape)
            agent_obs = np.concatenate((agent_obs, t), axis=2)

        agent_obs = np.pad(agent_obs, ((0, 0), (0, self.max_agent_size - num_agents), (0, 0)), 'constant')

        agent_dict = file["agent_dict"].item()
        num_timesteps = agent_obs.shape[0]
        one_hots = self._create_one_hot_agents(agent_dict, num_timesteps)
        one_hots = np.pad(one_hots, ((0, 0), (0, self.max_agent_size - num_agents), (0, 0)), 'constant')

        if self.process_first_graph:
            self.num_agents = [num_agents] * agent_obs.shape[0]
            self.agent_obs = agent_obs
            self.hideouts = file["hideout_observations"]
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.red_locs = file["red_locations"]/2428
            self.detected_location = detected_location
            self.one_hots = one_hots
            self.process_first_graph = False
            if self.last_k_fugitive_detection_bool:
                self.last_k_fugitive_detections = file["last_k_fugitive_detections"]

            if self.get_start_location_bool:
                # TODO Sean: this is a hack to get the start location
                # Need to ensure this will work when the start location is not in the last few locations
                self.start_locations = file["blue_observations"][:, -2:]
        else:
            self.num_agents.extend([num_agents] * agent_obs.shape[0])
            self.agent_obs = np.concatenate((self.agent_obs, agent_obs), 0)
            self.hideouts = np.append(self.hideouts, file["hideout_observations"], 0)
            self.red_locs = np.append(self.red_locs, file["red_locations"]/2428, 0)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.detected_location = np.append(self.detected_location, detected_location)
            self.one_hots = np.concatenate((self.one_hots, one_hots), 0)
            self.dones = np.append(self.dones, file["dones"])
            if self.last_k_fugitive_detection_bool:
                self.last_k_fugitive_detections = np.append(self.last_k_fugitive_detections, file["last_k_fugitive_detections"], 0)
            
            if self.get_start_location_bool:
                self.start_locations = np.append(self.start_locations, file["blue_observations"][:, -2:], 0)

    def _create_one_hot_agents(self, agent_dict, timesteps):
        one_hot_base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        agents = [agent_dict["num_known_cameras"] + agent_dict["num_unknown_cameras"],
                    agent_dict["num_helicopters"], agent_dict["num_search_parties"]]
        a = np.repeat(one_hot_base, agents, axis=0) # produce [num_agents, 3] (3 for each in one-hot)
        # one_hot = np.repeat(np.expand_dims(a, 0), self.sequence_length, axis=0) # produce [seq_len, num_agents, 3]
        one_hot = np.repeat(np.expand_dims(a, 0), timesteps, axis=0) # produce [timesteps, num_agents, 3]
        return one_hot

    def _produce_input(self, idx):
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        if idx - episode_start_idx >= self.sequence_length:
            agent_obs = self.agent_obs[idx - self.sequence_length:idx]
            if self.one_hot_bool:
                # Produce for each agent in agent_obs, include one-hot of the agent type
                # Agents obs is [Seq_len x Num_agents x 3]
                # One hot should be [Seq_len x Num_agents x 3]
                # Agent obs is in order known cameras, unknown cameras, helicopters, search parties
                agent_obs = np.concatenate((agent_obs, self.one_hots[idx - self.sequence_length:idx]), axis=2)

            sample = [agent_obs,
                        # self.detected_location[idx - self.sequence_length: idx],
                        self.hideouts[idx],
                        np.expand_dims(self.timesteps[idx], 0),
                        np.expand_dims(self.num_agents[idx], 0)]
        else:
            agent_obs = self._process_start_observations(self.agent_obs, idx, episode_start_idx)
            if self.one_hot_bool:
                agent_obs = np.concatenate((agent_obs, self._process_start_observations(self.one_hots, idx, episode_start_idx)), axis=2)

            sample = [agent_obs,
                    #   self.process_start_observations(self.detected_location, idx, episode_start_idx),
                      self.hideouts[idx], np.expand_dims(self.timesteps[idx], 0), np.expand_dims(self.num_agents[idx], 0)]
        
        if self.last_k_fugitive_detection_bool:
            sample.append(self.last_k_fugitive_detections[idx])# Append the last k fugitive detections

        if self.get_start_location_bool:
            sample.append(self.start_locations[idx])

        return sample

class HeterogeneousGNNDataset(GNNPrisonerDataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head,
                    one_hot=False, timestep=False, detected_location=False, get_start_location=False):

        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head,
                    one_hot, timestep, detected_location, get_start_location=get_start_location)

    def _produce_input(self, idx):
        # Output (agent_obs, hideouts, timesteps, num_agents)
        # However, unlike the gnn output, the timesteps will be a vector of t x 1 instead of a single timestep
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        if idx - episode_start_idx >= self.sequence_length:
            agent_obs = self.agent_obs[idx - self.sequence_length:idx]
            if self.one_hot_bool:
                # Produce for each agent in agent_obs, include one-hot of the agent type
                # Agents obs is [Seq_len x Num_agents x 3]
                # One hot should be [Seq_len x Num_agents x 3]
                # Agent obs is in order known cameras, unknown cameras, helicopters, search parties
                agent_obs = np.concatenate((agent_obs, self.one_hots[idx - self.sequence_length:idx]), axis=2)

            sample = [agent_obs,
                        self.hideouts[idx],
                        self.timesteps[idx - self.sequence_length: idx],
                        np.expand_dims(self.num_agents[idx], 0)]
        else:
            agent_obs = self._process_start_observations(self.agent_obs, idx, episode_start_idx)
            if self.one_hot_bool:
                agent_obs = np.concatenate((agent_obs, self._process_start_observations(self.one_hots, idx, episode_start_idx)), axis=2)

            sample = [agent_obs,
                      self.hideouts[idx],
                      self._process_start_observations(self.timesteps, idx, episode_start_idx),
                      np.expand_dims(self.num_agents[idx], 0)]

        if self.get_start_location_bool:
            sample.append(self.start_locations[idx])
        return sample



class VectorPrisonerDataset(BasePrisonerDataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head, view='blue'):
        self.view = view
        print(self.view)
        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head)


    def _load_data(self, folder_path):
        np_file = np.load(folder_path, allow_pickle=True)
        if self.view == "blue":
            self.blue_obs_np = np_file["blue_observations"]
        elif self.view == "red":
            self.red_obs_np = np_file["red_observations"]
        elif self.view == "both":
            self.blue_obs_np = np_file["blue_observations"]
            self.red_obs_np = np_file["red_observations"]
        else:
            raise ValueError("View must be either red or blue")

        self.red_locs = np_file["red_locations"]
        self.dones = np_file["dones"]

    def _produce_input(self, idx):
            # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        if idx - episode_start_idx >= self.sequence_length:
            if self.view == "red":
                red_sequence = self.red_obs_np[idx - self.sequence_length:idx]
            blue_sequence = self.blue_obs_np[idx - self.sequence_length:idx]
        else:
            if self.view == "red":
                red_sequence = self._process_start_observations(self.red_obs_np, idx, episode_start_idx)
            blue_sequence = self._process_start_observations(self.blue_obs_np, idx, episode_start_idx)

        if self.view == "red":
            return red_sequence
        elif self.view == "blue":
            return blue_sequence
        elif self.view == "both":
            return blue_sequence, red_sequence
        else:
            raise ValueError("View must be either red or blue")

if __name__ == "__main__":
    # path = "/nethome/sye40/PrisonerEscape/datasets/gnn_map_0_run_300_eps_0.1_norm.npz"
    # np_file = np.load(path, allow_pickle=True)

    seq_len = 16
    step_length = 5
    num_heads = 12

    # path = "/nethome/sye40/PrisonerEscape/datasets/train/gnn_map_0_run_300_eps_0.1_norm_random_cameras"
    # path = "/nethome/sye40/PrisonerEscape/datasets/small_train"
    path = "/nethome/sye40/PrisonerEscape/datasets/test_same/gnn_map_0_run_100_eps_0.1_norm"
    dataset = GNNPrisonerDataset(path, seq_len, num_heads, step_length,
            include_current=True, multi_head = True, one_hot=True,
            timestep=True, detected_location=True)
    print(len(dataset))
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    x, y = dataset[120]
    print(x[0].shape)
    # print(x[0][15, 0]*4320)


    # np_file_path = "/nethome/sye40/PrisonerEscape/datasets/seed_corrected/map_0_run_100_eps_0_norm.npz"
    # seq_len = 4
    # future_step = 10
    # red_blue_dataset = VectorPrisonerDataset(np_file_path, seq_len, num_heads, step_length, include_current=True, multi_head = True, view='blue')

    # print(red_blue_dataset[0])

    # import time
    # now = time.time()
    for x, y in train_dataloader:
        # print(x[3].shape)
        print(x[0].shape)
        break
        # for i in x:
        #     print(i.shape)
        # break