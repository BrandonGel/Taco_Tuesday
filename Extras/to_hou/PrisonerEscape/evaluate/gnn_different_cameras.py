import numpy as np
from models.configure_model import configure_model
import yaml
import os
import torch

from datasets.dataset import VectorPrisonerDataset, GNNPrisonerDataset
from datasets.old_gnn_dataset import LSTMGNNSequence
from torch.utils.data import DataLoader

from utils import get_configs, set_seeds

# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220612-2315'
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220615-0429'
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220615-0506'
# model_folder_path = "/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220616-1719"
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220617-0016'

# first filtering model
# model_folder_path = "/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220612-2315"

model_folder_path = 'logs/gnn/total/20220701-1537'

config_path = os.path.join(model_folder_path, "config.yaml")
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
batch_size = config["batch_size"]

# Configure dataloaders
# _, test_dataloader = load_datasets(config["datasets"], batch_size)

# test_path = "/nethome/sye40/PrisonerEscape/datasets/gnn_map_0_run_100_eps_0.1_norm_random_cameras.npz"
# test_path = "/nethome/sye40/PrisonerEscape/datasets/seed_corrected/gnn_map_0_run_100_eps_0.1_norm.npz"
# test_path = "/nethome/sye40/PrisonerEscape/datasets/gnn_map_0_run_100_eps_0.1_norm.npz"

# test_path = "/nethome/sye40/PrisonerEscape/datasets/test_same/gnn_map_0_run_100_eps_0.1_norm"
# test_path = "/nethome/sye40/PrisonerEscape/datasets/validation/gnn_map_0_run_10_eps_0.1_norm_random_cameras"
# test_path = "/nethome/sye40/PrisonerEscape/datasets/test/gnn_map_0_run_100_eps_0.1_norm_random_cameras"

    
seq_len = 16
step_length = 1
num_heads = 0
include_current=False
multi_head = False
one_hot=False

# np_file = np.load("/nethome/sye40/PrisonerEscape/datasets/seed_corrected/gnn_map_0_run_100_eps_0.1_norm.npz", allow_pickle=True)
np_file = np.load("/nethome/sye40/PrisonerEscape/datasets/seed_corrected/gnn_map_0_run_300_eps_0.1_norm.npz", allow_pickle=True)
seq_len = 4
future_step = 0
dataset = LSTMGNNSequence(
    np_file["agent_observations"], 
    np_file["hideout_observations"], 
    np_file["timestep_observations"],
    np_file["red_locations"], 
    np_file["dones"], 
    seq_len, future_step)

# dataset = GNNPrisonerDataset(test_path, seq_len, num_heads, step_length, include_current=include_current, multi_head = multi_head, one_hot=one_hot)
test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = config["device"]
# Load model
model = configure_model(config).to(device)

model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))
logprob_stack = []

for x_test, y_test in test_dataloader:

    # mean, std = model.forward(x_test)

    # print(mean.)
    logprob = model.get_stats(x_test, y_test)
    loss = model.compute_loss(x_test, y_test)
    print(loss.mean())
    # print(logprob)
    # take mean across a dimension
    # torch to numpy
    logprob = logprob.detach().cpu().numpy()
    logprob_stack.append(logprob)
    del logprob
    # logprob = logprob.view(logprob.shape[0], 13, 2)
    # print(logprob)
    # take mean across a dimension
    # logprob = logprob.sum(dim=2)
    
    
    # logprob = logprob.mean(dim=0)
    # print(logprob.shape)

# logprob_stack = torch.cat(logprob_stack, dim=0)
logprob_stack = np.concatenate(logprob_stack, axis=0)
print(logprob_stack.shape)
print(np.mean(logprob_stack, axis=0))

# print(logprob_stack.shape)
# Mean of the logprobability ordered by log probability

# print(logprob_stack.std(dim=0))