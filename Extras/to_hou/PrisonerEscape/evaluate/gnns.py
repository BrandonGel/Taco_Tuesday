import sys, os
sys.path.append(os.getcwd())

import numpy as np
from models.configure_model import configure_model
import yaml
import os
import torch

from datasets.dataset import VectorPrisonerDataset, GNNPrisonerDataset
# from datasets.old_gnn_dataset import LSTMGNNSequence
from torch.utils.data import DataLoader
from datasets.load_datasets import load_datasets, load_dataset_with_config_and_file_path

from utils import get_configs, set_seeds


# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220612-2315'
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220615-0429'
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220615-0506'
# model_folder_path = "/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220616-1719"


# first filtering model
# model_folder_path = "/nethome/sye40/PrisonerEscape/logs/gnn/filtering/20220618-1336"

# random cams all
# model_folder_path = "logs/gnn/filtering/20220618-0231"

# fixed cams
# model_folder_path = 'logs/gnn/total/20220701-1537'

# RRT fixed cams
# model_folder_path = 'logs/gnn/total/20220705-0946'
# get_start_location = False

# fixed cams heterogeneous lstm
# model_folder_path = 'logs/hetero_gnn/total/20220711-0344'

# random cams lstm front 
# model_folder_path = 'logs/hetero_gnn_lstm_front/random_cams/total/20220711-1417'

# fixed cameras lstm front
# model_folder_path = 'logs/hetero_gnn_lstm_front/random_cams/total/20220711-1459'

# random cams UPDATED
# model_folder_path = 'logs/gnn/total/20220701-1715'


# fixed cams all
# model_folder_path = "logs/gnn/filtering/20220618-1336"


# same cams all
# model_folder_path = "logs/gnn/filtering/20220618-1336"



############  hetero with Random Start Locations net
# model_folder_path = "logs/random_start_no_net/hetero_gnn/20220718-0449"
################# Random start locations #############################
# vector baseline
# model_folder_path = "logs/random_start/baseline/20220716-1829"

# homogeneous gnn
# model_folder_path = "logs/random_start/homo_gnn/20220716-2011"

# heterogeneous gnn

############## Random start locations with no net ####################
# model_folder_path = "logs/random_start_no_net/homo_gnn/20220717-1854"


model_folder_path = 'logs/hybrid_gnn/20220727-2331'
get_start_location = False

config_path = os.path.join(model_folder_path, "config.yaml")
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
batch_size = config["batch_size"]

# Configure dataloaders
# _, test_dataloader = load_datasets(config["datasets"], batch_size)


# file_path = "/nethome/mnatarajan30/codes/PrisonerEscape/datasets/map_0_run_20_heuristic_RRT.npz"
seq_len = config["datasets"]["seq_len"]
num_heads = config["datasets"]["num_heads"]
step_length = config["datasets"]["step_length"]
include_current = config["datasets"]["include_current"]
view = config["datasets"]["view"]
multi_head = config["datasets"]["multi_head"]


# dataset_path = "/workspace/PrisonerEscape/datasets/test/small_subset"
dataset_path = "/workspace/PrisonerEscape/datasets/small_subset"
dataset = load_dataset_with_config_and_file_path(config["datasets"], dataset_path)
# dataset = GNNPrisonerDataset(file_path, 
#     seq_len, 
#     num_heads, 
#     step_length, 
#     include_current=include_current, 
#     multi_head = multi_head,
#     one_hot=config["datasets"]["one_hot_agents"], 
#     timestep=config["datasets"]["timestep"], 
#     detected_location=config["datasets"]["detected_location"],
#     get_start_location=get_start_location)


test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = config["device"]
# Load model
model = configure_model(config).to(device)

model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))
logprob_stack = []

for x_test, y_test in test_dataloader:

    # print(x_test[0].shape)
    # mean, std = model.forward(x_test)
    # print(y_test)

    # print(mean.)
    logprob = model.get_stats(x_test, y_test)
    # loss = model.compute_loss(x_test, y_test)
    # print(loss.mean())
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
means = np.mean(logprob_stack, axis=0).tolist()
print(",".join(map(str, means)))

# print(logprob_stack.shape)
# Mean of the logprobability ordered by log probability

# print(logprob_stack.std(dim=0))