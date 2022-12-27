import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from datasets.load_datasets import load_datasets
from models.configure_model import configure_model
import math
import yaml
import argparse
import os
import cv2

import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import math
import random

from utils import get_configs, set_seeds

# Load configs
# config, config_path = get_configs()
# model_folder_path = "/nethome/sye40/PrisonerClean/logs/prediction/20220518-1718"

# red obs prediction
# model_folder_path = "/nethome/sye40/PrisonerEscape/logs/prediction/20220601-2348"

# blue obs in, single gaussian out
# model_folder_path = "/nethome/sye40/PrisonerEscape/logs/filtering/20220602-0118"

# blue obs in, mixture out
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/filtering/20220602-0157'

# blue obs in, mixture out for every timestep
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/filtering/20220606-2213'
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/filtering/20220606-2222'
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/filtering/20220606-2231'
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/prediction/20220615-2236'
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/filtering/20220613-0028'

# blue obs in, mixture out
# model_folder_path = 'logs/vector/baseline/20220617-2028'

# blue obs in, connected
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/connected/20220602-2339'

########### Post update fix red heuristic ##############
# RRT prediction
# model_folder_path = "logs/vector/baseline/20220705-0917"

model_folder_path = "logs/vector/baseline/20220812-2121"

config_path = os.path.join(model_folder_path, "config.yaml")
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
batch_size = config["batch_size"]

# Configure dataloaders
_, test_dataloader = load_datasets(config["datasets"], batch_size)

device = config["device"]
# Load model
model = configure_model(config).to(device)

model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))

logprob_stack = []

for x_test, y_test in test_dataloader:
    x_test = x_test.to(device).float()
    y_test = y_test.to(device).float()
    print(y_test.shape)
    # mean, std = model.forward(x_test)

    # print(mean.)
    logprob = model.get_stats(x_test, y_test)
    # logprob = logprob.view(logprob.shape[0], 13, 2)
    # print(logprob)
    # take mean across a dimension
    # logprob = logprob.sum(dim=2)
    logprob_stack.append(logprob)
    
    # logprob = logprob.mean(dim=0)
    # print(logprob.shape)

logprob_stack = torch.cat(logprob_stack, dim=0)
print(logprob_stack.mean(dim=0))
print(logprob_stack.mean(dim=0).sum())
# print(logprob_stack.shape)
# Mean of the logprobability ordered by log probability

# print(logprob_stack.std(dim=0))