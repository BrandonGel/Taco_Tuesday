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

# model_folder_path = "/nethome/sye40/PrisonerEscape/logs/connected/20220602-0010"

# hidden not connected
# blue obs in, mixture in middle, single gaussian out
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/connected/20220602-1446'
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/connected/20220602-1508'

# hidden connected
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/connected/20220602-1422'
model_folder_path = '/nethome/sye40/PrisonerEscape/logs/connected/20220603-1128'

config_path = os.path.join(model_folder_path, "config.yaml")
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
batch_size = config["batch_size"]

# Configure dataloaders
_, test_dataloader = load_datasets(config["datasets"], batch_size)

device = config["device"]
# Load model
model = configure_model(config["model"], config["datasets"]["num_heads"]).to(device)

model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))

logprob_stack = []

for x_test, y_test in test_dataloader:
    # x_test = x_test.to(device).float()
    # y_test = y_test.to(device).float()

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

# print(logprob_stack.shape)
# Mean of the logprobability ordered by log probability
print(logprob_stack.mean(dim=0))
# print(logprob_stack.std(dim=0))