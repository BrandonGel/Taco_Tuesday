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
from datasets.load_datasets import load_datasets, load_dataset, load_dataset_with_config_and_file_path

from utils import get_configs, set_seeds
from visualize.render_utils import plot_mog_heatmap

import matplotlib.pyplot as plt
from heatmap import generate_heatmap_img
import cv2

from evaluate.mixture_evaluation import mix_multinorm_cdf_nn
from utils import save_video

def get_probability_grid(nn_output, index):
    pi, mu, sigma = nn_output
    pi = pi.detach().squeeze().cpu().numpy()
    sigma = sigma.detach().squeeze().cpu().numpy()
    mu = mu.detach().squeeze().cpu().numpy()
    # print(mu.shape)
    grid = plot_mog_heatmap(mu[index], sigma[index], pi[index])
    return grid

set_seeds(0)

model_folder_path = 'logs/hybrid_gnn/20220727-2331'

config_path = os.path.join(model_folder_path, "config.yaml")
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
batch_size = config["batch_size"]

seq_len = config["datasets"]["seq_len"]
num_heads = config["datasets"]["num_heads"]
step_length = config["datasets"]["step_length"]
include_current = config["datasets"]["include_current"]
view = config["datasets"]["view"]
multi_head = config["datasets"]["multi_head"]

# test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# _, test_dataloader = load_datasets(config["datasets"], batch_size)
# test_dataset = load_dataset(config["datasets"], "test_path")

# dataset_path = "/workspace/PrisonerEscape/datasets/test/small_subset"
dataset_path = "datasets/small_subset"
# dataset_path = "datasets/ilrt_test/gnn_map_0_run_3_RRT"

test_dataset = load_dataset_with_config_and_file_path(config["datasets"], dataset_path)

device = config["device"]
# Load model
model = configure_model(config).to(device)

model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))
logprob_stack = []

index = 0 # For filtering

bin_count = 0

heatmap_images = []

prob_vals = []
ll_vals = []

for i in range(len(test_dataset)):
    x_test, y_test = test_dataset[i]
    # print(x_test)
    x_test = [torch.from_numpy(x).to(device).unsqueeze(0) for x in x_test]
    # x_test = torch.from_numpy(x_test).to(device).unsqueeze(0)
    # 
    
    ############## Binary Count
    nn_output = model.forward(x_test)
    # print(nn_output[0].shape)

    val = mix_multinorm_cdf_nn(nn_output, 0, y_test)
    prob_vals.append(val)
    if val >= 0.5:
        bin_count += 1
    

    ################## Heatmaps
    # grid = get_probability_grid(nn_output, index)
    # heatmap_img = generate_heatmap_img(grid, true_location=y_test[index]*2428)
    # # cv2.imwrite("visualize/ilrt_test/heatmap_{}.png".format(i), heatmap_img)
    # heatmap_images.append(heatmap_img)

    ###### Log likelihood
    y_test = torch.from_numpy(y_test).to(device)
    logprob = model.get_stats(x_test, y_test)
    ll_val = logprob[0][index]
    ll_vals.append(ll_val.cpu().item())

    print(val, ll_val.cpu().item())

# save_video(heatmap_images, "visualize/94.mp4", fps=15.0)
print(bin_count / len(test_dataset))

import pandas as pd 
d = {'probabilities': prob_vals, "log likelihoods": ll_vals}
df = pd.DataFrame(d) 
    
# saving the dataframe 
df.to_csv('tmp/91_results.csv') 