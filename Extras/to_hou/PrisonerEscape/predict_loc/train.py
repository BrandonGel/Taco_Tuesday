"""
Copyright (2022)
Georgia Tech Research Corporation (Sean Ye, Manisha Natarajan, Rohan Paleja, Letian Chen, Matthew Gombolay)
All rights reserved.
"""
import sys, os
project_path = os.getcwd()
sys.path.append(str(project_path))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from datasets.load_datasets import load_datasets
from models.configure_model import configure_model
import math
import os


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
import shutil

from torch.optim.lr_scheduler import MultiStepLR

def train(seed, 
          device, 
          train_dataloader, 
          test_dataloader, 
        #   batch_size,
          model,
          learning_rate,
          n_epochs,
          l2_lambda,
          log_dir,
          config_path):
    set_seeds(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer, milestones=[10,80], gamma=0.1)
    losses = []
    train_loss, prob_true_acts = 0, 0
    best_test_loss = np.inf
    weight_regularization = 1

    # Initialize for writing on tensorboard
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(log_dir, str(time))

    summary_dir = os.path.join(log_dir, 'summary')
    writer = SummaryWriter(log_dir=summary_dir)

    # copy config to log dir
    shutil.copy(config_path, os.path.join(log_dir, "config.yaml"))

    i = 0
    for epoch in tqdm(range(1, n_epochs+1)):
        batch_loss = 0
        num_batches = 0
        for x_train, y_train in train_dataloader:
            if x_train.shape[0] != batch_size:
                continue
            train_loss = 0
            i += 1
            num_batches += 1
            # x_train = x_train.float().to(device)
            # y_train = y_train.float().to(device)
            loss = model.compute_loss(x_train, y_train)

            # l2_reg = torch.tensor(0.).to(device)
            # for param in model.parameters():
                # l2_reg += torch.linalg.norm(param)
            # train_loss = loss + l2_lambda * l2_reg
            train_loss = loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            writer.add_scalar('loss/train/neg_logp_iter', loss.item(), i)
            # writer.add_scalar('loss/train/sigma', )
        scheduler.step()
        losses.append(batch_loss)
        writer.add_scalar('loss/train/neg_logp_epoch', batch_loss / (num_batches), epoch)

        # After every n epochs evaluate
        if epoch % 2 == 0:
            # model.eval()
            batch_test = 0
            num_batches_test = 0
            for x_test, y_test in test_dataloader:
                if x_test.shape[0] != batch_size:
                    continue
                num_batches_test += 1
                # x_test = x_test.float().to(device)
                # y_test = y_test.float().to(device)
                with torch.no_grad():
                    batch_test += model.compute_loss(x_test, y_test).item()
            # save the model
            torch.save(model.state_dict(), os.path.join(log_dir,  f"epoch_{epoch}.pth"))

            writer.add_scalar('loss/test/neglogp_epoch', batch_test / (num_batches_test), epoch)
            model.train()
            if log_dir:
                if batch_test < best_test_loss:
                    best_test_loss = batch_test
                    print(f"Saving Best Model... {batch_test / num_batches_test}")
                    torch.save(model.state_dict(), os.path.join(log_dir,  "best.pth"))

if __name__ == "__main__":
    # Load configs
    config, config_path = get_configs()
    batch_size = config["batch_size"]

    # Configure dataloaders
    train_dataloader, test_dataloader = load_datasets(config["datasets"], batch_size)

    device = config["device"]

    # Load model
    model = configure_model(config).to(device)
    # model = nn.DataParallel(model).to(device)

    if config["model"]["load_pth"] is not None:
        model.load_state_dict(torch.load(config["model"]["load_pth"]))

    train_configs = config["training"]
    seed = train_configs["seed"]
    learning_rate = train_configs["learning_rate"]
    epochs = train_configs["epochs"]
    log_dir = train_configs["log_dir"]
    l2_lambda = train_configs["l2_lambda"]
    print(l2_lambda)

    train(seed,
        device,
        train_dataloader, 
        test_dataloader, 
        model,
        learning_rate = learning_rate,
        n_epochs = epochs,
        l2_lambda = l2_lambda,
        log_dir = log_dir,
        config_path = config_path)