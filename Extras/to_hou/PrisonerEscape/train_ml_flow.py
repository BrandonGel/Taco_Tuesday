"""
Copyright (2022)
Georgia Tech Research Corporation (Sean Ye, Manisha Natarajan, Rohan Paleja, Letian Chen, Matthew Gombolay)
All rights reserved.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from datasets.load_datasets import load_datasets
from models.configure_model import configure_model
import math
import os

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
import shutil
import mlflow
from urllib.parse import urlparse

def train(seed, 
          device, 
          train_dataloader, 
          test_dataloader, 
        #   batch_size,
          model,
          learning_rate,
          n_epochs,
          log_dir,
          config_path,
          weight_decay,
          l2_lambda,
          scheduler=None, 
          save_epoch=True):
    set_seeds(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=n_epochs)
    
    losses = []

    train_loss, prob_true_acts = 0, 0
    best_test_loss = np.inf
    weight_regularization = 1

    # Initialize for writing on tensorboard
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(log_dir, str(time))
    mlflow.log_param("log_dir_time", log_dir)

    summary_dir = os.path.join(log_dir, 'summary')
    writer = SummaryWriter(log_dir=summary_dir)
    model.writer = writer

    # copy config to log dir
    shutil.copy(config_path, os.path.join(log_dir, "config.yaml"))
    i = 0
    for epoch in tqdm(range(1, n_epochs+1)):
        batch_loss = 0
        num_batches = 0
        for x_train, y_train in train_dataloader:
            i += 1
            num_batches += 1
            # x_train = x_train.float().to(device)
            # y_train = y_train.float().to(device)
            train_loss = model.compute_loss(x_train, y_train, i)

            # l2_reg = torch.tensor(0.).to(device)
            # for param in model.parameters():
            #     l2_reg += torch.linalg.norm(param)
            # train_loss += l2_lambda * l2_reg

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            batch_loss += train_loss.item()
            writer.add_scalar('loss/train/neg_logp_iter', train_loss.item(), i)
        losses.append(batch_loss)
        writer.add_scalar('loss/train/neg_logp_epoch', batch_loss / (num_batches), epoch)

        # After every n epochs evaluate
        if epoch % 2 == 0:
            # model.eval()
            batch_test = 0
            num_batches_test = 0
            for x_test, y_test in test_dataloader:
                num_batches_test += 1
                # x_test = x_test.float().to(device)
                # y_test = y_test.float().to(device)
                with torch.no_grad():
                    batch_test += model.eval_loss(x_test, y_test).item()
            if save_epoch:
                # save the model
                torch.save(model.state_dict(), os.path.join(log_dir,  f"epoch_{epoch}.pth"))

            writer.add_scalar('loss/test/neglogp_epoch', batch_test / (num_batches_test), epoch)
            model.train()
            if log_dir:
                if batch_test < best_test_loss:
                    best_test_loss = batch_test
                    print(f"Saving Best Model... {batch_test / num_batches_test}")
                    torch.save(model.state_dict(), os.path.join(log_dir,  "best.pth"))
                    mlflow.log_metric("nll", batch_test/num_batches_test)
            # writer.add_scalars(f'loss/info', {
            #     'score_train': batch_loss / num_batches,
            #     'score_test': batch_test / num_batches_test,
            # }, epoch)

if __name__ == "__main__":
    # Load configs
    config, config_path = get_configs()
    batch_size = config["batch_size"]

    # Configure dataloaders
    train_dataloader, test_dataloader = load_datasets(config["datasets"], batch_size)

    device = config["device"]

    # Load model
    model = configure_model(config).to(device)
    print(model)

    if config["model"]["load_pth"] is not None:
        model.load_state_dict(torch.load(config["model"]["load_pth"]))

    train_configs = config["training"]
    
    seed = train_configs["seed"]
    learning_rate = train_configs["learning_rate"]
    epochs = train_configs["epochs"]
    log_dir = train_configs["log_dir"]
    l2_lambda = train_configs["l2_lambda"]
    weight_decay = train_configs["weight_decay"]
    ml_flow_experiment = str(train_configs["ml_flow_experiment"])
    save_epoch = train_configs["save_epoch"]
    print(ml_flow_experiment)

    with mlflow.start_run(experiment_id = ml_flow_experiment):
        for k, v in config["datasets"].items():
            mlflow.log_param(k, v)

        for k, v in config["model"].items():
            mlflow.log_param(k, v)

        for k, v in config["training"].items():
            mlflow.log_param(k, v)
        train(seed,
            device,
            train_dataloader, 
            test_dataloader, 
            model,
            learning_rate = learning_rate,
            n_epochs = epochs,
            weight_decay = weight_decay,
            log_dir = log_dir,
            config_path = config_path,
            l2_lambda = l2_lambda,
            scheduler=config["training"]["scheduler"],
            save_epoch=save_epoch)