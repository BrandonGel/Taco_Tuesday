# from requests import post
import shutil
from dataset import RedBlueSequence, RedBlueSequenceSkip, RedBlueSequenceEncoded
from shared_latent.models.rssm import RSSMCore
from torch.utils.data import DataLoader
from torch import Tensor
import torch
import torch.distributions as D
from functions import logavgexp, flatten_batch, unflatten_batch
from shared_latent.models.decoder import DenseNormalDecoder, SingleGaussianDecoder
from shared_latent.models.dreamer import Dreamer
from torch.cuda.amp import GradScaler
from shared_latent.utils import get_configs
from models.shared_latent import SharedLatent
from shared_latent.models.decoder import SingleGaussianDecoder, MixtureDensityDecoder, DenseNormalDecoder, SingleGaussianDecoderStd
import random

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import yaml

import os
from datetime import datetime
from torch.distributions import Normal
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def test_stochastic_rollouts(model, obs, reset, red_locs, num_warmstart_timesteps, this_batch_size):
    """ Given the rssm and a decoder model, check our prediction a few steps into the future """

    # T = obs.shape[0]
    # post_warmstart = T - num_warmstart_timesteps
    start_obs = obs[:num_warmstart_timesteps]
    start_reset = reset[:num_warmstart_timesteps]

    hidden_state = model.warm_start_rssm(start_obs, start_reset, None, this_batch_size)

    predict_obs = obs[num_warmstart_timesteps:]
    predict_locs = red_locs[num_warmstart_timesteps:]
    predict_resets = reset[num_warmstart_timesteps:]

    _, decoder_loss = model.compute_loss(predict_obs, predict_resets, predict_locs, hidden_state, this_batch_size, do_open_loop=True)

    return decoder_loss

def initialize_rssm(conf, rssm_path):
    rssm = RSSMCore(input_dim=conf["in_dim"],
                embed_dim=conf["embed_dim"],
                deter_dim=conf["deter_dim"],
                stoch_dim=conf["stoch_dim"],
                stoch_discrete=conf["stoch_discrete"],
                hidden_dim=conf["hidden_dim"],
                gru_layers=conf["gru_layers"],
                gru_type=conf["gru_type"],
                layer_norm=conf["layer_norm"]).to(device)

    rssm.load_state_dict(torch.load(rssm_path))
    return rssm

def create_shared_latent_model(red_conf, blue_conf, red_rssm_path, blue_rssm_path):

    red_rssm = initialize_rssm(red_conf, red_rssm_path)
    blue_rssm = initialize_rssm(blue_conf, blue_rssm_path)
    red_rssm.load_state_dict(torch.load(red_rssm_path))
    blue_rssm.load_state_dict(torch.load(blue_rssm_path))

    conf = red_conf
    features_dim = conf["deter_dim"] + conf["stoch_dim"] * (conf["stoch_discrete"] or 1)
    if conf["decoder"] == "SingleParam":
        decoder = SingleGaussianDecoder(features_dim, conf["output_dim"]).to(device)
    elif conf["decoder"] == "SingleStd":
        decoder = SingleGaussianDecoderStd(features_dim, conf["output_dim"]).to(device)
    elif conf["decoder"] == "Mixture":
        decoder = MixtureDensityDecoder(features_dim, output_dim=conf["output_dim"]).to(device)
    else:
        raise NotImplementedError
    model = SharedLatent(red_rssm, blue_rssm, decoder).to(device)
    return model
    

def train(red_conf, blue_conf, red_rssm_path, blue_rssm_path):

    conf = red_conf
    # Set seeds
    torch.manual_seed(conf["seed"])
    np.random.seed(conf["seed"])
    random.seed(conf["seed"])


    np_file = np.load(conf["train_dataset_path"], allow_pickle=True)
    test_file = np.load(conf["test_dataset_path"], allow_pickle=True)

    total_resets = np.sum(np_file['dones'])
    print(total_resets)

    batch_size = conf["batch_size"] # B
    # seq_len = conf["seq_len"] # T
    loss_kl_weight = conf["loss_kl_weight"] # KL weight

    outer_seq = conf["outer_seq"] # outer sequence length
    inner_seq = conf["inner_seq"] # inner sequence length

    min_decoder_loss = conf["min_decoder_loss"] # min decoder loss

    train_length = 300
    red_blue_dataset = RedBlueSequenceEncoded(np_file['red_observations'], np_file['blue_observations'], np_file['red_locations'], np_file['dones'], 
                                outer_seq, inner_seq)

    # red_blue_dataset = RedBlueSequenceEncoded(np_file['red_observations'][:train_length], 
    #                                           np_file['blue_observations'][:train_length], 
    #                                           np_file['red_locations'][:train_length], 
    #                                           np_file['dones'][:train_length], 
    #                             outer_seq, inner_seq)
    train_dataloader = DataLoader(red_blue_dataset, batch_size=batch_size, shuffle=True)

    num_warmup = 16
    predict_horizon = 30
    num_outer_warmups = max(num_warmup // inner_seq, 1)
    num_outer_predict = max(predict_horizon // inner_seq, 1)
    total_outer = num_outer_warmups + num_outer_predict
    print(f"number outer warmups: {num_outer_warmups} and number outer predict: {num_outer_predict}")
    print(f"total warmup timesteps: {num_outer_warmups * inner_seq} and total predict timesteps: {num_outer_predict * inner_seq}")
    test_dataset = RedBlueSequenceEncoded(test_file['red_observations'], 
                                          test_file['blue_observations'], 
                                          test_file['red_locations'], 
                                          test_file['dones'], 
                            total_outer, inner_seq)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    summary_dir = f"logs/shared_latent/combined"

    # Initialize for writing on tensorboard
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(summary_dir, str(time))

    summary_dir = os.path.join(log_dir, 'summary')
    writer = SummaryWriter(log_dir=summary_dir)

    os.mkdir(log_dir + '/red')
    os.mkdir(log_dir + '/blue')
    # os.mkdir(log_dir + '/decoder')

    # copy config file to summary directory
    # shutil.copy(config_path, os.path.join(log_dir, 'config.yaml'))
    model = create_shared_latent_model(red_conf, blue_conf, red_rssm_path, blue_rssm_path)

    # # Freeze the decoder weights
    # decoder_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220510-1455/summary/9.pth'
    # model.load_state_dict(torch.load(decoder_path))
    # for param in model.decoder.parameters():
        # param.requires_grad = False

    # opt_1 = torch.optim.Adam(rssm.parameters(), lr=0.001)
    # opt_2 = torch.optim.Adam(decoder.parameters(), lr=0.001)

    # opt = torch.optim.Adam([{'params': rssm.parameters()}, {'params': decoder.parameters()}], lr=3e-4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = ExponentialLR(opt, gamma=0.9)


    # optimizers = [opt_1, opt_2]
    # scaler = GradScaler(enabled=True)

    i = 0
    best_test_loss = float('inf')
    for epoch in tqdm(range(conf["epochs"])):
        epoch_loss = 0
        for data in train_dataloader:
            i += 1
            red_obs, blue_obs, red_locs, reset = data
            # B x O x N x D
            this_batch_size = red_obs.shape[0]

            blue_obs = blue_obs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
            red_obs = red_obs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
            reset = reset.permute(1, 2, 0).to(device) #BxOxN -> OxNxB
            red_locs = red_locs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
            red_locs = red_locs[:, -1, :, :] # grab the last location in the inner dimension (OxBxD)

            shared, red_kl_loss, blue_kl_loss, red_decoder_loss, blue_decoder_loss = model.compute_loss(red_obs, blue_obs, reset, red_locs, None, this_batch_size, writer)
            total_loss = 3*shared + 5*red_kl_loss + 5*blue_kl_loss + red_decoder_loss + blue_decoder_loss
            writer.add_scalar('loss/train/red_decoder', red_decoder_loss.mean(), i)
            writer.add_scalar('loss/train/red_model', red_kl_loss.mean(), i)
            writer.add_scalar('loss/train/blue_decoder', blue_decoder_loss.mean(), i)
            writer.add_scalar('loss/train/blue_model', blue_kl_loss.mean(), i)
            writer.add_scalar('loss/train/shared', shared.mean(), i)

            opt.zero_grad()
        
            # schedule = False
            # if schedule:
            #     if epoch < 4:
            #         total_loss = torch.clamp(decoder_loss.mean(), min=min_decoder_loss)
            #     elif 4 <= epoch < 10 :
            #         for param in model.decoder.parameters():
            #             param.requires_grad = False
            #         total_loss =  loss_kl_weight * loss_model.mean() + torch.clamp(decoder_loss.mean(), min=min_decoder_loss)      
            #     else:
            #         for param in model.decoder.parameters():
            #             param.requires_grad = True
            #         total_loss =  loss_kl_weight * loss_model.mean() + torch.clamp(decoder_loss.mean(), min=min_decoder_loss)
            # else:
            #     total_loss =  loss_kl_weight * loss_model.mean() + torch.clamp(decoder_loss.mean(), min=min_decoder_loss)
            total_loss.backward()
            opt.step()
            epoch_loss += total_loss

        ##### Test losses ##### 
        # model.eval()
        # with torch.no_grad():
        #     test_decoder_losses = 0
        #     test_loss_model = 0
        #     test_total_loss = 0
        #     test_prior_decoder_loss = 0
        #     j = 0
        #     for data in test_dataloader:
        #         j += 1
        #         red_obs, blue_obs, red_locs, reset = data
        #         this_batch_size = red_obs.shape[0]

        #         blue_obs = blue_obs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
        #         red_obs = red_obs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
        #         reset = reset.permute(1, 2, 0).to(device) #BxOxN -> OxNxB
        #         red_locs = red_locs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
        #         red_locs = red_locs[:, -1, :, :] # grab the last location in the inner dimension

        #         if obs_type == "red":
        #             loss_model, decoder_loss = model.compute_loss(red_obs, reset, red_locs, None, this_batch_size)
        #             prior_stochastic_decoder_loss = test_stochastic_rollouts(model, red_obs, reset, red_locs, num_outer_warmups, this_batch_size)
        #         else:
        #             loss_model, decoder_loss = model.compute_loss(blue_obs, reset, red_locs, None, this_batch_size)
        #             prior_stochastic_decoder_loss = test_stochastic_rollouts(model, blue_obs, reset, red_locs, num_outer_warmups, this_batch_size)
                
        #         test_decoder_losses += decoder_loss.mean() 
        #         test_loss_model += loss_model.mean()
        #         test_total_loss = loss_kl_weight * loss_model.mean() + decoder_loss.mean()

        #         test_prior_decoder_loss += prior_stochastic_decoder_loss.mean()
            
        #         writer.add_scalar('loss/test/prior_decoder', prior_stochastic_decoder_loss.mean(), j)
        #         writer.add_scalar('loss/test/decoder', decoder_loss.mean(), j)
        #         writer.add_scalar('loss/test/model', loss_model.mean(), j)
            
        #     overall_test_loss = test_total_loss / j
        #     writer.add_scalar('loss/test/overall', overall_test_loss, epoch)
        #     writer.add_scalar('loss/test/prior_decoder_epoch', test_prior_decoder_loss / j, epoch)
        #     if overall_test_loss < best_test_loss:
        #         best_test_loss = overall_test_loss
        #         torch.save(model.state_dict(), log_dir + '/model_best.pth')

        torch.save(model.red_rssm.state_dict(), log_dir + f'/red/{epoch}.pth')
        torch.save(model.blue_rssm.state_dict(), log_dir + f'/blue/{epoch}.pth')
        # torch.save(model.decoder.state_dict(), log_dir, f'/decoder/{epoch}.pth')
        torch.save(model.state_dict(), log_dir + f'/{epoch}.pth')
        torch.save(model, log_dir + f'/{epoch}_whole.pth')
                # torch.save(decoder.state_dict(), summary_dir + '/decoder_best.pth')
        # model.train()

        writer.add_scalar('loss/train/overall', epoch_loss/batch_size, epoch)
        scheduler.step()

if __name__ == "__main__":
    # conf, config_path = get_configs()
    red_rssm_path = "/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220512-1348/rssm/17.pth"
    blue_rssm_path = "/nethome/sye40/PrisonerEscape/logs/shared_latent/blue/20220512-1340/rssm/44.pth"
    config_path = os.path.join(os.path.dirname(os.path.dirname(red_rssm_path)), 'config.yaml')
    with open(config_path, 'r') as stream:
        red_conf = yaml.safe_load(stream)

    config_path = os.path.join(os.path.dirname(os.path.dirname(blue_rssm_path)), 'config.yaml')
    with open(config_path, 'r') as stream:
        blue_conf = yaml.safe_load(stream)
    train(red_conf, blue_conf, red_rssm_path, blue_rssm_path)