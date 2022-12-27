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
import random

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os
from datetime import datetime
from torch.distributions import Normal
from torch.optim.lr_scheduler import ExponentialLR
from shared_latent.filter_dataset import filter_dataset

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

    _, decoder_loss, _ = model.compute_loss(predict_obs, predict_resets, predict_locs, hidden_state, this_batch_size, do_open_loop=True)

    return decoder_loss
    

def train(conf, config_path):

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
    # remove the camera locations from the dataset
    feature_names = ['time', 'prisoner_loc', 'search_party_detect', 'helicopter_detect', 'prev_action', 'hideout_loc']
    prediction_obs_dict = np_file['prediction_dict'].item()
    # red_file = filter_dataset(np_file['red_observations'][:train_length], prediction_obs_dict, feature_names, split_categorical=False)
    # red_blue_dataset = RedBlueSequenceEncoded(red_file, np_file['blue_observations'][:train_length], np_file['red_locations'][:train_length], np_file['dones'][:train_length], 
    #                             outer_seq, inner_seq)

    red_file = filter_dataset(np_file['red_observations'], prediction_obs_dict, feature_names, split_categorical=False)
    red_blue_dataset = RedBlueSequenceEncoded(red_file, np_file['blue_observations'], np_file['red_locations'], np_file['dones'], 
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
    
    feature_names = ['time', 'prisoner_loc', 'search_party_detect', 'helicopter_detect', 'prev_action', 'hideout_loc']
    prediction_obs_dict = test_file['prediction_dict'].item()
    red_file_test = filter_dataset(test_file['red_observations'], prediction_obs_dict, feature_names, split_categorical=False)
    
    test_dataset = RedBlueSequenceEncoded(red_file_test, 
                                          test_file['blue_observations'], 
                                          test_file['red_locations'], 
                                          test_file['dones'], 
                            total_outer, inner_seq)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    obs_type = conf["obs_type"]

    summary_dir = f"logs/shared_latent/{obs_type}"

    # Initialize for writing on tensorboard
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(summary_dir, str(time))

    decoder_path = os.path.join(log_dir, "decoder")
    os.makedirs(decoder_path, exist_ok=True)

    rssm_path = os.path.join(log_dir, "rssm")
    os.makedirs(rssm_path, exist_ok=True)

    summary_dir = os.path.join(log_dir, 'summary')
    writer = SummaryWriter(log_dir=summary_dir)

    # copy config file to summary directory
    shutil.copy(config_path, os.path.join(log_dir, 'config.yaml'))

    if obs_type == "red":
        in_dim =  red_blue_dataset[0][0].shape[-1]
    else:
        in_dim =  red_blue_dataset[0][1].shape[-1]

    model = Dreamer(conf)

    # Load the model where reconstruction was trained
    # model.load_state_dict(torch.load("/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220515-2057/78.pth"))
    # for param in model.reconstructor.parameters():
    #     param.requires_grad = False

    # # Freeze the decoder weights
    # decoder_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220510-1455/summary/9.pth'
    # model.load_state_dict(torch.load(decoder_path))
    # for param in model.decoder.parameters():
    #     param.requires_grad = False

    # opt_1 = torch.optim.Adam(rssm.parameters(), lr=0.001)
    # opt_2 = torch.optim.Adam(decoder.parameters(), lr=0.001)

    # opt = torch.optim.Adam([{'params': rssm.parameters()}, {'params': decoder.parameters()}], lr=3e-4)
    opt = torch.optim.Adam(model.parameters(), lr=conf["learning_rate"])
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

            # print(red_locs[0, 0, :], red_locs[1, 0, :])

            if obs_type == "red":
                loss_model, decoder_loss, reconstruction_loss = model.compute_loss(red_obs, reset, red_locs, None, this_batch_size)
            else:
                loss_model, decoder_loss, reconstruction_loss = model.compute_loss(blue_obs, reset, red_locs, None, this_batch_size)
                

            writer.add_scalar('loss/train/reconstruction', reconstruction_loss.mean(), i)
            writer.add_scalar('loss/train/decoder', decoder_loss.mean(), i)
            writer.add_scalar('loss/train/model', loss_model.mean(), i)

            # for opt in optimizers:
            opt.zero_grad()
        
            schedule = True
            if schedule:
                if epoch < 4:
                    total_loss = torch.clamp(decoder_loss.mean(), min=min_decoder_loss) + reconstruction_loss.mean()
                    # total_loss = reconstruction_loss.mean()
                elif 4 <= epoch < 10 :
                    for param in model.decoder.parameters():
                        param.requires_grad = False
                    total_loss =  loss_kl_weight * loss_model.mean() + torch.clamp(decoder_loss.mean(), min=min_decoder_loss) + reconstruction_loss.mean()  
                else:
                    for param in model.decoder.parameters():
                        param.requires_grad = True
                    total_loss =  loss_kl_weight * loss_model.mean() + torch.clamp(decoder_loss.mean(), min=min_decoder_loss) + reconstruction_loss.mean()
            else:
                # total_loss =  loss_kl_weight * loss_model.mean() + torch.clamp(decoder_loss.mean(), min=min_decoder_loss) + reconstruction_loss.mean()
                total_loss =  loss_kl_weight * loss_model.mean() + decoder_loss.mean() + reconstruction_loss.mean()
                # total_loss = reconstruction_loss.mean()
            #     for param in model.reconstructor.parameters():
            #         param.requires_grad = False
            #     total_loss = loss_kl_weight * loss_model.mean() + torch.clamp(decoder_loss.mean(), min=-4)
            # loss = decoder_loss.mean()
            total_loss.backward()
            # scaler.scale(loss).backward()
            
            # for opt in optimizers:
            opt.step()
            # scaler.unscale_(opt)
            # scaler.step(opt)
            # scaler.update()

            # print(loss)
            epoch_loss += total_loss

        ##### Test losses ##### 
        model.eval()
        with torch.no_grad():
            test_decoder_losses = 0
            test_loss_model = 0
            test_total_loss = 0
            test_prior_decoder_loss = 0
            test_reconstruction_loss = 0
            j = 0
            for data in test_dataloader:
                j += 1
                red_obs, blue_obs, red_locs, reset = data
                this_batch_size = red_obs.shape[0]

                blue_obs = blue_obs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
                red_obs = red_obs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
                reset = reset.permute(1, 2, 0).to(device) #BxOxN -> OxNxB
                red_locs = red_locs.permute(1, 2, 0, 3).to(device) #BxOxNxD -> OxNxBxD
                red_locs = red_locs[:, -1, :, :] # grab the last location in the inner dimension

                if obs_type == "red":
                    loss_model, decoder_loss, reconstruction_loss = model.compute_loss(red_obs, reset, red_locs, None, this_batch_size)
                    prior_stochastic_decoder_loss = test_stochastic_rollouts(model, red_obs, reset, red_locs, num_outer_warmups, this_batch_size)
                else:
                    loss_model, decoder_loss, reconstruction_loss = model.compute_loss(blue_obs, reset, red_locs, None, this_batch_size)
                    prior_stochastic_decoder_loss = test_stochastic_rollouts(model, blue_obs, reset, red_locs, num_outer_warmups, this_batch_size)
                
                test_decoder_losses += decoder_loss.mean() 
                test_loss_model += loss_model.mean()
                test_total_loss = loss_kl_weight * loss_model.mean() + decoder_loss.mean()

                test_prior_decoder_loss += prior_stochastic_decoder_loss.mean()
                test_reconstruction_loss += reconstruction_loss.mean()
            
                writer.add_scalar('loss/test/prior_decoder', prior_stochastic_decoder_loss.mean(), j)
                writer.add_scalar('loss/test/decoder', decoder_loss.mean(), j)
                writer.add_scalar('loss/test/model', loss_model.mean(), j)
            
            overall_test_loss = test_total_loss / j
            writer.add_scalar('loss/test/overall', overall_test_loss, epoch)
            writer.add_scalar('loss/test/prior_decoder_epoch', test_prior_decoder_loss / j, epoch)
            writer.add_scalar('loss/test/reconstruction_epoch', test_reconstruction_loss / j, epoch)
            if overall_test_loss < best_test_loss:
                best_test_loss = overall_test_loss
                torch.save(model.state_dict(), log_dir + '/model_best.pth')

            torch.save(model.state_dict(), log_dir + f'/{epoch}.pth')
            torch.save(model.rssm.state_dict(), rssm_path + f'/{epoch}.pth')
            torch.save(model.decoder.state_dict(), decoder_path + f'/{epoch}.pth')
                # torch.save(decoder.state_dict(), summary_dir + '/decoder_best.pth')
        model.train()

        writer.add_scalar('loss/train/overall', epoch_loss/batch_size, epoch)
        scheduler.step()

if __name__ == "__main__":
    conf, config_path = get_configs()
    train(conf, config_path)