from dataset import RedBlueSequence, RedBlueSequenceSkip
from shared_latent.models.rssm import RSSMCore
from torch.utils.data import DataLoader
from torch import Tensor
import torch
import torch.distributions as D
from functions import logavgexp, flatten_batch, unflatten_batch
from shared_latent.models.decoder import MixtureDensityDecoder, DenseNormalDecoder, SingleGaussianDecoder
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def test_stochastic_rollouts(model, obs, reset, red_locs, this_batch_size):
    """ Given the rssm and a decoder model, check our prediction a few steps into the future """

    start_obs = obs[:4]
    start_reset = reset[:4]

    hidden_state = model.warm_start_rssm(start_obs, start_reset, None, this_batch_size)

    predict_obs = obs[4:]
    predict_locs = red_locs[4:]
    predict_resets = reset[4:]

    _, _, decoder_loss = model.compute_loss(predict_obs, predict_resets, predict_locs, hidden_state, this_batch_size, do_open_loop=True)

    return decoder_loss
    

def train(conf):

    # Set seeds
    torch.manual_seed(conf["seed"])
    np.random.seed(conf["seed"])
    random.seed(conf["seed"])


    np_file = np.load(conf["train_dataset_path"], allow_pickle=True)
    test_file = np.load(conf["test_dataset_path"], allow_pickle=True)

    total_resets = np.sum(np_file['dones'])
    print(total_resets)

    batch_size = conf["batch_size"] # B
    seq_len = conf["seq_len"] # T
    loss_kl_weight = conf["loss_kl_weight"] # KL weight

    # train_length = 124
    # red_blue_dataset = RedBlueSequence(np_file['red_observations'][:train_length], np_file['blue_observations'][:train_length], np_file['red_locations'][:train_length], np_file['dones'][:train_length], seq_len)
    red_blue_dataset = RedBlueSequence(np_file['red_observations'], np_file['blue_observations'], np_file['red_locations'], np_file['dones'], seq_len)
    # skip_step = 5
    # red_blue_dataset = RedBlueSequenceSkip(np_file['red_observations'], np_file['blue_observations'], np_file['red_locations'], np_file['dones'], skip_step, seq_len)
    train_dataloader = DataLoader(red_blue_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = RedBlueSequence(test_file['red_observations'], test_file['blue_observations'], test_file['red_locations'], test_file['dones'], seq_len)
    # test_dataset = RedBlueSequenceSkip(test_file['red_observations'], test_file['blue_observations'], test_file['red_locations'], test_file['dones'], skip_step, seq_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    obs_type = conf["obs_type"]

    summary_dir = f"logs/shared_latent/{obs_type}"

    # Initialize for writing on tensorboard
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(summary_dir, str(time))

    summary_dir = os.path.join(log_dir, 'summary')
    writer = SummaryWriter(log_dir=summary_dir)

    if obs_type == "red":
        in_dim =  red_blue_dataset[0][0].shape[-1]
    else:
        in_dim =  red_blue_dataset[0][1].shape[-1]

    model = Dreamer(conf)

    # Freeze the decoder weights
    # decoder_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220427-1508/summary/decoder_best.pth'
    # decoder_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/blue/20220502-1034/summary/10.pth'
    # model.load_state_dict(torch.load(decoder_path))
    # for param in model.decoder.parameters():
    #     param.requires_grad = False

    # opt_1 = torch.optim.Adam(rssm.parameters(), lr=0.001)
    # opt_2 = torch.optim.Adam(decoder.parameters(), lr=0.001)

    # opt = torch.optim.Adam([{'params': rssm.parameters()}, {'params': decoder.parameters()}], lr=3e-4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    # optimizers = [opt_1, opt_2]
    scaler = GradScaler(enabled=True)

    i = 0
    best_test_loss = float('inf')
    for epoch in tqdm(range(100)):
        epoch_loss = 0
        for data in train_dataloader:
            i += 1
            red_obs, blue_obs, red_locs, reset = data
            this_batch_size = red_obs.shape[0]

            blue_obs = blue_obs.permute(1, 0, 2).to(device)
            red_obs = red_obs.permute(1, 0, 2).to(device)
            reset = reset.permute(1, 0).to(device)
            red_locs = red_locs.permute(1, 0, 2).to(device)

            if obs_type == "red":
                loss_model, decoder_loss, reconstruction_loss = model.compute_loss(red_obs, reset, red_locs, None, this_batch_size)
            else:
                loss_model, decoder_loss, reconstruction_loss = model.compute_loss(blue_obs, reset, red_locs, None, this_batch_size)
                

            writer.add_scalar('loss/train/reconstruction', reconstruction_loss, i)
            writer.add_scalar('loss/train/decoder', decoder_loss.mean(), i)
            writer.add_scalar('loss/train/model', loss_model.mean(), i)

            # for opt in optimizers:
            opt.zero_grad()
        
            # loss = total_loss.mean(dim=0).mean()
            # if epoch < 10:
            #     total_loss = torch.clamp(decoder_loss.mean(), min=-4)
            # elif 20 < epoch < 40: 
            #     for param in model.decoder.parameters():
            #         param.requires_grad = False
            #     total_loss = torch.clamp(decoder_loss.mean(), min=-4) + reconstruction_loss.mean()            
            # else:
            #     for param in model.reconstructor.parameters():
            #         param.requires_grad = False
            total_loss = loss_kl_weight * loss_model.mean() + torch.clamp(decoder_loss.mean(), min=-4) + reconstruction_loss.mean()
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
            j = 0
            for data in test_dataloader:
                j += 1
                red_obs, blue_obs, red_locs, reset = data
                this_batch_size = red_obs.shape[0]

                blue_obs = blue_obs.permute(1, 0, 2).to(device)
                red_obs = red_obs.permute(1, 0, 2).to(device)
                reset = reset.permute(1, 0).to(device)
                red_locs = red_locs.permute(1, 0, 2).to(device)

                if obs_type == "red":
                    loss_model, decoder_loss, reconstruction_loss = model.compute_loss(red_obs, reset, red_locs, None, this_batch_size)
                    prior_stochastic_decoder_loss = test_stochastic_rollouts(model, red_obs, reset, red_locs, this_batch_size)
                else:
                    loss_model, decoder_loss, reconstruction_loss = model.compute_loss(blue_obs, reset, red_locs, None, this_batch_size)
                    prior_stochastic_decoder_loss = test_stochastic_rollouts(model, blue_obs, reset, red_locs, this_batch_size)
                
                test_decoder_losses += decoder_loss.mean() 
                test_loss_model += loss_model.mean()
                test_total_loss = loss_kl_weight * loss_model.mean() + decoder_loss.mean()

                test_prior_decoder_loss += prior_stochastic_decoder_loss.mean()
            
                writer.add_scalar('loss/test/prior_decoder', prior_stochastic_decoder_loss.mean(), j)
                writer.add_scalar('loss/test/decoder', decoder_loss.mean(), j)
                writer.add_scalar('loss/test/model', loss_model.mean(), j)
            
            overall_test_loss = test_total_loss / j
            writer.add_scalar('loss/test/overall', overall_test_loss, epoch)
            writer.add_scalar('loss/test/prior_decoder_epoch', test_prior_decoder_loss / j, epoch)
            if overall_test_loss < best_test_loss:
                best_test_loss = overall_test_loss
                torch.save(model.state_dict(), summary_dir + '/model_best.pth')

            torch.save(model.state_dict(), summary_dir + f'/{epoch}.pth')
                # torch.save(decoder.state_dict(), summary_dir + '/decoder_best.pth')
        model.train()

        writer.add_scalar('loss/train/overall', epoch_loss/batch_size, epoch)

if __name__ == "__main__":
    conf, config_path = get_configs()
    train(conf)