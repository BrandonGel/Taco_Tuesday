import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch import distributions

# experiment to see vae can work with this dataset
# from datasets.dataset_old import RedBlueDataset, RedBlueSequence, RedBlueSequenceOriginal
# from datasets.multi_head_dataset_old import MultiHeadDataset

from datasets.
from torch.utils.data import DataLoader
from experimental.filter_dataset import filter_dataset


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.enc_mu = torch.nn.Linear(hidden_dim, latent_size)
        self.enc_log_sigma = torch.nn.Linear(hidden_dim, latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        # sigma = nn.ELU()(sigma) + 1 + 1e-15
        return torch.distributions.Normal(loc=mu, scale=sigma)

class Decoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        self.linear1 = torch.nn.Linear(latent_size, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = torch.tanh(self.linear2(x))
        return torch.distributions.Normal(mu, torch.ones_like(mu))

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        q_z = self.encoder(x)
        z = q_z.rsample()
        return self.decoder(z), q_z
    
if __name__ == '__main__':
    # load data
    # pull the continuous data from the red dataset
    # path = "/nethome/sye40/PrisonerClean/datasets/eps_0/map_0_run_300_eps_0_normalized.npz"
    path = "/nethome/sye40/PrisonerEscape/datasets/map_0_run_100_eps_0.1_norm.npz"
    np_file = np.load(path, allow_pickle=True)

    prediction_obs_dict = np_file['prediction_dict'].item()
    blue_obs_dict = np_file['blue_dict'].item()

    # Include time, prisoner_location
    # feature_names = ['time', 'prisoner_loc', 'camera_loc', 'search_party_detect', 'helicopter_detect', 'prev_action', 'hideout_loc']
    # feature_names = ['time', 'prisoner_loc', 'search_party_detect', 'helicopter_detect', 'prev_action', 'hideout_loc']
    feature_names = ['camera_loc']
    # print(prediction_obs_names._idx_dict)

    continuous, discrete = filter_dataset(np_file['red_observations'], prediction_obs_dict, feature_names, split_categorical=True)
    # print(discrete.shape)
    print(continuous.shape)
    # print(continuous)
    # print(discrete[0])

    seq_len = 1
    future_step = 0
    red_blue_dataset = RedBlueSequence(
        continuous, 
        np_file["blue_observations"], 
        np_file["red_locations"], 
        np_file["dones"], 
        seq_len, future_step, "red")

    dataloader = DataLoader(red_blue_dataset, batch_size=128, shuffle=True)
    # print(red_blue_dataset[0][0].shape)
    print(red_blue_dataset[0][0])

    input_dim = continuous.shape[1]
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.001
    hidden_size = 32
    latent_size = 8

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder = Encoder(input_dim, hidden_size, latent_size)
    decoder = Decoder(latent_size, hidden_size, input_dim)

    vae = VAE(encoder, decoder).to(device)

    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for inputs, y in dataloader:
            inputs = inputs.view(-1,  input_dim).to(device).float()
            optimizer.zero_grad()
            p_x, q_z = vae(inputs)
            log_likelihood = p_x.log_prob(inputs).sum(-1).mean()
            kl = torch.distributions.kl_divergence(
                q_z, 
                torch.distributions.Normal(0, 1.)
            ).sum(-1).mean()
            loss = -(log_likelihood - kl)
            loss.backward()
            optimizer.step()
            l = loss.item()
        print(epoch, l, log_likelihood.item(), kl.item())
        # take a sample from the decoder
        z = torch.randn(1, latent_size).to(device)
        x = vae.decoder(z).loc.detach().cpu().numpy()
        # print(x)