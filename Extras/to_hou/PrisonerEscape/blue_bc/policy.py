from turtle import forward
import numpy as np
import torch
from torch import embedding, nn
import torch.nn.functional as F
# from blue_bc.utils import gumbel_softmax_soft_hard
from blue_bc.bc_utils import build_mlp, build_mlp_hidden_layers, reparameterize, reparameterize_sigmoid, evaluate_lop_pi, evaluate_lop_pi_para, atanh, calculate_log_pi


class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0])) # initialized as all zero

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))


class LSTMMDNBluePolicy(nn.Module):

    def __init__(self, num_features, num_actions, hidden_dims, mdn_num, hidden_act_func, output_func, device):
        super(LSTMMDNBluePolicy, self).__init__()
        self.device = device
        self.hidden_dims = hidden_dims
        self.mdn_num = mdn_num # m
        self.output_dim = num_actions # c
        self.lstm_layer_num = 1
        self.eps = 1e-8

        """MLP layer"""
        self.mlp0 = nn.Linear(num_features, self.hidden_dims[0])
        self.nonlinear0 = hidden_act_func
        """LSTM layer"""
        self.lstm1 = nn.LSTM(input_size=self.hidden_dims[0], hidden_size=self.hidden_dims[1], batch_first=True)
        self.nonlinear1 = hidden_act_func
        """MLP layer to generate alpha, sigma, mu"""
        self.alpha_layer = nn.Linear(self.hidden_dims[1], mdn_num)
        self.sigma_layer = nn.Linear(self.hidden_dims[1], self.output_dim * mdn_num)
        self.mu_layer = nn.Linear(self.hidden_dims[1], self.output_dim * mdn_num)
        """output layer"""
        self.output_layer = output_func
        """utils layer"""
        self.softmax = nn.Softmax(dim=2)


        # # self.h1 = nn.Linear(hidden_dim, hidden_dim)
        # self.out_heading = nn.Linear(hidden_dim, 1)
        # self.out_speed = nn.Linear(hidden_dim, 1)

        # self.embeddings_num = hidden_dim
        # self.alpha_layer = nn.Linear(self.embeddings_num, mdn_num)
        # self.sigma_layer = nn.Linear(self.embeddings_num, mdn_num)
        # self.mu_layer = nn.Linear(self.embeddings_num, self.output_dim * mdn_num)

        # self.nonlinear_relu = nn.ReLU()
        # self.nonlinear_tanh = nn.Tanh()
        # self.nonlinear_sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=2)
    def sample(self, x):
        x = x.unsqueeze(0)
        alpha, log_sigma, mu = self.alpha_sigma_mu(x)
        alpha = alpha.squeeze()
        a = torch.tensor(range(self.mdn_num))
        # p = torch.tensor([0.1, 0.1, 0.1, 0.7])
        i = alpha.multinomial(num_samples=1, replacement=True)
        log_sigma_each_gs = log_sigma[:,:,self.output_dim*i:self.output_dim*(i+1)].squeeze()
        mu_each_gs = mu[:,:,self.output_dim*i:self.output_dim*(i+1)].squeeze()
        reparameterize(mu_each_gs, log_sigma_each_gs)
        return 

    def forward(self, x):
        x = x.unsqueeze(0)
        alpha, log_sigma, mu = self.alpha_sigma_mu(x)
        if alpha.numel() != 1:
            alpha = alpha.squeeze()
            i = alpha.multinomial(num_samples=1, replacement=True)
            mu_each_gs = mu[:,:,self.output_dim*i:self.output_dim*(i+1)].squeeze()
        else:
            mu_each_gs = mu.squeeze()
        return torch.tanh(mu_each_gs)

    def alpha_sigma_mu(self, x):
        # input of LSTM: tensor of shape (L, N, H_{in}) when batch_first is false, otherwise (N, L, H_{in})
        # output of LSTM: tensor of shape (L, N, D * H_{out}) when batch_first is false, otherwise (N, L, D * H_{out})
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        """pass into first MLP layer"""
        x = self.mlp0(x)
        x = self.nonlinear0(x)
        """initialize hidden states and cell states"""
        h0 = torch.zeros(self.lstm_layer_num, batch_size, self.hidden_dims[1]).requires_grad_().to(self.device)
        c0 = torch.zeros(self.lstm_layer_num, batch_size, self.hidden_dims[1]).requires_grad_().to(self.device)
        """pass into lstm layer"""
        hiddens, (hn, _) = self.lstm1(x, (h0, c0)) # hiddens: (N, L, D * H_{out}), L is the sequence length, batch first
        hiddens = self.nonlinear1(hiddens[:,seq_len-1:seq_len,:]) # (N, L, D * H_{out})
        """pass into MDN parameter calculation layers"""
        alpha = self.softmax(self.alpha_layer(hiddens)) # (N, L=1, m=5)
        log_sigma = self.sigma_layer(hiddens) # (N, L=1, c * m = 12 * 5)
        mu = self.mu_layer(hiddens) # (N, L=1, c * m = 12 * 5)
        return alpha, log_sigma, mu

    def evaluate_action_prob(self, x, y):
        alpha, log_sigma, mu = self.alpha_sigma_mu(x)
        sum_all_log_gs = torch.Tensor(np.array([0])).to(self.device)
        sum_all_gs = torch.Tensor(np.array([self.eps])).to(self.device)
        for i in range(self.mdn_num):
            alpha_each_gs = alpha[:,:,i]
            log_sigma_each_gs = log_sigma[:,:,self.output_dim*i:self.output_dim*(i+1)].squeeze()
            mu_each_gs = mu[:,:,self.output_dim*i:self.output_dim*(i+1)].squeeze()
            """noises should obey N(mu=0, sigma=1)"""
            noises = (atanh(y) - mu_each_gs) / (log_sigma_each_gs.exp() + self.eps)
            log_p_noises_standard_gs = calculate_log_pi(log_sigma_each_gs, noises, y)
            # sum_all_gs = sum_all_gs + alpha_each_gs * log_p_noises_standard_gs.exp()
            sum_all_log_gs = sum_all_log_gs + alpha_each_gs * log_p_noises_standard_gs
        neg_ln_mdn = -sum_all_log_gs.mean()
        mu_average = mu.mean()
        log_sigma_average = log_sigma.mean()
        # print("sum_all_gs_max = ", sum_all_gs.max())
        # neg_ln_mdn = -sum_all_gs.log().mean()
        return neg_ln_mdn, mu_average, log_sigma_average

    def mdn_gaussian_distribution(self, alpha, sigma, mu, y):
        prob = alpha * ((1.0 / (torch.pow(2 * torch.pi, torch.Tensor(np.array(self.output_dim / 2.0)).to(self.device)) * sigma))) * torch.exp(-torch.pow(torch.linalg.norm(y - mu, dim=1, keepdim=True), 2) / (2 * torch.pow(sigma, 2)))
        return prob

    def evaluate_loss(self, x, y):
        loss, mu_average, log_sigma_average = self.evaluate_action_prob(x, y)
        # loss = torch.mean(neg_ln_mdn)
        stats_dict = dict(neglogp=loss.item(), mu_average=mu_average, log_sigma_average=log_sigma_average)
        return loss, stats_dict

    def predict(self, x, deterministic=None):
        obs = torch.from_numpy(x).float().unsqueeze(0).to("cuda")
        return self.alpha_sigma_mu(obs).cpu().detach().numpy()

class HighLevelPolicy(nn.Module):

    def __init__(self, state_shape, agent_num, subpolicy_shape, para_shape, hidden_units=(64, 64), hidden_activation=nn.Tanh()):
        super().__init__()
        self.agent_num = agent_num
        self.subpolicy_shape = subpolicy_shape
        self.para_shape = para_shape
        self.net = build_mlp_hidden_layers(
            input_dim=state_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.subpolicy_fc1 = build_mlp_hidden_layers(input_dim=hidden_units[-1], hidden_units=(hidden_units[-1],), hidden_activation=hidden_activation)
        self.subpolicy_fc2 = nn.Linear(hidden_units[-1], subpolicy_shape[0] * agent_num)
        self.para_fc1 = build_mlp_hidden_layers(input_dim=hidden_units[-1], hidden_units=(hidden_units[-1],), hidden_activation=hidden_activation)
        self.para_fc2 = nn.Linear(hidden_units[-1], para_shape[0] * agent_num)
        self.subpolicy_output = F.gumbel_softmax
        self.para_output = nn.Sigmoid()
        # self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0])) # initialized as all zero

    def forward(self, states):
        x = self.net(states)
        subpolicy = self.subpolicy_fc2(self.subpolicy_fc1(x)).view(-1, self.agent_num, self.subpolicy_shape[0])
        subpolicy = self.subpolicy_output(subpolicy, tau=1, hard=True, eps=1e-10, dim=-1)
        # subpolicy
        paras = self.para_fc2(self.para_fc1(x)).view(-1, self.agent_num, self.para_shape[0])
        paras = self.para_output(paras)
        # x = self.output(x)
        print("subpolicy = ", subpolicy)
        print("paras = ", paras)
        return subpolicy, paras


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, norm_in=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class HierRandomPolicy(nn.Module):
    def __init__(self, state_shape, agent_num, subpolicy_shape, para_shape, hidden_units=(64, 64), hidden_activation=nn.ReLU()) -> None:
        super().__init__()
        self.agent_num = agent_num
        self.subpolicy_shape = subpolicy_shape
        self.para_shape = para_shape
        self.common_layer = build_mlp_hidden_layers(input_dim=state_shape[0], hidden_units=hidden_units, hidden_activation=hidden_activation)
        self.subpolicy_fc1 = build_mlp_hidden_layers(input_dim=hidden_units[-1], hidden_units=(hidden_units[-1],), hidden_activation=hidden_activation)
        self.subpolicy_fc2 = nn.Linear(hidden_units[-1], subpolicy_shape[0] * agent_num)
        self.para_fc1 = build_mlp_hidden_layers(input_dim=hidden_units[-1], hidden_units=(hidden_units[-1],), hidden_activation=hidden_activation)
        self.para_fc2 = nn.Linear(hidden_units[-1], para_shape[0] * agent_num)
        self.subpolicy_output = gumbel_softmax_soft_hard
        self.para_output = nn.Sigmoid()
        self.log_stds = nn.Parameter(torch.zeros(1, self.agent_num * para_shape[0])) # initialized as all zero

    def forward(self, states):
        # x = self.common_layer(states)
        # subpolicy = self.subpolicy_fc2(self.subpolicy_fc1(x)).view(-1, self.agent_num, self.subpolicy_shape[0])
        # subpolicy_prob, subpolicy = self.subpolicy_output(subpolicy, tau=1, eps=1e-10, dim=-1)
        # # subpolicy
        # paras_mean_to_sigmoid = self.para_fc2(self.para_fc1(x)).view(-1, self.agent_num, self.para_shape[0])
        # paras = self.para_output(paras_mean_to_sigmoid)
        subpolicy_prob, subpolicy, paras_mean_to_sigmoid = self.generate_dist(states)
        paras, log_paras = reparameterize_sigmoid(paras_mean_to_sigmoid, self.log_stds)
        paras = paras.view(-1, self.agent_num, self.para_shape[0])
        log_subpolicy = torch.log(subpolicy_prob[subpolicy==1] + 1e-6).sum()
        log_pi = log_subpolicy
        # print("subpolicy_prob, subpolicy = ", (subpolicy_prob, subpolicy))
        # print("subpolicy_prob = ", subpolicy_prob)
        # print("log_subpolicy = %f, log_paras = %f" % (log_subpolicy, log_paras))
        return subpolicy, paras, log_pi

    def generate_dist(self, states):
        x = self.common_layer(states)
        subpolicy = self.subpolicy_fc2(self.subpolicy_fc1(x)).view(-1, self.agent_num, self.subpolicy_shape[0])
        subpolicy_prob, subpolicy = self.subpolicy_output(subpolicy, tau=3, eps=1e-10, dim=-1)
        # subpolicy
        paras_mean_to_sigmoid = self.para_fc2(self.para_fc1(x)).view(-1, self.agent_num, self.para_shape[0])
        return subpolicy_prob, subpolicy, paras_mean_to_sigmoid

    # def sample(self, states):
    #     return reparameterize(self.net(states), self.log_stds)

    # def evaluate_log_subpolicy_para_pi(self, states, subpolicy_prob, subpolicy, paras_mean_to_sigmoid):
    #     act_policy_prob = subpolicy_prob[subpolicy==1]
    #     act_para_prob = evaluate_lop_pi_para(paras_mean_to_sigmoid, self.log_stds, paras_act)
    #     # subpolicy_prob, paras_mean_to_sigmoid = self.generate_dist(states)
    #     # subpolicy_act = actions[0]
    #     # paras_act = actions[1]

    #     return 

    # def evaluate_log_subpolicy_pi()

def gumbel_softmax_soft_hard(logits, tau=1, eps=1e-10, dim=-1):
    probs = F.gumbel_softmax(logits, tau, hard=False, eps=eps, dim=dim)
    index = probs.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    one_hot = y_hard - probs.detach() + probs
    return probs, one_hot.detach()
       



