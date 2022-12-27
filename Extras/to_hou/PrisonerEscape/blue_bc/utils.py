import yaml
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
import os
import copy
from blue_bc.policy import MLPNetwork

def save_video(ims, filename, fps=30.0):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

def gumbel_softmax_soft_hard(logits, tau=1, eps=1e-10, dim=-1):
    probs = F.gumbel_softmax(logits, tau, hard=False, eps=eps, dim=dim)
    one_hot = F.gumbel_softmax(logits, tau, hard=True, eps=eps, dim=dim)
    return probs, one_hot.detach()


class HierScheduler(object):
    def __init__(self, env, subpolicy, paras, new_detection, detection_history, timesteps) -> None:
        self.env = env
        self.subpolicy = subpolicy.squeeze().detach().cpu().numpy()
        self.paras = paras.squeeze().detach().cpu().numpy()
        self.agent_num = self.env.num_search_parties + self.env.num_helicopters 
        self.option_num = subpolicy.shape[-1]
        self.option_names = ["plan_path_to_stop", "plan_path_to_random_para", "plan_path_to_loc_para", "plan_spiral_para", "plan_path_to_intercept_para"]
        self.para_indices = [[], [0], [[0],[1,2]], [0], [0]]
        self.new_detection = new_detection
        self.detection_history = detection_history
        self.timesteps = timesteps
    
    @property
    def update_path_flag(self):
        if self.new_detection is not None:
            flag = True
        else:
            flag = False
        return flag

    @property
    def options(self):
        # opt = self.subpolicy.squeeze()
        options = np.nonzero(self.subpolicy)[-1]
        return options

    def plan_path(self):
        if self.update_path_flag:
            blue_ag_idx = 0
            for search_party in self.env.search_parties_list:
                opt_idx = self.options[blue_ag_idx]
                self.command_agent(opt_idx, blue_ag_idx, search_party)
                # if self.para_indices[opt_idx] == []:
                #     getattr(search_party, self.option_names[opt_idx])()
                # else:
                #     getattr(search_party, self.option_names[opt_idx])(*([self.paras[blue_ag_idx,self.para_indices[opt_idx][i]] for i in range(len(self.para_indices[opt_idx]))]))
                blue_ag_idx = blue_ag_idx + 1

            if self.env.is_helicopter_operating():
                for helicopter in self.env.helicopters_list:
                    opt_idx = self.options[blue_ag_idx]
                    self.command_agent(opt_idx, blue_ag_idx, helicopter)
                    # if self.para_indices[opt_idx] == []:
                    #     getattr(helicopter, self.option_names[opt_idx])()
                    # else:
                    #     getattr(search_party, self.option_names[opt_idx])(*([self.paras[blue_ag_idx,self.para_indices[opt_idx][i]] for i in range(len(self.para_indices[opt_idx]))]))
                    blue_ag_idx = blue_ag_idx + 1
        else:
            pass
         
    def command_agent(self, opt_idx, blue_ag_idx, blue_agent):
        if opt_idx == 0:
            blue_agent.plan_path_to_stop_para()
        elif opt_idx == 1:
            blue_agent.plan_path_to_random_para(self.paras[blue_ag_idx,0])
        elif opt_idx == 2:
            blue_agent.plan_path_to_loc_para(self.paras[blue_ag_idx,0], self.paras[blue_ag_idx,[1,2]])
        elif opt_idx == 3:
            blue_agent.plan_spiral_para(self.paras[blue_ag_idx,0])
        elif opt_idx == 4:
            vector = np.array(self.new_detection) - np.array(self.detection_history[-2][0])
            speed = np.sqrt(np.sum(np.square(vector))) / (self.timesteps - self.detection_history[-2][1])
            direction = np.arctan2(vector[1], vector[0])
            blue_agent.plan_path_to_intercept_para(self.paras[blue_ag_idx,0], speed, direction, self.new_detection)
        else:
            raise NotImplementedError

    def get_each_action(self):
        # get the action for each party
        actions = []
        for search_party in self.env.search_parties_list:
            action = np.array(search_party.get_action_according_to_plan_para())
            actions.append(action)
        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                action = np.array(helicopter.get_action_according_to_plan_para())
                actions.append(action)
        else:
            for helicopter in self.env.helicopters_list:
                action = np.array([0, 0, 0])
                actions.append(action)
        return actions

MSELoss = torch.nn.MSELoss()
MSELoss_each = torch.nn.MSELoss(reduction='none')
class BaseTrainer(object):
    def __init__(self, blue_policy, input_dim, out_dim, hidden_dim, update_target_period, lr, device) -> None:
        self.blue_policy = blue_policy.to(device)
        self.target_blue_policy = copy.deepcopy(self.blue_policy).to(device)
        self.blue_critic = MLPNetwork(input_dim, out_dim, hidden_dim, nonlin=torch.nn.functional.relu, constrain_out=False, norm_in=False, discrete_action=False).to(device)
        self.target_blue_critic = copy.deepcopy(self.blue_critic).to(device)
        self.policy_optimizer = torch.optim.Adam(self.blue_policy.parameters(), lr=lr) # original: lr
        self.critic_optimizer = torch.optim.Adam(self.blue_critic.parameters(), lr=lr)

        self.niter = 0
        self.update_target_period = update_target_period
        self.gamma = 0.95
        self.tau = 0.01

    def update(self, sample, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, hier_act, rewards, dones, next_obs = sample
        batch_size = obs.shape[0]
        vf_in = torch.cat((obs, hier_act.detach()), dim=1)
        actual_value = self.blue_critic(vf_in) # reward_prediction(from t)

        trgt_acs = self.target_blue_policy(next_obs)
        # trgt_acs = torch.cat((trgt_acs[0].view(batch_size, -1), trgt_acs[1].view(batch_size, -1)), dim=1)
        trgt_vf_in = torch.cat((next_obs, trgt_acs), dim=1)
        target_value = (rewards.view(-1, 1) + self.gamma * self.target_blue_critic(trgt_vf_in) * (1 - dones.view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)

        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        self.critic_optimizer.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm(self.blue_critic.parameters(), 0.5)
        self.critic_optimizer.step()
        # curr_agent.scheduler_critic_optimizer.step()

        curr_pol_out = self.blue_policy(obs)
        # curr_pol_out = torch.cat((curr_pol_out[0].view(batch_size, -1), curr_pol_out[1].view(batch_size, -1)), dim=1)
        vf_in = torch.cat((obs, curr_pol_out), dim=1)
        pol_loss = -self.blue_critic(vf_in).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        self.policy_optimizer.zero_grad()
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm(self.blue_policy.parameters(), 0.5)
        self.policy_optimizer.step()

        # if self.niter % self.update_target_period == 0:
        self.update_all_targets()

        if logger is not None:
            logger.add_scalars('blue/losses',
                               {'vf_loss': torch.mean(vf_loss_each),
                                'td_error': torch.mean(td_error_each),
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        self.soft_update(self.target_blue_critic, self.blue_critic, self.tau)
        self.soft_update(self.target_blue_policy, self.blue_policy, self.tau)
        self.niter += 1

    def soft_update(self, target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class HierTrainer(object):
    def __init__(self, blue_policy, input_dim, out_dim, hidden_dim, update_target_period, lr, device) -> None:
        self.blue_policy = blue_policy.to(device)
        self.target_blue_policy = copy.deepcopy(self.blue_policy).to(device)
        self.blue_critic = MLPNetwork(input_dim, out_dim, hidden_dim, nonlin=torch.nn.functional.relu, constrain_out=False, norm_in=False, discrete_action=False).to(device)
        self.target_blue_critic = copy.deepcopy(self.blue_critic).to(device)
        self.policy_optimizer = torch.optim.Adam(self.blue_policy.parameters(), lr=0.1 * lr) # original: lr
        self.critic_optimizer = torch.optim.Adam(self.blue_critic.parameters(), lr=lr)

        self.niter = 0
        self.update_target_period = update_target_period
        self.gamma = 0.95
        self.tau = 0.01

    def update(self, sample, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, hier_act, rewards, dones, next_obs = sample
        batch_size = obs.shape[0]
        vf_in = torch.cat((obs, hier_act.detach()), dim=1)
        actual_value = self.blue_critic(vf_in) # reward_prediction(from t)

        trgt_acs = self.target_blue_policy(next_obs)
        trgt_acs = torch.cat((trgt_acs[0].view(batch_size, -1), trgt_acs[1].view(batch_size, -1)), dim=1)
        trgt_vf_in = torch.cat((next_obs, trgt_acs), dim=1)
        target_value = (rewards.view(-1, 1) + self.gamma * self.target_blue_critic(trgt_vf_in)) # current_reward_in_buffer + reward_prediction(from t+1)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)

        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        self.critic_optimizer.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm(self.blue_critic.parameters(), 0.5)
        self.critic_optimizer.step()
        # curr_agent.scheduler_critic_optimizer.step()

        curr_pol_out = self.blue_policy(obs)
        curr_pol_out = torch.cat((curr_pol_out[0].view(batch_size, -1), curr_pol_out[1].view(batch_size, -1)), dim=1)
        vf_in = torch.cat((obs, curr_pol_out), dim=1)
        pol_loss = -self.blue_critic(vf_in).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        self.policy_optimizer.zero_grad()
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm(self.blue_policy.parameters(), 0.5)
        self.policy_optimizer.step()

        # if self.niter % self.update_target_period == 0:
        self.update_all_targets()

        if logger is not None:
            logger.add_scalars('blue/losses',
                               {'vf_loss': torch.mean(vf_loss_each),
                                'td_error': torch.mean(td_error_each),
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        self.soft_update(self.target_blue_critic, self.blue_critic, self.tau)
        self.soft_update(self.target_blue_policy, self.blue_policy, self.tau)
        self.niter += 1

    def soft_update(self, target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)