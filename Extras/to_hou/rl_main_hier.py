import os
import argparse
import time
from pathlib import Path

from cv2 import VIDEOWRITER_PROP_FRAMEBYTES
import sys
import yaml
import copy

project_path = os.getcwd()
sys.path.append(str(project_path))
from simulator.forest_coverage.autoencoder import train
from simulator import BlueSequenceEnv
from simulator.prisoner_env import PrisonerBothEnv
from simulator.prisoner_perspective_envs import PrisonerBlueEnv
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.a_star_avoid import AStarAdversarialAvoid
from heuristic import HierRLBlue
import matplotlib
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from blue_bc.policy import MLPNetwork, HighLevelPolicy
from blue_bc.maddpg_filtering import MADDPGFiltering

matplotlib.use('agg')
import matplotlib.pylab
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from utils import save_video
from blue_bc.utils import BaseTrainer, HierTrainer, blue_obs_type_from_estimator, get_modified_blue_obs
from config_loader import config_loader
import random
from simulator.load_environment import load_environment
from buffer import ReplayBuffer, Buffer
from prioritized_memory import Memory
from maddpg import BaseMADDPG, MADDPG, BaseDDPG
from enum import Enum, auto

class Estimator(Enum):
    DETECTIONS = auto()
    LINEAR_ESTIMATOR = auto()
    NO_DETECTIONS = auto()

def main_reg_NeLeGt(config, env_config):
    # set up trainer condition
    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)    
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    # log_dir = os.path.join("logs", "bc", str(time))
    """Specify the writer"""
    logger = SummaryWriter(log_dir=log_dir)
    """Save the config into the para dir"""
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    """Load the environment"""
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.seed(config["environment"]["seed"])
    if config["environment"]["fugitive_policy"] == "heuristic":
        fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    elif config["environment"]["fugitive_policy"] == "a_star":
        fugitive_policy = AStarAdversarialAvoid(env)
    else:
        raise ValueError("fugitive_policy should be heuristic or a_star")
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    _, blue_partial_observation = env.reset()
    blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
    """Load the model"""
    agent_num = env.num_helicopters + env.num_search_parties
    action_dim_per_agent = 2
    obs_dims=[blue_observation[i].shape[0] for i in range(agent_num)]
    ac_dims=[action_dim_per_agent for i in range(agent_num)]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims]
    # hier_high_act_dim = agent_num * env.subpolicy_num
    # hier_low_act_dim = agent_num * env.max_para_num

    maddpg = BaseMADDPG(agent_num = agent_num, 
                        num_in_pol = blue_observation[0].shape[0], 
                        num_out_pol = action_dim_per_agent, 
                        num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
                        discrete_action = False, 
                        gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["lr"], policy_lr=0.5*config["train"]["lr"], hidden_dim=config["train"]["hidden_dim"], device=device)

    # blue_hier_policy = HierRLBlue(env, maddpg, device)
    """Initialize the buffer"""
    replay_buffer = ReplayBuffer(config["train"]["buffer_size"], agent_num, obs_ac_filter_loc_dims=obs_ac_filter_loc_dims, is_cuda=config["environment"]["cuda"])
    # imgs = []
    # last_t = 0
    # t = 0
    # done = False
    for ep in range(config["train"]["episode_num"]):
        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        """Start a new episode"""
        _, blue_partial_observation = env.reset()
        blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
        t = 0
        imgs = []
        done = False
        while not done:
            """Start Revising"""
            t = t + 1
            torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
            # rearrange actions to be per environment (each [] element corresponds to an environment id)
            # actions = [[ac[i] for ac in agent_actions] for i in range(config["train"]["n_rollout_threads"])]     
    #         # print("current t = ", t)
    #         # red_action = red_policy.predict(red_observation)
    #         # blue_actions = blue_heuristic.step_observation(blue_observation)
    #         """Partial Blue Obs"""
    #         # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
    #         """Full Blue Obs"""
            # action, new_detection, hier_action = blue_hier_policy.predict_full_observation(blue_observation)
            _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions)))
            next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            rewards = dist_reward + detect_reward
            replay_buffer.push(blue_observation, agent_actions, rewards, next_blue_observation, done)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            # print("blue rewards: ", rewards)
            if ep % config["train"]["video_step"] == 0:
                game_img = env.render('Policy', show=False, fast=True)
                imgs.append(game_img)
            
        print("complete %f of the training" % (ep/float(config["train"]["episode_num"])))
        if ep % config["train"]["video_step"] == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
        if ep % config["train"]["save_interval"] == 0:
            maddpg.save(model_dir / (str(ep) + ".pth"))
            maddpg.save(base_dir / ("model.pth"))

        if len(replay_buffer) >= config["train"]["batch_size"]: # update every config["train"]["steps_per_update"] steps
            if config["environment"]["cuda"]:
                maddpg.prep_training(device='gpu')
            else:
                maddpg.prep_training(device='cpu')

            for a_i in range(maddpg.nagents):
                sample = replay_buffer.sample(config["train"]["batch_size"], to_gpu=config["environment"]["cuda"])
                maddpg.update(sample, a_i, logger=logger)
            maddpg.update_all_targets()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)
  
    #     """update Q func"""
    #     if hier_buffer._n > config["train"]["batch_size"]:
    #         for _ in range(config["train"]["steps_per_update"]):
    #             hier_trainer.update(hier_buffer.sample(config["train"]["batch_size"]), logger=logger)
    return

def main_per_NeLeGt(config, env_config):
    # set up trainer condition
    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)  

    print("Running with PER")
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    """Specify the writer"""
    logger = SummaryWriter(log_dir=log_dir)
    """Save the config into the para dir"""
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    """Load the environment"""
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.seed(config["environment"]["seed"])
    if config["environment"]["fugitive_policy"] == "heuristic":
        fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    elif config["environment"]["fugitive_policy"] == "a_star":
        fugitive_policy = AStarAdversarialAvoid(env)
    else:
        raise ValueError("fugitive_policy should be heuristic or a_star")
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    _, blue_partial_observation = env.reset()
    blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
    """Load the model"""
    agent_num = env.num_helicopters + env.num_search_parties
    action_dim_per_agent = 2
    # hier_high_act_dim = agent_num * env.subpolicy_num
    # hier_low_act_dim = agent_num * env.max_para_num

    # maddpg = BaseMADDPG(agent_num = agent_num, 
    #                     num_in_pol = blue_observation[0].shape[0], 
    #                     num_out_pol = action_dim_per_agent, 
    #                     num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
    #                     discrete_action = False, 
    #                     gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["lr"], policy_lr=0.5*config["train"]["lr"], hidden_dim=config["train"]["hidden_dim"], device=device)

    maddpg = BaseDDPG(agent_num = agent_num, 
                        num_in_pol = blue_observation[0].shape[0], 
                        num_out_pol = action_dim_per_agent, 
                        num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent), 
                        discrete_action = False, 
                        gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["lr"], policy_lr=0.5*config["train"]["lr"], hidden_dim=config["train"]["hidden_dim"], device=device)

    # blue_hier_policy = HierRLBlue(env, maddpg, device)
    """Initialize the buffer"""
    # configs_agents_feats_minMaxIntervals = [[[0, 1, 2, 3, 4], [0, 1], [[-1, 0.5, 0.1], [0, 10, 0.5]]], [[5], [0, 1], [[-1, 1, 0.1], [0, 10, 0.5]]]]
    all_agents_ind = [0, 1, 2, 3, 4, 5]
    td_minMaxInterval = [0, 10, 0.5]
    td_class_name = "td_error"
    replay_buffers = [Memory(capacity=config["train"]["buffer_size"], feature_minMaxInterval=td_minMaxInterval, feature_class_name=td_class_name, e=config["per"]["e"], a=config["per"]["a"], beta=config["per"]["beta"], beta_increment_per_sampling=config["per"]["beta_increment_per_sampling"]) for _ in range(len(all_agents_ind))]

    for ep in range(config["train"]["episode_num"]):
        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        """Start a new episode"""
        # blue_observation, blue_partial_observation = env.reset()
        _ , blue_partial_observation = env.reset()
        blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
    
        t = 0
        imgs = []
        done = False
        while not done:
            """Start Revising"""
            t = t + 1
            torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
            # rearrange actions to be per environment (each [] element corresponds to an environment id)
            # actions = [[ac[i] for ac in agent_actions] for i in range(config["train"]["n_rollout_threads"])]     
    #         # print("current t = ", t)
    #         # red_action = red_policy.predict(red_observation)
    #         # blue_actions = blue_heuristic.step_observation(blue_observation)
    #         """Partial Blue Obs"""
    #         # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
    #         """Full Blue Obs"""
            # action, new_detection, hier_action = blue_hier_policy.predict_full_observation(blue_observation)
            _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions)))
            next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)           
            
            rewards = dist_reward + detect_reward
            for a_i in all_agents_ind:
                dones = (np.ones(maddpg.nagents) * done == 1)
                td_error = np.array(replay_buffers[a_i].max_td_error)
                td_errors = np.ones(maddpg.nagents) * td_error
                sample = [blue_observation, agent_actions, rewards, next_blue_observation, dones, td_errors]
                """Add experience tuple and associated TD error to sumtree buffer"""
                replay_buffers[a_i].add(td_error, sample)

            # replay_buffer.push(blue_observation, agent_actions, rewards, next_blue_observation, done)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            # print("blue rewards: ", rewards)
            if ep % config["train"]["video_step"] == 0:
                game_img = env.render('Policy', show=False, fast=True)
                imgs.append(game_img)
            
        print("complete %f of the training" % (ep/float(config["train"]["episode_num"])))
        if ep % config["train"]["video_step"] == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
        if ep % config["train"]["save_interval"] == 0:
            maddpg.save(model_dir / (str(ep) + ".pth"))
            maddpg.save(base_dir / ("model.pth"))

        if replay_buffers[0].tree.n_entries >= config["train"]["batch_size"]: # update every config["train"]["steps_per_update"] steps
            if config["environment"]["cuda"]:
                maddpg.prep_training(device='gpu')
            else:
                maddpg.prep_training(device='cpu')

            for a_i in range(maddpg.nagents):
                replay_buffer = replay_buffers[a_i]
                samples_idxs_weights = list(replay_buffer.sample(config["train"]["batch_size"]))
                samples_idxs_weights = split_to_batch(samples_idxs_weights, device)
                updated_td_error = maddpg.update(samples_idxs_weights, a_i, train_option="per", logger=logger)
                # update priority
                for i in range(config["train"]["batch_size"]):
                    idx = samples_idxs_weights[1][i]
                    replay_buffer.update(idx, updated_td_error[i].detach().cpu().numpy())               
            maddpg.update_all_targets()
            ep_rews = replay_buffer.get_average_rewards(t)
            for a_i, a_ep_rew in enumerate(ep_rews):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)
    return

def main_hier(config):
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    """Specify the writer"""
    logger = SummaryWriter(log_dir=log_dir)
    """Save the config into the para dir"""
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    """Load the environment"""
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment('simulator/configs/balance_game.yaml')
    env.seed(config["environment"]["seed"])
    if config["environment"]["fugitive_policy"] == "heuristic":
        fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    elif config["environment"]["fugitive_policy"] == "a_star":
        fugitive_policy = AStarAdversarialAvoid(env)
    else:
        raise ValueError("fugitive_policy should be heuristic or a_star")
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    blue_observation, blue_partial_observation = env.reset()
    """Load the model"""
    agent_num = env.num_helicopters + env.num_search_parties

    hier_high_obs_dim_each = blue_observation[0].shape[0]
    hier_high_obs_dim = agent_num * hier_high_obs_dim_each
    hier_low_obs_dim_each = blue_observation[0].shape[0] + env.subpolicy_num
    hier_low_obs_dim = agent_num * hier_low_obs_dim_each

    hier_high_act_dim_each = env.subpolicy_num
    hier_high_act_dim = agent_num * hier_high_act_dim_each
    hier_low_act_dim_each = env.max_para_num
    hier_low_act_dim = agent_num * hier_low_act_dim_each
    maddpg = MADDPG(agent_num = agent_num, 
                    num_in_high_pol = blue_observation[0].shape[0], 
                    num_in_low_pol = blue_observation[0].shape[0] + hier_high_act_dim_each,
                    subpolicy_num_out_pol = env.subpolicy_num, 
                    para_num_out_pol = env.max_para_num, 
                    num_in_high_critic = hier_high_obs_dim + hier_high_act_dim, 
                    num_in_low_critic = hier_low_obs_dim + hier_low_act_dim, 
                    gamma=config["train"]["gamma"], tau=config["train"]["tau"], lr=config["train"]["lr"], hidden_dim=(config["train"]["hidden_dim"], config["train"]["hidden_dim"]), device=device)
    blue_hier_policy = HierRLBlue(env, maddpg, device)
    """Initialize the buffer"""
    hier_high_buffer = Buffer(
        buffer_size=config["train"]["buffer_size"],
        # state_shape=env.blue_partial_observation_space.shape,
        state_shape=(hier_high_obs_dim,),
        action_shape=(hier_high_act_dim,),
        reward_shape=(agent_num,),
        device=device
    )
    hier_low_buffer = Buffer(
        buffer_size=config["train"]["buffer_size"],
        # state_shape=env.blue_partial_observation_space.shape,
        state_shape=(hier_low_obs_dim,),
        action_shape=(hier_low_act_dim,),
        reward_shape=(agent_num,),
        device=device
    )

    imgs = []
    t = 0
    subepisode_t = 1
    subepisode_detect_return = np.zeros((6))
    subepisode_dist_return = np.zeros((6))
    episode_return = np.zeros((6))
    done = False
    first_high_update = True
    for ep in range(config["train"]["episode_num"]):
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        while not done:
            """Start Revising"""
            t = t + 1
            
            # print("current t = ", t)
            # red_action = red_policy.predict(red_observation)
            # blue_actions = blue_heuristic.step_observation(blue_observation)
            """Partial Blue Obs"""
            # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
            """Full Blue Obs"""
            update_high_policy_flag, high_policy_output, high_policy_input, update_low_policy_flag, low_policy_output, low_policy_input, actions = blue_hier_policy.predict_full_observation_period(blue_observation)
            next_blue_observation, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(actions)
            """Calculate subepisode reward"""
            if not update_high_policy_flag:
                subepisode_detect_return = subepisode_detect_return + detect_reward
                subepisode_dist_return = subepisode_dist_return + dist_reward
                subepisode_t = subepisode_t + 1
            else:
                if not first_high_update:
                    # Save subepisode_return to old buffer position
                    # Save blue_observation as 'next observation' to old buffer position
                    old_names = ["rewards", "next_states"]
                    # subepisode_return = np.concatenate((subepisode_detect_return, subepisode_dist_return))
                    subepisode_return = np.array(subepisode_detect_return) + np.array(subepisode_dist_return)
                    # subepisode_return = np.array(subepisode_detect_return)
                    print("subepisode_return = ", subepisode_return)
                    old_return_nextObs = [subepisode_return/subepisode_t, np.concatenate(high_policy_input)]
                    # print("subepisode_t = ", subepisode_t)
                    pointer_move_flag = True
                    hier_high_buffer.individual_append(old_names, old_return_nextObs, pointer_move_flag)
                else:
                    first_high_update = False
                # Save blue_observation to new buffer position
                # Save hier action to new buffer position
                new_names = ["states", "actions"]
                new_states_actions = [np.concatenate(high_policy_input), high_policy_output]
                print("high_policy_output = ", high_policy_output.view(agent_num, -1))
                pointer_move_flag = False
                hier_high_buffer.individual_append(new_names, new_states_actions, pointer_move_flag)            
                # Set subepisode_return to reward
                subepisode_detect_return = np.array(detect_reward)
                subepisode_dist_return = np.array(dist_reward)
                subepisode_t = 1
            low_level_reward = dist_reward
            next_low_obs = np.concatenate((np.array(next_blue_observation), high_policy_output.detach().cpu().numpy().reshape(agent_num, -1)), axis=-1)
            hier_low_buffer.append(np.concatenate(low_policy_input), low_policy_output.detach().cpu().numpy(), low_level_reward, done, next_low_obs.reshape(-1))

            # print("helicopter_actions = ", blue_actions[5])
            # print("blue_actions = ", blue_actions)
            # mask = False if t == env.max_timesteps else done
            # print("blue_partial_observation = ", blue_partial_observation)
            # print("next_blue_partial_observation = ", next_blue_partial_observation)
            # print("to_velocity_vector(blue_actions)) = ", to_velocity_vector(blue_actions))
            # print("helicopter_action_theta_speed", blue_actions[5])
            episode_return += np.array(detect_reward)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            if ep % config["train"]["video_step"] == 0:
                game_img = env.render('Policy', show=False, fast=True)
                imgs.append(game_img)
            print("t = ", t)
            if done:
                # print("go into done branch")
                if ep % config["train"]["video_step"] == 0:
                    video_path = video_dir / (str(ep) + ".mp4")
                    save_video(imgs, str(video_path), fps=10)
            if ep % config["train"]["save_interval"] == 0:
                maddpg.save(model_dir / (str(ep) + ".pth"))
                maddpg.save(base_dir / ("model.pth"))
                """Terminate one episode"""
                # Save subepisode_return to old buffer position
                # Save blue_observation as 'next observation' to old buffer position
                old_names = ["rewards", "next_states"]
                subepisode_return = np.array(subepisode_detect_return) + np.array(subepisode_dist_return)
                # subepisode_return = np.array(subepisode_detect_return)
                old_return_nextObs = [subepisode_return/subepisode_t, np.concatenate(blue_observation)]
                pointer_move_flag = True
                hier_high_buffer.individual_append(old_names, old_return_nextObs, pointer_move_flag)
                for i in range(agent_num):
                    logger.add_scalar('agent%i/episode_reward' % i, episode_return[i], ep)
                    logger.add_scalar('agent%i/episode_reward_per_step' % i, episode_return[i]/t, ep)
                # logger.add_scalar('episode_reward', episode_return, ep)
                """Start a new episode"""
                blue_observation, blue_partial_observation = env.reset()
                blue_hier_policy.reset()
                imgs = []
                t = 0
                subepisode_t = 1
                subepisode_detect_return = np.zeros((6))
                subepisode_dist_return = np.zeros((6))
                episode_return = np.zeros((6))
                done = False
                first_high_update = True
                break
        """update Q func"""
        if hier_high_buffer._n > config["train"]["batch_size"]:
            for _ in range(config["train"]["steps_per_update"]):
                for i in range(agent_num):
                    maddpg.update_high(hier_high_buffer.sample(config["train"]["batch_size"]), agent_i=i, logger=logger)
                maddpg.update_high_targets()
        if hier_low_buffer._n > config["train"]["batch_size"]:
            for _ in range(config["train"]["steps_per_update"]):
                for i in range(agent_num):
                    maddpg.update_low(hier_low_buffer.sample(config["train"]["batch_size"]), agent_i=i, logger=logger)
                maddpg.update_low_targets()
    return

def main_per_filtering(config, env_config):
    # set up trainer condition
    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)
    print("Running with PER")
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    """Specify the writer"""
    logger = SummaryWriter(log_dir=log_dir)
    """Save the config into the para dir"""
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    """Load the environment"""
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])

    env = load_environment(env_config)
    env.seed(config["environment"]["seed"])

    if config["environment"]["fugitive_policy"] == "heuristic":
        fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    elif config["environment"]["fugitive_policy"] == "a_star":
        fugitive_policy = AStarAdversarialAvoid(env)
    else:
        raise ValueError("fugitive_policy should be heuristic or a_star")
    
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    _ , blue_partial_observation = env.reset()
    blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
    # if blue_obs_type == Estimator.LINEAR_ESTIMATOR:
    #     blue_observation = env.get_modified_blue_observation_linear_estimator()
    # elif blue_obs_type == Estimator.DETECTIONS:
    #     blue_observation = env.get_modified_blue_obs_last_detections()
    
    filtering_input = copy.deepcopy(env.get_last_k_fugitive_with_timestep())
    prisoner_loc = copy.deepcopy(env.get_prisoner_location())
    
    """Load the model"""
    agent_num = env.num_helicopters + env.num_search_parties
    action_dim_per_agent = 2
    # hier_high_act_dim = agent_num * env.subpolicy_num
    # hier_low_act_dim = agent_num * env.max_para_num

    maddpg = MADDPGFiltering(
                        filtering_model_config = config["train"]["filtering_model_config"],
                        filtering_model_path = config["train"]["filtering_model_path"],
                        agent_num = agent_num, 
                        num_in_pol = blue_observation[0].shape[0], 
                        num_out_pol = action_dim_per_agent, 
                        num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
                        discrete_action = False, 
                        gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["lr"], policy_lr=0.5*config["train"]["lr"], hidden_dim=config["train"]["hidden_dim"], device=device)

    # blue_hier_policy = HierRLBlue(env, maddpg, device)
    """Initialize the buffer"""
    # configs_agents_feats_minMaxIntervals = [[[0, 1, 2, 3, 4], [0, 1], [[-1, 0.5, 0.1], [0, 10, 0.5]]], [[5], [0, 1], [[-1, 1, 0.1], [0, 10, 0.5]]]]
    all_agents_ind = [0, 1, 2, 3, 4, 5]
    td_minMaxInterval = [0, 10, 0.5]
    td_class_name = "td_error"
    replay_buffers = [Memory(capacity=config["train"]["buffer_size"], feature_minMaxInterval=td_minMaxInterval, feature_class_name=td_class_name, e=config["per"]["e"], a=config["per"]["a"], beta=config["per"]["beta"], beta_increment_per_sampling=config["per"]["beta_increment_per_sampling"]) for _ in range(len(all_agents_ind))]

    for ep in range(config["train"]["episode_num"]):
        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        """Start a new episode"""
        _ , blue_partial_observation = env.reset()
        blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
        # if blue_obs_type == Estimator.LINEAR_ESTIMATOR:
        #     blue_observation = env.get_modified_blue_observation_linear_estimator()
        # elif blue_obs_type == Estimator.DETECTIONS:
        #     blue_observation = env.get_modified_blue_obs_last_detections()
    
        filtering_input = copy.deepcopy(env.get_last_k_fugitive_with_timestep())
        prisoner_loc = copy.deepcopy(env.get_prisoner_location())

        t = 0
        imgs = []
        done = False
        while not done:
            """Start Revising"""
            t = t + 1
            torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
            filtering_input_tensor = torch.tensor(filtering_input).unsqueeze(0).to(device)
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, filtering_input_tensor, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
            # rearrange actions to be per environment (each [] element corresponds to an environment id)
            # actions = [[ac[i] for ac in agent_actions] for i in range(config["train"]["n_rollout_threads"])]     
    #         # print("current t = ", t)
    #         # red_action = red_policy.predict(red_observation)
    #         # blue_actions = blue_heuristic.step_observation(blue_observation)
    #         """Partial Blue Obs"""
    #         # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
    #         """Full Blue Obs"""
            # action, new_detection, hier_action = blue_hier_policy.predict_full_observation(blue_observation)
            _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions)))
            next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)          
            # if blue_obs_type == Estimator.LINEAR_ESTIMATOR:
            #     next_blue_observation = env.get_modified_blue_observation_linear_estimator()
            # elif blue_obs_type == Estimator.DETECTIONS:
            #     next_blue_observation = env.get_modified_blue_obs_last_detections()
            # else:
            #     raise NotImplementedError
            
            next_filtering_input = copy.deepcopy(env.get_last_k_fugitive_with_timestep())
            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            
            rewards = dist_reward + detect_reward
            filter_save = np.expand_dims(np.array(filtering_input), 0)
            prisoner_save = np.array(prisoner_loc)/2428
            for a_i in all_agents_ind:
                dones = (np.ones(maddpg.nagents) * done == 1)
                td_error = np.array(replay_buffers[a_i].max_td_error)
                td_errors = np.ones(maddpg.nagents) * td_error
                sample = [blue_observation, agent_actions, rewards, next_blue_observation, dones, td_errors, filter_save, prisoner_save]
                # sample = [blue_observation, agent_actions, rewards, next_blue_observation, dones, td_errors]
                """Add experience tuple and associated TD error to sumtree buffer"""
                replay_buffers[a_i].add(td_error, sample)
            
            # replay_buffer.push(blue_observation, agent_actions, rewards, next_blue_observation, done)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            filtering_input = next_filtering_input
            prisoner_loc = next_prisoner_loc
            # print("blue rewards: ", rewards)
            if ep % config["train"]["video_step"] == 0:
                game_img = env.render('Policy', show=False, fast=True)
                imgs.append(game_img)
            
            # print("t = ", t)
        print("complete %f of the training" % (ep/float(config["train"]["episode_num"])))
        if ep % config["train"]["video_step"] == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)

        if ep % config["train"]["save_interval"] == 0:
            maddpg.save(model_dir / (str(ep) + ".pth"))
            maddpg.save(base_dir / ("model.pth"))

        if replay_buffers[0].tree.n_entries >= config["train"]["batch_size"]: # update every config["train"]["steps_per_update"] steps
            if config["environment"]["cuda"]:
                maddpg.prep_training(device='gpu')
            else:
                maddpg.prep_training(device='cpu')

            for a_i in range(maddpg.nagents):
                replay_buffer = replay_buffers[a_i]
                samples_idxs_weights = list(replay_buffer.sample(config["train"]["batch_size"]))
                # samples_idxs_weights = split_to_batch(samples_idxs_weights, device)
                samples_idxs_weights, filtering_input_update, prisoner_locs_update = split_to_batch_filtering(samples_idxs_weights, device)
                updated_td_error = maddpg.update(samples_idxs_weights, filtering_input_update, prisoner_locs_update, a_i, train_option="per", logger=logger)
                # update priority
                for i in range(config["train"]["batch_size"]):
                    idx = samples_idxs_weights[1][i]
                    replay_buffer.update(idx, updated_td_error[i].detach().cpu().numpy())               
            maddpg.update_all_targets()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)
    return

def main_reg_filtering(config, env_config):
    # set up trainer condition
    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)    
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    """Specify the writer"""
    logger = SummaryWriter(log_dir=log_dir)
    """Save the config into the para dir"""
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    """Load the environment"""
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.seed(config["environment"]["seed"])
    if config["environment"]["fugitive_policy"] == "heuristic":
        fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    elif config["environment"]["fugitive_policy"] == "a_star":
        fugitive_policy = AStarAdversarialAvoid(env)
    else:
        raise ValueError("fugitive_policy should be heuristic or a_star")
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    _, blue_partial_observation = env.reset()
    blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
    filtering_input = copy.deepcopy(env.get_last_k_fugitive_with_timestep())
    prisoner_loc = copy.deepcopy(env.get_prisoner_location())
    """Load the model"""
    agent_num = env.num_helicopters + env.num_search_parties
    action_dim_per_agent = 2
    filtering_input_dims = [[len(filtering_input), len(filtering_input[0])] for i in range(agent_num)]
    obs_dims=[blue_observation[i].shape[0] for i in range(agent_num)]
    ac_dims=[action_dim_per_agent for i in range(agent_num)]
    loc_dims = [len(prisoner_loc) for i in range(agent_num)]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]
    # hier_high_act_dim = agent_num * env.subpolicy_num
    # hier_low_act_dim = agent_num * env.max_para_num

    maddpg = MADDPGFiltering(
                        filtering_model_config = config["train"]["filtering_model_config"],
                        filtering_model_path = config["train"]["filtering_model_path"],
                        agent_num = agent_num, 
                        num_in_pol = blue_observation[0].shape[0], 
                        num_out_pol = action_dim_per_agent, 
                        num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
                        discrete_action = False, 
                        gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["lr"], policy_lr=0.5*config["train"]["lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
    
    # blue_hier_policy = HierRLBlue(env, maddpg, device)
    """Initialize the buffer"""
    replay_buffer = ReplayBuffer(config["train"]["buffer_size"], agent_num, obs_ac_filter_loc_dims=obs_ac_filter_loc_dims, is_cuda=config["environment"]["cuda"])
    # imgs = []
    # last_t = 0
    # t = 0
    # done = False
    for ep in range(config["train"]["episode_num"]):
        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        """Start a new episode"""
        _, blue_partial_observation = env.reset()
        blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
        filtering_input = copy.deepcopy(env.get_last_k_fugitive_with_timestep())
        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False
        while not done:
            """Start Revising"""
            t = t + 1
            torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
            filtering_input_tensor = torch.tensor(filtering_input).unsqueeze(0).to(device)
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, filtering_input_tensor, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
            # rearrange actions to be per environment (each [] element corresponds to an environment id)
            # actions = [[ac[i] for ac in agent_actions] for i in range(config["train"]["n_rollout_threads"])]     
    #         # print("current t = ", t)
    #         # red_action = red_policy.predict(red_observation)
    #         # blue_actions = blue_heuristic.step_observation(blue_observation)
    #         """Partial Blue Obs"""
    #         # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
    #         """Full Blue Obs"""
            # action, new_detection, hier_action = blue_hier_policy.predict_full_observation(blue_observation)
            _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions)))
            next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            next_filtering_input = copy.deepcopy(env.get_last_k_fugitive_with_timestep())
            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            rewards = dist_reward + detect_reward
            filter_save = [np.array(filtering_input) for i in range(maddpg.nagents)]
            prisoner_save = [np.array(prisoner_loc)/2428 for i in range(maddpg.nagents)]
            # sample = [blue_observation, agent_actions, rewards, next_blue_observation, done, filter_save, prisoner_save]
            replay_buffer.push_filter(blue_observation, agent_actions, rewards, next_blue_observation, done, filter_save, prisoner_save)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            filtering_input = next_filtering_input
            prisoner_loc = next_prisoner_loc
            # print("blue rewards: ", rewards)
            if ep % config["train"]["video_step"] == 0:
                game_img = env.render('Policy', show=False, fast=True)
                imgs.append(game_img)
            
        print("complete %f of the training" % (ep/float(config["train"]["episode_num"])))
        if ep % config["train"]["video_step"] == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
        if ep % config["train"]["save_interval"] == 0:
            maddpg.save(model_dir / (str(ep) + ".pth"))
            maddpg.save(base_dir / ("model.pth"))

        if len(replay_buffer) >= config["train"]["batch_size"]: # update every config["train"]["steps_per_update"] steps
            if config["environment"]["cuda"]:
                maddpg.prep_training(device='gpu')
            else:
                maddpg.prep_training(device='cpu')

            for a_i in range(maddpg.nagents):
                sample = replay_buffer.sample_filter(config["train"]["batch_size"], to_gpu=config["environment"]["cuda"])
                maddpg.update(sample[0:5], sample[5][0], sample[6][0], a_i, train_option="regular", logger=logger)
                # maddpg.update(sample, a_i, logger=logger)
            maddpg.update_all_targets()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)
  
    #     """update Q func"""
    #     if hier_buffer._n > config["train"]["batch_size"]:
    #         for _ in range(config["train"]["steps_per_update"]):
    #             hier_trainer.update(hier_buffer.sample(config["train"]["batch_size"]), logger=logger)
    return

def split_directions_to_direction_speed(directions):
    blue_actions_norm_angle_vel = []
    blue_actions_directions = np.split(directions, 6)
    search_party_v_limit = 20
    helicopter_v_limit = 127
    for idx in range(len(blue_actions_directions)):
        if idx < 5:
            search_party_direction = blue_actions_directions[idx]
            if np.linalg.norm(search_party_direction) > 1:
                search_party_direction = search_party_direction / np.linalg.norm(search_party_direction)
            search_party_speed = np.maximum(np.linalg.norm(search_party_direction), 1.0) * search_party_v_limit
            blue_actions_norm_angle_vel.append(np.array(search_party_direction.tolist() + [search_party_speed]))
        elif idx < 6:
            helicopter_direction = blue_actions_directions[idx]
            if np.linalg.norm(helicopter_direction) > 1:
                helicopter_direction = helicopter_direction / np.linalg.norm(helicopter_direction)
            helicopter_speed = np.maximum(np.linalg.norm(helicopter_direction), 1.0) * helicopter_v_limit
            blue_actions_norm_angle_vel.append(np.array(helicopter_direction.tolist()+ [helicopter_speed]))  

    return blue_actions_norm_angle_vel 

def split_to_batch(per_sample, device):
    """per_sample[0]: [[obs(6), agent_actions(6), rewards(6), next_obs(6), dones(6), td(6)], [...], [...], ..., [...]]
       desired_sample: [[[obs0(batch_size)], [obs1(batch_size)], ..., [obs5(batch_size)]], [], [], [], []]"""
    per_sample_np = np.transpose(np.array(per_sample[0]), (1, 2, 0))
    per_sample_np = per_sample_np.tolist()
    item_num = len(per_sample_np)
    agent_num = len(per_sample_np[0])
    for item_idx in range(item_num):
        for agent_idx in range(agent_num):
            per_sample_np[item_idx][agent_idx] = torch.Tensor(np.vstack(per_sample_np[item_idx][agent_idx])).to(device)
    per_sample[0] = per_sample_np
    # obs = []
    # acs = []
    # rews = []
    # next_obs = []
    # dones = []
    # desired_sample = [[], [], [], [], []]
    # batch_size = len(per_sample)
    # agent_num = len(per_sample[0])
    # for item_idx in range(len(desired_sample)):
    #     for agent_idx in range(agent_num):
    #         for batch_idx in range(batch_size):
    #             it_ag_ba = per_sample[batch_idx][item_idx]
    return per_sample

def split_to_batch_filtering(per_sample, device):
    """ 
       per_sample[0]: [[obs(6), agent_actions(6), rewards(6), next_obs(6), dones(6), td(6), filtering_input(1), prisoner_loc(1)], [...], [...], ..., [...]]
       desired_sample: [[[obs0(batch_size)], [obs1(batch_size)], ..., [obs5(batch_size)]], [], [], [], []]
    """
    filtering_inputs = []
    prisoner_locs = []
    batched_agents = []

    for sample in per_sample[0]:
        # obs, agent_actions, rewards, next_obs, dones, td, filtering_input, prisoner_loc = sample
        original_batch = sample[:-2]
        filtering_input = sample[-2]
        prisoner_loc = sample[-1]
        filtering_inputs.append(filtering_input)
        prisoner_locs.append(prisoner_loc)
        batched_agents.append(original_batch)

    per_sample_np = np.array(batched_agents)
    per_sample_np = np.transpose(per_sample_np, (1, 2, 0))
    per_sample_np = per_sample_np.tolist()
    item_num = len(per_sample_np)
    agent_num = len(per_sample_np[0])
    for item_idx in range(item_num):
        for agent_idx in range(agent_num):
            per_sample_np[item_idx][agent_idx] = torch.Tensor(np.vstack(per_sample_np[item_idx][agent_idx])).to(device)
    per_sample[0] = per_sample_np
    filtering_inputs = torch.Tensor(np.vstack(filtering_inputs)).to(device)
    prisoner_locs = torch.Tensor(np.vstack(prisoner_locs)).to(device)
    return per_sample, filtering_inputs, prisoner_locs


if __name__ == '__main__':
    config = config_loader(path="./blue_bc/parameters_training.yaml")  # load model configuration
    env_config = config_loader(path=config["environment"]["env_config_file"])
    """create base dir"""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path("./logs/marl") / timestr
    os.makedirs(base_dir, exist_ok=True)
    """Benchmark Starts Here"""
    # Specify the benchmarking parameters: random seeds, sharing strategies, env type, learning rates, PER parameters
    seeds = [0] # 1, 2, 3, 4, 5
    structures = ["regular"] # "regular", "per", hier
    fugitive_policys = ["heuristic"] # "a_star", "heuristic"
    estimators = ["ground_truth"] # "no_estimator", "linear_estimator", "filtering", "predicting", "ground_truth", "detections"
    lrs = [0.0005]
    alpha = [0.3] # 0.1, 0.2, 0.3, 0.4, 0.5
    beta_increment_per_sampling = [0.00001] # 0, 0.00001, 0.00002, 0.00003, 0.00004
    
    for seed in seeds:
        for lr in lrs:
            for structure in structures:
                for policy in fugitive_policys:
                    for est in estimators:
                        for a in alpha:
                            for beta_increment in beta_increment_per_sampling:
                                """Modify the config"""
                                config["environment"]["seed"] = seed
                                config["train"]["lr"] = lr
                                config["environment"]["structure"] = structure
                                config["environment"]["fugitive_policy"] = policy
                                config["per"]["a"] = a
                                config["per"]["beta_increment_per_sampling"] = beta_increment
                                config["environment"]["estimator"] = est
                                if est == "ground_truth":
                                    env_config["include_fugitive_location_in_blue_obs"] = True
                                else:
                                    env_config["include_fugitive_location_in_blue_obs"] = False

                                """create base dir name for each setting"""    
                                base_dir = Path("./logs/marl") / timestr / (structure+"_"+policy+"_"+est)
                                config["environment"]["dir_path"] = str(base_dir)

                                if config["environment"]["structure"] == "regular" and config["environment"]["estimator"] in ["detections", "no_estimator", "linear_estimator", "ground_truth"]:
                                    main_reg_NeLeGt(config, env_config)
                                # if config["environment"]["structure"] == "regular" and config["environment"]["estimator"] == "linear_estimator":
                                #     main_reg_linearEstimator(config)
                                elif config["environment"]["structure"] == "regular" and config["environment"]["estimator"] == "filtering":
                                    main_reg_filtering(config, env_config)
                                elif config["environment"]["structure"] == "regular" and config["environment"]["estimator"] == "predicting":
                                    main_reg_predicting(config)
                                # if config["environment"]["structure"] == "regular" and config["environment"]["estimator"] == "ground_truth":
                                #     main_reg_groundTruth(config)

                                elif config["environment"]["structure"] == "per" and config["environment"]["estimator"] in ["detections", "no_estimator", "linear_estimator", "ground_truth"]:
                                    main_per_NeLeGt(config, env_config)
                                # if config["environment"]["structure"] == "per" and config["environment"]["estimator"] == "linear_estimator":
                                #     main_per_linearEstimator(config)
                                elif config["environment"]["structure"] == "per" and config["environment"]["estimator"] == "filtering":
                                    main_per_filtering(config, env_config)
                                elif config["environment"]["structure"] == "per" and config["environment"]["estimator"] == "predicting":
                                    main_per_predicting(config)
                                # if config["environment"]["structure"] == "per" and config["environment"]["estimator"] == "ground_truth":
                                #     main_per_groundTruth(config)



                                # elif config["environment"]["structure"] == "per":
                                #     main_per(config)
                                # elif config["environment"]["structure"] == "filter_per":
                                #     main_filter_per(config, env_config)
                                # elif config["environment"]["structure"] == "hier":
                                #     main_hier(config)

                                else:
                                    raise NotImplementedError