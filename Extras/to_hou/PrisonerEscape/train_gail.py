from simulator import PrisonerEnv, PrisonerGoalEnv
from stable_baselines3.sac import SAC
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from hier import OptionWorker

from behavioral_cloning.bc import train_bc
from utils import evaluate_mean_reward
from behavioral_cloning.collect_demonstrations import collect_demonstrations, save_buffer
from behavioral_cloning.rrt import collect_demonstrations_rrt
from behavioral_cloning.collect_mountain_data import collect_demonstrations_mountain
from behavioral_cloning.bc_lstm import train_bc_lstm

from dagger.dagger import train_dagger

from fugitive_policies.heuristic_goal import HeuristicPolicyGoal
from fugitive_policies.heuristic import HeuristicPolicy
from utils import load_environment

import numpy as np
import argparse, os
import pickle
from copy import deepcopy
import torch
import random

import yaml

def get_configs():
    """
    Parse command line arguments and return the resulting args namespace
    """
    parser = argparse.ArgumentParser("Train Prisoner agent using MA-GAIL")
    parser.add_argument("--config", type=str, required=True, help="Path to .yaml config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded, args.config

def main():
    # Load configs
    config, config_path = get_configs()

    seed = config["seed"]
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    show = config["show"]
    # Create environment
    env = load_environment(config_path, config["environment"]["observation_step_type"])
    print(env.observation_space.shape)

    # Collect Data
    dataset_path = config['dataset']['path']
    if config['dataset']['collect']:
        print("Collecting data")
        buffer_size = config['dataset']['buffer_size']
        buffer_save_type = config['dataset']["buffer_save_type"]
        if config['heuristic'] == 'rrt':
            print("Using RRT")
            buffer = collect_demonstrations_rrt(env, buffer_size, show)
        elif config['heuristic'] == 'direct':
            print("Using direct")
            
            heuristic_policy = HeuristicPolicyGoal(env)
            buffer = collect_demonstrations(env, buffer_save_type, heuristic_policy, buffer_size=buffer_size)
        elif config['heuristic'] == 'avoid':
            print("Using avoid")
            repeat_stops = config['dataset']['repeat_stops']
            heuristic_policy = HeuristicPolicy(env)
            buffer = collect_demonstrations(env, buffer_save_type, heuristic_policy, buffer_size=buffer_size, repeat_stops=repeat_stops)
        elif config['heuristic'] == 'mountain':
            print("Using mountain")
            heuristic_policy = HeuristicPolicyGoal(env)
            buffer = collect_demonstrations_mountain(env, heuristic_policy, buffer_size=buffer_size, p_rand=0.0, std=0.0, show=show) 
        elif config['heuristic'] == 'rl':
            # assert goal_env_bool == False, "RL heuristic cannot be used with goal environments"
            policies = [torch.load(f'bc_policies/goal_{i}_policy.pth') for i in range(3)]
            # goals = env.hideout_locations[:]
            env_option = OptionWorker(env, policies, env.hideout_locations[:])
            model = PPO.load(config['rl_path'])
            buffer = collect_demonstrations(env_option, buffer_save_type, model.policy, buffer_size=buffer_size)
        else:
            raise ValueError("Invalid heuristic")
        save_buffer(buffer, dataset_path)
    else:
        with open(dataset_path, 'rb') as handle:
            buffer = pickle.load(handle)


    # Train BC policy
    log_dir = config['log_dir']
    if config['bc']['train']:
        print("Training BC")
        lr = config['bc']['lr']
        policy_kwargs = {}
        policy_kwargs['net_arch'] = config['bc']['net_arch']

        if config['bc']['model_type'] == 'SAC':
            model = SAC('MlpPolicy', env, tensorboard_log=log_dir, verbose=1, learning_rate=lr)
        elif config['bc']['model_type'] == 'A2C':
            model = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, verbose=1, learning_rate=lr)
        elif config['bc']['model_type'] == 'LSTM' or 'dagger':
            pass
        else:
            raise ValueError("Invalid model type")

        if config['bc']['model_type'] == 'SAC' or config['bc']['model_type'] == 'A2C':
            batch_size = config['bc']['batch_size']
            epochs = config['bc']['epochs']
            video_step = config['bc']['video_step']
            goal = None
            print(model.policy.observation_space.shape)
            returned_policy = train_bc(env, buffer, model.policy, epochs=epochs, batch_size=batch_size, save_folder=log_dir, goal=goal, video_step=video_step, config_file=config_path)
            model.policy = returned_policy
            mean_reward = evaluate_mean_reward(env, model.policy.predict, num_iter = 100)
            print(mean_reward)
        elif config['bc']['model_type'] == 'LSTM':
            num_observations = config['bc']['num_observations']
            batch_size = config['bc']['batch_size']
            epochs = config['bc']['epochs']
            video_step = config['bc']['video_step']
            returned_policy = train_bc_lstm(env, buffer, num_observations, learning_rate=lr, epochs=epochs, batch_size=batch_size, video_step=video_step, config_file=config_path)
        elif config['bc']['model_type'] == 'dagger':
            from dagger.dagger import reconstruct_policy
            batch_size = config['bc']['batch_size']
            epochs = config['bc']['epochs']
            video_step = config['bc']['video_step']

            if config['dagger_warmstart_policy'] is not None:
                model = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, verbose=1, learning_rate=lr)
                model.policy = reconstruct_policy(config['dagger_warmstart_policy'])
            else:
                model = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, verbose=1, learning_rate=lr)
            # model.policy = policy
            returned_policy = train_dagger(env, model.policy, epochs=epochs, batch_size=batch_size, save_folder=log_dir, config_file=None)
                
        else:
            raise ValueError(f"Invalid model type: {config['bc']['model_type']}")

if __name__ == '__main__':
    main()