""" We save a file with both the observations of the blue and red agents. """

import gym
import numpy as np
import torch
import pickle

from tqdm import tqdm
from simulator.prisoner_env import PrisonerEnv, PrisonerGoalEnv
from simulator.terrain import Terrain, TerrainType
from hier import OptionWorker
from fugitive_policies.heuristic import HeuristicPolicy

from simulator import initialize_prisoner_environment
import argparse

def collect_demonstrations(env, policy, num_runs, show=False):
    """ Collect demonstrations for multi-step prediction. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """        
    red_observations = []
    blue_observations = []
    red_locations = []
    dones = []
    

    for _ in tqdm(range(num_runs)):
        # print(len(buffer))
        observation = env.reset()
        red_obs_names = env.prediction_obs_names
        # print(red_obs_names._idx_dict['prisoner_loc'])
        done = False
        while not done:
            action = policy.predict(observation)[0]

            observation, reward, done, infos = env.step(action)
            prisoner_location = env.get_prisoner_location()
            blue_observation = env.get_blue_observation()
            
            red_observation = env.get_prediction_observation()
            wrapped_red = red_obs_names(red_observation)
            print(wrapped_red['prisoner_loc'], np.array(prisoner_location)/2428)

            red_observations.append(red_observation)
            blue_observations.append(blue_observation)
            red_locations.append(prisoner_location)
            dones.append(done)
            if show:
                env.render('heuristic', show=True, fast=True)

            if done or env.timesteps > 1000:
                if env.unwrapped.timesteps >= 2000:
                    print("Got stuck")
                observation = env.reset()
                break
    
    red_observations = np.stack(red_observations)
    blue_observations = np.stack(blue_observations)
    red_locations = np.stack(red_locations)

    return red_observations, blue_observations, red_locations/2428, dones

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=0, help='Environment to use')
    
    args = parser.parse_args()
    map_num = args.map_num

    epsilon = 0
    env = initialize_prisoner_environment(map_num)

    heuristic = HeuristicPolicy(env, epsilon=epsilon)

    print(f"Collecting data for {map_num} with training shape {env.prediction_observation_space.shape}")

    # num_runs = 100
    # red_observations, blue_observations, red_locations, dones = collect_demonstrations(env, heuristic, num_runs=num_runs, show=False)
    # np.savez(f"shared_latent/map_{map_num}_run_{num_runs}_eps_{epsilon}.npz", 
    #                 red_observations=red_observations, 
    #                 blue_observations=blue_observations, 
    #                 red_locations=red_locations, 
    #                 dones=dones)

    # print(red_observations.shape, blue_observations.shape, red_locations.shape)

    num_runs = 1000
    red_observations, blue_observations, red_locations, dones = collect_demonstrations(env, heuristic, num_runs=num_runs, show=False)
    blue_obs_dict = env.blue_obs_names._idx_dict
    prediction_obs_dict = env.prediction_obs_names._idx_dict
    np.savez(f"shared_latent/map_{map_num}_run_{num_runs}_eps_{epsilon}_norm.npz", 
                red_observations=red_observations, 
                blue_observations=blue_observations, 
                red_locations=red_locations, 
                dones=dones,
                prediction_dict = prediction_obs_dict,
                blue_dict = blue_obs_dict
                )