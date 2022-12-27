""" This script collects a triple of blue observations, red observations, and red locations. """

import gym
import numpy as np
import torch
import pickle

import os
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
from simulator.prisoner_env import PrisonerBothEnv
from simulator.terrain import Terrain, TerrainType
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.rrt_star_adversarial_heuristic import RRTStarAdversarial
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid

from blue_policies.heuristic import BlueHeuristic
from red_bc.heuristic import SimplifiedBlueHeuristic
from simulator.prisoner_perspective_envs import PrisonerEnv
from simulator.load_environment import load_environment

# from simulator import initialize_prisoner_environment
import argparse

def collect_demonstrations(env, policy, num_runs, show=False):
    """ Collect demonstrations for multi-step prediction. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """     

    """ Initialize observations and ground truth red locations and dones as empty lists """   
    red_observations = []
    blue_observations = []
    red_locations = []
    dones = []
    
    """ What is num_runs? """
    for _ in tqdm(range(num_runs)):
        """ Initialize the environment, observation, and done """
        # print(len(buffer))
        observation = env.reset()
        red_obs_names = env.prediction_obs_names
        # print(red_obs_names._idx_dict['prisoner_loc'])
        done = False
        """ When the episode is not ended """
        while not done:
            """ Generate the red action and update the enviornment accordingly """
            action = policy.predict(observation)[0]
            observation, reward, done, infos = env.step(action)
            """ Get the ground truth prisoner location, blue observation and red observation """
            prisoner_location = env.get_prisoner_location()
            # print("prisoner_location.shape = ", len(prisoner_location))
            blue_observation = env.get_blue_observation()
            # print("blue_observation.shape = ", len(blue_observation))
            red_observation = env.get_prediction_observation()
            # print("red_observation.shape = ", len(red_observation))

            wrapped_red = red_obs_names(red_observation)
            # print(wrapped_red['prisoner_loc'], np.array(prisoner_location)/2428)
            """ Save the ground truth prisoner location, blue observation and red observation and dones into the empty lists """
            red_observations.append(red_observation)
            blue_observations.append(blue_observation)
            red_locations.append(prisoner_location)
            dones.append(done)
            if show:
                env.render('heuristic', show=True, fast=True)

            """ When the episode ends, reset the environment and the observation """
            if done:
                if env.unwrapped.timesteps >= 2000:
                    print("Got stuck")
                observation = env.reset()
                break
    
    """ Every location/observation is converted to a line in an numpy array"""
    red_observations = np.stack(red_observations)
    blue_observations = np.stack(blue_observations)
    red_locations = np.stack(red_locations)/2428

    return red_observations, blue_observations, red_locations, dones

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=0, help='Environment to use')
    
    seed = 0; num_runs = 300
    # seed = 1; num_runs = 100
    args = parser.parse_args()
    map_num = args.map_num

    epsilon = 0.1
    # env = initialize_prisoner_environment(map_num, seed=seed, heuristic_type="Normal")
    env = load_environment('simulator/configs/fixed_cams_random_uniform_start_camera_net.yaml')
    blue_policy = SimplifiedBlueHeuristic(env, debug=False)
    env = PrisonerEnv(env, blue_policy)

    heuristic = HeuristicPolicy(env, epsilon=epsilon)
    # heuristic = RRTStarAdversarialAvoid(env, max_speed=7.5, n_iter=2000)

    print(f"Collecting data for {map_num} with training shape {env.prediction_observation_space.shape}")
    blue_obs_dict = env.blue_obs_names._idx_dict
    prediction_obs_dict = env.prediction_obs_names._idx_dict

    red_observations, blue_observations, red_locations, dones = \
        collect_demonstrations(env, heuristic, num_runs=num_runs, show=False)
    assert np.round(blue_observations.max()) <= 1
    assert np.round(blue_observations.min()) >= -1

    assert np.round(red_observations.max()) <= 1
    assert np.round(red_observations.min()) >= -1
    np.savez(f"datasets/random_start_locations/map_{map_num}_run_{num_runs}_heuristic_eps_{epsilon}.npz",
                red_observations=red_observations, 
                blue_observations=blue_observations, 
                red_locations=red_locations, 
                dones=dones,
                prediction_dict = prediction_obs_dict,
                blue_dict = blue_obs_dict
                )
