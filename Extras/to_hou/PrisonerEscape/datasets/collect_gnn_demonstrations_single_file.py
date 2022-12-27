""" This script collects a triple of agent observations, hideout location, and timestep. """
import os, sys
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
from simulator.gnn_wrapper import PrisonerGNNEnv

from simulator.load_environment import load_environment
from simulator.prisoner_perspective_envs import PrisonerEnv
from simulator import PrisonerBothEnv, PrisonerBlueEnv, PrisonerEnv
from blue_policies.heuristic import BlueHeuristic
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid

from simulator import initialize_prisoner_environment
import argparse
import random

def collect_demonstrations(epsilon, num_runs, 
                    starting_seed, 
                    random_cameras, 
                    folder_name,
                    heuristic_type,
                    env_path,
                    show=False):
    """ Collect demonstrations for the homogeneous gnn where we assume all agents are the same. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """
    num_detections = 0
    total_timesteps = 0
    for seed in tqdm(range(starting_seed, starting_seed + num_runs)):
        detect = 0
        t = 0
        agent_observations = []
        hideout_observations = []
        timestep_observations = []
        detected_locations = []
        blue_observations = []
        red_observations = []
        last_k_fugitive_detections = []

        red_locations = []
        dones = []

        num_random_known_cameras = random.randint(25, 30)
        num_random_unknown_cameras = random.randint(25, 30)

        # env = initialize_prisoner_environment(map_num,
        #                                     observation_step_type = observation_step_type, 
        #                                     epsilon=epsilon, 
        #                                     random_cameras=random_cameras,
        #                                     num_random_known_cameras = num_random_known_cameras,
        #                                     num_random_unknown_cameras = num_random_unknown_cameras,
        #                                     seed=seed,
        #                                     heuristic_type=heuristic_type,
        #                                     store_last_k_fugitive_detections=store_last_k_fugitive_detections)

        env = load_environment(env_path)
        env.seed(seed)
        print("Running with seed {}".format(seed))
        np.random.seed(seed)
        random.seed(seed)


        if heuristic_type == 'Normal':
            red_policy = HeuristicPolicy(env, epsilon=epsilon)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_eps_{epsilon}_{heuristic_type}"
        else:
            red_policy = RRTStarAdversarialAvoid(env, max_speed=7.5, n_iter=2000)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"
        if random_cameras:
            path += "_random_cameras"

        env = PrisonerBlueEnv(env, red_policy)
        if not os.path.exists(path):
            os.makedirs(path)

        env = PrisonerGNNEnv(env)
        policy = BlueHeuristic(env, debug=False)

        gnn_obs, blue_obs = env.reset()
        policy.reset()
        policy.init_behavior()

        done = False
        while not done:
            t += 1
            blue_actions = policy.predict(blue_obs)
            gnn_obs, blue_obs, reward, done, _ = env.step(blue_actions)

            prisoner_location = env.get_prisoner_location()
            detected_location = blue_obs[-2:]
            blue_observation = env.get_blue_observation()
            red_observation = env.get_prediction_observation()

            red_observations.append(red_observation)
            blue_observations.append(blue_observation)
            agent_observations.append(gnn_obs[0])
            hideout_observations.append(gnn_obs[1])
            timestep_observations.append(gnn_obs[2])
            red_locations.append(prisoner_location)
            dones.append(done)
            detected_locations.append(detected_location)

            if env.is_detected:
                num_detections += 1
                detect += 1

            if store_last_k_fugitive_detections:
                last_k_fugitive_detections.append(np.array(env.get_last_k_fugitive_detections()))

            if show:
                env.render('heuristic', show=True, fast=True)

            if done:
                if env.unwrapped.timesteps >= 2000:
                    print(f"Got stuck, {env.unwrapped.timesteps}")
                print(f"{detect}/{t} = {detect/t} detection rate")
                break
                
        agent_dict = {"num_known_cameras": env.num_known_cameras,
                      "num_unknown_cameras": env.num_unknown_cameras,
                      "num_helicopters": env.num_helicopters,
                      "num_search_parties": env.num_search_parties}

        blue_obs_dict = env.blue_obs_names._idx_dict
        prediction_obs_dict = env.prediction_obs_names._idx_dict
        np.savez(path + f"/seed_{seed}_known_{env.num_known_cameras}_unknown_{env.num_unknown_cameras}.npz", 
            blue_observations = blue_observations,
            red_observations = red_observations,
            agent_observations=agent_observations,
            hideout_observations=hideout_observations,
            timestep_observations=timestep_observations, 
            detected_locations = detected_locations,
            red_locations=red_locations, 
            dones=dones,
            agent_dict = agent_dict,
            detect=detect,
            last_k_fugitive_detections=last_k_fugitive_detections,
            blue_obs_dict = blue_obs_dict,
            prediction_obs_dict = prediction_obs_dict
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=0, help='Environment to use')
    
    args = parser.parse_args()
    map_num = args.map_num

    store_last_k_fugitive_detections = True

    # starting_seed = 90; num_runs = 400; epsilon=0.1; folder_name = "train"
    # starting_seed = 500; num_runs = 100; epsilon = 0.1; folder_name = "test"
    # starting_seed = 0; num_runs = 2; epsilon=0.1; folder_name = "train_same_new"

    starting_seed=0; num_runs=400; epsilon=0; folder_name = "random_start_location_no_net"

    heuristic_type = "RRT"
    random_cameras=False
    observation_step_type = "Blue"

    # env_path = None
    # env_path = "simulator/configs/fixed_cams_random_uniform_start_camera_net.yaml"
    env_path = "simulator/configs/fixed_cams_random_uniform_start_no_camera_net.yaml"
    
    # camera_configuration = "/nethome/sye40/PrisonerEscape/simulator/camera_locations/reduced.txt"
    # print(f"Collecting data for {map_num} with training shape {env.prediction_observation_space.shape}")

    # print(agent_observations.shape)
    collect_demonstrations(epsilon, num_runs, starting_seed, random_cameras, folder_name, heuristic_type, env_path, show=False)