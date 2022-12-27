from utils import evaluate_mean_reward
# from simulator import PrisonerBothEnv, initialize_prisoner_environment
from simulator.load_environment import load_environment
from fugitive_policies.heuristic import HeuristicPolicy
from blue_policies.heuristic import BlueHeuristic
import os
import numpy as np
import cv2
import csv
from pathlib import Path
from utils import save_video
from datetime import datetime
import random
from utils import save_video
from simulator.prisoner_env_variations import intialize_multimodal_env


os.makedirs("logs/temp/", exist_ok=True)
now = datetime.now()
log_location = f"logs/run_heuristic/{now.strftime('%d_%m_%Y_%H_%M_%S')}/"
os.makedirs("logs/run_heuristic/", exist_ok=True)
os.makedirs(log_location, exist_ok=True)
# demo_record_csv_path = Path(log_location) / "demo_test.csv"
# # env_kwargs = {}
# # env_kwargs['spawn_mode'] = "corner"
# # env_kwargs['spawn_range'] = 350
# # env_kwargs['helicopter_battery_life'] = 200
# # env_kwargs['helicopter_recharge_time'] = 40
# # env_kwargs['num_search_parties'] = 5
# # # env_kwargs['reward_scheme'] = reward_scheme
# # env_kwargs['random_cameras'] = False
# # env_kwargs['observation_step_type'] = "Fugitive"
# # env_kwargs['debug'] = False
# # env_kwargs['observation_terrain_feature']=False

# # # Directory to randomly cycle between all the maps
# # # env_kwargs['terrain_map'] = 'simulator/forest_coverage/maps'

# # # Single map to always test on one map
# # # env_kwargs['terrain_map'] = 'simulator/forest_coverage/maps_0.2/1.npy'
# # env_kwargs['terrain_map'] = 'simulator/forest_coverage/perlin_2/3.npy'
# # env_kwargs['camera_file_path'] = "simulator/camera_locations/40_percent_cameras.txt"
# # # env_kwargs['mountain_locations'] = [(400, 300), (1600, 1800)] # original mountain setup
# # # env_kwargs['mountain_locations'] = [(400, 2000), (1500, 1000)] # second mountain setup
# # # env_kwargs['mountain_locations'] = [(1000, 900), (1500, 1300)] # third mountain setup
# # env_kwargs['mountain_locations'] = [(600, 1100), (1000, 1900)] # fourth mountain setup

# # env = PrisonerEnv(**env_kwargs)

# epsilon = 0.1
# seed = 2
# variation = 0
# print(f"Loaded environment variation {variation} with seed {seed}")

# # set seeds
# np.random.seed(seed)
# random.seed(seed)


# terrain_map = f'simulator/forest_coverage/map_set/{variation}.npy'
# if variation == 0:
#     mountain_locations = [(400, 300), (1600, 1800)] # original mountain setup
# elif variation == 1:
#     mountain_locations = [(400, 2000), (1500, 1000)]
# elif variation == 2:
#     mountain_locations = [(1000, 900), (1500, 1300)]
# elif variation == 3:
#     mountain_locations = [(600, 1100), (1000, 1900)]
# else:
#     raise ValueError(f'Invalid variation {variation}')

# camera_configuration="simulator/camera_locations/original_and_more.txt"
# observation_step_type="Fugitive" 
# step_reset=True 
# terrain=None

# env = PrisonerBothEnv(terrain=terrain,
#                     spawn_mode='corner',
#                     observation_step_type=observation_step_type,
#                     random_cameras=False,
#                     camera_file_path=camera_configuration,
#                     mountain_locations=mountain_locations,
#                     camera_range_factor=1.0,
#                     observation_terrain_feature=False,
#                     random_hideout_locations=False,
#                     spawn_range=350,
#                     helicopter_battery_life=200,
#                     helicopter_recharge_time=40,
#                     num_search_parties=5,
#                     terrain_map=terrain_map,
#                     step_reset = step_reset
#                     )
# env.seed(seed)

# # blue_heuristic = BlueHeuristic(env, env.search_parties_list, env.helicopters_list, debug=False)
# # blue_heuristic.reset(env.search_parties_list, env.helicopters_list)

epsilon = 0
env = load_environment('simulator/configs/fixed_cams_random_uniform_start_camera_net.yaml')
# env = load_environment('simulator/configs/1.yaml')

blue_heuristic = BlueHeuristic(env, debug=False)
blue_heuristic.reset()

num_iter = 3


show = True
mean_return = 0.0

start_locs = []

for episode_num in range(num_iter):
    imgs = []

    policy = HeuristicPolicy(env, epsilon=epsilon)
    red_observation = env.reset()
    start_locs.append(env.get_prisoner_location())
    blue_heuristic.init_behavior()
    blue_observation = env.get_blue_observation()
    done = False
    episode_return = 0.0
    i = 0
    imgs = []

    while not done:
        i += 1
        red_action = policy.predict(red_observation)
        """blue_actions: len(blue_actions) = 5(N_search_parties) + 1(N_helicopter); blue_actions[0] = [dx, dy, speed]"""
        blue_actions = blue_heuristic.predict(blue_observation)

        red_observation, blue_observation,  reward, done, _ = env.step_both(red_action[0], blue_actions)
        """blue_observation: normalized_time, [camera_x, camera_y] * 77, [known_hideout_x, known_hideout_y] * 1, [helicopter_x, helicopter_y] * 1, [search_party_x, search_party_y] * 5, 
        [parties_detection_of_fugitive_one_hot = [b (0/1) * N_blue=83: if detected, prisoner_x, prisoner_y]] total dim = 254"""
        blue_observation = env.get_blue_observation()
        # with open(str(demo_record_csv_path), "a") as csvfile:
        #     demonstration_obs_acs = np.concatenate([blue_observation, np.concatenate(blue_actions)])
        #     writer = csv.writer(csvfile)
        #     writer.writerow(demonstration_obs_acs)
       
        # blue_observation = env.get_blue_observation()
        episode_return += reward

        game_img = env.render('Policy', show=False, fast=True)
        imgs.append(game_img)

        mountain_map = env.terrain.world_representation[0, :, :]
        # np.save('temp/mountain.npy', mountain_map)

            # blue_plan_img = cv2.imread("logs/temp/debug_plan.png")  # the blue heuristic plan debug figure
            # combine both images to one
            # max_x = max(game_img.shape[0], blue_plan_img.shape[0])
            # max_y = max(game_img.shape[1], blue_plan_img.shape[1])
            # game_img_reshaped = np.pad(game_img, ((0, max_x - game_img.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
            # blue_plan_reshaped = np.pad(blue_plan_img, ((0, max_x - blue_plan_img.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)

            # img = np.concatenate((game_img_reshaped, blue_plan_reshaped), axis=1)
            # imgs.append(img)

        if done:
            break

    save_video(imgs, log_location + "%d.mp4" % episode_num, fps=10)

    mean_return += episode_return / num_iter
    print(episode_return)

print(start_locs)
# sl = np.array(start_locs)
# import matplotlib.pyplot as plt
# heatmap, xedges, yedges = np.histogram2d(sl[:, 0], sl[:, 1], bins=50)
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# plt.clf()
# plt.imshow(heatmap.T, extent=extent, origin='lower')
# plt.show()