# from os import pathconf_names
# from simulator.prisoner_env import PrisonerEnv, PrisonerGoalEnv
# from simulator.prisoner_batch_wrapper import PrisonerBatchEnv
# from stable_baselines3.ppo import PPO
# import sklearn.metrics as metrics
import matplotlib
import torch
import matplotlib.pyplot as plt
# import gym
import numpy as np
from tqdm import tqdm
import pickle
import cv2
import yaml

import os
import sys
sys.path.append(os.getcwd())

from math import log2

matplotlib.use('agg')
import matplotlib.pylab
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from utils import save_video


def generate_heatmap_img(matrix, sigma=0, vmin_vmax=None, true_location=None):
    """ Given a matrix generate the heatmap image 
    True location plots an X where the fugitive's true location is
    
    """
    smoothed = []
    # fig, ax = plt.subplots()
    # matrix = np.transpose(matrix)
    # smooth the matrix

    smoothed_matrix = gaussian_filter(matrix, sigma=sigma)
    smoothed.append(smoothed_matrix)
    # Set 0s to None as they will be ignored when plotting
    # smoothed_matrix[smoothed_matrix == 0] = None
    matrix[matrix == 0] = None
    # Plot the data
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            sharex=False, sharey=True,
                            figsize=(5, 5))
    # ax1.matshow(display_matrix, cmap='hot')
    # ax1.set_title("Original matrix")

    x_min = 0;
    x_max = 2428;
    y_min = 0;
    y_max = 2428
    extent = [x_min, x_max, y_min, y_max]

    if vmin_vmax is not None:
        im = ax1.matshow(smoothed_matrix, extent=extent, vmin=vmin_vmax[0], vmax=vmin_vmax[1])
    else:
        im = ax1.matshow(smoothed_matrix, extent=extent)

    if true_location is not None:
        # plt.plot(true_location[0], true_location[1], 'x', color='red')
        ax1.scatter(true_location[0], true_location[1], color='r', s=50)
    # print(im.get_clim())
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([])
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    cbar.ax.invert_yaxis()
    # plt.show()

    # plt.savefig("simulator/temp" + str(frame_i) + ".png")
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close('all')
    return img


def generate_policy_heatmap_video(env, policy, num_timesteps=2520, num_rollouts=20, path='simulator/temp.mp4'):
    """
    Generates the heatmap displaying probabilities of ending up in certain cells
    :param current_state: current location of prisoner, current state of world
    :param policy: must input state, output action
    :param num_timesteps: how far in time ahead, remember time is in 15 minute intervals.
    """
    time_between_frames = 60
    num_frames = num_timesteps // time_between_frames
    # print(num_frames)
    # Create 3D matrix
    display_matrix = np.zeros((num_frames, env.dim_x + 1, env.dim_y + 1))

    for num_traj in tqdm(range(num_rollouts), desc="generating_heatmap"):
        _, observation = env.reset()
        frame_index = 0
        for j in range(num_timesteps):
            # action = policy.predict(observation, deterministic=False)[0]
            action = policy(torch.Tensor(observation).cuda())
            _, observation, _, done, _ = env.step(split_directions_to_direction_speed(action.cpu().detach().numpy()))
            
            # update count
            if frame_index >= num_frames:
                break
            elif j % time_between_frames == 0:
                display_matrix[frame_index, env.prisoner.location[0], env.dim_y - env.prisoner.location[1]] += 4
                frame_index += 1
            if done:
                break
        if done:
            for frame_i in range(frame_index, num_frames):
                display_matrix[frame_i, env.prisoner.location[0], env.dim_y - env.prisoner.location[1]] += 4
            # self.render('human', show=True)
    imgs = []
    smoothed = []
    # norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    for frame_i in tqdm(range(num_frames)):
        matrix = display_matrix[frame_i]
        fig, ax = plt.subplots()
        matrix = np.transpose(matrix)
        # smooth the matrix
        smoothed_matrix = gaussian_filter(matrix, sigma=50)
        smoothed.append(smoothed_matrix)
        # Set 0s to None as they will be ignored when plotting
        # smoothed_matrix[smoothed_matrix == 0] = None
        matrix[matrix == 0] = None
        # Plot the data
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                sharex=False, sharey=True,
                                figsize=(5, 5))
        # ax1.matshow(display_matrix, cmap='hot')
        # ax1.set_title("Original matrix")
        im = ax1.matshow(smoothed_matrix, vmin=0.0, vmax=0.001)
        # print(im.get_clim())
        num_hours = str((frame_i * time_between_frames / 60).__round__(2))

        ax1.set_title("Heatmap at Time t=" + num_hours + ' hours')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks([])
        plt.tight_layout()
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        cbar.ax.invert_yaxis()
        # plt.show()

        # plt.savefig("simulator/temp" + str(frame_i) + ".png")
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs.append(img)
        plt.close('all')

    save_video(imgs, path, fps=2)
    return smoothed

def split_directions_to_direction_speed(directions):
    blue_actions_norm_angle_vel = []
    blue_actions_directions = np.split(directions, 6)
    search_party_v_limit = 6.5
    helicopter_v_limit = 127
    for idx in range(len(blue_actions_directions)):
        if idx < 5:
            search_party_direction = blue_actions_directions[idx]
            if np.linalg.norm(search_party_direction) > 1:
                search_party_direction = search_party_direction / np.linalg.norm(search_party_direction)
            search_party_speed = search_party_v_limit
            blue_actions_norm_angle_vel.append(np.array(search_party_direction.tolist() + [search_party_speed]))
        elif idx < 6:
            helicopter_direction = blue_actions_directions[idx]
            if np.linalg.norm(helicopter_direction) > 1:
                helicopter_direction = helicopter_direction / np.linalg.norm(helicopter_direction)
            helicopter_speed = helicopter_v_limit
            blue_actions_norm_angle_vel.append(np.array(helicopter_direction.tolist()+ [helicopter_speed]))  

    return blue_actions_norm_angle_vel    

def compare_heatmaps_l2(path_1, path_2, save_path):
    with open(path_1, 'rb') as handle:
        bc = pickle.load(handle)

    with open(path_2, 'rb') as handle:
        gt = pickle.load(handle)

    norms = []
    hours = list(range(0, len(bc)))
    for bc_matrix, gc_matrix in zip(bc, gt):
        fig, ax = plt.subplots()
        l2_norm = np.linalg.norm(bc_matrix - gc_matrix)
        norms.append(l2_norm)

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.style.use('ggplot')
    params = {
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'legend.fontsize': 'xx-large',
        # 'figure.figsize': (6, 5),
        'axes.labelsize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'xx-large',
        'ytick.labelsize': 'xx-large'}
    matplotlib.pylab.rcParams.update(params)

    plt.figure()
    # plt.plot(model_history.epoch, loss, 'r', linewidth=5.0, label='Training loss')
    plt.plot(hours, norms, 'b', linewidth=5.0)
    plt.title('Error')
    plt.xlabel('Hours')
    plt.ylabel('L2 Norm')
    plt.ylim([0, 0.65])
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def save_heatmap_video_rl():
    ppo_model_path = "/star-data/prisoner-policies/hier/trained-workers/manager/ppo.zip"
    env = PrisonerEnv(spawn_mode='normal', random_cameras=False, observation_step_type='Fugitive')
    policies = [torch.load(f'bc_policies/goal_{i}_policy.pth') for i in range(3)]
    env = OptionWorker(env, policies, env.hideout_locations[:])
    model = PPO.load(ppo_model_path)

    smoothed_matrices = generate_policy_heatmap_video(env, policy=model.policy, num_timesteps=1200,
                                                      path='temp/heatmap_rl.mp4')
    with open("scripts/heatmaps/rl_matrix.pkl", 'wb') as f:
        pickle.dump(smoothed_matrices, f)


def save_heatmap_bc():
    from behavioral_cloning.bc import reconstruct_policy
    path = "weights/bc_rl_ground_truth_obs.pth"
    policy = reconstruct_policy(path)

    env = PrisonerEnv(spawn_mode='normal', random_cameras=False, observation_step_type="GroundTruth")
    smoothed_matrices = generate_policy_heatmap_video(env, policy=policy, num_timesteps=600,
                                                      path="scripts/videos/heatmap_bc_ground_truth_obs.mp4")
    with open("scripts/heatmaps/bc_gt_matrix.pkl", 'wb') as f:
        pickle.dump(smoothed_matrices, f)


def save_heatmap_ground_truth_heuristic():
    from fugitive_policies.heuristic import HeuristicPolicy

    # set random seed
    np.random.seed(0)

    env_kwargs = {}
    env_kwargs['spawn_mode'] = "normal"
    # env_kwargs['reward_scheme'] = reward_scheme
    env_kwargs['random_cameras'] = False
    env_kwargs['observation_step_type'] = "Fugitive"

    # Directory to randomly cycle between all the maps
    # env_kwargs['terrain_map'] = 'simulator/forest_coverage/maps'

    # Single map to always test on one map
    env_kwargs['terrain_map'] = 'simulator/forest_coverage/maps/1.npy'
    env_kwargs['camera_file_path'] = "simulator/camera_locations/fill_camera_locations.txt"
    env_kwargs['observation_terrain_feature'] = True
    env_kwargs['random_hideout_locations'] = True
    env = PrisonerEnv(**env_kwargs)
    heuristic_policy = HeuristicPolicy(env)
    generate_policy_heatmap_video(env, policy=heuristic_policy, num_timesteps=1200,
                                  path="scripts/videos/heatmap_ground_truth_heuristic_uniform.mp4")


if __name__ == '__main__':
    # save_heatmap_video_rl()
    # save_heatmap_bc()
    # compare_heatmaps_l2('scripts/heatmaps/bc_matrix.pkl', 'scripts/heatmaps/rl_matrix.pkl', 'scripts/figures/bc_rl_l2_norm.png')

    # save_heatmap_ground_truth_heuristic()

    # Behavioral Cloning
    from behavioral_cloning.bc import reconstruct_policy
    import os
    import random

    # set random seed
    np.random.seed(0)
    random.seed(0)

    # Behavioral Cloning
    # path = "/nethome/sye40/PrisonerEscape/logs/bc_train/20220217-1434/policy_epoch_84.pth"
    # path = "/nethome/sye40/PrisonerEscape/logs/bc_train/20220217-1434/bc_best.pth"
    # path = "/nethome/sye40/PrisonerEscape/logs/dagger/20220218-1602/bc_best.pth"

    # Dagger with normal
    # path = "/nethome/sye40/PrisonerEscape/logs/dagger/20220221-1018/policy_epoch_75.pth"

    # policy = reconstruct_policy(path)

    # env = PrisonerEnv(spawn_mode='normal', 
    #             random_cameras=False, 
    #             observation_step_type="Fugitive",
    #             terrain_map = 'simulator/forest_coverage/maps/1.npy',
    #             camera_file_path = 'simulator/camera_locations/fill_camera_locations.txt',
    #             random_hideout_locations = True,
    #             observation_terrain_feature = True)
    # smoothed_matrices = generate_policy_heatmap_video(env, num_rollouts=15, policy=policy, num_timesteps=2100, path="scripts/videos/heatmap_dagger_uniform_normal.mp4")

    # Behavioral Cloning with set hideouts and stop condition
    # path = '/nethome/sye40/PrisonerEscape/logs/bc_train/20220222-2232/policy_epoch_4.pth'

    # Behavioral Cloning with LSTM
    # path = "/nethome/sye40/PrisonerEscape/logs/bc/20220223-0110/policy_epoch_2000.pth"
    
    # DAGGER
    # path = "/nethome/sye40/PrisonerEscape/logs/dagger/20220222-2219/policy_epoch_50.pth"

    # BC without LSTM
    path = "/nethome/sye40/PrisonerEscape/logs/bc_train/20220223-0930/policy_epoch_610.pth"
    policy = reconstruct_policy(path)

    config_path = os.path.join(os.path.dirname(path), 'config.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # if goal_env_bool:
    #     env = PrisonerGoalEnv()
    #     env.set_hideout_goal(goal)
    # else:
    env = PrisonerEnv(spawn_mode=config["environment"]["spawn_mode"], 
                        observation_step_type="Fugitive", 
                        random_cameras=config['environment']['random_cameras'], 
                        camera_file_path=config['environment']['camera_file_path'], 
                        place_mountains_bool=config['environment']['place_mountains_bool'], 
                        camera_range_factor=config['environment']['camera_range_factor'],
                        terrain_map=config['environment']['terrain_map'],
                        observation_terrain_feature=config['environment']['observation_terrain_feature'],
                        random_hideout_locations=config['environment']['random_hideout_locations'],
                        spawn_range = config['environment']['spawn_range'],
                        helicopter_battery_life=config['environment']['helicopter_battery_life'],
                        helicopter_recharge_time=config['environment']['helicopter_recharge_time'],
                        known_hideout_locations = config['environment']['known_hideout_locations'],
                        unknown_hideout_locations = config['environment']['unknown_hideout_locations'])
    
    # batch_env = PrisonerBatchEnv(env, batch_size=config['bc']['num_observations'])
    smoothed_matrices = generate_policy_heatmap_video(env, num_rollouts=15, policy=policy, num_timesteps=2100, path="scripts/videos/bc_new.mp4")
    
    # from fugitive_policies.heuristic import HeuristicPolicy
    # heuristic_policy = HeuristicPolicy(env)
    # smoothed_matrices = generate_policy_heatmap_video(env, num_rollouts=15, policy=heuristic_policy, num_timesteps=2100, path="scripts/videos/ground_truth.mp4")