import os
import argparse

from cv2 import VIDEOWRITER_PROP_FRAMEBYTES
import sys

project_path = os.getcwd()
sys.path.append(str(project_path))
from simulator import BlueSequenceEnv
from simulator.prisoner_env import PrisonerBothEnv
from simulator.prisoner_perspective_envs import PrisonerBlueEnv
from fugitive_policies.heuristic import HeuristicPolicy
from heuristic import HierRLBlue
import matplotlib
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from blue_bc.policy import MLPNetwork, HighLevelPolicy

matplotlib.use('agg')
import matplotlib.pylab
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from utils import save_video
from blue_bc.utils import BaseTrainer, HierTrainer
import random
from simulator.load_environment import load_environment
from buffer import Buffer

def main(args):
    """Specify the writer"""
    logger = SummaryWriter(log_dir=args.video_dir)
    """Load the environment"""
    device = 'cuda' if args.cuda else 'cpu'
    epsilon = 0.1
    variation = 0
    print(f"Loaded environment variation {variation} with seed {args.seed}")

    # set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = load_environment('simulator/configs/blue_bc.yaml')
    env.seed(args.seed)
    fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    blue_observation, next_blue_partial_observation = env.reset()
    """Load the model"""
    blue_policy = torch.load(args.model_path).to(device)
    blue_hier_policy = HierRLBlue(env, blue_policy, device)
    """Initialize the buffer"""
    hier_act_dim = blue_policy.agent_num * (blue_policy.subpolicy_shape[0] + blue_policy.para_shape[0])
    hier_buffer = Buffer(
        buffer_size=args.subpolicy_buffer_size,
        # state_shape=env.blue_partial_observation_space.shape,
        state_shape=env.blue_observation_space.shape,
        action_shape=(hier_act_dim,),
        device=device
    )
    """Initialize HierTrainer"""
    hier_trainer = HierTrainer(blue_policy, input_dim=hier_act_dim+env.blue_observation_space.shape[0], out_dim=1, hidden_dim=128, update_target_period=5, lr=args.lr, device=device)
    for ep in range(args.episode_num):
        blue_hier_policy.reset()
        blue_hier_policy.init_behavior()

        imgs = []
        t = 0
        # total_return = 0.0
        # num_episodes = 0
        subepisode_return = np.array([0.0])
        episode_return = np.array([0.0])
        done = False
        first_detection = True
        while not done:
            """Start Revising"""
            t = t + 1
            # print("current t = ", t)
            # red_action = red_policy.predict(red_observation)
            # blue_actions = blue_heuristic.step_observation(blue_observation)
            """Partial Blue Obs"""
            # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
            """Full Blue Obs"""
            action, new_detection, hier_action = blue_hier_policy.predict_full_observation(blue_observation)
            next_blue_observation, next_blue_partial_observation, reward, done, _ = env.step(action)
            """Calculate subepisode reward"""
            if new_detection == None:
                subepisode_return = subepisode_return + np.array(reward)
            else:
                if first_detection == False:
                    # Save subepisode_return to old buffer position
                    # Save blue_observation as 'next observation' to old buffer position
                    old_names = ["rewards", "next_states"]
                    old_return_nextObs = [np.array(subepisode_return), blue_observation]
                    pointer_move_flag = True
                    hier_buffer.individual_append(old_names, old_return_nextObs, pointer_move_flag)
                else:
                    first_detection = False
                # Save blue_observation to new buffer position
                # Save hier action to new buffer position
                new_names = ["states", "actions"]
                new_states_actions = [blue_observation, hier_action]
                pointer_move_flag = False
                hier_buffer.individual_append(new_names, new_states_actions, pointer_move_flag)            
                # Set subepisode_return to reward
                subepisode_return = np.array(reward)

            # print("helicopter_actions = ", blue_actions[5])
            # print("blue_actions = ", blue_actions)
            mask = False if t == env.max_timesteps else done
            # print("blue_partial_observation = ", blue_partial_observation)
            # print("next_blue_partial_observation = ", next_blue_partial_observation)
            # print("to_velocity_vector(blue_actions)) = ", to_velocity_vector(blue_actions))
            # print("helicopter_action_theta_speed", blue_actions[5])
            episode_return += np.array(reward)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            if ep % args.video_step == 0:
                game_img = env.render('Policy', show=False, fast=True)
                imgs.append(game_img)
            print("t = ", t)
            if done:
                # print("go into done branch")
                if ep % args.video_step == 0:
                    video_path = args.video_dir + "_" + str(ep) + ".mp4"
                    save_video(imgs, video_path, fps=10)
                    torch.save(blue_policy, args.video_dir + "/rl_policy/" + "_" + str(ep) + ".pth")
                """Terminate one episode"""
                # Save subepisode_return to old buffer position
                # Save blue_observation as 'next observation' to old buffer position
                old_names = ["rewards", "next_states"]
                old_return_nextObs = [np.array(subepisode_return), blue_observation]
                pointer_move_flag = True
                hier_buffer.individual_append(old_names, old_return_nextObs, pointer_move_flag)
                logger.add_scalar('episode_reward', episode_return/t, ep)
                """Start a new episode"""
                blue_observation, blue_partial_observation = env.reset()
                t = 0
                subepisode_return = np.array([0.0])
                episode_return = np.array([0.0])
                imgs = []
                done = False
                first_detection = True

                

                break
        """update Q func"""
        if hier_buffer._n > args.batch_size:
            for _ in range(args.update_num_one_episode):
                hier_trainer.update(hier_buffer.sample(args.batch_size), logger=logger)
    return

def main_scratch(args):
    """Specify the writer"""
    logger = SummaryWriter(log_dir=args.video_dir)
    """Load the environment"""
    device = 'cuda' if args.cuda else 'cpu'
    epsilon = 0.1
    variation = 0
    print(f"Loaded environment variation {variation} with seed {args.seed}")

    # set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = load_environment('simulator/configs/blue_bc.yaml')
    env.seed(args.seed)
    fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    blue_observation, next_blue_partial_observation = env.reset()
    """Load the model"""
    units_actor = (args.units_actor, args.units_actor, args.units_actor)
    blue_policy = HighLevelPolicy(
        # state_shape=env.blue_partial_observation_space.shape,
        state_shape=env.blue_observation_space.shape,
        agent_num = env.num_helicopters + env.num_search_parties,
        subpolicy_shape=(env.subpolicy_num, ),
        para_shape=(env.max_para_num,),
        hidden_units=units_actor,
        hidden_activation=nn.ReLU() # original: nn.Tanh()
        ).to(device)
    blue_hier_policy = HierRLBlue(env, blue_policy, device)
    """Initialize the buffer"""
    hier_act_dim = blue_policy.agent_num * (blue_policy.subpolicy_shape[0] + blue_policy.para_shape[0])
    hier_buffer = Buffer(
        buffer_size=args.subpolicy_buffer_size,
        # state_shape=env.blue_partial_observation_space.shape,
        state_shape=env.blue_observation_space.shape,
        action_shape=(hier_act_dim,),
        device=device
    )
    """Initialize HierTrainer"""
    hier_trainer = HierTrainer(blue_policy, input_dim=hier_act_dim+env.blue_observation_space.shape[0], out_dim=1, hidden_dim=128, update_target_period=5, lr=args.lr, device=device)
    for ep in range(args.episode_num):
        blue_hier_policy.reset()
        blue_hier_policy.init_behavior()

        imgs = []
        t = 0
        # total_return = 0.0
        # num_episodes = 0
        subepisode_return = np.array([0.0])
        episode_return = np.array([0.0])
        done = False
        first_detection = True
        while not done:
            """Start Revising"""
            t = t + 1
            # print("current t = ", t)
            # red_action = red_policy.predict(red_observation)
            # blue_actions = blue_heuristic.step_observation(blue_observation)
            """Partial Blue Obs"""
            # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
            """Full Blue Obs"""
            action, new_detection, hier_action = blue_hier_policy.predict_full_observation(blue_observation)
            next_blue_observation, next_blue_partial_observation, reward, done, _ = env.step(action)
            """Calculate subepisode reward"""
            if new_detection == None:
                subepisode_return = subepisode_return + np.array(reward)
            else:
                if first_detection == False:
                    # Save subepisode_return to old buffer position
                    # Save blue_observation as 'next observation' to old buffer position
                    old_names = ["rewards", "next_states"]
                    old_return_nextObs = [np.array(subepisode_return), blue_observation]
                    pointer_move_flag = True
                    hier_buffer.individual_append(old_names, old_return_nextObs, pointer_move_flag)
                else:
                    first_detection = False
                # Save blue_observation to new buffer position
                # Save hier action to new buffer position
                new_names = ["states", "actions"]
                new_states_actions = [blue_observation, hier_action]
                pointer_move_flag = False
                hier_buffer.individual_append(new_names, new_states_actions, pointer_move_flag)            
                # Set subepisode_return to reward
                subepisode_return = np.array(reward)

            # print("helicopter_actions = ", blue_actions[5])
            # print("blue_actions = ", blue_actions)
            mask = False if t == env.max_timesteps else done
            # print("blue_partial_observation = ", blue_partial_observation)
            # print("next_blue_partial_observation = ", next_blue_partial_observation)
            # print("to_velocity_vector(blue_actions)) = ", to_velocity_vector(blue_actions))
            # print("helicopter_action_theta_speed", blue_actions[5])
            episode_return += np.array(reward)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            if ep % args.video_step == 0:
                game_img = env.render('Policy', show=False, fast=True)
                imgs.append(game_img)
            print("t = ", t)
            if done:
                # print("go into done branch")
                if ep % args.video_step == 0:
                    video_path = args.video_dir + "_" + str(ep) + ".mp4"
                    save_video(imgs, video_path, fps=10)
                    torch.save(blue_policy, args.video_dir + "/rl_policy/" + "_" + str(ep) + ".pth")
                """Terminate one episode"""
                # Save subepisode_return to old buffer position
                # Save blue_observation as 'next observation' to old buffer position
                old_names = ["rewards", "next_states"]
                old_return_nextObs = [np.array(subepisode_return), blue_observation]
                pointer_move_flag = True
                hier_buffer.individual_append(old_names, old_return_nextObs, pointer_move_flag)
                logger.add_scalar('episode_reward_per_step', episode_return/t, ep)
                logger.add_scalar('episode_reward', episode_return, ep)
                """Start a new episode"""
                blue_observation, blue_partial_observation = env.reset()
                t = 0
                subepisode_return = np.array([0.0])
                episode_return = np.array([0.0])
                imgs = []
                done = False
                first_detection = True

                

                break
        """update Q func"""
        if hier_buffer._n > args.batch_size:
            for _ in range(args.update_num_one_episode):
                hier_trainer.update(hier_buffer.sample(args.batch_size), logger=logger)
    return


def main_ddpg(args):
    """Specify the writer"""
    logger = SummaryWriter(log_dir=(args.video_dir + "/test"))
    """Load the environment"""
    device = 'cuda' if args.cuda else 'cpu'
    epsilon = 0.1
    variation = 0
    print(f"Loaded environment variation {variation} with seed {args.seed}")

    # set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = load_environment('simulator/configs/blue_bc.yaml')
    env.seed(args.seed)
    fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    blue_observation, blue_partial_observation = env.reset()
    """Load the model"""
    blue_obs_dim = blue_partial_observation.shape[0]
    blue_act_dim = env.action_space.shape[0]
    blue_policy = MLPNetwork(input_dim=blue_obs_dim, out_dim=blue_act_dim, hidden_dim=128, constrain_out=True, discrete_action=False).to(device)
    # blue_hier_policy = HierRLBlue(env, blue_policy, device)
    """Initialize the buffer"""
    # hier_act_dim = blue_policy.agent_num * (blue_policy.subpolicy_shape[0] + blue_policy.para_shape[0])
    replay_buffer = Buffer(
        buffer_size=args.subpolicy_buffer_size,
        # state_shape=env.blue_partial_observation_space.shape,
        state_shape=blue_partial_observation.shape,
        action_shape=env.action_space.shape,
        device=device
    )
    base_trainer = BaseTrainer(blue_policy, input_dim=blue_obs_dim+blue_act_dim, out_dim=1, hidden_dim=128, update_target_period=5, lr=args.lr, device=device)
    for ep in range(args.episode_num):
        imgs = []
        t = 0
        # total_return = 0.0
        # num_episodes = 0
        subepisode_return = np.array([0.0])
        episode_return = np.array([0.0])
        done = False
        while not done:
            """Start Revising"""
            t = t + 1
            # print("current t = ", t)
            # red_action = red_policy.predict(red_observation)
            # blue_actions = blue_heuristic.step_observation(blue_observation)
            """Partial Blue Obs"""
            # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
            """Full Blue Obs"""
            action = blue_policy(torch.Tensor(blue_partial_observation).to(device))
            next_blue_observation, next_blue_partial_observation, reward, done, _ = env.step(split_directions_to_direction_speed(action.cpu().detach().numpy()))
            """save to buffer and calculate episode reward"""
            replay_buffer.append(blue_partial_observation, action.cpu().detach().numpy(), reward, done, next_blue_partial_observation)
            # print("helicopter_actions = ", blue_actions[5])
            # print("blue_actions = ", blue_actions)
            mask = False if t == env.max_timesteps else done
            # print("blue_partial_observation = ", blue_partial_observation)
            # print("next_blue_partial_observation = ", next_blue_partial_observation)
            # print("to_velocity_vector(blue_actions)) = ", to_velocity_vector(blue_actions))
            # print("helicopter_action_theta_speed", blue_actions[5])
            episode_return += np.array(reward)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            if ep % args.video_step == 0:
                game_img = env.render('Policy', show=True, fast=True)
                # imgs.append(game_img)
            print("t = ", t)
            if done:
                # print("go into done branch")
                # if ep % args.video_step == 0:
                #     video_path = args.video_dir + "_" + str(ep) + ".mp4"
                #     save_video(imgs, video_path, fps=10)
                """Terminate one episode"""
                logger.add_scalar('episode_reward', episode_return/t, ep)
                """Start a new episode"""
                blue_observation, blue_partial_observation = env.reset()
                t = 0
                episode_return = np.array([0.0])
                imgs = []
                done = False
                break
        """update Q func"""
        if replay_buffer._n > args.batch_size:
            for _ in range(args.update_num_one_episode):
                base_trainer.update(replay_buffer.sample(args.batch_size), logger=logger)
    return


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
            search_party_speed = np.maximum(np.linalg.norm(search_party_direction), 1.0) * search_party_v_limit
            blue_actions_norm_angle_vel.append(np.array(search_party_direction.tolist() + [search_party_speed]))
        elif idx < 6:
            helicopter_direction = blue_actions_directions[idx]
            if np.linalg.norm(helicopter_direction) > 1:
                helicopter_direction = helicopter_direction / np.linalg.norm(helicopter_direction)
            helicopter_speed = np.maximum(np.linalg.norm(helicopter_direction), 1.0) * helicopter_v_limit
            blue_actions_norm_angle_vel.append(np.array(helicopter_direction.tolist()+ [helicopter_speed]))  

    return blue_actions_norm_angle_vel 

if __name__ == '__main__':
    """add some configurations"""
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--video_dir', type=str, required=True)

    
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--subpolicy_buffer_size', type=int, default=500000) # original 50000
    p.add_argument('--episode_num', type=int, default=10000) # original 5000
    p.add_argument('--update_num_one_episode', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=128) # original 32
    p.add_argument('--video_step', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.002) # original 0.01
    p.add_argument('--units_actor', type=int, default=128)
    """if we use lstm"""
    p.add_argument('--seq_len', type=int, default=50)
    args = p.parse_args()

    main_scratch(args)
    # main_ddpg(args)


