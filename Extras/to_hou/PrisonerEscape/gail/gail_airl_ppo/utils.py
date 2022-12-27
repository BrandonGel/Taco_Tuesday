from tqdm import tqdm
import numpy as np
import torch

from .buffer import Buffer
from utils import save_video


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, blue_heuristic, red_policy, buffer_size, device, std, p_rand, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    """Start to Revise"""
    t = 0
    total_return = 0.0
    num_episodes = 0
    episode_return = 0.0
    # red_observation = env.reset()
    blue_observation, blue_partial_observation = env.reset()
    blue_heuristic.reset()
    blue_heuristic.init_behavior()
    # blue_observation = env.get_blue_observation()
    # blue_act_sp_shape = blue_heuristic.joint_action_space_len(blue_observation)

    buffer = Buffer(
        buffer_size=buffer_size,
        # state_shape=env.blue_partial_observation_space.shape,
        state_shape=env.blue_observation_space.shape,
        action_shape=env.blue_action_space_shape,
        device=device
    )
    """End Revising"""
    # total_return = 0.0
    # num_episodes = 0

    # state = env.reset()
    # t = 0
    # episode_return = 0.0
    imgs = []
    filled_buffer_len = 0
    # for _ in tqdm(range(1, buffer_size + 1)):
    while filled_buffer_len < buffer_size:
        """Start Revising"""
        t = t + 1
        
        """Partial Blue Obs"""
        # blue_actions = blue_heuristic.predict(blue_partial_observation)
        """Full Blue Obs"""
        blue_actions = blue_heuristic.predict_full_observation(blue_observation)
        
        # print("helicopter_actions = ", blue_actions[5])
        # print("blue_actions = ", blue_actions)
        next_blue_observation, next_blue_partial_observation, reward, done, _ = env.step(blue_actions)
        # next_blue_obs = env.get_blue_observation()
        mask = False if t == env.max_timesteps else done
        """Partial Blue Obs"""
        # buffer.append(blue_partial_observation, np.concatenate(to_velocity_vector(blue_actions)), reward, mask, next_blue_partial_observation)
        """Full Blue Obs"""
        if env.prisoner_detected_loc_history2[0:2] != [-1, -1]:
            buffer.append(blue_observation, np.concatenate(to_velocity_vector(blue_actions)), reward, mask, next_blue_observation)
            filled_buffer_len = filled_buffer_len + 1
            print("Now the buffer filling process completes %f" % (filled_buffer_len/buffer_size))
        # print("blue_partial_observation = ", blue_observation)
        # print("next_blue_partial_observation = ", next_blue_observation)
        # print("to_velocity_vector(blue_actions)) = ", to_velocity_vector(blue_actions))
        # print("helicopter_action_theta_speed", blue_actions[5])
        episode_return += reward

        blue_observation = next_blue_observation
        blue_partial_observation = next_blue_partial_observation
        game_img = env.render('Policy', show=True, fast=True)
        imgs.append(game_img)
        if done:
            num_episodes += 1
            total_return += episode_return

            blue_observation, blue_partial_observation = env.reset()
            blue_heuristic.reset()
            blue_heuristic.init_behavior()

            t = 0
            episode_return = 0.0
        
        """End Revising"""
        # t += 1

        # if np.random.rand() < p_rand:
        #     action = env.action_space.sample()
        # else:
        #     action = algo.exploit(state)
        #     action = add_random_noise(action, std)

        # next_state, reward, done, _ = env.step(action)
        # mask = False if t == env._max_episode_steps else done
        # buffer.append(state, action, reward, mask, next_state)
        # episode_return += reward

        # if done:
        #     num_episodes += 1
        #     total_return += episode_return
        #     state = env.reset()
        #     t = 0
        #     episode_return = 0.0

        # state = next_state
    save_video(imgs, "/home/wu/GatechResearch/Zixuan/PrisonerEscape/buffers/video" + "/%d.mp4" % buffer_size, fps=10)
    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer

def to_angle_speed(blue_actions):
    for idx in range(len(blue_actions)):
        if idx < 5:
            theta = np.arctan2(blue_actions[idx][1], blue_actions[idx][0])
            if theta < 0:
                theta = np.pi * 2 + theta
            theta = (theta / (2 * np.pi)) * 2 + (-1)
            speed = (blue_actions[idx][2] / 6.5) * 2 + (-1)
            blue_actions[idx] = np.array([theta, speed])
            # if blue_actions[idx][2] == 0:
            #     blue_actions[idx] = np.array([-1, -1])
        elif idx < 6:
            theta = np.arctan2(blue_actions[idx][1], blue_actions[idx][0])
            if theta < 0:
                theta = np.pi * 2 + theta
            theta = (theta / (2 * np.pi)) * 2 + (-1)
            speed = (blue_actions[idx][2] / 127) * 2 + (-1)
            blue_actions[idx] = np.array([theta, speed])   
            # if blue_actions[idx][2] == 0:
            #     blue_actions[idx] = np.array([-1, -1])      
    return blue_actions

def to_velocity_vector(blue_actions):
    velocities = []
    for idx in range(len(blue_actions)):
        direction_vector = np.array([blue_actions[idx][0], blue_actions[idx][1]])
        normalized_v_x = np.clip(direction_vector[0], a_min=-1, a_max=1)
        normalized_v_y = np.clip(direction_vector[1], a_min=-1, a_max=1)
        velocities.append(np.array([normalized_v_x, normalized_v_y]))
    return velocities
            


            
