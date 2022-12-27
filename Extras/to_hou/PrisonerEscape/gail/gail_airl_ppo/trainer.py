import os
import numpy as np
from time import time, sleep
from datetime import timedelta
from utils import save_video
from torch.utils.tensorboard import SummaryWriter

from blue_policies.heuristic import BlueHeuristic


class Trainer:

    def __init__(self, env, env_test, blue_heuristic, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=1):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2**31-seed)
        """search parties + helicopter"""
        self.search_party_num = len(self.env.search_parties_list)
        self.search_party_vel = self.env.search_parties_list[0].speed
        self.helicopter_num = len(self.env.helicopters_list)
        self.helicopter_vel = self.env.helicopters_list[0].speed

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

        self.blue_heuristic = blue_heuristic
        # self.red_policy = red_policy

    def train(self):

        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        # red_observation = self.env.reset()
        self.blue_heuristic.reset()
        self.blue_heuristic.init_behavior()
        blue_observation, blue_partial_observation = self.env.reset()

        for step in range(1, self.num_steps + 1):
            print("current step is: ", step)
            # Pass to the algorithm to update state and episode timestep.
            # blue_observation, _, done, t = self.algo.step(self.env, blue_observation, self.blue_heuristic, t, step)
            blue_partial_observation, _, done, t = self.algo.step(self.env, blue_partial_observation, self.blue_heuristic, t, step)
            
            # blue_observation = next_blue_obs
            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}.pkl'))

            if done:
                t = 0
                # red_observation = self.env.reset()
                self.blue_heuristic.reset()
                self.blue_heuristic.init_behavior()
                blue_observation, blue_partial_observation = self.env.reset()

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        for _ in range(self.num_eval_episodes):
            """Start to Evaluate"""
            imgs = []
            self.blue_heuristic.reset()
            self.blue_heuristic.init_behavior()
            blue_observation, blue_partial_observation = self.env_test.reset()

            episode_return = 0.0
            done = False
            """End to Evaluate"""
            # state = self.env_test.reset()
            # episode_return = 0.0
            # done = False

            while (not done):

                # blue_actions = self.algo.exploit(blue_observation)
                blue_actions = self.algo.exploit(blue_partial_observation)
                
                # print("blue_actions from NN output = ", blue_actions)
                # blue_actions = self.split_to_direction_speed(blue_actions)
                blue_actions = self.split_directions_to_direction_speed(blue_actions)
                # red_action = self.red_policy.predict(red_observation)
                # blue_actions = blue_actions + 3
                # red_observation, reward, done, _ = self.env_test.step(red_action[0], blue_actions)
                blue_observation, blue_partial_observation, reward, done, _ = self.env_test.step(blue_actions)
                print("blue_partial_observation = ", blue_partial_observation)
                # blue_observation = self.env_test.get_blue_observation()
                # print("blue_observation = ", blue_observation)
                # print("blue_actions = ", blue_actions)

                
                game_img = self.env_test.render('Policy', show=True, fast=True)
                imgs.append(game_img)

                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        save_video(imgs, self.log_dir + "/videos/%d.mp4" % step, fps=10)
        self.writer.add_scalar('return/test', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    # def split_normalize_direction_denormalize_vel(self, blue_actions):
    #     blue_actions = np.split(blue_actions, self.search_party_num + self.helicopter_num)
    #     for idx, omega_v in enumerate(blue_actions):
    #         if idx < self.search_party_num:
    #             blue_actions[idx][0:2] = blue_actions[idx][0:2] / np.linalg.norm(blue_actions[idx][0:2])
    #             blue_actions[idx][2] = blue_actions[idx][2] * self.search_party_vel
    #         elif idx < self.search_party_num + self.helicopter_num:
    #             blue_actions[idx][0:2] = blue_actions[idx][0:2] / np.linalg.norm(blue_actions[idx][0:2])
    #             blue_actions[idx][2] = blue_actions[idx][2] * self.helicopter_vel
    #     return blue_actions

    def split_directions_to_direction_speed(self, directions):
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
        
    def split_to_direction_speed(self, blue_actions_norm_angle_vel):
        blue_actions_norm_angle_vel = np.split(blue_actions_norm_angle_vel, 6)
        for idx in range(len(blue_actions_norm_angle_vel)):
            if idx < 5:
                speed = (blue_actions_norm_angle_vel[idx][1] + 1) / 2.0 * 6.5
                angle = (blue_actions_norm_angle_vel[idx][0] + 1) / 2.0 * 2.0 * np.pi 
                x_direction_normalized = np.cos(angle)
                y_direction_normalized = np.sin(angle)
                blue_actions_norm_angle_vel[idx] = np.array([x_direction_normalized, y_direction_normalized, speed])
            elif idx < 6:
                speed = (blue_actions_norm_angle_vel[idx][1] + 1) / 2.0 * 127
                angle = (blue_actions_norm_angle_vel[idx][0] + 1) / 2.0 * 2.0 * np.pi 
                x_direction_normalized = np.cos(angle)
                y_direction_normalized = np.sin(angle)
                blue_actions_norm_angle_vel[idx] = np.array([x_direction_normalized, y_direction_normalized, speed])

        return blue_actions_norm_angle_vel

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
