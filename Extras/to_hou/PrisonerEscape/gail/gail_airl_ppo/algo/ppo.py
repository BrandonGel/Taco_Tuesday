import torch
from torch import nn
from torch.optim import Adam
import numpy as np

from simulator import helicopter


from .base import Algorithm
from gail_airl_ppo.buffer import RolloutBuffer
from gail_airl_ppo.network import StateIndependentPolicy, StateFunction


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU() # original: nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU() # original: nn.Tanh()
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, blue_observation, blue_heuristic, t, step):
        t += 1
        # red_action = red_policy.predict(red_observation)

        blue_actions, log_pi = self.explore(blue_observation)
        
        # blue_actions_direction_speed = self.split_to_direction_speed(blue_actions)
        blue_actions_direction_speed = self.split_directions_to_direction_speed(blue_actions)

        # next_red_obs, reward, done, _ = env.step(red_action[0], blue_actions_direction_speed)
        blue_obs, partial_blue_obs, reward, done, _  = env.step(blue_actions_direction_speed)
        next_blue_obs = partial_blue_obs
        # next_blue_obs = blue_obs
        mask = False if t == env.max_timesteps else done

        self.buffer.append(blue_observation, blue_actions, reward, mask, log_pi, next_blue_obs)

        # if done:
        #     t = 0
        #     red_observation = env.reset()

        return next_blue_obs, _, done, t

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

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states) # MLP
            next_values = self.critic(next_states) # MLP
            # print("states = ", states)
            # print("next_states = ", next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        torch.save(self.actor, save_dir)
