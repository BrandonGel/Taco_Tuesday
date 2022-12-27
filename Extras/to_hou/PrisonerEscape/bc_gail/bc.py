import os
import gym
import numpy as np
import torch as th
import shutil
# import tqdm.autonotebook as tqdm
from tqdm import tqdm

from datetime import datetime
from torch.distributions import Normal
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Type, Union
# from . import policies as policy_base
from stable_baselines3.common import policies, utils, vec_env
from heatmap import generate_policy_heatmap_video, generate_demo_video
import sys
sys.path.append("/home/wu/GatechResearch/Zixuan/PrisonerEscape")
from simulator.tl_coverage.autoencoder import train
from buffer import Buffer, SerializedBuffer
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common import policies
from gym import spaces
import random




from simulator import PrisonerBothEnv, PrisonerBlueEnv
from fugitive_policies.heuristic import HeuristicPolicy
from blue_policies.heuristic import BlueHeuristic, SimplifiedBlueHeuristic
from gail.gail_airl_ppo.network import StateIndependentPolicy
import torch
import torch.nn as nn
import argparse

# def evaluate_mean_reward(env, policy, num_iter, goal):
#     mean_return = 0.0

#     episode_return_list = []
#     for _ in tqdm(range(1, num_iter+1)):
#         # if goal < 0:
#         #     goal_hideout = num_iter % 3
#         #     env.set_hideout_goal(goal_hideout)
#             # print(goal_hideout)
#         # env = gym.make("PrisonerEscape-v0")

#         state = env.reset()
#         done = False
#         episode_return = 0.0
#         while not done:
#             action = policy(state)
#             state, reward, done, _ = env.step(action[0])
#             episode_return += reward

#             if done:
#                 break

#         mean_return += episode_return / num_iter
#         episode_return_list.append([episode_return])
#     return mean_return

# def split_buffer(buffer: ReplayBuffer, split_percentage = 0.8):
#     """ Splits a ReplayBuffer into two, one for test and one for train"""
#     total_size = buffer.buffer_size
#     # print(buffer.actions.shape, total_size)

#     indices = np.random.permutation(total_size)

#     num_train = int(total_size * split_percentage)
    
#     train_indices = indices[-num_train:]
#     val_indices = indices[:-num_train]

#     observations_train = buffer.observations[train_indices.tolist()]
#     next_observations_train = buffer.next_observations[train_indices.tolist()]
#     actions_train = buffer.actions[train_indices.tolist()]
#     dones_train = buffer.dones[train_indices.tolist()]


#     observations_test = buffer.observations[val_indices.tolist()]
#     next_observations_test = buffer.next_observations[val_indices.tolist()]
#     actions_test = buffer.actions[val_indices.tolist()]
#     dones_test = buffer.dones[val_indices.tolist()]

#     train_buffer = ReplayBuffer(
#         buffer_size=num_train,
#         observation_space=buffer.observation_space,
#         action_space=buffer.action_space,
#         handle_timeout_termination=False
#     )

#     train_buffer.observations = observations_train
#     train_buffer.actions = actions_train
#     train_buffer.dones = dones_train
#     train_buffer.next_observations = next_observations_train
#     train_buffer.pos = num_train

#     test_buffer = ReplayBuffer(
#         buffer_size=len(val_indices),
#         observation_space=buffer.observation_space,
#         action_space=buffer.action_space,
#         handle_timeout_termination=False
#     )

#     test_buffer.observations = observations_test
#     test_buffer.actions = actions_test
#     test_buffer.dones = dones_test
#     test_buffer.next_observations = next_observations_test
#     test_buffer.pos = len(val_indices)

#     return train_buffer, test_buffer

def main(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    epsilon = 0.1
    seed = 1
    variation = 0
    print(f"Loaded environment variation {variation} with seed {seed}")

    # set seeds
    np.random.seed(seed)
    random.seed(seed)


    terrain_map = f'simulator/tl_coverage/map_set/{variation}.npy'
    if variation == 0:
        mountain_locations = [(400, 300), (1600, 1800)] # original mountain setup
    elif variation == 1:
        mountain_locations = [(400, 2000), (1500, 1000)]
    elif variation == 2:
        mountain_locations = [(1000, 900), (1500, 1300)]
    elif variation == 3:
        mountain_locations = [(600, 1100), (1000, 1900)]
    else:
        raise ValueError(f'Invalid variation {variation}')

    camera_configuration="simulator/camera_locations/original_and_more.txt"
    observation_step_type="Fugitive" 
    step_reset=True 
    terrain=None

    env = PrisonerBothEnv(terrain=terrain,
                        spawn_mode='corner',
                        observation_step_type=observation_step_type,
                        random_cameras=False,
                        camera_file_path=camera_configuration,
                        mountain_locations=mountain_locations,
                        camera_range_factor=1.0,
                        observation_terrain_feature=False,
                        random_hideout_locations=False,
                        spawn_range=350,
                        helicopter_battery_life=200,
                        helicopter_recharge_time=40,
                        num_search_parties=5,
                        terrain_map=terrain_map,
                        step_reset = step_reset
                        )
    env.seed(seed)
    red_policy = HeuristicPolicy(env, epsilon=epsilon)
    env = PrisonerBlueEnv(env, red_policy)

    # blue_heuristic = BlueHeuristic(env, debug=False)
    # blue_heuristic.reset()
    # red_policy = HeuristicPolicy(env, epsilon=epsilon)
    # blue_obs = env.reset()
    # blue_obs = env.reset()
    blue_heuristic = SimplifiedBlueHeuristic(env, debug=False)
    blue_partial_observation = env.reset()
    blue_heuristic.reset()
    blue_heuristic.init_behavior()

    buffer_exp = SerializedBuffer(path=args.buffer, device=device)
    units_actor = (args.units_actor, args.units_actor, args.units_actor)
    bc_policy = StateIndependentPolicy(
        # state_shape=env.blue_partial_observation_space.shape,
        state_shape=env.blue_observation_space.shape,
        action_shape=env.blue_action_space_shape,
        hidden_units=units_actor,
        hidden_activation=nn.ReLU() # original: nn.Tanh()
        ).to(device)

    train_bc(args=args, env=env, replay_buffer=buffer_exp, policy=bc_policy, epochs=5, video_step=128, save_folder=None)


def train_bc(args, env, replay_buffer, policy, epochs, video_step, save_folder=None):
    """Train a stable baselines 3 policy on a stable baselines 3 replay buffer."""
    
    time = datetime.now().strftime("%Y%m%d-%H%M")
    if save_folder:
        log_dir = os.path.join(save_folder, str(time))
    else:
        log_dir = os.path.join("logs", "bc", str(time))

    summary_dir = os.path.join(log_dir, 'summary')
    writer = SummaryWriter(log_dir=summary_dir)

    # if config_file:
    #     shutil.copy(config_file, os.path.join(log_dir, "config.yaml"))

    bc_trainer = BC(
        args=args,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=replay_buffer,
        device='cuda' if args.cuda else 'cpu',
        writer = writer,
        log_dir = log_dir,
        policy=policy,
        env = env, 
    )
    bc_trainer.train(n_epochs=args.epochs, batch_size=args.batch_size, video_step = args.video_step)

    if save_folder is not None:
        save_path = os.path.join(save_folder, "bc.pth")
    else:
        save_path = os.path.join(log_dir, "bc.pth")
        
    bc_trainer.save_policy(save_path)
    return bc_trainer.policy

def reconstruct_policy(
    policy_path: str,
    device: Union[th.device, str] = "auto",
) -> policies.BasePolicy:
    """Reconstruct a saved policy.

    Args:
        policy_path: path where `.save_policy()` has been run.
        device: device on which to load the policy.

    Returns:
        policy: policy with reloaded weights.
    """
    policy = th.load(policy_path, map_location=utils.get_device(device))
    # assert isinstance(policy, policies.BasePolicy)
    return policy


class ConstantLRSchedule:
    """A callable that returns a constant learning rate."""

    def __init__(self, lr: float = 1e-3):
        """Builds ConstantLRSchedule.

        Args:
            lr: the constant learning rate that calls to this object will return.
        """
        self.lr = lr

    def __call__(self, _):
        """Returns the constant learning rate."""
        return self.lr

class BC:
    """Behavioral cloning (BC).

    Recovers a policy via supervised learning from observation-action pairs.
    """

    def __init__(
        self,
        args,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Optional[policies.BasePolicy] = None,
        demonstrations: Buffer, 
        # batch_size: int = 32,
        # optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, th.device] = "auto",
        writer,
        log_dir,
        env
        # goal
    ):
        """Builds BC.

        Args:
            observation_space: the observation space of the environment.
            action_space: the action space of the environment.
            policy: a Stable Baselines3 policy; if unspecified,
                defaults to `FeedForward32Policy`.
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            batch_size: The number of samples in each batch of expert data.
            optimizer_cls: optimiser to use for supervised training.
            optimizer_kwargs: keyword arguments, excluding learning rate and
                weight decay, for optimiser construction.
            ent_weight: scaling applied to the policy's entropy regularization.
            l2_weight: scaling applied to the policy's L2 regularization.
            device: name/identity of device to place policy on.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: If `weight_decay` is specified in `optimizer_kwargs` (use the
                parameter `l2_weight` instead.)
        """
        self.batch_size = args.batch_size

        self.demonstrations = demonstrations
        self.writer = writer
        self.log_dir = log_dir
        self.env = env
        # self.goal = goal

        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError("Use the parameter l2_weight instead of weight_decay.")
        self.tensorboard_step = 0

        self.action_space = action_space
        self.observation_space = observation_space
        self.device = device

        # if policy is None:
        #     # policy = policy_base.FeedForward32Policy(
        #     #     observation_space=observation_space,
        #     #     action_space=action_space,
        #     #     # Set lr_schedule to max value to force error if policy.optimizer
        #     #     # is used by mistake (should use self.optimizer instead).
        #     #     lr_schedule=ConstantLRSchedule(th.finfo(th.float32).max),
        #     # )

        #     policy = SACPolicy(           
        #         observation_space=observation_space,
        #         action_space=action_space,
        #         # Set lr_schedule to max value to force error if policy.optimizer
        #         # is used by mistake (should use self.optimizer instead).
        #         lr_schedule=ConstantLRSchedule(th.finfo(th.float32).max),
        #         net_arch=[32, 32]
        #     )
        # self._policy = policy.to(self.device)
        self.policy = policy
        # TODO(adam): make policy mandatory and delete observation/action space params?
        # assert self.policy.observation_space == self.observation_space
        # assert self.policy.action_space == self.action_space

        # optimizer_kwargs = optimizer_kwargs or {}
        # self.optimizer = optimizer_cls(
        #     self.policy.parameters(),
        #     **optimizer_kwargs,
        # )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr_actor)
        self.ent_weight = ent_weight
        self.l2_weight = l2_weight

    # @property
    # def policy(self) -> policies.BasePolicy:
    #     return self._policy

    def _calculate_loss(
        self,
        obs: Union[th.Tensor, np.ndarray],
        acts: Union[th.Tensor, np.ndarray],
    ) -> Tuple[th.Tensor, Mapping[str, float]]:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            obs: The observations seen by the expert. If this is a Tensor, then
                gradients are detached first before loss is calculated.
            acts: The actions taken by the expert. If this is a Tensor, then its
                gradients are detached first before loss is calculated.

        Returns:
            loss: The supervised learning loss for the behavioral clone to optimize.
            stats_dict: Statistics about the learning process to be logged.

        """
        obs = th.as_tensor(obs, device=self.device).detach()
        acts = th.as_tensor(acts, device=self.device).detach()
       	            
        # if isinstance(self.policy, SACPolicy):
        #     mean_actions, log_std, kwargs = self.policy.actor.get_action_dist_params(obs)
        #     # print(mean_actions.shape, log_std.shape)
        #     distribution = Normal(mean_actions, th.exp(log_std))
        #     log_prob = distribution.log_prob(acts)
        #     entropy = distribution.entropy()
        # else:
            # log_pis = self.actor.evaluate_log_pi(states, actions)
        log_prob = self.policy.evaluate_log_pi(obs, acts)
            # _, log_prob, entropy = self.policy.evaluate_actions(obs, acts)

        prob_true_act = th.exp(log_prob).mean()
        log_prob = log_prob.mean()
        # entropy = entropy.mean()

        # l2_norms = [th.sum(th.square(w)) for w in self.policy.parameters()]
        # l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square

        # ent_loss = -self.ent_weight * entropy
        neglogp = -log_prob
        # l2_loss = self.l2_weight * l2_norm
        # loss = neglogp + ent_loss
        # loss = neglogp + ent_loss + l2_loss
        loss = neglogp

        stats_dict = dict(
            neglogp=neglogp.item(),
            loss=loss.item(),
            # entropy=entropy.item(),
            # ent_loss=ent_loss.item(),
            prob_true_act=prob_true_act.item(),
            # l2_norm=l2_norm.item(),
            # l2_loss=l2_loss.item(),
        )

        return loss, stats_dict

    # def get_test_statistics(self, test_buffer: ReplayBuffer):
    #     with th.no_grad():
    #         obs = th.as_tensor(test_buffer.observations, device=self.device).detach().squeeze(1)
    #         acts = th.as_tensor(test_buffer.actions, device=self.device).detach().squeeze(1)

    #         # print(acts.shape)
    #         _, log_prob, entropy = self.policy.evaluate_actions(obs, acts)

    #         prob_test_act = th.exp(log_prob).mean()
    #         log_prob = log_prob.mean()
    #         neglogp = -log_prob

    #         return prob_test_act, neglogp

    def train(
        self,
        *,
        n_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        video_step,
        # reset_tensorboard: bool = False,
        
    ):
        """Train with supervised learning for some number of epochs.

        Here an 'epoch' is just a complete pass through the expert data loader,
        as set by `self.set_expert_data_loader()`.

        Args:
            n_epochs: Number of complete passes made through expert data before ending
                training. Provide exactly one of `n_epochs` and `n_batches`.
            n_batches: Number of batches loaded from dataset before ending training.
                Provide exactly one of `n_epochs` and `n_batches`.
            on_epoch_end: Optional callback with no parameters to run at the end of each
                epoch.
            on_batch_end: Optional callback with no parameters to run at the end of each
                batch.
            log_interval: Log stats after every log_interval batches.
            log_rollouts_venv: If not None, then this VecEnv (whose observation and
                actions spaces must match `self.observation_space` and
                `self.action_space`) is used to generate rollout stats, including
                average return and average episode length. If None, then no rollouts
                are generated.
            log_rollouts_n_episodes: Number of rollouts to generate when calculating
                rollout stats. Non-positive number disables rollouts.
            progress_bar: If True, then show a progress bar during training.
            reset_tensorboard: If True, then start plotting to Tensorboard from x=0
                even if `.train()` logged to Tensorboard previously. Has no practical
                effect if `.train()` is being called for the first time.
        """

        # if reset_tensorboard:
        #     self.tensorboard_step = 0
        max_mean_reward = -np.inf
        # train_buffer, test_buffer = split_buffer(self.demonstrations, 0.8)
        # train_buffer = Buffer(
        # buffer_size=buffer_size,
        # state_shape=env.blue_partial_observation_space.shape,
        # action_shape=env.blue_action_space_shape,
        # device=device
        # )
        buffer_exp = self.demonstrations
        for epoch_num in tqdm(range(1, n_epochs+1)):
            number_batches = buffer_exp.states.shape[0] // batch_size
            batch_loss = 0
            for _ in range(number_batches):
                observations, actions, _, _, _ = buffer_exp.sample(batch_size)
                # replay_buffer_samples = buffer_exp.sample(batch_size)
                # observations = replay_buffer_samples.observations
                # actions = replay_buffer_samples.actions

                loss, stats_dict_loss = self._calculate_loss(observations, actions)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss += stats_dict_loss['neglogp']

            self.writer.add_scalar('loss/neglogp', batch_loss/number_batches, epoch_num)
            
            if epoch_num % 20 == 0:
                # try:
                # mean_reward = evaluate_mean_reward(self.env, self.policy.predict, 10, self.goal)
                # prob_test, logp_test = self.get_test_statistics(test_buffer)

                self.writer.add_scalar('prob/true_act', stats_dict_loss["prob_true_act"], epoch_num)
                # self.writer.add_scalar('prob/test_act', prob_test, epoch_num)
                # self.writer.add_scalar('mean_reward', mean_reward, epoch_num)
                self.save_policy(os.path.join(self.log_dir, "policy_epoch_" + str(epoch_num) + ".pth"))
                # if mean_reward >= max_mean_reward:
                #     self.save_policy(os.path.join(self.log_dir, "bc_best.pth"))
                #     print(f'New {mean_reward} is greater than previous {max_mean_reward}')
                # max_mean_reward = max(max_mean_reward, mean_reward)
                # except:
                #     print('Error in evaluation')

            # produce a video of the current policy
            if epoch_num % video_step == 0:
                video_path = os.path.join(self.log_dir, "video_" + str(epoch_num) + ".mp4")
                # generate_policy_heatmap_video(self.env, policy=self.policy, num_timesteps=1200, path=video_path) #generate video with normal spawn mode
                generate_demo_video(env=self.env, blue_policy=self.policy, path=video_path)

            self.tensorboard_step += 1

    def save_policy(self, policy_path) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            policy_path: path to save policy to.
        """
        th.save(self.policy, policy_path)

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=512) # original 1024
    p.add_argument('--epochs', type=int, default=2500) 
    p.add_argument('--video_step', type=int, default=100) 
    p.add_argument('--num_steps', type=int, default=30000000) # original 10**7
    p.add_argument('--eval_interval', type=int, default=50000)

    p.add_argument('--lr_actor', type=float, default=1e-5) # original 5r-5
    p.add_argument('--lr_critic', type=float, default=5e-5)
    p.add_argument('--lr_disc', type=float, default=1e-5)
    p.add_argument('--units_actor', type=int, default=64)
    p.add_argument('--units_critic', type=int, default=64)
    p.add_argument('--units_disc', type=int, default=64)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    

    p.add_argument('--env_id', type=str, default='blue_gail')
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main(args)
    # observation_low =np.array([0, 0, 0], np.float)
    # observation_high = np.array([1, 1, 1], dtype=np.float)
    # observation_space = spaces.Box(observation_low, observation_high)

    # action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]))
    # buffer = ReplayBuffer(
    #     buffer_size=10,
    #     observation_space=observation_space,
    #     action_space=action_space,
    #     handle_timeout_termination=False
    # )
    # for i in range(10):
    #     state = np.array([[i, i, i]])/10.
    #     next_state = np.array([[i, i, i]])/10.
    #     action = np.array([[i, i]])/10.
    #     reward = 0
    #     done = True
    #     infos = None
    #     buffer.add(state, next_state, action, reward, done, infos)


    # train, test = split_buffer(buffer, 0.8)
    # # print("buffer observations")
    # # print(buffer.observations)
    # # print("train observations")
    # # print(train.observations, train.actions)
    # # print("test observations")
    # # print(test.observations)
    
