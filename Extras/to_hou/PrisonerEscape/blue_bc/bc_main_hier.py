import os
import gym
import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import argparse
from tqdm import tqdm

from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Type, Union
from stable_baselines3.common import policies, utils, vec_env
from video_generator import generate_demo_video
from policy import StateIndependentPolicy, LSTMMDNBluePolicy, HighLevelPolicy

import sys
project_path = os.getcwd()
sys.path.append(str(project_path))
# from simulator.tl_coverage.autoencoder import train
from buffer import Buffer, SerializedBuffer
from stable_baselines3.common import policies
from simulator import PrisonerBothEnv, PrisonerBlueEnv, BlueSequenceEnv
from simulator.load_environment import load_environment
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
from heuristic import BlueHeuristic, SimplifiedBlueHeuristicPara
from bc_utils import sample_sequence_from_buffer, idx_to_onehot, para_agents



def main(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    
    print(f"Loaded environment with seed {args.seed}")
    env = load_environment('simulator/configs/blue_bc.yaml')

    env.seed(args.seed)
    epsilon = 0.1
    red_policy = HeuristicPolicy(env, epsilon=epsilon)
    # red_policy = RRTStarAdversarialAvoid(env, terrain_cost_coef=2000, n_iter=2000, max_speed=7.5)
    env = PrisonerBlueEnv(env, red_policy)
    env = BlueSequenceEnv(env, sequence_len=args.seq_len)

    blue_heuristic = SimplifiedBlueHeuristicPara(env, debug=False)
    blue_partial_observation = env.reset()
    blue_heuristic.reset()
    blue_heuristic.init_behavior()

    buffer_exp = SerializedBuffer(path=args.buffer, device=device)
    units_actor = (args.units_actor, args.units_actor, args.units_actor)
    """MLP"""
    hier_policy = HighLevelPolicy(
        # state_shape=env.blue_partial_observation_space.shape,
        state_shape=env.blue_observation_space.shape,
        agent_num = env.num_helicopters + env.num_search_parties,
        subpolicy_shape=(env.subpolicy_num, ),
        para_shape=(env.max_para_num,),
        hidden_units=units_actor,
        hidden_activation=nn.ReLU() # original: nn.Tanh()
        ).to(device)
    """LSTM"""
    # hidden_dims = (args.units_actor, args.units_actor // 2)
    # bc_policy = LSTMMDNBluePolicy(num_features=env.blue_observation_space.shape[0], num_actions=env.blue_action_space_shape[0], hidden_dims=hidden_dims, mdn_num=args.mdn_num, 
    #                               hidden_act_func=nn.ReLU(), output_func=nn.Tanh(), device=device).to(device)
    train_bc(args=args, env=env, replay_buffer=buffer_exp, policy=hier_policy, epochs=5, video_step=128, save_folder=None)


def train_bc(args, env, replay_buffer, policy, epochs, video_step, save_folder=None):
    """Train a stable baselines 3 policy on a stable baselines 3 replay buffer."""
    
    time = datetime.now().strftime("%Y%m%d-%H%M")
    if save_folder:
        log_dir = os.path.join(save_folder, str(time))
    else:
        log_dir = os.path.join("logs", "bc", str(time))

    summary_dir = os.path.join(log_dir, 'summary')
    writer = SummaryWriter(log_dir=summary_dir)

    bc_trainer = BC(
        args=args,
        observation_space=env.blue_observation_space,
        action_space=env.action_space,
        demonstrations=replay_buffer,
        device='cuda' if args.cuda else 'cpu',
        writer = writer,
        log_dir = log_dir,
        policy=policy,
        env = env, 
    )
    bc_trainer.train(n_epochs=args.epochs, batch_size=args.batch_size, seq_len=args.seq_len, observation_space=env.blue_observation_space, max_timesteps=env.max_timesteps, video_step = args.video_step)

    if save_folder is not None:
        save_path = os.path.join(save_folder, "bc.pth")
    else:
        save_path = os.path.join(log_dir, "bc.pth")
        
    bc_trainer.save_policy(save_path)
    return bc_trainer.policy

def reconstruct_policy(
    policy_path: str,
    device: Union[torch.device, str] = "auto",
) -> policies.BasePolicy:
    """Reconstruct a saved policy.

    Args:
        policy_path: path where `.save_policy()` has been run.
        device: device on which to load the policy.

    Returns:
        policy: policy with reloaded weights.
    """
    policy = torch.load(policy_path, map_location=utils.get_device(device))
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
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, torch.device] = "auto",
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

        self.policy = policy

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr_actor)
        self.ent_weight = ent_weight
        self.l2_weight = l2_weight
        self.device = 'cuda' if args.cuda else 'cpu'


    def _calculate_loss(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        acts: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, Mapping[str, float]]:
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
        obs = torch.as_tensor(obs, device=self.device).detach()

        acts_gt = torch.as_tensor(acts, device=self.device).detach()
        subpolicy, paras = self.policy(obs)

        option_num = subpolicy.shape[2]
        agent_num = subpolicy.shape[1]
        subpolicy_gt_onehot = torch.as_tensor(idx_to_onehot(acts_gt[:,0], option_num).unsqueeze(1).expand(-1,agent_num,-1), device=self.device)
        paras_gt = acts_gt[:,1:]
        paras_gt = para_agents(paras_gt, paras)

        # loss_func = torch.nn.CrossEntropyLoss()
        loss_func = torch.nn.MSELoss()
        loss = loss_func(subpolicy, subpolicy_gt_onehot) + loss_func(paras, paras_gt)
        # print("subpolicy[0,:,:]= ", subpolicy[0,:,:])
        # print("paras[0,:,:] = ", paras[0,:,:])
        # print("paras_gt[0,:,:] = ", paras_gt[0,:,:])
        

        # log_prob = self.policy.evaluate_log_pi(obs, acts)
        # log_prob = log_prob.mean()
        # neglogp = -log_prob
        # loss = neglogp

        stats_dict = dict(
            loss=loss.item()
        )

        return loss, stats_dict

    def train(
        self,
        *,
        n_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len,
        observation_space,
        max_timesteps,
        video_step,
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

        max_mean_reward = -np.inf

        buffer_exp = self.demonstrations
        for epoch_num in tqdm(range(1, n_epochs+1)):
            # iters = 100
            iters = buffer_exp.states.shape[0] // (batch_size)
            batch_loss = 0
            mu_average = 0
            log_sigma_average = 0
            for _ in range(iters):
                """MLP Blue Centralized Training"""
                observations, actions, _, _, _ = buffer_exp.sample(batch_size)
                loss, stats_dict_loss = self._calculate_loss(observations, actions)
                """LSTM Blue Centralized Training"""
                # observations, actions = sample_sequence_from_buffer(buffer_exp, batch_size, seq_len, observation_space, max_timesteps)
                # loss, stats_dict_loss = self.policy.evaluate_loss(observations.detach(), actions.detach())


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss += stats_dict_loss['loss']
                # mu_average += stats_dict_loss['mu_average']
                # log_sigma_average += stats_dict_loss['log_sigma_average']

            # self.writer.add_scalar('loss/neglogp', batch_loss/number_batches, epoch_num)
            self.writer.add_scalar('loss/loss', batch_loss/iters, epoch_num)
            # self.writer.add_scalar('loss/mu_average', mu_average/iters, epoch_num)
            # self.writer.add_scalar('loss/log_std_average', log_sigma_average/iters, epoch_num)
            
            if epoch_num % 100 == 0:

                self.save_policy(os.path.join(self.log_dir, "policy_epoch_" + str(epoch_num) + ".pth"))


            # produce a video of the current policy
            if epoch_num % video_step == 0:
                video_path = os.path.join(self.log_dir, "video_" + str(epoch_num) + ".mp4")
                # generate_policy_heatmap_video(self.env, policy=self.policy, num_timesteps=1200, path=video_path) #generate video with normal spawn mode
                # generate_demo_video(env=self.env, blue_policy=self.policy, path=video_path, device=self.device)

            self.tensorboard_step += 1

    def save_policy(self, policy_path) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            policy_path: path to save policy to.
        """
        torch.save(self.policy, policy_path)

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=32) # original 32
    p.add_argument('--epochs', type=int, default=100) 
    p.add_argument('--video_step', type=int, default=100) 

    p.add_argument('--mdn_num', type=int, default=1)
    p.add_argument('--seq_len', type=int, default=50)
    p.add_argument('--lr_actor', type=float, default=1e-4) # original 5r-5
    p.add_argument('--units_actor', type=int, default=128)
    
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main(args)

    
