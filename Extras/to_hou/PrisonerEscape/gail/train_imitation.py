from distutils.command.config import config
import os
import argparse
from datetime import datetime
import torch
import numpy as np
import random
import copy
import sys
sys.path.append("/home/wu/GatechResearch/Zixuan/PrisonerEscape/")

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer

from simulator import PrisonerBothEnv, PrisonerBlueEnv
from fugitive_policies.heuristic import HeuristicPolicy
from blue_policies.heuristic import BlueHeuristic, SimplifiedBlueHeuristic


def run(args):
    """Revision Begins Here"""
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
    fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    env = PrisonerBlueEnv(env, fugitive_policy)
    
    env_test = copy.deepcopy(env)
    buffer_exp = SerializedBuffer(path=args.buffer, device=torch.device("cuda" if args.cuda else "cpu"))
    units_actor = (args.units_actor, args.units_actor, args.units_actor)
    units_critic = (args.units_critic, args.units_critic)
    units_disc = (args.units_disc, args.units_disc)
    algo = ALGOS[args.algo](buffer_exp=buffer_exp, 
        state_shape=env.blue_partial_observation_space.shape,
        action_shape=env.blue_action_space_shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
        lr_actor = args.lr_actor,
        lr_critic = args.lr_critic,
        lr_disc = args.lr_disc,
        units_disc=units_disc,
        units_actor=units_actor,
        units_critic=units_critic,
        max_grad_norm=args.max_grad_norm
    )
    """Revision Ends Here"""
    # env = make_env(args.env_id)
    # env_test = make_env(args.env_id)
    # buffer_exp = SerializedBuffer(
    #     path=args.buffer,
    #     device=torch.device("cuda" if args.cuda else "cpu")
    # )

    # algo = ALGOS[args.algo](
    #     buffer_exp=buffer_exp,
    #     state_shape=env.observation_space.shape,
    #     action_shape=env.action_space.shape,
    #     device=torch.device("cuda" if args.cuda else "cpu"),
    #     seed=args.seed,
    #     rollout_length=args.rollout_length
    # )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}')

    blue_heuristic = SimplifiedBlueHeuristic(env, debug=False)
    # blue_heuristic.reset()
    # red_policy = HeuristicPolicy(env, epsilon=epsilon)

    trainer = Trainer(
        env=env,
        env_test=env_test,
        blue_heuristic=blue_heuristic,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        # video_save_path = args.buffer
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=512) # original 1024
    p.add_argument('--num_steps', type=int, default=30000000) # original 10**7
    p.add_argument('--eval_interval', type=int, default=50000)

    p.add_argument('--lr_actor', type=float, default=5e-5)
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
    run(args)
