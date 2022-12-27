import os
import argparse
import numpy as np
import random
import torch

import sys
sys.path.append("/home/wu/GatechResearch/Zixuan/PrisonerEscape/gail")

from gail.gail_airl_ppo.env import make_env
from gail.gail_airl_ppo.algo import SACExpert
from gail.gail_airl_ppo.utils import collect_demo
from simulator import PrisonerBothEnv, PrisonerBlueEnv
from fugitive_policies.heuristic import HeuristicPolicy
from blue_policies.heuristic import BlueHeuristic, SimplifiedBlueHeuristic


def run(args):
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
    # blue_heuristic.reset()
    # blue_heuristic.init_behavior()


    buffer = collect_demo(
        env=env,
        blue_heuristic=blue_heuristic,
        red_policy=red_policy,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )
    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        f'size{args.buffer_size}_std{args.std}_prand{args.p_rand}.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--weight', type=str, required=True)
    p.add_argument('--env_id', type=str, default='Prisoner')
    p.add_argument('--buffer_size', type=int, default=3000)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
