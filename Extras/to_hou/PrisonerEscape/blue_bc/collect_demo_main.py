import os
import argparse
import numpy as np
import random
import torch
from pathlib import Path
from collect_demo_utils import collect_demo
import sys
project_path = os.getcwd()
sys.path.append(str(project_path))
from simulator import PrisonerBothEnv, PrisonerBlueEnv
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
from heuristic import BlueHeuristic, SimplifiedBlueHeuristicPara
from simulator.load_environment import load_environment

def run(args):

    print(f"Loaded environment with seed {args.seed}")

    env = load_environment('simulator/configs/blue_bc.yaml')
    
    # set seeds
    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    epsilon = 0.1
    red_policy = HeuristicPolicy(env, epsilon=epsilon)
    # red_policy = RRTStarAdversarialAvoid(env, terrain_cost_coef=2000, n_iter=2000, max_speed=7.5)
    """Environment for blue team is constructed here"""
    env = PrisonerBlueEnv(env, red_policy)
    """Blue heuristic is used to generate expert trajectory"""
    blue_heuristic = SimplifiedBlueHeuristicPara(env, debug=False)


    buffer, buffer_subpolicy = collect_demo(
        env=env,
        blue_heuristic=blue_heuristic,
        red_policy=red_policy,
        buffer_size=args.buffer_size,
        subpolicy_buffer_size = args.subpolicy_buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed
    )
    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        f'size{args.buffer_size}.pth'
    ))
    buffer_subpolicy.save(os.path.join(
        'subpolicy_buffers',
        args.env_id,
        f'size{args.subpolicy_buffer_size}.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='BlueTeamHierPara')
    p.add_argument('--buffer_size', type=int, default=1000000)
    p.add_argument('--subpolicy_buffer_size', type=int, default=50000)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=2)
    args = p.parse_args()
    run(args)
