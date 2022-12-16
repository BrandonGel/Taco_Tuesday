import sys
import argparse
import multiprocessing
from multiprocessing import Pool
import time
import gym
import gym_cap
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import *

# modules needed to generate policies
import gym_cap.heuristic as policy

description = "Evaluate two different policy."
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--episode', type=int, help='number of episodes to run', default=100)
# parser.add_argument('--blue_policy', type=str, help='blue policy', default='AStar') #AStar policy for Blue
# parser.add_argument('--blue_policy', type=str, help='blue policy', default='Defense') #Defense policy for Blue
# parser.add_argument('--blue_policy', type=str, help='blue policy', default='Patrol') #Patrol policy for Blue
parser.add_argument('--blue_policy', type=str, help='blue policy', default='Defense') #Patrol policy for Blue
parser.add_argument('--red_policy', type=str, help='red policy', default='Invariant')
parser.add_argument('--config_path', type=str, help='configuration path', default='base_settings.ini')
parser.add_argument('--map_size', type=int, help='size of the board', default=20)
parser.add_argument('--time_step', type=int, help='maximum time step', default=100)
parser.add_argument('--fair_map', help='run on fair map', action='store_true')
parser.add_argument('--cores', type=int, help='number of cores (-1 to use all)', default=1)
args = parser.parse_args()


fair_maps = ['test_maps/board{}.txt'.format(i) for i in range(1,2)]

# initialize the environment

# TODO: Primi
def _roll(n):
    num_episode = int(args.episode)//cores
    blue_policy = getattr(policy, args.blue_policy)()
    red_policy = getattr(policy, args.red_policy)()
    env = gym.make(
            "cap-v0",
            map_size = args.map_size,
            policy_blue = blue_policy,
            policy_red = red_policy,
            config_path=args.config_path
        )
    stat_win = np.array([0, 0, 0])
    stat_flag = np.array([0, 0, 0]) # Win mode
    stat_eliminated = np.array([0, 0, 0]) # Win mode
    ave_time = []
    ave_step = []

    for iterate in trange(num_episode, ncols=50, position=n):
    #for iterate in range(num_episode):
        if args.fair_map:
            env.reset(custom_board=random.choice(fair_maps))
        else:
            env.reset()

        iter_time = time.time()
        for steps in range(int(args.time_step)):
            # feedback from environment
            action = env.action_space.sample()
            observation, reward, game_finish, info  = env.step(action)

            if game_finish:
                break

        ave_time.append(time.time() - iter_time)
        ave_step.append(steps)

        stat_win += np.array([
                env.blue_win,
                not (env.blue_win or env.red_win),
                env.red_win
            ])
        stat_flag += np.array([
                env.red_flag_captured,
                not (env.blue_flag_captured or env.red_flag_captured),
                env.blue_flag_captured
            ])
        stat_eliminated += np.array([
                env.red_eliminated,
                not (env.blue_eliminated or env.red_eliminated),
                env.blue_eliminated
            ])
    env.close()

    return stat_win, stat_flag, stat_eliminated, ave_time, ave_step

stat_win = np.array([0, 0, 0])
stat_flag = np.array([0, 0, 0]) # Win mode
stat_eliminated = np.array([0, 0, 0]) # Win mode
ave_time = []
ave_step = []

if args.cores == -1:
    cores = multiprocessing.cpu_count()
else:
    cores = args.cores

if cores > 1:
    L = list(range(cores))
    with Pool(processes=cores) as p:
        for i, result in enumerate(p.imap_unordered(_roll, L)):
            stat_win += result[0]
            stat_flag += result[1]
            stat_eliminated += result[2]
            ave_time.extend(result[3])
            ave_step.extend(result[4])
else:
    result = _roll(0)
    stat_win += result[0]
    stat_flag += result[1]
    stat_eliminated += result[2]
    ave_time.extend(result[3])
    ave_step.extend(result[4])

stat_win = np.stack([stat_win, 100*stat_win/sum(stat_win)]).flatten('F')
stat_flag = np.stack([stat_flag, 100*stat_flag/sum(stat_flag)]).flatten('F')
stat_eliminated = np.stack([stat_eliminated, 100*stat_eliminated/sum(stat_eliminated)]).flatten('F')
print('\n'*(cores-1))
print("--------------------------------------- Statistics ---------------------------------------")
print("TEAM 1 : {}, TEAM 2 : {}".format(args.blue_policy, args.red_policy))
str_format = "{:<12}     | BLUE : {:<6.0f}({:<4.1f}%) | DRAW : {:<6.0f}({:<4.1f}%) | RED : {:<6.0f}({:<4.1f}%) |"
print(str_format.format('OVERALL', *stat_win))
print(str_format.format('WIN BY FLAG', *stat_flag))
print(str_format.format('WIN BY KILL', *stat_eliminated))
print("Average Run Time : {:.5f} ± {:.5f} sec".format(np.mean(ave_time), np.std(ave_time)))
print("Average Step     : {:.5f} ± {:.5f} sec".format(np.mean(ave_step), np.std(ave_step)))
