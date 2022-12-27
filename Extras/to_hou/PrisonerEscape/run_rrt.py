import gym
import numpy as np

from utils import save_video
from time import sleep
from simulator.terrain import Terrain, TerrainType
from fugitive_policies.rrt_star_adversarial_heuristic import RRTStarAdversarial
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid

from simulator import PrisonerEnv, PrisonerBothEnv, PrisonerRedEnv
from fugitive_policies.heuristic import HeuristicPolicy
from red_bc.heuristic import BlueHeuristic, SimplifiedBlueHeuristic
import random

from simulator.load_environment import load_environment

def run_environment():
    
    # blue_policy = SimplifiedBlueHeuristic(env, debug=False)

    env = load_environment('simulator/configs/fixed_cams_random_uniform_start_camera_net.yaml')
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    blue_policy = SimplifiedBlueHeuristic(env, debug=False)
    env = PrisonerRedEnv(env, blue_policy)
    
    # env = PrisonerEnv(env, blue_policy)
    # print(env.observation_space.shape)
    # heuristic_policy = RRTStarAdversarial(env, n_iter=2500, max_speed=15)
    heuristic_policy = RRTStarAdversarialAvoid(env, terrain_cost_coef=2000, n_iter=2000, max_speed=7.5)

    full_red_observation, partial_red_observation = env.reset()

    done = False

    imgs = []
    while not done:
        action = heuristic_policy.predict(full_red_observation)
        full_red_observation, reward, done, info, partial_red_observation = env.step(action[0])
        img = env.render('policy', show=True, fast=True)
        imgs.append(img)
        if done:
            print("Finished")
            save_video(imgs, "figures/video.mp4")
            break

def generate_heatmap():
    np.random.seed(4)
    env = PrisonerEnv()
    policy = HeuristicPolicy(env)
    env.generate_policy_heatmap(env.reset(), policy=policy.get_action, num_timesteps=500, end=True)

if __name__ == "__main__":
    seed = 888
    # seed = 45

    # for seed in range(116,):

    for seed in [888]:
        print(f"Running seed: {seed}")
        run_environment()