import argparse
import torch
import time
import imageio
import json
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import seaborn as sns
from pathlib import Path
from torch.autograd import Variable
from tensorboard.backend.event_processing import event_accumulator
from tabulate import tabulate


def run(config):
    summary_root = Path('./logs').absolute()
    summary_dir = summary_root / config.proj_name / config.rl_type / config.timed_folder/ 'summary'
    for file_name in os.listdir(summary_dir.absolute()):
        if file_name.startswith("events"):
            binary_path = (summary_dir.absolute() / file_name)
    result_analysis(config, binary_path)
    return

def result_analysis(config, binary_path):

    # Calculate average reward, loss
    fig_save_path = str(binary_path.parent)
    binary_path = str(binary_path)
    ea=event_accumulator.EventAccumulator(binary_path, size_guidance={'scalars': 0})
    ea.Reload()
    print(ea.scalars.Keys())

 
    weight_smooth = 0.999
    disc_loss_event = ea.scalars.Items('loss/disc')
    critic_loss_event = ea.scalars.Items('loss/critic')
    actor_loss_event = ea.scalars.Items('loss/actor')
    acc_exp_event = ea.scalars.Items('stats/acc_exp')
    acc_pi_event = ea.scalars.Items('stats/acc_pi')

    disc_loss = np.array([event.value for event in disc_loss_event])
    critic_loss = np.array([event.value for event in critic_loss_event])
    actor_loss = np.array([event.value for event in actor_loss_event])
    acc_exp = np.array([event.value for event in acc_exp_event])
    acc_pi = np.array([event.value for event in acc_pi_event])

    disc_loss = smooth(disc_loss, weight_smooth)
    critic_loss = smooth(critic_loss, weight_smooth)
    actor_loss = smooth(actor_loss, weight_smooth)
    acc_exp = smooth(acc_exp, weight_smooth)
    acc_pi = smooth(acc_pi, weight_smooth)

    disc_loss_axis = np.array(range(len(disc_loss)))
    critic_loss_axis = np.array(range(len(critic_loss)))
    actor_loss_axis = np.array(range(len(actor_loss)))
    acc_exp_axis = np.array(range(len(acc_exp)))
    acc_pi_axis = np.array(range(len(acc_pi)))


    plt.figure()
    plt.plot(disc_loss_axis, disc_loss, 'b-', label='disc_loss')
    # plt.fill_between(episode_axis0, agent0_reward_mean - agent0_reward_std, agent0_reward_mean + agent0_reward_std, color='b', alpha=0.2)
    plt.title("disc_loss v.s. step")
    plt.xlabel("Step")
    plt.ylabel("Disc_loss")
    plt.savefig(fig_save_path + "/disc_loss.png")

    plt.figure()
    plt.plot(critic_loss_axis, critic_loss, 'b-', label='critic_loss')
    # plt.fill_between(episode_axis0, agent0_reward_mean - agent0_reward_std, agent0_reward_mean + agent0_reward_std, color='b', alpha=0.2)
    plt.title("critic_loss v.s. step")
    plt.xlabel("Step")
    plt.ylabel("Critic_loss")
    plt.savefig(fig_save_path + "/critic_loss.png")

    plt.figure()
    plt.plot(actor_loss_axis, actor_loss, 'b-', label='actor_loss')
    # plt.fill_between(episode_axis0, agent0_reward_mean - agent0_reward_std, agent0_reward_mean + agent0_reward_std, color='b', alpha=0.2)
    plt.title("actor_loss v.s. step")
    plt.xlabel("Step")
    plt.ylabel("Actor_loss")
    plt.savefig(fig_save_path + "/actor_loss.png")

    plt.figure()
    plt.plot(acc_exp_axis, acc_exp, 'b-', label='acc_exp')
    # plt.fill_between(episode_axis0, agent0_reward_mean - agent0_reward_std, agent0_reward_mean + agent0_reward_std, color='b', alpha=0.2)
    plt.title("acc_exp v.s. step")
    plt.xlabel("Step")
    plt.ylabel("acc_exp")
    plt.savefig(fig_save_path + "/acc_exp.png")

    plt.figure()
    plt.plot(acc_pi_axis, acc_pi, 'b-', label='acc_pi')
    # plt.fill_between(episode_axis0, agent0_reward_mean - agent0_reward_std, agent0_reward_mean + agent0_reward_std, color='b', alpha=0.2)
    plt.title("acc_pi v.s. step")
    plt.xlabel("Step")
    plt.ylabel("acc_pi")
    plt.savefig(fig_save_path + "/acc_pi.png")
    plt.show()

    return 

# Smooth the reward data
def smooth(scalar, weight=0.75):
    if weight == 0:
        smoothed = scalar
        return smoothed
    last = scalar[0]
    smoothed = []
    for ind, point in enumerate(scalar):
        if ind > 0:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
    return smoothed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("--proj_name", default="blue_gail", type=str, help="blue or red team")
    parser.add_argument("--rl_type", default="gail", type=str, help="gial or other rl methods")
    parser.add_argument("--timed_folder", help="Name of log folder stamped by time")

    config = parser.parse_args()

    run(config)
