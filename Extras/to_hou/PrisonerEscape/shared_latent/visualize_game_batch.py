from simulator.prisoner_env_variations import initialize_prisoner_environment
import numpy as np
import torch
from simulator.prisoner_batch_wrapper import PrisonerBatchEnv
from fugitive_policies.heuristic import HeuristicPolicy
from heatmap import generate_heatmap_img
from utils import plot_gaussian_heatmap, save_video
import os
from shared_latent.models.dreamer import Dreamer
        
def render_and_collect(env, policy, model, seq_len, obs_type, device, path, seed, render=False):
    show = True
    imgs = []

    episode_return = 0.0
    done = False

    # Initialize empty observations
    observation_space = env.blue_observation_space
    observation = env.reset()

    # Convert from numpy to torch tensor
    print(observation.shape)
    in_state = model.rssm.cell.init_state(1)

    i = 0
    timestep = 0
    while not done:
        i += 1
        fugitive_observation = env.get_fugitive_observation()
        action = policy.predict(fugitive_observation)
        observation, reward, done, _ = env.step(action[0])
        episode_return += reward

        observation = torch.from_numpy(observation).to(device).view(1, 8, -1)
        obs = observation.permute(1, 0, 2)
        r = torch.full((8, 1), False).to(device)

        in_state = model.warm_start_rssm(obs, r, None, 1)
        reset = torch.full((10, 1), False).to(device)
        mean, log_std = model.rollout_prior(in_state, reset, 1, 10)
        
        # convert torch to numpy
        mean = mean.detach().cpu().numpy()
        log_std = log_std.detach().cpu().numpy()
        # mean = decoder_output.squeeze()
        # true_location = np.array(env.prisoner.location)
        # print(mean[0]*2428, true_location)

    
        if render:
            grid = plot_gaussian_heatmap(mean[0], log_std[0], res=10)
            heatmap_img = generate_heatmap_img(grid, true_location=env.get_prisoner_location())
            imgs.append(heatmap_img)
            # print(heatmap_img.shape)
        #     game_img = env.render('Policy', show=False, fast=True)
        #     img = combine_game_heatmap(game_img, heatmap_img)
        #     imgs.append(img)
        # timestep += 1
        if done:
            break

    if render:
        save_video(imgs, os.path.dirname(path) + f"/heatmap_{seed}.mp4", fps=5)
    #     # save_video(imgs, f'temp/mog/test_{seed}.mp4', fps=5)


def run_single_seed(seed, path, render=False):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(os.path.dirname(path), 'config.yaml')
    import yaml
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    seq_len = config['seq_len']
    obs_type = config['obs_type']

    if obs_type == "red":
        observation_step_type = "Prediction"
    else:
        observation_step_type = "Blue"

    env = initialize_prisoner_environment(variation=0,
                                        observation_step_type=observation_step_type,
                                        seed=seed)

    env = PrisonerBatchEnv(env, batch_size=8)
    policy = HeuristicPolicy(env)

    model = Dreamer(config)
    model.load_state_dict(torch.load(os.path.join(path, '32.pth')))
    # model.rssm.load_state_dict(torch.load(os.path.join(os.path.dirname(path), 'rssm_best.pth')))
    # model.decoder.load_state_dict(torch.load(os.path.join(os.path.dirname(path), 'decoder_best.pth')))
    # model_path = os.path.join(os.path.dirname(path), 'model_ckpt_2.pth')
    # model.load_state_dict(torch.load(model_path))
    model.eval()
    stats = render_and_collect(env, policy, model, seq_len, obs_type, device, path, seed, render=render)
 
if __name__ == "__main__":
    # path = "/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220501-0059/summary/"
    # path = 'logs/shared_latent/red/20220501-0419/summary/'
    path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220501-1425/summary/'
    seed = 0
    run_single_seed(seed, path, render=True)