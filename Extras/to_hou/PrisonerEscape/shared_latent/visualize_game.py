from simulator.prisoner_env_variations import initialize_prisoner_environment
import numpy as np
import torch
from simulator.prisoner_batch_wrapper import PrisonerBatchEnv
from fugitive_policies.heuristic import HeuristicPolicy
from heatmap import generate_heatmap_img
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

        observation = torch.from_numpy(observation).to(device).view(1, 1, -1)
        r = torch.full((1, 1), False).to(device)

        priors, posts, samples, features, states, in_state = model.rssm.forward(
                    observation,       # tensor(T, B, E)
                    r,       # tensor(T, B)
                    in_state,# Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)]
                    iwae_samples =1,#: int = 1,
                    do_open_loop=False)

        mean, std = model.rollout_prior((torch.clone(in_state[0]), torch.clone(in_state[1])), 1, 10)
        
        true_location = np.array(env.prisoner.location)
        print(mean[0]*2428, true_location)

    #     if render:
    #         # heatmap_img = module.render(output, true_location)
    #         grid = module.get_probability_grid(output, true_location)
    #         heatmap_img = generate_heatmap_img(grid, true_location=env.get_prisoner_location())
    #         game_img = env.render('Policy', show=False, fast=True)
    #         img = combine_game_heatmap(game_img, heatmap_img)
    #         imgs.append(img)
    #     timestep += 1
    #     if done:
    #         break

    # if render:
    #     save_video(imgs, os.path.dirname(path) + f"/heatmap_{seed}.mp4", fps=5)
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

    # env = PrisonerEnv(env, batch_size=config["seq_len"])
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
    # path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220501-0419'
    path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220501-1425/summary/'
    seed = 0
    run_single_seed(seed, path)