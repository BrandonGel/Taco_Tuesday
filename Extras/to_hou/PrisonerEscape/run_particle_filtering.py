from utils import evaluate_mean_reward
from simulator import PrisonerEnv
from fugitive_policies.heuristic import HeuristicPolicy
import os
import numpy as np
import cv2
from utils import save_video
from datetime import datetime
import random
from utils import save_video
from particle_filter.particle_filter import ParticleFilter
from particle_filter.evaluation_metric import get_ground_truth_likelihood_particle_filter, binary_prob_metric_particle_filter, rmse_particle_filter, kl_divergence_particle_filter, plot_metrics, generate_delta_plot, nll_particle_filter
from simulator.prisoner_env_variations import initialize_prisoner_environment
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from blue_policies.heuristic import BlueHeuristic
# plt.rcParams['text.usetex'] = True


def calculate_metrics(num_rollouts, env, heuristic_policy, debug=False, calculate_delta=False):
    """
    Calculate Filtering metrics (RMSE, KL, binary_prob_metric) across rollouts
    and report mean and std across multiple rollouts for a given filtering model
    :param num_rollouts:
    :param env:
    :return:
    """

    seeds = np.arange(100)
    all_metrics = defaultdict(dict)

    deltas = np.linspace(0, 0.5, 11)
    log_likelihoods = defaultdict(dict)  # For storing likelihood each time step (in each rollout) and each delta
    avg_likelihoods = np.zeros((num_rollouts, len(deltas)))  # For mean likelihood across rollout for each delta

    for k in range(num_rollouts):
        np.random.seed(seeds[k])
        random.seed(seeds[k])
        env.seed(seeds[k])
        particle_filter = ParticleFilter(env)
        blue_observation = env.reset()

        heuristic_policy.reset()
        heuristic_policy.init_behavior()
        done = False

        # Initialize vars to store values of metrics for each rollout
        binary_count = 0
        RMSE, KL, NLL = [], [], []

        for d in deltas:
            log_likelihoods[k][str(d)] = []

        timestep = 0
        detected_x, detected_y = [], []
        undetected_x, undetected_y = [], []
        t = []
        detected_timesteps = []  # Store the timesteps where the fugitive was detected in the rollout
        imgs = []
        while not done:
            # fugitive_observation = env.get_fugitive_observation()
            action = heuristic_policy.predict(blue_observation)
            blue_observation, reward, done, _ = env.step(action)

            # Pass the blue observation to the filtering model
            particle_filter.propagate()
            if env.is_detected:
                particle_filter.update_weight_according_to_detection(env.prisoner.location)
                particle_filter.normalize_all_weights()
            if debug:
                particle_filter.plot(save_name="logs/temp/debug_particle_filter_%d.png" % timestep, ground_truth_location=env.prisoner.location)
                game_img = env.render('Policy', show=True, fast=True)
                particle_img = cv2.imread("logs/temp/debug_particle_filter_%d.png" % timestep)
                if particle_img is not None:
                    max_x = max(game_img.shape[0], particle_img.shape[0])
                    max_y = max(game_img.shape[1], particle_img.shape[1])
                    game_img_reshaped = np.pad(game_img, ((0, max_x - game_img.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
                    particle_img_reshaped = np.pad(particle_img, ((0, max_x - particle_img.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)

                    img = np.concatenate((game_img_reshaped, particle_img_reshaped), axis=1)
                    imgs.append(img)
                else:
                    imgs.append(game_img)


            # demonstration_tensor = torch.from_numpy(blue_observation).to(device).float().unsqueeze(0)
            # prediction_params = model(demonstration_tensor).detach().cpu().numpy()
            # mean_location, log_std = prediction_params[:2], prediction_params[2:]

            true_location = np.array(env.prisoner.location)  # get prisoner's true location from the env

            # Compute all metrics
            RMSE.append(rmse_particle_filter(particle_filter.particles, true_location))
            KL.append(kl_divergence_particle_filter(particle_filter.particles, true_location=true_location))
            NLL.append(nll_particle_filter(particle_filter.particles, true_location=true_location))

            result = get_ground_truth_likelihood_particle_filter(particle_filter.particles, true_location, dist_threshold=0.05)
            binary_count += 1 if result >= 0.5 else 0
            log_likelihoods[k]["0.05"].append(result)
            t.append(timestep)
            timestep += 1

            # For plotting with respect to delta=0.05
            if env.is_detected:
                detected_timesteps.append(timestep)
                # timestep = 0
                detected_x.append(result)
                detected_y.append(timestep)
            else:
                undetected_x.append(result)
                undetected_y.append(timestep)

            if calculate_delta:
                for d in deltas:
                    if d == 0.05:
                        continue  # We've already computed the value for delta=0.05 above
                    result = get_ground_truth_likelihood_particle_filter(particle_filter.particles,
                                                                         true_location=true_location,
                                                                         dist_threshold=d)
                    log_likelihoods[k][str(d)].append(result)

        # ------------------------------------- Generate Likelihood plot --------------------------------------------- #
        # print(first_detection)
        # fig, ax = plt.subplots()
        # ax.scatter(undetected_y, undetected_x, c='coral', alpha=0.5, label='undetected')
        # ax.scatter(detected_y, detected_x, c='lightblue', alpha=0.5, label='detected')
        #
        # # For plot since last detection
        # # first_detection = detected_timesteps[0]
        # # ax.scatter(t[first_detection:], log_likelihoods[first_detection:], c='tab:blue', alpha=0.5, label='detected')

        # ax.legend()
        # plt.ylabel(r'$P(Likelihood \leq \delta$)')
        # plt.xlabel('time since episode start')
        # plt.title(r'$\delta = 0.05$')
        # plt.savefig('tmp/matt_plot_{}.png'.format(k))
        # ------------------------------------------------------------------------------------------------------------ #

        num_timesteps_in_rollout = len(RMSE)
        num_detections = len(detected_timesteps)

        print("Seed: {}".format(seeds[k]))
        print(f"NLL: mean: {np.mean(NLL)}, std: {np.std(NLL)}")
        print("MSE: mean:{}, std:{}".format(np.mean(RMSE), np.std(RMSE)))
        print("KL: mean:{}, std:{}".format(np.mean(KL), np.std(KL)))
        print("Binary Count: count:{}, total:{}".format(binary_count, num_timesteps_in_rollout))
        print("num detected timesteps: {}".format(num_detections))
        print("---------------------------------------------------------")

        # End of Episode... Calculate Aggregate metrics for filtering
        all_metrics[k]['KL'] = KL
        all_metrics[k]['RMSE'] = RMSE
        all_metrics[k]['binary_count_ratio'] = binary_count / num_timesteps_in_rollout

        # To plot on x-axis
        all_metrics[k]['detection_ratio'] = num_detections / num_timesteps_in_rollout
        all_metrics[k]['detected_timesteps_std'] = np.std(detected_timesteps)

        # Calculate time between detections
        time_between_detections = []
        for t in range(num_detections - 1):
            if t == 0:
                time_between_detections.append(detected_timesteps[t + 1] - 0)
            else:
                time_between_detections.append(detected_timesteps[t + 1] - detected_timesteps[t])

        time_between_detections.append(num_timesteps_in_rollout - detected_timesteps[-1])

        all_metrics[k]['time_between_detections'] = np.mean(time_between_detections)

        if debug:
            save_video(imgs, "logs/temp/particle_density_%d.mp4" % k, fps=10)
        with open("tmp/all_metrics.pkl", "wb") as f:
            pickle.dump(all_metrics, f)

        if calculate_delta:
            for idx, d in enumerate(deltas):
                avg_likelihoods[k, idx] = np.mean(log_likelihoods[k][str(d)])

    return all_metrics, avg_likelihoods


if __name__=='__main__':
    # Here: Testing the evaluation metric on output from a filtering model
    env = initialize_prisoner_environment(variation=0,
                                          epsilon=0.1,
                                          observation_step_type="Blue")
    # epsilon = 0.1
    # heuristic_policy = HeuristicPolicy(env)
    blue_heuristic = BlueHeuristic(env, debug=False)
    blue_heuristic.reset()
    blue_heuristic.init_behavior()


    env.reset()

    num_rollouts = 20
    all_metrics, avg_likelihoods = calculate_metrics(num_rollouts=num_rollouts, env=env,
                                                     heuristic_policy=blue_heuristic, debug=False, calculate_delta=True)
    # plot_metrics(all_metrics, metric='RMSE', num_rollouts=num_rollouts)
    # plot_metrics(all_metrics, metric='KL', num_rollouts=num_rollouts)
    # plot_metrics(all_metrics, metric='binary_count_ratio', num_rollouts=num_rollouts)
    # generate_delta_plot(avg_likelihoods, deltas=np.linspace(0, 0.5, 11))
