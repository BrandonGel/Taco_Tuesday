"""
This file includes metrics for evaluating the filtering module (outputs a Gaussian Distribution)
Current metrics implemented:
-- Binary value for predicting true fugitive's location (only for Gaussian Distribution)
"""
import sys, os

sys.path.append(os.getcwd())
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.distributions import Normal
# -------------------- Imports to test the metric ----------------------- #
# from filtering.models.blue_state_nll_adaptive_std_model import NLLModel
# from filtering.models.blue_state_filtering_sequence import NLLSeqModel
# from filtering.models.blue_state_gru_autoregressive_model import NLLSeq2Seq, Encoder, Decoder
# from simulator.prisoner_env import PrisonerEnv
# from simulator.prisoner_batch_wrapper import PrisonerBatchEnv
from fugitive_policies.heuristic import HeuristicPolicy
from scipy.stats import multivariate_normal
from simulator.utils import distance
import random
from collections import defaultdict
from simulator.prisoner_env_variations import initialize_prisoner_environment
# from filtering.models.particle_filter import ParticleFilter
import pickle
import cv2
from particle_filter.plot_fugitive_likelihood_from_gaussian import plot_gaussian_heatmap
from heatmap import generate_heatmap_img
from utils import save_video
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def get_ground_truth_likelihood_particle_filter(all_particles, true_location, dist_threshold=0.05, top_particles=500):
    dist_threshold = 2428 * dist_threshold
    # first get top weighted particles
    particle_with_weight = []
    for particle in all_particles:
        particle_with_weight.append((particle.weight, particle))
    particle_with_weight_sorted = sorted(particle_with_weight, key=lambda x: x[0], reverse=True)

    total_weights = 0
    closer_than_dist_threshold_weights = 0
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        if distance(true_location, particle.location) < dist_threshold:
            closer_than_dist_threshold_weights += weight
    return closer_than_dist_threshold_weights / total_weights


def binary_prob_metric_particle_filter(all_particles, true_location, dist_threshold=0.05, top_particles=500,
                                       threshold=0.5):
    true_prob = get_ground_truth_likelihood_particle_filter(all_particles, true_location, dist_threshold, top_particles)
    # print("threshold_prob", true_prob)
    if true_prob >= threshold:
        return 1
    else:
        return 0


def get_ground_truth_likelihood(mean, logstd, true_location, dist_threshold=0.05):
    """
    Given mean, and logstd predicted from the filtering module, calculate the likelihood
    of the fugitive's ground truth location from the predicted distribution
    :param mean: (np.array) Mean of the predicted distribution from the filtering module
    :param logstd: (np.array) Logstd of the predicted distribution from the filtering module
    :param true_location: (np.array) Ground Truth location of the fugitive (x,y)
    :return:
        prob: The probability of the fugitive being at the ground truth location
              as predicted by the filtering module's distribution
    """
    n_dims = 2  # Location: (x,y)
    var = np.exp(logstd) ** 2
    cov = np.eye(n_dims) * var

    # CDF calculates from -infty to the upper limit.
    # Therefore subtracting the lower limit to calculate the cdf between lower limit to upper limit instead of -infty to upper limit
    prob = multivariate_normal.cdf(true_location + dist_threshold, mean=mean, cov=cov) - multivariate_normal.cdf(
        true_location - dist_threshold, mean=mean, cov=cov)
    return prob


def binary_prob_metric(mean, logstd, true_location, threshold=0.5):
    """
    Provides a binary score based on the probability distribution output from the filtering model.
    returns 1 if P(true_location) >= threshold else returns 0

    :param mean: (np.array) Mean of the predicted distribution from the filtering module
    :param logstd: (np.array) Logstd of the predicted distribution from the filtering module
    :param true_location: (np.array) Ground truth location of the fugitive at the current timestep
    :param threshold: probability threshold
    :return:
        binary value based on the probability threshold
    """
    true_prob = get_ground_truth_likelihood(mean, logstd, true_location)
    # print(true_prob)
    if true_prob >= threshold:
        return 1
    else:
        return 0


def rmse_particle_filter(all_particles, true_location, top_particles=500):
    particle_with_weight = []
    for particle in all_particles:
        particle_with_weight.append((particle.weight, particle))
    particle_with_weight_sorted = sorted(particle_with_weight, key=lambda x: x[0], reverse=True)

    total_weights = 0
    weighted_average_location = np.array([0.0, 0.0])
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        weighted_average_location += particle.location * weight
    weighted_average_location /= total_weights
    # print("mean location", weighted_average_location)
    # print("true location", true_location)
    distance_between = distance(weighted_average_location, true_location)
    # print("distance", distance_between)
    return distance_between


def kl_divergence_particle_filter(all_particles, true_location, top_particles=500):
    particle_with_weight = []
    for particle in all_particles:
        particle_with_weight.append((particle.weight, particle))
    particle_with_weight_sorted = sorted(particle_with_weight, key=lambda x: x[0], reverse=True)

    total_weights = 0
    weighted_average_location = np.array([0.0, 0.0])
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        weighted_average_location += particle.location * weight
    weighted_average_location /= total_weights

    total_weights = 0
    weighted_average_location_std = np.array([0.0, 0.0])
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        weighted_average_location_std += (particle.location - weighted_average_location) ** 2 * weight
    weighted_average_location_std /= total_weights
    # # TODO: Manisha check this... My neural net typically outputs low logstd of -4.5 --> 0.011 std --> 0.011 * 2428 = 27
    weighted_average_location_std = np.maximum(np.sqrt(weighted_average_location_std), np.ones_like(weighted_average_location_std)*30)
    # weighted_average_location_std = np.sqrt(weighted_average_location_std) / 2428
    # print("mean location", weighted_average_location, "std location", weighted_average_location_std)
    # print("true location", true_location)
    return kl_divergence(weighted_average_location, np.log(weighted_average_location_std), true_location)

def nll_particle_filter(all_particles, true_location, normalization_constant=2428, top_particles=500):
    """ calculate the negative log likelihood of the particle filter output assuming a Gaussian centered
        around the particle's weighted mean and std

        Normalization constant used to normalize the locations onto 2428 grid
    
    """
    particle_with_weight = []
    for particle in all_particles:
        particle_with_weight.append((particle.weight, particle))
    particle_with_weight_sorted = sorted(particle_with_weight, key=lambda x: x[0], reverse=True)

    total_weights = 0
    weighted_average_location = np.array([0.0, 0.0])
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        weighted_average_location += particle.location/normalization_constant * weight
    weighted_average_location /= total_weights

    total_weights = 0
    weighted_average_location_std = np.array([0.0, 0.0])
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        weighted_average_location_std += (particle.location/normalization_constant - weighted_average_location) ** 2 * weight
    weighted_average_location_std /= total_weights
    # # TODO: Manisha check this... My neural net typically outputs low logstd of -4.5 --> 0.011 std --> 0.011 * 2428 = 27
    # weighted_average_location_std = np.maximum(np.sqrt(weighted_average_location_std), np.ones_like(weighted_average_location_std)*30)
    weighted_average_location_std = np.sqrt(weighted_average_location_std)
    return nll(weighted_average_location, weighted_average_location_std, true_location/normalization_constant)

def nll(mean, std, true_location):
    mean = torch.from_numpy(mean).float()
    std = torch.from_numpy(std).float()
    true_location = torch.from_numpy(true_location).float()
    distribution = Normal(mean, std)
    logprob = distribution.log_prob(true_location)
    res = -logprob.sum().item()
    print(mean, std, true_location, res)
    # if res > 10:
    #     print(mean, std, true_location, res)
    return res

def rmse_from_mode(mean, true_location, scale=1):
    """
    Calculates the Root Mean Squared Error between the predicted mean (or mode) from the filtering
    module and the ground truth location of the fugitive.
    :param mean: (np.array) Mean of the predicted distribution from the filtering module (0-1)
    :param true_location: (np.array) Normalized Ground truth location of the fugitive at the current timestep (0-1)
    :param scale: At what scale the MSE should be computed. By default, the value is 1 which means that we will
                  compute the MSE in the value range 0-1. If we set to the env dim, then we calculate the MSE w.r.t. the
                  actual positions on the map
    :return:
        mse_error: Mean squared Difference between the predicted mean (or mode) from the filtering
                   module and the ground truth location of the fugitive.
    """

    mean = mean * scale
    true_location = true_location * scale

    mse_error = np.linalg.norm(mean - true_location)
    return mse_error


def kl_divergence(predicted_mean, predicted_logstd, true_location):
    """
    Calculates the KL Divergence between the predicted distribution from the filtering location
    and the true distribution (centered on the true location)

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )

         where m0, S0 are the mean and covariance of the true distribution and
               m1, S1 are the mean and covariance from the predicted distribution

    Code Reference: https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv

    :param predicted_mean: (np.array) Mean of the predicted distribution from the filtering module
    :param predicted_logstd: (np.array) Logstd of the predicted distribution from the filtering module
    :param true_location: (np.array) Ground truth location of the fugitive at the current timestep
    :return:
        KL Divergence between the two distributions
    """
    n_dims = 2  # x, y dimension for prisoner location
    predicted_var = np.exp(predicted_logstd) ** 2
    predicted_cov = np.eye(n_dims) * predicted_var

    true_mean = true_location
    true_cov = np.eye(n_dims) * 0.001  # Assume very small covariance for the true distribution

    # Calculate KL Divergence
    inverse_pred_cov = np.linalg.inv(predicted_cov)
    diff = predicted_mean - true_mean

    #  KL Divergence has three terms
    trace_term = np.trace(inverse_pred_cov @ true_cov)
    determinant_term = np.log(
        np.linalg.det(predicted_cov) / np.linalg.det(true_cov))  # np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ inverse_pred_cov @ diff  # np.sum( (diff*diff) * iS1, axis=1)

    KL_div = 0.5 * (trace_term + determinant_term + quad_term - n_dims)
    return KL_div

def generate_delta_plot(avg_ground_truth_likelihoods, pf_ground_truth_likelihoods, deltas):
    """
    Generate the plot for comparing the probability of ground truth with respect to varying
    distance thresholds (delta)
    :param avg_ground_truth_likelihoods: (np array of shape: n_rollouts x n_deltas) Ground truth likelihoods averaged across all timesteps for each delta
    :param deltas: Delta values for which we are computing the ground truth likelihoods
    :return:
        None. Saves a matplotlib figure
    """
    fig, ax = plt.subplots()
    likelihood_means = np.mean(avg_ground_truth_likelihoods, axis=0)
    likelihood_stds = np.std(avg_ground_truth_likelihoods, axis=0)

    pf_likelihood_means = np.mean(pf_ground_truth_likelihoods, axis=0)
    pf_likelihood_stds = np.std(pf_ground_truth_likelihoods, axis=0)

    plt.plot(deltas, likelihood_means, c="tab:blue", label='filtering')
    plt.fill_between(deltas, likelihood_means - likelihood_stds,
                     likelihood_means + likelihood_stds, alpha=0.3, color="tab:blue")

    plt.plot(deltas, pf_likelihood_means, c="tab:orange", label='particle_filter')
    plt.fill_between(deltas, pf_likelihood_means - pf_likelihood_stds,
                     pf_likelihood_means + pf_likelihood_stds, alpha=0.3, color="tab:orange")
    ax.legend()
    ax.set_ylabel(r'$P(Likelihood \leq \delta$)')
    ax.set_xlabel(r'$\delta$')
    plt.savefig('tmp/delta_all.png')


def calculate_metrics(num_rollouts, env, policy_path, model,
                      device, heuristic_policy, calculate_delta=False):
    """
    Calculate Filtering metrics (RMSE, KL, binary_prob_metric) across rollouts
    and report mean and std across multiple rollouts for a given filtering model
    :param num_rollouts:
    :param env:
    :param policy_path:
    :return:
    """
    policy = torch.load(policy_path, map_location=device)
    model.load_state_dict(policy)
    model.eval()

    seeds = np.arange(100)
    seeds = [66]
    all_metrics = defaultdict(dict)
    pf_metrics = defaultdict(dict)

    deltas = np.linspace(0, 0.5, 11)
    log_likelihoods = defaultdict(dict)  # For storing likelihood each time step (in each rollout) and each delta
    pf_log_likelihoods = defaultdict(
        dict)  # For storing likelihood each time step (in each rollout) and each delta for pf

    avg_likelihoods = np.zeros((num_rollouts, len(deltas)))  # For mean likelihood across rollout for each delta
    pf_avg_likelihoods = np.zeros(
        (num_rollouts, len(deltas)))  # For mean likelihood across rollout for each delta for pf

    for k in range(num_rollouts):
        torch.manual_seed(seeds[k])
        np.random.seed(seeds[k])
        random.seed(seeds[k])
        env.seed(seeds[k])
        env.reset()
        done = False

        particle_filter = ParticleFilter(env)

        # Initialize vars to store values of metrics for each rollout
        final_binary_count, pf_final_binary_count = 0, 0
        RMSE, KL, binary_counts = [], [], []
        pf_RMSE, pf_KL, pf_binary_counts = [], [], []

        for d in deltas:
            log_likelihoods[k][str(d)] = []
            pf_log_likelihoods[k][str(d)] = []

        timestep = 0
        detected_x, detected_y = [], []
        undetected_x, undetected_y = [], []

        pf_detected_x, pf_detected_y = [], []
        pf_undetected_x, pf_undetected_y = [], []

        game_imgs, pf_imgs, imgs = [], [], []

        t = []
        detected_timesteps = []  # Store the timesteps where the fugitive was detected in the rollout
        while not done:
            fugitive_observation = env.get_fugitive_observation()
            action = heuristic_policy.predict(fugitive_observation)
            blue_observation, reward, done, _ = env.step(action[0])

            # Propagate particle filter
            particle_filter.propagate()
            if env.is_detected:
                particle_filter.update_weight_according_to_detection(env.prisoner.location)
                particle_filter.normalize_all_weights()

            # Pass the blue observation to the filtering model
            demonstration_tensor = torch.from_numpy(blue_observation).to(device).float().unsqueeze(0)
            prediction_params = model.predict(demonstration_tensor).detach().cpu().numpy()
            prediction_params = prediction_params[:, -1, :]
            mean_location, log_std = prediction_params[0, :2], prediction_params[0, 2:]
            true_location = np.array(env.prisoner.location) / 2428  # get prisoner's true location from the env

            # Compute all metrics for filtering
            RMSE.append(rmse_from_mode(mean=mean_location, true_location=true_location, scale=2428))
            KL.append(
                kl_divergence(predicted_mean=mean_location, predicted_logstd=log_std, true_location=true_location))

            result = get_ground_truth_likelihood(mean=mean_location, logstd=log_std, true_location=true_location,
                                                 dist_threshold=0.05)

            binary_counts.append(result >= 0.5)

            log_likelihoods[k]["0.05"].append(result)
            t.append(timestep)
            timestep += 1

            grid = plot_gaussian_heatmap(mean_location, log_std)
            # For plotting with respect to delta=0.05
            if env.is_detected:
                detected_timesteps.append(timestep)
                # # timestep = 0
                detected_x.append(result)
                detected_y.append(timestep)
            else:
                undetected_x.append(result)
                undetected_y.append(timestep)

            game_img = env.render('Policy', show=False, fast=True)
            heatmap_img = generate_heatmap_img(grid, true_location=env.prisoner.location)

            # generate ground truth live plot
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            fig.set_size_inches(4, 4)
            ax.scatter(detected_y, detected_x, c='tab:blue', alpha=0.5, label='detected')
            ax.scatter(undetected_y, undetected_x, c='tab:red', alpha=0.5, label='undetected')
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.set_xlim([0, 400])
            ax.set_ylabel(r'$P(Likelihood \leq \delta$)')
            ax.set_xlabel('time since episode start')
            ax.set_title(r'$\delta = 0.05$')
            canvas.draw()  # draw the canvas, cache the renderer
            likelihood_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            likelihood_img = likelihood_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Convert game image from RGB to BGR
            likelihood_img = cv2.cvtColor(likelihood_img, cv2.COLOR_RGB2BGR)

            # Stack Heatmap and likelihood images one below the other
            combined_img = np.vstack([heatmap_img, likelihood_img])

            # combine both images to one
            max_x = max(game_img.shape[0], combined_img.shape[0])
            max_y = max(game_img.shape[1], combined_img.shape[1])

            # # pad
            game_img_reshaped = np.pad(game_img, ((0, max_x - game_img.shape[0]), (0, 0), (0, 0)), 'constant',
                                       constant_values=0)
            combined_img_reshaped = np.pad(combined_img, ((0, max_x - combined_img.shape[0]), (0, 0), (0, 0)),
                                           'constant',
                                           constant_values=0)

            img = np.concatenate((game_img_reshaped, combined_img_reshaped), axis=1)
            imgs.append(img)

            # ------------------------------------Compute all metrics for particle_filter-------------------------------------------#
            pf_true_location = np.array(env.prisoner.location)
            pf_RMSE.append(rmse_particle_filter(particle_filter.particles, pf_true_location))
            pf_KL.append(kl_divergence_particle_filter(particle_filter.particles, true_location=pf_true_location))

            result = get_ground_truth_likelihood_particle_filter(particle_filter.particles, pf_true_location,
                                                                 dist_threshold=0.05)
            pf_binary_counts.append(result >= 0.5)
            pf_log_likelihoods[k]["0.05"].append(result)
            #
            # pf_heatmap_img = particle_filter.plot(save_name=None,
            #                                       to_cv=True,
            #                                       ground_truth_location=env.prisoner.location)

            # if env.is_detected:
            #     # timestep = 0
            #     pf_detected_x.append(result)
            #     pf_detected_y.append(timestep)
            # else:
            #     pf_undetected_x.append(result)
            #     pf_undetected_y.append(timestep)

            # generate ground truth live plot
            # fig = Figure()
            # canvas = FigureCanvas(fig)
            # ax = fig.gca()
            # fig.set_size_inches(4, 4)
            # ax.scatter(detected_y, detected_x, c='tab:blue', alpha=0.5, label='detected')
            # ax.scatter(undetected_y, undetected_x, c='tab:red', alpha=0.5, label='undetected')
            # ax.legend()
            # ax.set_ylim([0, 1.1])
            # ax.set_xlim([0, 400])
            # ax.set_ylabel(r'$P(Likelihood \leq \delta$)')
            # ax.set_xlabel('time since episode start')
            # ax.set_title(r'$\delta = 0.05$')
            # canvas.draw()  # draw the canvas, cache the renderer
            # likelihood_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            # likelihood_img = likelihood_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #
            # # Convert game image from RGB to BGR
            # likelihood_img = cv2.cvtColor(likelihood_img, cv2.COLOR_RGB2BGR)
            #
            # # Stack Heatmap and likelihood images one below the other
            # pf_combined_img = np.vstack([pf_heatmap_img, likelihood_img])
            #
            # # combine both images to one
            # max_x = max(game_img.shape[0], pf_combined_img.shape[0])
            # max_y = max(game_img.shape[1], pf_combined_img.shape[1])
            #
            # # # pad
            # game_img_reshaped = np.pad(game_img, ((0, max_x - game_img.shape[0]), (0, 0), (0, 0)), 'constant',
            #                            constant_values=0)
            # pf_combined_img_reshaped = np.pad(pf_combined_img, ((0, max_x - pf_combined_img.shape[0]), (0, 0), (0, 0)),
            #                                   'constant',
            #                                   constant_values=0)
            #
            # pf_img = np.concatenate((game_img_reshaped, pf_combined_img_reshaped), axis=1)
            # pf_imgs.append(pf_img)

            if calculate_delta:
                for d in deltas:
                    if d == 0.05:
                        continue  # We've already computed the value for delta=0.05 above
                    result = get_ground_truth_likelihood(mean=mean_location, logstd=log_std,
                                                         true_location=true_location,
                                                         dist_threshold=d)
                    log_likelihoods[k][str(d)].append(result)

                    # For particle filter
                    result = get_ground_truth_likelihood_particle_filter(particle_filter.particles,
                                                                         true_location=pf_true_location,
                                                                         dist_threshold=d)
                    pf_log_likelihoods[k][str(d)].append(result)

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
        final_binary_count = np.sum(binary_counts)
        pf_final_binary_count = np.sum(pf_binary_counts)
        print("Seed: {}".format(seeds[k]))
        print("Filtering Results...")
        print("MSE: mean:{}, std:{}".format(np.mean(RMSE), np.std(RMSE)))
        print("KL: mean:{}, std:{}".format(np.mean(KL), np.std(KL)))
        print("Binary Count: count:{}, total:{}".format(final_binary_count, num_timesteps_in_rollout))
        print("num detected timesteps: {}".format(num_detections))
        print("---------------------------------------------------------")
        print("Particle Filtering Results...")
        print("MSE: mean:{}, std:{}".format(np.mean(pf_RMSE), np.std(pf_RMSE)))
        print("KL: mean:{}, std:{}".format(np.mean(pf_KL), np.std(pf_KL)))
        print("Binary Count: count:{}, total:{}".format(pf_final_binary_count, num_timesteps_in_rollout))
        print("num detected timesteps: {}".format(num_detections))
        print("---------------------------------------------------------")

        save_video(imgs, 'tmp/seq_model_66.mp4'.format(seeds[k]), fps=5)
        # save_video(pf_imgs, 'tmp/pf_sponsor_{}_map_0_more_cams.mp4'.format(seeds[k]), fps=5)

        # End of Episode... Calculate Aggregate metrics for filtering
        all_metrics[k]['KL'] = KL
        all_metrics[k]['RMSE'] = RMSE
        all_metrics[k]['binary_count_ratio'] = final_binary_count / num_timesteps_in_rollout

        pf_metrics[k]['KL'] = pf_KL
        pf_metrics[k]['RMSE'] = pf_RMSE
        pf_metrics[k]['binary_count_ratio'] = pf_final_binary_count / num_timesteps_in_rollout

        # To plot on x-axis
        all_metrics[k]['detection_ratio'] = num_detections / num_timesteps_in_rollout
        pf_metrics[k]['detection_ratio'] = num_detections / num_timesteps_in_rollout

        all_metrics[k]['detected_timesteps_std'] = np.std(detected_timesteps)
        pf_metrics[k]['detected_timesteps_std'] = np.std(detected_timesteps)

        # Calculate time between detections
        time_between_detections = []
        for t in range(num_detections - 1):
            if t == 0:
                time_between_detections.append(detected_timesteps[t])  # First detection since start of episode (t=0)
                time_between_detections.append(detected_timesteps[t + 1] - detected_timesteps[t])
            else:
                time_between_detections.append(detected_timesteps[t + 1] - detected_timesteps[t])
        if len(detected_timesteps) > 0:
            time_between_detections.append(num_timesteps_in_rollout - detected_timesteps[-1])

        all_metrics[k][
            'time_between_detections'] = time_between_detections  # Store time between detections instead of average time between detections
        pf_metrics[k][
            'time_between_detections'] = time_between_detections  # Store time between detections instead of average time between detections

        if calculate_delta:
            for idx, d in enumerate(deltas):
                avg_likelihoods[k, idx] = np.mean(log_likelihoods[k][str(d)])
                pf_avg_likelihoods[k, idx] = np.mean(pf_log_likelihoods[k][str(d)])

        # Calculate average metric value with respect to time between detections
        all_metrics[k]['average_RMSE_between_detections'] = []
        all_metrics[k]['average_KL_between_detections'] = []
        all_metrics[k]['average_binary_count_between_detections'] = []

        pf_metrics[k]['average_RMSE_between_detections'] = []
        pf_metrics[k]['average_KL_between_detections'] = []
        pf_metrics[k]['average_binary_count_between_detections'] = []

        start_idx = np.array([0] + detected_timesteps)
        end_idx = np.array(detected_timesteps + [num_timesteps_in_rollout])

        for dt in range(num_detections + 1):
            all_metrics[k]['average_RMSE_between_detections'].append(np.mean(RMSE[start_idx[dt]: end_idx[dt]]))
            all_metrics[k]['average_KL_between_detections'].append(np.mean(KL[start_idx[dt]: end_idx[dt]]))
            all_metrics[k]['average_binary_count_between_detections'].append(
                np.mean(binary_counts[start_idx[dt]: end_idx[dt]]))

            pf_metrics[k]['average_RMSE_between_detections'].append(np.mean(pf_RMSE[start_idx[dt]: end_idx[dt]]))
            pf_metrics[k]['average_KL_between_detections'].append(np.mean(pf_KL[start_idx[dt]: end_idx[dt]]))
            pf_metrics[k]['average_binary_count_between_detections'].append(
                np.mean(pf_binary_counts[start_idx[dt]: end_idx[dt]]))

    return all_metrics, avg_likelihoods, pf_metrics, pf_avg_likelihoods


# def calculate_metrics_particle_filter(num_rollouts, env, heuristic_policy, debug=False, calculate_delta=False):
#     """
#     Calculate Filtering metrics (RMSE, KL, binary_prob_metric) across rollouts
#     and report mean and std across multiple rollouts for a given filtering model
#     :param num_rollouts:
#     :param env:
#     :return:
#     """
#
#     seeds = np.arange(100)
#     all_metrics = defaultdict(dict)
#
#     deltas = np.linspace(0, 0.5, 11)
#     log_likelihoods = defaultdict(dict)  # For storing likelihood each time step (in each rollout) and each delta
#     avg_likelihoods = np.zeros((num_rollouts, len(deltas)))  # For mean likelihood across rollout for each delta
#
#     for k in range(num_rollouts):
#         np.random.seed(seeds[k])
#         random.seed(seeds[k])
#         env.seed(seeds[k])
#         env.reset()
#         particle_filter = ParticleFilter(env)
#         done = False
#
#         # Initialize vars to store values of metrics for each rollout
#         binary_count = 0
#         RMSE, KL = [], []
#
#         for d in deltas:
#             log_likelihoods[k][str(d)] = []
#
#         timestep = 0
#         detected_x, detected_y = [], []
#         undetected_x, undetected_y = [], []
#         t = []
#         detected_timesteps = []  # Store the timesteps where the fugitive was detected in the rollout
#         imgs = []
#         while not done:
#             fugitive_observation = env.get_fugitive_observation()
#             action = heuristic_policy.predict(fugitive_observation)
#             blue_observation, reward, done, _ = env.step(action[0])
#
#             # Pass the blue observation to the filtering model
#             particle_filter.propagate()
#             if env.is_detected:
#                 particle_filter.update_weight_according_to_detection(env.prisoner.location)
#                 particle_filter.normalize_all_weights()
#             if debug:
#                 particle_filter.plot(save_name="logs/temp/debug_particle_filter_%d.png" % timestep,
#                                      ground_truth_location=env.prisoner.location)
#                 game_img = env.render('Policy', show=True, fast=True)
#                 particle_img = cv2.imread("logs/temp/debug_particle_filter_%d.png" % timestep)
#                 if particle_img is not None:
#                     max_x = max(game_img.shape[0], particle_img.shape[0])
#                     max_y = max(game_img.shape[1], particle_img.shape[1])
#                     game_img_reshaped = np.pad(game_img, ((0, max_x - game_img.shape[0]), (0, 0), (0, 0)), 'constant',
#                                                constant_values=0)
#                     particle_img_reshaped = np.pad(particle_img, ((0, max_x - particle_img.shape[0]), (0, 0), (0, 0)),
#                                                    'constant', constant_values=0)
#
#                     img = np.concatenate((game_img_reshaped, particle_img_reshaped), axis=1)
#                     imgs.append(img)
#                 else:
#                     imgs.append(game_img)
#
#             # demonstration_tensor = torch.from_numpy(blue_observation).to(device).float().unsqueeze(0)
#             # prediction_params = model(demonstration_tensor).detach().cpu().numpy()
#             # mean_location, log_std = prediction_params[:2], prediction_params[2:]
#
#             true_location = np.array(env.prisoner.location)  # get prisoner's true location from the env
#
#             # Compute all metrics
#             RMSE.append(rmse_particle_filter(particle_filter.particles, true_location))
#             KL.append(kl_divergence_particle_filter(particle_filter.particles, true_location=true_location))
#
#             result = get_ground_truth_likelihood_particle_filter(particle_filter.particles, true_location,
#                                                                  dist_threshold=0.05)
#             binary_count += 1 if result >= 0.5 else 0
#             log_likelihoods[k]["0.05"].append(result)
#             t.append(timestep)
#             timestep += 1
#
#             # For plotting with respect to delta=0.05
#             if env.is_detected:
#                 detected_timesteps.append(timestep)
#                 # timestep = 0
#                 detected_x.append(result)
#                 detected_y.append(timestep)
#             else:
#                 undetected_x.append(result)
#                 undetected_y.append(timestep)
#
#             if calculate_delta:
#                 for d in deltas:
#                     if d == 0.05:
#                         continue  # We've already computed the value for delta=0.05 above
#                     result = get_ground_truth_likelihood_particle_filter(particle_filter.particles,
#                                                                          true_location=true_location,
#                                                                          dist_threshold=d)
#                     log_likelihoods[k][str(d)].append(result)
#
#         # ------------------------------------- Generate Likelihood plot --------------------------------------------- #
#         # print(first_detection)
#         # fig, ax = plt.subplots()
#         # ax.scatter(undetected_y, undetected_x, c='coral', alpha=0.5, label='undetected')
#         # ax.scatter(detected_y, detected_x, c='lightblue', alpha=0.5, label='detected')
#         #
#         # # For plot since last detection
#         # # first_detection = detected_timesteps[0]
#         # # ax.scatter(t[first_detection:], log_likelihoods[first_detection:], c='tab:blue', alpha=0.5, label='detected')
#
#         # ax.legend()
#         # plt.ylabel(r'$P(Likelihood \leq \delta$)')
#         # plt.xlabel('time since episode start')
#         # plt.title(r'$\delta = 0.05$')
#         # plt.savefig('tmp/matt_plot_{}.png'.format(k))
#         # ------------------------------------------------------------------------------------------------------------ #
#
#         num_timesteps_in_rollout = len(RMSE)
#         num_detections = len(detected_timesteps)
#
#         print("Seed: {}".format(seeds[k]))
#         print("MSE: mean:{}, std:{}".format(np.mean(RMSE), np.std(RMSE)))
#         print("KL: mean:{}, std:{}".format(np.mean(KL), np.std(KL)))
#         print("Binary Count: count:{}, total:{}".format(binary_count, num_timesteps_in_rollout))
#         print("num detected timesteps: {}".format(num_detections))
#         print("---------------------------------------------------------")
#
#         # End of Episode... Calculate Aggregate metrics for filtering
#         all_metrics[k]['KL'] = KL
#         all_metrics[k]['RMSE'] = RMSE
#         all_metrics[k]['binary_count_ratio'] = binary_count / num_timesteps_in_rollout
#
#         # To plot on x-axis
#         all_metrics[k]['detection_ratio'] = num_detections / num_timesteps_in_rollout
#         all_metrics[k]['detected_timesteps_std'] = np.std(detected_timesteps)
#
#         # Calculate time between detections
#         time_between_detections = []
#         for t in range(num_detections - 1):
#             if t == 0:
#                 time_between_detections.append(detected_timesteps[t + 1] - 0)
#             else:
#                 time_between_detections.append(detected_timesteps[t + 1] - detected_timesteps[t])
#
#         time_between_detections.append(num_timesteps_in_rollout - detected_timesteps[-1])
#
#         all_metrics[k]['time_between_detections'] = np.mean(time_between_detections)
#
#         if debug:
#             save_video(imgs, "logs/temp/particle_density_%d.mp4" % k, fps=10)
#         with open("tmp/all_metrics.pkl", "wb") as f:
#             pickle.dump(all_metrics, f)
#
#         if calculate_delta:
#             for idx, d in enumerate(deltas):
#                 avg_likelihoods[k, idx] = np.mean(log_likelihoods[k][str(d)])
#
#     return all_metrics, avg_likelihoods


def plot_for_time_between_detections(all_metrics, pf_metrics, metric, num_rollouts):
    # Plot across all rollouts
    x, y, pf_y = [], [], []
    assert metric in ['RMSE', 'KL', 'binary_count']
    for k in range(num_rollouts):
        x.extend(all_metrics[k]['time_between_detections'])
        y.extend(all_metrics[k]['average_{}_between_detections'.format(metric)])
        pf_y.extend(pf_metrics[k]['average_{}_between_detections'.format(metric)])

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.5, color='tab:blue', label='filtering')
    ax.scatter(x, pf_y, alpha=0.5, color='tab:orange', label='particle_filter')
    ax.legend()
    plt.xlabel('time between detections across rollouts')
    plt.ylabel("Average {} between detections".format(metric))
    plt.title('LSTM - Current Timestep')
    plt.savefig('tmp/{}_time_bw_detections_std.png'.format(metric))


def plot_metrics(all_metrics, pf_metrics, metric, num_rollouts):
    """
    Plot a specific metric (rmse, Kl, etc.) mean and std across multiple rollouts
    :param all_metrics: dictionary of dictionaries indexed by rollout id and the metric corresponding to that rollout
    :param metric: (str) 'KL', 'RMSE', 'binary_count'
    :param num_rollouts: Number of rollouts in the all_metrics dictionary whose metrics have been evaluated
    :return:
        None
    """
    metric_means, pf_metric_means = [], []
    metric_std, pf_metric_std = [], []
    assert metric in all_metrics[0].keys()
    plot_x1, plot_x2, plot_x3 = [], [], []
    for k in range(num_rollouts):
        metric_means.append(np.mean(all_metrics[k][metric]))
        metric_std.append(np.std(all_metrics[k][metric]))
        pf_metric_means.append(np.mean(pf_metrics[k][metric]))
        pf_metric_std.append(np.std(pf_metrics[k][metric]))
        plot_x1.append(all_metrics[k]['detection_ratio'])
        plot_x2.append(all_metrics[k]['detected_timesteps_std'])
        # plot_x3.append(all_metrics[k]['time_between_detections'])

    # Plot with respect to proportion of timesteps detected
    fig, ax = plt.subplots()
    ax.scatter(plot_x1, metric_means, alpha=0.5, c="tab:blue", label='filtering')
    ax.scatter(plot_x1, pf_metric_means, alpha=0.5, c="tab:orange", label='particle_filter')

    if metric == "binary_count_ratio":
        ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.legend()
    plt.xlabel('Ratio of timesteps detected across rollouts')
    plt.ylabel(metric)
    plt.title('LSTM - Current Timestep')
    plt.savefig('tmp/{}_detection_ratio.png'.format(metric))

    # Plot with respect to proportion of timesteps detected
    fig, ax = plt.subplots()
    ax.scatter(plot_x2, metric_means, alpha=0.5, c="tab:blue", label='filtering')
    ax.scatter(plot_x2, pf_metric_means, alpha=0.5, c="tab:orange", label='particle_filter')
    ax.legend()
    plt.xlabel('Std: detected timesteps across rollouts')
    plt.ylabel(metric)
    plt.title('LSTM - Current Timestep')
    plt.savefig('tmp/{}_detected_timestep_std.png'.format(metric))

    # # Plot with respect to proportion of timesteps detected
    # fig, ax = plt.subplots()
    # ax.scatter(plot_x3, metric_means, alpha=0.5)
    # plt.xlabel('time between detections across rollouts')
    # plt.ylabel(metric)
    # plt.title('LSTM - Current Timestep')
    # plt.savefig('tmp/{}_time_bw_detections_std.png'.format(metric))


if __name__ == '__main__':
    # Here: Testing the evaluation metric on output from a filtering model

    device = 'cpu'
    # Load a pre-trained filtering model
    # Regular cameras model
    # path = "logs/filtering/normal/mean/adaptive-std/new_terrain/maps_0.2/20220309-0915/filter_policy_epoch_100.pth"

    # Seq2Seq for regular camera model
    # path = "logs/filtering/normal/seq2seq/adaptive-std/new_terrain/maps_0.2/seq_output/20220304-0738/filter_policy_epoch_100.pth"

    # Autoregressive model
    path = "logs/filtering/normal/seq2seq/autoregressive/adaptive-std/new_terrain/maps_0.2/20220314-1154/filter_policy_epoch_100.pth"

    # More cameras model
    # path = "logs/filtering/normal/mean/adaptive-std/new_terrain/maps_0.2/more_cameras/20220310-1319/filter_policy_epoch_100.pth"

    batch_size = 10
    seq_len = 4

    # Load the env to get observations
    # Include terrain variations for analysis
    env = initialize_prisoner_environment(variation=0,
                                          # camera_configuration="simulator/camera_locations/70_per_cent_cameras_w_camera_net.txt",
                                          observation_step_type="Blue")

    # env.seed(seed=SEED)

    input_shape = env.reset().reshape(-1, 1).shape[0]

    # model = NLLModel(lstm_input_dim=input_shape, lstm_hidden_dim=64, lstm_num_layers=1,
    #                  batch_size=batch_size).to(device)

    # model = NLLSeqModel(lstm_input_dim=input_shape, lstm_hidden_dim=64, lstm_num_layers=1,
    #                     batch_size=1, seq_len=seq_len, device=device).to(device)

    encoder = Encoder(input_dim=input_shape,
                      hid_dim=64,
                      n_layers=1)

    # Output Dim = 2 because output is the prisoner location (x,y)
    decoder = Decoder(output_dim=2,
                      hid_dim=64,
                      n_layers=1)

    # Define the neural network model
    model = NLLSeq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)

    env = PrisonerBatchEnv(env, batch_size=seq_len)
    heuristic_policy = HeuristicPolicy(env)

    env.reset()
    done = False

    num_rollouts = 1
    all_metrics, avg_likelihoods, pf_metrics, pf_likelihoods = calculate_metrics(num_rollouts=num_rollouts, env=env,
                                                                                 policy_path=path,
                                                                                 device='cpu', model=model,
                                                                                 heuristic_policy=heuristic_policy,
                                                                                 calculate_delta=True)
    data = {
        'all_metrics': all_metrics,
        'avg_likelihoods': avg_likelihoods,
        'pf_metrics': pf_metrics,
        'pf_likelihoods': pf_likelihoods
    }

    # with open("tmp/plot_data_metrics_{}_map_0.pkl".format(num_rollouts), "wb") as f:
    #     pickle.dump(data, f)

    # pf_metrics, pf_avg_likelihoods = calculate_metrics_particle_filter(num_rollouts=num_rollouts, env=env,
    #                                                                    heuristic_policy=heuristic_policy, debug=False, calculate_delta=True)

    # plot_metrics(all_metrics, pf_metrics, metric='RMSE', num_rollouts=num_rollouts)
    # plot_metrics(all_metrics, pf_metrics, metric='KL', num_rollouts=num_rollouts)
    # plot_metrics(all_metrics, pf_metrics, metric='binary_count_ratio', num_rollouts=num_rollouts)
    #
    # plot_for_time_between_detections(all_metrics, pf_metrics, metric='binary_count', num_rollouts=num_rollouts)
    # plot_for_time_between_detections(all_metrics, pf_metrics, metric='RMSE', num_rollouts=num_rollouts)
    # plot_for_time_between_detections(all_metrics, pf_metrics, metric='KL', num_rollouts=num_rollouts)
    #
    # generate_delta_plot(avg_likelihoods, pf_likelihoods, deltas=np.linspace(0, 0.5, 11))
