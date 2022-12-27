import os
import numpy as np
from tqdm import tqdm
import glob
from particle_filter.particle_filter import ParticleFilter
from particle_filter.evaluation_metric import *
from simulator.prisoner_env_variations import initialize_prisoner_environment
from utils import save_video
import copy
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def load_data(path):
    # run_path = "test.npz"
    # print(run_path)
    run = np.load(path)
    locations = run['locations']
    directions = run['directions']
    speeds = run['speeds']
    detections = run['detections']
    return locations, directions, speeds, detections


def get_stats_per_run(locations, directions, speeds, detections, env):
    """ Given heatmaps, true locations, and timesteps, compute the following:

    Binary count
    KL Divergence
    Mean Squared Error
    Probability around the agent

    """
    timestep = locations.shape[0]
    probabilities = [[], [], [], [], [], []]
    kls = [[], [], [], [], [], []]
    mses = [[], [], [], [], [], []]
    nls = [[], [], [], [], [], []]
    binary_counts = [0, 0, 0, 0, 0, 0]
    video_frames = []
    timestamps = []
    probability_imgs = []
    mse_imgs = []

    # Initialize particle filter once...
    particle_filter = ParticleFilter(env, direction_mode=False, initialize_within_camera_grid=True)
    particle_filter.normalize_all_weights()

    for t in tqdm(range(0, timestep)):
        # for t in tqdm(range(0, 10)):
        timestamps.append(t)

        particle_filter.propagate_direction()

        # Re-weight the particles only if there is a detection
        if detections[t]:
            particle_filter.update_weight_according_to_detection(locations[t])
            # particle_filter.update_weight_according_to_direction([np.cos(directions[t]), np.sin(directions[t])])
            particle_filter.normalize_all_weights()

        # Create a copy of the particle filter to propagate and compare with the future timesteps
        propagated_particle_filter = copy.deepcopy(particle_filter)
        for propagated_particles in propagated_particle_filter.particles:
            propagated_particles.speed *= 10  # Increase the speed to propagate particles into the future

        for n_step_in_future in [10, 20, 30, 40, 50, 60]:
            if t + n_step_in_future >= timestep:
                break
            propagated_particle_filter.propagate_direction()
            true_location = np.array(locations[t + n_step_in_future])
            likelihood = get_ground_truth_likelihood_particle_filter(propagated_particle_filter.particles, true_location,
                                                                     dist_threshold=0.05)
            probabilities[int(n_step_in_future // 10) - 1].append(likelihood)

            mse = rmse_particle_filter(propagated_particle_filter.particles, true_location)
            mses[int(n_step_in_future // 10) - 1].append(mse)
            binary_counts[int(n_step_in_future // 10) - 1] += 1 if likelihood >= 0.5 else 0

            kl = kl_divergence_particle_filter(propagated_particle_filter.particles, true_location=true_location)
            kls[int(n_step_in_future // 10) - 1].append(kl)

            nl = nll_particle_filter(propagated_particle_filter.particles, true_location)
            nls[int(n_step_in_future // 10) - 1].append(nl)

            if n_step_in_future == 30:
                # print(likelihood)
                if t in np.arange(273, 350):
                    continue
                else:
                    # pass
                    img = propagated_particle_filter.plot(save_name=f"logs/temp/debug_particle_filter_n_{n_step_in_future}_{t}.png",
                                               to_cv=True, ground_truth_location=true_location, hideout_locations=env.hideout_list)
                    video_frames.append(img)

    average_probabilities = []
    for probability in probabilities:
        average_probabilities.append(np.mean(probability))

    average_mses = []
    for mse in mses:
        average_mses.append(np.mean(mse))

    binary_counts = np.array(binary_counts) / [timestep - 10, timestep - 20, timestep - 30,
                                               timestep - 40, timestep - 50, timestep - 60]

    return video_frames, probabilities, mses, kls, binary_counts, nls


# path = '/star-data/sye40/0308-1826-runs/'
# path = '/star-data/sye40/logs/msp_EBM_Sequence/20220309-1319/runs'
# map_num = 0
# path = '/nethome/sye40/PrisonerEscape/prisoner-prediction-epsilon-04/map_0'

# map_num = 3
# path = '/nethome/sye40/PrisonerEscape/prisoner-prediction-epsilon-04-map-3/map_3'

map_num = 0
# path = '/nethome/sye40/PrisonerEscape/prisoner-prediction-epsilon-04/map_0'
# path = '/nethome/sye40/PrisonerEscape/temp/prisoner-prediction-epsilon-04/map_0'
path = '/star-data/mnatarajan30/prisoner-prediction-epsilon-04-map-0/map_0'

files = glob.glob(path)

probabilities = [[], [], [], [], [], []]
mses = [[], [], [], [], [], []]
binary_counts = [[], [], [], [], [], []]

for i in tqdm(range(11, 12)):
    # for i in tqdm():
    save_folder = os.path.dirname(path) + '/%d' % i
    # check if folder exists
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    run = os.path.join(path, f"{i}.npz")
    locations, directions, speeds, detections = load_data(run)
    env = initialize_prisoner_environment(map_num,
                                          observation_step_type="Blue",
                                          seed=i)

    print("Hideout locations: ")
    for k, hideout in enumerate(env.hideout_list):
        print("hideout {}, known:{}: {}".format(k, hideout.known_to_good_guys, hideout.location))
    video_frames, probabilities_run, mses_run, kl_run, bin_counts, nls = get_stats_per_run(locations, directions,
                                                                                           speeds, detections, env)

    np.savez(os.path.join(save_folder, f"{i}_stats.npz"),
             probabilities=np.array(probabilities_run),
             mses=np.array(mses_run),
             kls=np.array(kl_run),
             binary_count=np.array(bin_counts),
             nls=np.array(nls))
    save_video(video_frames, os.path.join(save_folder, f"{i}_random_heatmaps.mp4"), fps=5)
    # save_video(probability_imgs, f'{save_folder}/prob.mp4', fps=5)
    # save_video(mse_imgs, f'{save_folder}/mses.mp4', fps=5)
    # print(np.std(probabilities), np.std(mses), np.std(binary_counts))
    # print(np.mean(probabilities), np.mean(mses), np.mean(binary_counts))

    for i in range(6):
        probabilities[i].append(probabilities_run[i])
        mses[i].append(mses_run[i])
        binary_counts[i].append(bin_counts[i])

probabilities = np.array(probabilities)
mses = np.array(mses)
binary_counts = np.array(binary_counts)

# np.savez(os.path.join(path, f"all_stats.npz"), probabilities=np.array(probabilities), mses=np.array(mses),
#          binary_count=np.array(binary_counts))

# print("Combined stats:")
# print(np.mean(probabilities), np.mean(mses), np.mean(binary_counts))
# print(np.std(probabilities), np.std(mses), np.std(binary_counts))

# run_path = os.path.join(path, f"{i}.npz")
# import matplotlib.pyplot as plt

# # print(np.array(probabilities).shape)
# # plt.figure()
# # plt.scatter(range(len(probabilities)), probabilities)


# # print(np.array(probabilities).shape)
# plt.figure()
# plt.scatter(range(len(mses)), mses)
