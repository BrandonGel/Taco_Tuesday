""" Convert a folder of gnn models to a single numpy file with blue observations """

import os
import numpy as np

# path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations/train"
# save_path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations/train_vector"

# path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations/test"
# save_path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations/test_vector.npy"

path = "/star-data/sye40/random_start_locations/small_subset_test"
save_path = "/star-data/sye40/random_start_locations/small_subset_test_vector"

path = "/star-data/sye40/random_start_locations_no_net/small_subset_test"
save_path = "/star-data/sye40/random_start_locations_no_net/small_subset_test_vector"
# save_path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations/test_vector"

path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations_no_net/train"
save_path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations_no_net/train_vector"

path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations_no_net/test"
save_path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations_no_net/test_vector.npz"

# get all files in the folder
files = os.listdir(path)
# print(files)

print("Number of files: ", len(files))

blue_obs = []
red_locs = []
dones = []

for file in files:
    arr = np.load(os.path.join(path, file), allow_pickle=True)
    blue_ob = arr["blue_observations"]
    red_loc = arr["red_locations"]/2428
    done = arr["dones"]

    dones.append(done)
    blue_obs.append(blue_ob)
    red_locs.append(red_loc)

blue_obs = np.concatenate(blue_obs)
red_locs = np.concatenate(red_locs)
dones = np.concatenate(dones)

print(blue_obs.shape, red_locs.shape, dones.shape)

np.savez(save_path,
            blue_observations = blue_obs,
            red_locations=red_locs, 
            dones=dones,
            agent_dict = arr["agent_dict"])