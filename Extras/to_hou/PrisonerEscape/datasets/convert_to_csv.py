import pandas as pd
import numpy as np

blue_obs_dict_sample = "datasets/ilrt_test/gnn_map_0_run_3_RRT/seed_1_known_44_unknown_33.npz"
data = np.load(blue_obs_dict_sample, allow_pickle=True)
blue_obs_dict = data['blue_obs_dict'].item()

seed = 91

# data_path = f"/nethome/mnatarajan30/codes/PrisonerEscape/datasets/train_same_new/gnn_map_0_run_300_RRT/seed_{seed}_known_44_unknown_33.npz "

# data_path = f"/nethome/mnatarajan30/codes/PrisonerEscape/datasets/train_same_new/gnn_map_0_run_300_RRT/seed_0_known_44_unknown_33.npz"
# data_path = f"/workspace/PrisonerEscape/datasets/random_start_location/gnn_map_0_run_100_RRT/seed_{seed}_known_17_unknown_85.npz"
data_path = f"/workspace/PrisonerEscape/datasets/arst/seed_{seed}_known_44_unknown_33.npz"
data = np.load(data_path, allow_pickle=True)

blue_observations = data['blue_observations']
red_locations = data['red_locations'] / 2428
# blue_obs_dict = data['blue_obs_dict'].item()

names = [""] * blue_observations.shape[1]

for key in blue_obs_dict:
    start_key, end_key = blue_obs_dict[key]
    print(start_key, end_key)
    if end_key - start_key == 2:
        names[start_key:end_key] = [key + "_x", key + "_y"]
    else:
        names[start_key:end_key] = [key] * (end_key - start_key) 

combined_observations = np.concatenate([blue_observations, red_locations], axis=1)
names.append("prisoner_loc_x")
names.append("prisoner_loc_y")

print(names)
df = pd.DataFrame(combined_observations, columns=names)


df.to_csv(f"datasets/ilrt_test/{seed}.csv")