from datasets.multi_head_dataset_old import MultiHeadDataset
from datasets.dataset_old import RedBlueSequence
import numpy as np

# Ensure the output that we're getting from both datasets is the same
np_file = np.load("/nethome/sye40/PrisonerEscape/datasets/seed_corrected/map_0_run_100_eps_0_norm.npz", allow_pickle=True)
seq_len = 4
future_step = 10
bad_dataset = MultiHeadDataset(
    np_file["red_observations"], 
    np_file["blue_observations"], 
    np_file["red_locations"], 
    np_file["dones"], 
    seq_len, 1, future_step, "red")

good_dataset = RedBlueSequence(
    np_file["red_observations"], 
    np_file["blue_observations"], 
    np_file["red_locations"], 
    np_file["dones"], 
    seq_len, future_step, "red")

print(bad_dataset.done_locations)
idx = 27385
print(bad_dataset[idx][1], good_dataset[idx][1])

for i in range(len(np_file["red_observations"])):
    b = bad_dataset[i][1][-2:]
    a = good_dataset[i][1]
    if not np.array_equal(b, a):
        print("something wrong")
    