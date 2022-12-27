# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# path = "/nethome/sye40/PrisonerEscape/datasets/post_red_obs_fix/map_0_run_300_heuristic_eps_0.1.npz"
# title = "Heuristic State Occupancy"
# save_location = "evaluate/figs/heuristic_occupancy.png"

# path = "/nethome/sye40/PrisonerEscape/datasets/post_red_obs_fix/map_0_400_RRT_gnn_save.npz"
# title = "RRT* State Occupancy"
# save_location = "evaluate/figs/rrt_occupancy.png"

# path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations/train_vector.npz"
# title = "Random Locations (camera net) State Occupancy"
# save_location = "evaluate/figs/random_locs.png"

path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations_no_net/train_vector.npz"
title = "Random Locations (no net) State Occupancy"
save_location = "evaluate/figs/random_locs_no_net_1.png"

unknown = [(2077, 2151), (2170, 603), (37, 1293), (1890, 30), (1151, 2369), (356, 78), (1751, 1433), (1638, 1028), (1482, 387), (457, 1221)]
known = [(234, 2082), (1191, 950), (563, 750), (2314, 86), (1119, 1623), (1636, 2136), (602, 1781), (2276, 1007), (980, 118), (2258, 1598)]

a = np.load(path)
red_locs =  a["red_locations"]
x = red_locs[:, 0]*2428
y = red_locs[:, 1]*2428
fig, ax = plt.subplots(figsize =(10, 7))
# Creating plot
# plt.hexbin(x, y, bins=30)
# plt.scatter([500], [500], c='r', s=100)
plt.hist2d(x, y, bins=100)

plt.scatter(np.array(known)[:, 0], np.array(known)[:, 1], marker='X', c='r', s=100, label="Known Hideouts")
plt.scatter(np.array(unknown)[:, 0], np.array(unknown)[:, 1], marker='D', c='yellow', s=100, label="Unknown Hideouts")
ax.set_aspect('equal')
plt.tight_layout()
plt.legend()
plt.savefig(save_location)

# show plot
# plt.show()