import numpy as np

import matplotlib.pyplot as plt

data_path = f"/workspace/PrisonerEscape/datasets/test/small_subset/seed_{500}_known_44_unknown_33.npz"
data = np.load(data_path, allow_pickle=True)

red_locations = data['red_locations']

plt.figure()
for loc in red_locations:
    plt.scatter(loc[0], loc[1])

plt.savefig("visualize/test.png")