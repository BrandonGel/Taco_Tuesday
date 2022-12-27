import numpy as np

import matplotlib.pyplot as plt

x = np.arange(50, 301, 50)

filtering = [-1.194, -1.644, -2.544, -3.042, -3.264, -3.292]
averaged_prediction = [-43.62, -49.42, -50.82, -54.38, -54.62, -56.29]

plt.figure()
# plot the data
plt.plot(x, -np.array(filtering), label='Filtering')
plt.title('Filtering Data Volume Comparisons')

#axes labels
plt.xlabel('Data Volume (Number of Simulation Runs)')
plt.ylabel('Log-Likelihood')

plt.savefig('evaluate/filtering.png')


# import glob
# npzes_path = "/nethome/sye40/PrisonerClean/datasets"
# npzes_path = 

# get all the npz files
# npz_files = glob.glob(npzes_path + "/*.npz")
# print(npz_files)

################ filtering plot
plt.figure()

data_volume_samples = []
for i in x:
    f = f"/nethome/sye40/PrisonerClean/datasets/map_0_run_{i}_eps_0.1_norm.npz"
    file = np.load(f, allow_pickle=True)
    data_volume_samples.append(len(file["dones"]))

plt.plot(data_volume_samples, -np.array(filtering), label='Filtering')
plt.title('Filtering Data Volume Comparisons')

#axes labels
plt.xlabel('Data Volume (Number of Samples)')
plt.ylabel('Log-Likelihood')
plt.ylim(bottom=0)

plt.savefig('evaluate/filtering_samples.png')


######### Prediction Graph
plt.figure()
plt.plot(data_volume_samples, -np.array(averaged_prediction)/12, label='Prediction')
plt.title('Prediction Data Volume Comparisons')

# set y axis min to 0
plt.ylim(bottom=0)

#axes labels
plt.xlabel('Data Volume (Number of Samples)')
plt.ylabel('Log-Likelihood')

plt.savefig('evaluate/prediction_samples.png')


######### combined
plt.figure()
plt.plot(data_volume_samples, -np.array(filtering), label='Filtering', color='blue')
plt.plot(data_volume_samples, -np.array(averaged_prediction)/12, label='Prediction', color='red')
plt.title('Prediction Data Volume Comparisons')

# add legend
plt.legend()

# set y axis min to 0
plt.ylim(bottom=0)

#axes labels
plt.xlabel('Data Volume (Number of Samples)')
plt.ylabel('Log-Likelihood')

plt.savefig('evaluate/combined.png')

