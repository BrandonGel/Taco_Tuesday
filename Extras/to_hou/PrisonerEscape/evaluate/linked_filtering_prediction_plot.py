import numpy as np
import matplotlib.pyplot as plt
import os

def particle_filter():
    data_all = [[], [], [], [], [], []]
    path = "/nethome/sye40/PrisonerEscape/temp/prisoner-prediction-epsilon-04"
    for i in range(10, 20):
        folder_path = os.path.join(path, str(i), f"{i}_stats.npz")
        data = np.load(folder_path, allow_pickle=True)
        for j in range(6):
            data_all[j].extend(data["nls"][j])

    x = (np.arange(6) + 1) * 10
    y = [-np.mean(data_all[i]) for i in range(6)]
    return x, y

# These were collected with eps = 0.1
# gaussian_mixture_in_middle_single_out = [3.7424, 3.8162, 3.8622, 3.8782, 3.8614, 3.8147, 3.7420, 3.6482, 3.5380,
#         3.4168, 3.2908, 3.1634, 3.0383]

# gaussian_mixture_in_middle_hidden_connected_single_out = [3.6174, 3.5747, 3.4915, 3.4160, 3.3135, 3.2441, 3.1568, 3.0785, 2.9903,
#         2.9024, 2.8158, 2.7430, 2.6605]

# blue_obs_in_single_gaussian_out = [3.1426, 3.1076, 3.0647, 3.0004, 2.9270, 2.8418, 2.7641, 2.6864, 2.6034,
#         2.5261, 2.4494, 2.3759, 2.3018]

# prediction_obs = [9.7733, 7.9770, 6.8855, 6.2316, 5.6782, 5.2838, 4.9311, 4.6227, 4.3669,
#         4.1512, 3.9435, 3.7669, 3.5906]

# blue_obs_in_mixture_out = [3.9839, 3.9722, 3.9469, 3.9066, 3.8534, 3.8134, 3.7609, 3.7132, 3.6698,
#         3.6052, 3.5253, 3.4437, 3.3483]

gaussian_mixture_in_middle_single_out = [1.7645, 1.7290, 1.6890, 1.6450, 1.6039, 1.5589, 1.5138, 1.4654, 1.4221,
        1.3775, 1.3324, 1.2875, 1.2171]
blue_obs_in_mixture_out = [2.4852, 2.5112, 2.5054, 2.4109, 2.4282, 2.3961, 2.3686, 2.3289, 2.3134,
        2.3481, 2.3230, 2.3033, 2.0323]
linked_filtering_prediction = [-89.3851, -31.9798, -15.3396,  -8.8868,  -6.1001,  -4.5767,  -3.6297,
        -3.0807,  -2.6568,  -2.4305,  -2.2463,  -2.1382]

x = np.arange(0, 65, 5)
x2 = np.arange(5, 65, 5)

# p_x, p_y = particle_filter()

p_y = [1.5912216176429002, 1.4506082924099086, 1.354078310711413, 1.26455701028462,
                   1.1745304166415595, 1.0833315460960808, 0.949393156866075, 0.8576705371667752,
                   0.7186138462809187, 0.6095632447809848, 0.48322872834307373, 0.36718507421709734,
                   0.2461767405587872]

plt.figure()
# plot the data
plt.plot(x2, np.array(linked_filtering_prediction), marker='o', label='1a: Linked Filtering and Prediction', color='blue')
plt.plot(x, np.array(gaussian_mixture_in_middle_single_out), marker = 'o', label='1b: Separate Filtering and Prediction Modules', color='orange')
plt.plot(x, np.array(blue_obs_in_mixture_out), marker='o', label='2: Fully End-to-End Prediction', color='green')
plt.plot(x, np.array(p_y), marker='o', label='Particle Filter', color='red')

# plt.plot(x, np.array(gaussian_mixture_in_middle_hidden_connected_single_out), label='2b: Blue Obs In, Mixture Middle, Hidden Connected, Single Out')
# plt.plot(x, np.array(blue_obs_in_single_gaussian_out), label='Blue Obs In, Single Out')
# plt.plot(p_x, p_y, label='Red Obs In, Particle Filter')
# plt.plot(x, np.array(prediction_obs), label='Red obs in, Single Out')


plt.title('Prediction with Blue Observations')

# add legend
plt.legend()

# set y axis min to 0
# plt.ylim(bottom=0)

#axes labels
plt.xlabel('Timesteps into the future')
plt.ylabel('Log-Likelihood')

plt.savefig('evaluate/figs/plot_comparison_2.png')