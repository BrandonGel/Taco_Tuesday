import numpy as np
import matplotlib.pyplot as plt
import os

# model_folder_path = 'logs/vector/baseline/20220617-2028'
# Fixed cameras using vector
# fixed_cams_baseline = [2.4852, 2.5112, 2.5054, 2.4109, 2.4282, 2.3961, 2.3686, 2.3289, 2.3134,
#         2.3481, 2.3230, 2.3033, 2.0323]

fixed_cams_baseline = [2.2598, 2.3444, 2.3117, 2.2975, 2.2830, 2.2622, 2.2437, 2.2181, 2.2014,
        2.1741, 2.1695, 2.1393, 2.0812]

# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/filtering/20220613-0028'
# Fixed cameras using GNN
# fixed_cams_gnn = [3.0703435, 3.0581698, 3.0823014, 3.1129918, 3.1118615, 3.1035755, 3.0168865,
#  3.057906,  2.9749818, 3.0282824, 2.9824426, 2.9855144, 2.9538584]
fixed_cams_gnn = [2.5263988971710205,2.516058921813965,2.516115427017212,2.5208775997161865,2.522369623184204,2.5242748260498047,2.518357276916504,2.5008060932159424,2.489193916320801,2.4823760986328125,2.469841480255127,2.3836238384246826,2.375403881072998]

# model_folder_path = "logs/gnn/filtering/20220618-0231"
# Random cams on random test set
rand_cams_gnn_on_fixed_cams = [2.2491044998168945,2.239176034927368,2.257249116897583,2.1872398853302,2.2249364852905273,2.239286422729492,2.2104313373565674,2.2142412662506104,2.2349958419799805,2.157381534576416,2.193288564682007,2.1972901821136475,2.1741623878479004]
rand_cams_gnn = [2.0730578899383545,2.031930923461914,2.0492210388183594,2.012028217315674,2.0193400382995605,1.9950172901153564,2.0065500736236572,1.9747785329818726,2.0092263221740723,1.9280071258544922,1.9709933996200562,1.9764113426208496,1.948766827583]

particle_filter = [1.5912216176429002, 1.4506082924099086, 1.354078310711413, 1.26455701028462,
                   1.1745304166415595, 1.0833315460960808, 0.949393156866075, 0.8576705371667752,
                   0.7186138462809187, 0.6095632447809848, 0.48322872834307373, 0.36718507421709734,
                   0.2461767405587872]

pf_rrt = [-3.4731810698816346, -3.7817059115555396, -3.9801599264247574, -4.293531998309871, -5.552689318770659, -6.82628508897337, -7.280548483105227, -7.749506290757518, -7.620393059020226,
-7.474444669291649, -7.48932098939193, -7.518469191196128, -8.361951565376097]

het_lstm_gnn_fixed_gnn = [2.286745548248291,2.2790145874023438,2.258436918258667,2.270951986312866,2.231358528137207,2.247321844100952,2.2425148487091064,2.2196104526519775,2.1231589317321777,2.1878769397735596,2.1717398166656494,2.161069869995117,2.1394221782684326]

lstm_front_gnn_random = [2.258619546890259,2.259122133255005,2.228590250015259,2.2273335456848145,2.195244789123535,2.1963493824005127,2.2085583209991455,2.184443712234497,2.186727523803711,2.1472997665405273,2.166353702545166,2.1115920543670654,2.1395742893218994]

lstm_front_gnn_fixed = [2.510695219039917,2.4773340225219727,2.488874912261963,2.4822418689727783,2.4623265266418457,2.4534504413604736,2.453538179397583,2.443192958831787,2.4194867610931396,2.4223735332489014,2.400930404663086,2.398042917251587,2.3792662620544434]

x = np.arange(0, 65, 5)

plt.figure()
# plot the data

# plt.plot(x, np.array(particle_filter),  marker="o", label='Particle Filter Heuristic')
plt.plot(x, np.array(pf_rrt),  marker="o", label='Particle Filter RRT')
plt.plot(x, np.array(fixed_cams_gnn),  marker="o", label='Homogeneous-GNN Fixed Cams')
plt.plot(x, np.array(lstm_front_gnn_fixed), marker="o", label='Decoupled LSTM-Heterogeneous Fixed Cams')
plt.plot(x, np.array(fixed_cams_baseline),  marker="o", label='RNN')
plt.plot(x, np.array(het_lstm_gnn_fixed_gnn), marker="o", label='Coupled Heterogeneous-LSTM Fixed Cams')
# plt.plot(x, np.array(rand_cams_gnn_on_fixed_cams), marker="o", label='Homogeneous-GNN Random Camera Tested on Fixed Cameras')

plt.title('Fixed Cameras')

# add legend
plt.legend()

# set y axis min to 0
# plt.ylim(bottom=0)
# plt.ylim(bottom=-9)
# plt.set_ylim([0])

#axes labels
plt.xlabel('Timesteps into the future')
plt.ylabel('Log-Likelihood')

# plt.savefig('evaluate/figs/random_cam_comparison.png')
plt.savefig('evaluate/figs/fixed_cam_comparison.png')


plt.figure()

##########################
plt.title('Random Cameras')
# add legend
plt.legend()

# set y axis min to 0
# plt.ylim(bottom=0)
# plt.ylim(bottom=0, top=2.5)

# plt.set_ylim([0])

#axes labels
plt.xlabel('Timesteps into the future')
plt.ylabel('Log-Likelihood')

# plt.plot(x, np.array(particle_filter),  marker="o", label='Particle Filter')
plt.plot(x, np.array(pf_rrt),  marker="o", label='Particle Filter RRT')
plt.plot(x, np.array(rand_cams_gnn),  marker="o", label='Homogeneous-GNN Random Cameras')
plt.plot(x, np.array(lstm_front_gnn_random), marker="o", label='Decoupled LSTM-Heterogeneous Random Cams')

plt.legend()
plt.savefig('evaluate/figs/random_cam_comparison.png')