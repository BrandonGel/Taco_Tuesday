from matplotlib import markers
from shared_latent.stats_nll import main
from filtering.models.evaluate_nll_seq2seq import evaluate_seq2seq_nll

prediction_length = 20

red_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/red/20220501-1425/summary/'
red_means, red_stds = main(red_path, prediction_length, render=True)

blue_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/blue/20220502-1108/summary/'
blue_means, blue_stds = main(blue_path, prediction_length, render=True)

model_path = '/nethome/sye40/PrisonerEscape/logs/filtering/normal/seq2seq_no_attention/adaptive-std/new_terrain/maps_0/20220502-2318/filter_policy_epoch_2.pth'
baseline_filtering = evaluate_seq2seq_nll(model_path, prediction_length)[-prediction_length:]

baseline_prediction = [3.933872493285105, 3.8429138854858667, 3.788228206879869, 3.708333307860406, 3.6160197109850443, 3.5104947272023157, 3.43867673344499, 3.3728456578940498, 3.2980110180133693, 3.2411462602895313, 3.1707240980995186, 3.111618413073326, 3.0549971292222327, 3.005897662214985, 2.957142498900415, 2.907817458018953, 2.8477324345890245, 2.7948278595338025, 2.7407521075550285, 2.6998837135458893]
print(red_means, blue_means)

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(blue_means.shape[0])
fig, ax = plt.subplots()

# plot lines
plt.plot(x, blue_means, label = "RSSM Filtering", color = "blue", marker='o')
plt.plot(x, red_means, label = "RSSM Prediction", color = "red", marker='o')
plt.plot(x, baseline_filtering, label = "Baseline Seq2Seq Filtering", color = "aqua", marker='o')
plt.plot(x, baseline_prediction, label = "Single Step Prediction", color = "maroon", marker='o')

#set ylim
plt.ylim(0, 4)

#add legend
plt.legend()

# ax.errorbar(x, mean,
#             # yerr=std/2,
#             fmt='-o')

ax.set_xlabel('Timesteps into the future')
ax.set_ylabel('Log-likelihood')
ax.set_title('Log-Likelihood with RSSM')
# save the figure
plt.savefig("nll_shared_latent.png")