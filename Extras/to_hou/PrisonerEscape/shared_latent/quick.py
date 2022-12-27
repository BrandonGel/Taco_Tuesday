from matplotlib import markers
from shared_latent.stats_nll import main
from filtering.models.evaluate_nll_seq2seq import evaluate_seq2seq_nll

blue_path = '/nethome/sye40/PrisonerEscape/logs/shared_latent/blue/20220502-1108/summary/'
blue_means, blue_stds = main(blue_path, 'model_best.pth', 60, render=True)

print(blue_means)
print(blue_stds)
