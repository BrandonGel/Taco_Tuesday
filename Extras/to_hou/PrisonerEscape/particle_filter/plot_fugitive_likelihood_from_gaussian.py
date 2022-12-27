import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

def plot_gaussian_heatmap(mean, logstd, res=1):
    """
    Plot the 2D gaussian from mean and logstd
    :param mean: (np array) Mean of the 2d gaussian
    :param logstd: (np array) log(std) of the 2d gaussian
    :return:
        img: grid of values for the 2D gaussian heatmap
    """

    # Calculate covariance matrix from logstd. Assuming covariance as a diagonal matrix
    var = np.exp(logstd) ** 2
    cov = np.eye(2) * var

    k1 = multivariate_normal(mean=mean, cov=cov)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    # Since fugitive locations are normalized, both x, y \in {0, 1}
    xlim = (0, 1)
    ylim = (0, 1)

    # Taking Resolution as 1/10th of the env grid size
    xres = math.ceil(2428/res)
    yres = math.ceil(2428/res)

    x = np.linspace(xlim[0], xlim[1], xres)
    y = ylim[1] - np.linspace(ylim[0], ylim[1], yres)  # Y-axis reversed
    xx, yy = np.meshgrid(x, y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = k1.pdf(xxyy)

    # reshape and plot image
    img = zz.reshape((xres, yres))
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()

    return img

#
# mean = np.array([0.2867, 0.1330])
# logstd = np.array([-3.4674, -3.4296])
# #
# plot_gaussian_heatmap(mean, logstd)

