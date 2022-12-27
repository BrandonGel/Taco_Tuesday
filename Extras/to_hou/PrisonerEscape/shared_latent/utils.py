import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math
import yaml
import argparse
import os
import cv2

def get_configs():
    """
    Parse command line arguments and return the resulting args namespace
    """
    parser = argparse.ArgumentParser("Train Filtering Modules")
    parser.add_argument("--config", type=str, required=True, help="Path to .yaml config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded, args.config

def save_video(ims, filename, fps=30.0):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

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

    return img

def plot_mog_heatmap(mean, std, pi, res=1):
    """
    Plot the 2D gaussian for mixture of gaussians
    :param mean: (np array) Mean of the 2d gaussian, Num Gaussian Mixtures x 2
    :param std: (np array) log(std) of the 2d gaussian Num Gaussian Mixtures x 2
    :param pi: (np array) Probability of each gaussian, [Num Gaussian Mixtures]
    :return:
        grid: grid of values for the 2D gaussian heatmap
    """
    # create a grid of (x,y) coordinates at which to evaluate the kernels
    # Since fugitive locations are normalized, both x, y \in {0, 1}
    xlim = (0, 1)
    ylim = (0, 1)

    # Taking Resolution as 1/10th of the env grid size
    xres = math.ceil(2428/10)
    yres = math.ceil(2428/10)

    x = np.linspace(xlim[0], xlim[1], xres)
    y = ylim[1] - np.linspace(ylim[0], ylim[1], yres)  # Y-axis reversed
    xx, yy = np.meshgrid(x, y)

    z_accum = np.zeros(xres*yres)
    # print("plotting map")
    # print(mean, std)
    for i in range(mean.shape[0]):

        mu = mean[i]
        s = std[i]

        # Calculate covariance matrix from logstd. Assuming covariance as a diagonal matrix
        var = s ** 2
        var = np.clip(var, 0.00001, 20) 
        print(var)
        cov = np.eye(2) * var

        k1 = multivariate_normal(mean=mu, cov=cov)

        # evaluate kernels at grid points
        xxyy = np.c_[xx.ravel(), yy.ravel()]
        zz = k1.pdf(xxyy)

        z_accum += pi[i] * zz

    # reshape and plot image
    grid = z_accum.reshape((xres, yres))
    return grid