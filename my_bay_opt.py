import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('/home/mz1482/project/vt-bayesian-opt-bopt_debug/my_bayes_opt/')
import bo_new
import math
from numpy.linalg import multi_dot
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from data_analysis import get_heart_bounds, correlation_coef, graph_3d
from graph import narrow,corrplot3axes,trend,nearest,plot_exploration, graph_dist_over_axis, graph_cc_distribution, cube,gp_plot
# from BayesOptLib.bayes_opt.bayesian_optimization import BayesianOptimization
# from RandomSampler import RandomSampler
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
from prettytable import PrettyTable
from scipy.stats import wasserstein_distance
from matplotlib import cm

def get_index(label):
    """
    Gets the idx of a label in the labels array
    :param label: label to check for
    :return: idx
    """
    idx = 0
    for coord in labels:
        if np.array_equal(label, coord):
            break
        idx += 1
    return idx

def black_box(x, y, z):
    """
    Represents a black box function to maximize for CC of two ECGs given an XYZ coordinate
    :param x:
    :param y: coordinates of the prediction
    :param z:
    :return: CC of that point and the target
    """
    sample_ecg = ecgs[get_index(np.array([x, y, z]))]
    return abs(correlation_coef(target_ecg, sample_ecg))

ecgs = pd.read_csv("simu-data/Heart3_SimuData.csv", header=None).to_numpy()
labels = pd.read_csv("simu-data/Heart3_XYZsub.csv", header=None).to_numpy() / 1000
# Get bounds of the heart mesh
bounds = get_heart_bounds(labels)
# Pick out a sample to use as a target
tidx = np.random.randint(0, labels.shape[0])
target, target_ecg = labels[tidx], ecgs[tidx]
# optimizer = optimize_point(labels,bounds)  
optimizer = bo_new.mybo(f=black_box,pbounds=bounds, real_set=labels)
gp = optimizer.maximize(init_points=10, n_iter=15,  acq="ucb", kappa = 2)
gp_plot(gp)

# color_gradient = []
# # Loop through all points to get CC with that point
# for ecg, coord in zip(ecgs, labels):
#     if np.array_equal(target_ecg, ecg):
#         true = coord
#         color_gradient.append(1)
#         continue

#     cc = correlation_coef(target_ecg, ecg)
#     color_gradient.append(cc)

# plot_exploration(target,labels,optimizer.X, color_gradient)

#     graph_cc_distribution(target_ecg,ecgs,labels)
# x,y,z,nn_cc = corrplot3axes(tidx,labels,ecgs,15)
# narrow(target,target_ecg,ecgs,labels,15)