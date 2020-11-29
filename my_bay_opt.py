import sys
sys.path.append('/home/mz1482/project/vt-bayesian-opt-bopt_debug/my_bayes_opt/')
import bo_new
import pandas as pd
import numpy as np
from data_analysis import get_heart_bounds, correlation_coef, graph_3d, get_index
from graph import narrow,corrplot3axes,trend,nearest,plot_exploration, graph_dist_over_axis, graph_cc_distribution, cube, gp_plot, gp_plot2,predicted_visited, init_gp_plot
from BayesOptLib.bayes_opt.bayesian_optimization import BayesianOptimization
from RandomSampler import RandomSampler
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer


def black_box(x, y, z):
    """
    Represents a black box function to maximize for CC of two ECGs given an XYZ coordinate
    :param x:
    :param y: coordinates of the prediction
    :param z:
    :return: CC of that point and the target
    """
    sample_ecg = ecgs[get_index(np.array([x, y, z]),labels)]
    return abs(correlation_coef(target_ecg, sample_ecg))

ecgs = pd.read_csv("new_simu-data/Heart1/Heart1_SimuData_4000.csv", header=None).to_numpy()
labels = pd.read_csv("new_simu-data/Heart1/Coord1_4000.csv", header=None).to_numpy() / 1000
# Get bounds of the heart mesh
bounds = get_heart_bounds(labels)
# Pick out a sample to use as a target
tidx = np.random.randint(0, labels.shape[0])
target, target_ecg = labels[tidx], ecgs[tidx]
# optimizer = optimize_point(labels,bounds)  
optimizer = bo_new.mybo(f=black_box,pbounds=bounds, real_set=labels)
gp,X = optimizer.gpfit(init_points=10, n_iter=0,  acq="ucb", kappa = 1)
print(X)
graph_cc_distribution(target_ecg,ecgs,labels)
init_gp_plot(10,gp,labels,X)