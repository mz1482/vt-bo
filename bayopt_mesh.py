"""
File that holds the Bayesian Opt experiments for the 3D heart mesh
"""
import pandas as pd
import numpy as np
from data_analysis import get_heart_bounds, correlation_coef, graph_3d
from graph import narrow,corrplot3axes,trend,nearest,plot_exploration, graph_dist_over_axis, graph_cc_distribution, cube
from BayesOptLib.bayes_opt.bayesian_optimization import BayesianOptimization
from RandomSampler import RandomSampler
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
from prettytable import PrettyTable
from scipy.stats import wasserstein_distance

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
    return correlation_coef(target_ecg, sample_ecg) - np.linalg.norm(sample_ecg-target_ecg)/7212


def optimize_point(labels,bounds):
    # Build the optimizer with the heart bounds
    optimizer = BayesianOptimization(
        f=black_box,
        pbounds=bounds,
        random_state=None, real_set=labels
    )

    # Maximize over x number of points
    optimizer.maximize(init_points=10, n_iter=5,  acq="ucb", kappa = 2)
    return optimizer


if __name__ == '__main__':
    # Read in ECGs and Coordinates
    ecgs = pd.read_csv("simu-data/Heart3_SimuData.csv", header=None).to_numpy()
    labels = pd.read_csv("simu-data/Heart3_XYZsub.csv", header=None).to_numpy() / 1000
    # Get bounds of the heart mesh
    bounds = get_heart_bounds(labels)
    # Pick out a sample to use as a target
    tidx = np.random.randint(0, labels.shape[0])
    target, target_ecg = labels[tidx], ecgs[tidx]
    print("Target: ", target)
#     cube(target_ecg,ecgs,labels)
    # Get plots of target CC distribution    
#     graph_dist_over_axis(target_ecg)
    # Optimize for target and plot path
    
    optimizer = optimize_point(labels,bounds)   
    
#     table=nearest(tidx,labels,ecgs,15)
#     x= PrettyTable()
#     x.field_names = ['x', 'y','z','Distance','Corr']
#     for row in table:
#         x.add_row(row)
#     print(x)
    
#     trend(target,optimizer.visited,optimizer.predicted)


    color_gradient = []
    # Loop through all points to get CC with that point
    for ecg, coord in zip(ecgs, labels):
        if np.array_equal(target_ecg, ecg):
            true = coord
            color_gradient.append(1)
            continue

        cc = correlation_coef(target_ecg, ecg)
        color_gradient.append(cc)

    plot_exploration(target,labels,optimizer.visited, color_gradient)
    
#     graph_cc_distribution(target_ecg,ecgs,labels)
    x,y,z,nn_cc = corrplot3axes(tidx,labels,ecgs,15)
    narrow(target,target_ecg,ecgs,labels,15)