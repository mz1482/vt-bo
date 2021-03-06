"""
@file bayopt_mesh
@author Ryan Missel

File that holds the Bayesian Opt experiments for the 3D heart mesh
"""
import pandas as pd
import numpy as np
from data_analysis import get_heart_bounds, correlation_coef, graph_3d, graph_cc_distribution, graph_dist_over_axis
from BayesOptLib.bayes_opt.bayesian_optimization import BayesianOptimization
from RandomSampler import RandomSampler
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
from prettytable import PrettyTable


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
    return correlation_coef(target_ecg, sample_ecg)


def optimize_point(labels):
    # Build the optimizer with the heart bounds
    optimizer = BayesianOptimization(
        f=black_box,
        pbounds=bounds,
        random_state=None, real_set=labels
    )

    # Maximize over x number of points
    optimizer.maximize(init_points=10, n_iter=10,  acq="ucb", kappa = 2)
    return optimizer




def plot_exploration(visited, color_gradient):
    """
    Handles plotting the predictions of the network over time
    :param visited:
    :return:
    """
    path = np.array(visited)
    color_gradient = np.array(color_gradient)
    # for i in range(len(path)):
    #     cur = np.array(path[:i])
    #     rest = np.delete(labels, np.where(np.isin(labels, cur)), axis=0)
    #
    #     fig = plt.figure(0)
    #     ax = fig.gca(projection='3d')
    #
    #     ax.scatter(xs=rest[:, 0], ys=rest[:, 1], zs=rest[:, 2], zdir='z', alpha=0.75, color='gray')
    #     ax.scatter(xs=cur[:, 0], ys=cur[:, 1], zs=cur[:, 2], zdir='z', color='blue')
    #     ax.scatter(xs=target[0], ys=target[1], zs=target[2], color='black')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #     plt.pause(1)
    #     fig.clear()


    # Plot final for viewing
    rest = np.delete(labels, np.where(np.isin(labels, path)), axis=0)
    color_gradient = np.delete(color_gradient, np.where(np.isin(labels, path)), axis=0)
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')

    ax.scatter(xs=rest[:, 0], ys=rest[:, 1], zs=rest[:, 2], zdir='z', alpha=0.75, c=color_gradient, cmap = plt.cm.Reds)
    ax.scatter(xs=path[:, 0], ys=path[:, 1], zs=path[:, 2], zdir='z', color='blue')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color = 'blue')

    m = path
    for i in range(len(m)):
        ax.text(m[i, 0], m[i, 1], m[i, 2], '%s' % (str(i)), size=10, zorder=1, color='k')
    ax.scatter(xs=target[0], ys=target[1], zs=target[2], color='black', s = 100)
    plt.show()


if __name__ == '__main__':
    # Read in ECGs and Coordinates
    ecgs = pd.read_csv("simu-data/Heart3_SimuData.csv", header=None).to_numpy()
    labels = pd.read_csv("simu-data/Heart3_XYZsub.csv", header=None).to_numpy() / 1000
    print(ecgs.shape)
    # Get bounds of the heart mesh
    bounds = get_heart_bounds()

    # Pick out a sample to use as a target
    tidx = np.random.randint(0, labels.shape[0])
    #tidx = 1
    target, target_ecg = labels[tidx], ecgs[tidx]
    print("Target: ", target)

    # Remove target from labels
    #labels = np.delete(labels, np.where(np.isin(labels, target)), axis=0)

    # Get plots of target CC distribution
    
    # graph_dist_over_axis(target_ecg)

    # Optimize for target and plot path
    optimizer = optimize_point(labels)




    # Test Random Sampler
    # rs = RandomSampler(ecgs, labels)
    # points, cc = rs.optimize(target, target_ecg, ecgs.shape[0] - 1)
    # print("RS # Points/CC: ", points, cc)
    


    
    graph_cc_distribution(target_ecg)
    color_gradient = []
    # Loop through all points to get CC with that point
    for ecg, coord in zip(ecgs, labels):
        if np.array_equal(target_ecg, ecg):
            true = coord
            color_gradient.append(1)
            continue

        cc = correlation_coef(target_ecg, ecg)
        color_gradient.append(cc)

    plot_exploration(optimizer.visited, color_gradient)