"""
File that holds the Bayesian Opt experiments for the 3D heart mesh
"""
import pandas as pd
import numpy as np
from data_analysis import get_heart_bounds, correlation_coef, graph_3d, get_index
from graph import narrow,corrplot3axes,trend,nearest,plot_exploration, graph_dist_over_axis, graph_cc_distribution, cube, gp_plot, gp_plot2,predicted_visited
from BayesOptLib.bayes_opt.bayesian_optimization import BayesianOptimization
from RandomSampler import RandomSampler
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
from prettytable import PrettyTable
from scipy.stats import wasserstein_distance


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

if __name__ == '__main__':
    # Read in ECGs and Coordinates
    ecgs = pd.read_csv("new_simu-data/Heart1/Heart1_SimuData_4000.csv", header=None).to_numpy()
    labels = pd.read_csv("new_simu-data/Heart1/Coord1_4000.csv", header=None).to_numpy() / 1000
    bounds = get_heart_bounds(labels) # getting bounds 
    tidx = np.random.randint(0, labels.shape[0])    # Pick out a sample to use as a target
    target, target_ecg = labels[tidx], ecgs[tidx]
    print("Target: ", target)
    init = 10 #initial random sample
    steps = 10 # total AL step
    af = "ucb" 
        
#     table=nearest(tidx,labels,ecgs,15)
#     x= PrettyTable()
#     x.field_names = ['x', 'y','z','Distance','Corr']
#     for row in table:
#         x.add_row(row)
#     print(x)    
    optimizer = BayesianOptimization(f=black_box,pbounds=bounds,random_state=None, real_set=labels)
    gp,X = optimizer.gpfit(init_points=init, n_iter=steps,  acq=af, kappa = 2.5,kappa_decay=0.75,kappa_decay_delay=2)
#     graph_cc_distribution(target_ecg,ecgs,labels)
#     gp_plot2(gp,labels)
    trend(target,optimizer.visited,optimizer.predicted)

#     plot_exploration(init,target,target_ecg,labels,ecgs,optimizer.visited)
    predicted_visited(init,target,target_ecg,labels,ecgs,optimizer.visited,optimizer.predicted)

