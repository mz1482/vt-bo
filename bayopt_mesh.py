import pandas as pd
import numpy as np
from data_analysis import get_heart_bounds, correlation_coef, graph_3d, get_index
from graph import *
from BayesOptLib.bayes_opt.bayesian_optimization import BayesianOptimization
from RandomSampler import RandomSampler
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
# from prettytable import PrettyTable

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
    uvc = pd.read_csv("new_simu-data/UVC/Coord1_UVC_4000.csv", header=None).to_numpy()
    bounds = get_heart_bounds(labels) # getting bounds 
    tidx = np.random.randint(0, labels.shape[0])    # Pick out a sample to use as a target
    target, target_ecg,target_uvc = labels[tidx], ecgs[tidx], uvc[tidx]
    print("Target: ", target)
    init = 5 #initial random sample
    steps = 10 # total AL step
    af = "ucb" 
   
    optimizer = BayesianOptimization(f=black_box,pbounds=bounds,random_state=None, real_set=labels)
    gp,X = optimizer.gpfit(init_points=init, n_iter=steps,  acq=af, kappa = 1,kappa_decay=0.75,kappa_decay_delay=2)
#     print(len(optimizer.predicted))
#     heart(target,labels)
    graph_cc_distribution(target_ecg,ecgs,labels)
    gp_plot(gp,labels,target)
    

#     plot_exploration(init,target,target_ecg,labels,ecgs,optimizer.visited)
#     print(optimizer.predicted)
    predicted_visited(init,target,target_ecg,labels,ecgs,optimizer.visited,optimizer.predicted)
#     trend(target,optimizer.visited,optimizer.predicted)
    labels = uvc[:,0:3]
    bounds = get_heart_bounds(labels) # getting bounds 
    print("Target: ", target_uvc)

    optimizer = BayesianOptimization(f=black_box,pbounds=bounds,random_state=None, real_set=labels)
    gp,X = optimizer.gpfit(init_points=init, n_iter=steps,  acq=af, kappa = 1,kappa_decay=0.75,kappa_decay_delay=2)
#     heart(target,labels)
    graph_cc_distribution(target_ecg,ecgs,labels)
    gp_plot(gp,labels,target_uvc)
    

#     plot_exploration(init,target,target_ecg,labels,ecgs,optimizer.visited)
#     print(optimizer.predicted)
    predicted_visited(init,target_uvc,target_ecg,labels,ecgs,optimizer.visited,optimizer.predicted)
#     trend(target,optimizer.visited,optimizer.predicted)


