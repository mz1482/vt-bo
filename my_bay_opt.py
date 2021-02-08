import sys
sys.path.append('/home/mz1482/project/vt-bayesian-opt-bopt_debug/my_bayes_opt/')
import bo_new
import pandas as pd
import numpy as np
from data_analysis import *
from graph import *
# from my_bayes_opt import bo_new
from RandomSampler import RandomSampler
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
# from prettytable import PrettyTable
import plotly.graph_objects as go

ecgs = pd.read_csv("data/simu_data_4000/Heart1/Heart1_SimuData_4000.csv", header=None).to_numpy()
labels = pd.read_csv("data/simu_data_4000/Heart1/Coord1_4000.csv", header=None).to_numpy() / 1000
uvc = pd.read_csv("data/simu_data_4000/UVC/Coord1_UVC_4000.csv", header=None).to_numpy()

idx=lv_rv(uvc,-1)
uvc_lv=uvc[idx]
uvc_lv = uvc_lv[:,0:3]
ecgs_lv = ecgs[idx]
labels_lv = labels[idx]


def black_box_xyz(x, y, z):
    """
    Represents a black box function to maximize for CC of two ECGs given an XYZ coordinate
    :param x:
    :param y: coordinates of the prediction
    :param z:
    :return: CC of that point and the target
    """
    sample_ecg = ecgs[get_index(np.array([x, y, z]),labels)]
    return abs(correlation_coef(target_ecg, sample_ecg))


def black_box_uvc(x, y, z):
    """
    Represents a black box function to maximize for CC of two ECGs given an XYZ coordinate
    :param x:
    :param y: coordinates of the prediction
    :param z:
    :return: CC of that point and the target
    """
    sample_ecg = ecgs_lv[get_index(np.array([x, y, z]),uvc_lv)]
    return abs(correlation_coef(target_ecg, sample_ecg))

if __name__ == '__main__':
    init = 5 #initial random sample
    steps = 15 # total AL step
    af = "ucb" 
    bounds = get_heart_bounds(uvc_lv)
    total_error,total_step,cc = [],[],[]
    for n in range(1):
        print("experiment:",n+1)
        tidx = np.random.randint(0, uvc_lv.shape[0])    # Pick out a sample to use as a target
        target_xyz, target_ecg,target_uvc = labels_lv[tidx], ecgs_lv[tidx], uvc_lv[tidx]
        print("target location",target_xyz) 
        optimizer = bo_new.mybo(f=black_box_uvc,pbounds=bounds, real_set=uvc_lv)
        gp,X,rs,predicted = optimizer.gpfit(init_points=init, n_iter=steps,  acq="ucb", kappa = .75,kappa_decay=0.75,kappa_decay_delay=2)
# print(optimizer.predicted)
        graph_cc_distribution(target_ecg,ecgs,labels)
