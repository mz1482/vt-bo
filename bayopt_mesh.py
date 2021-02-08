import pandas as pd
import numpy as np
from data_analysis import *
from graph import *
from BayesOptLib.bayes_opt.bayesian_optimization import BayesianOptimization
from RandomSampler import RandomSampler
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
import time
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
    ecgs = pd.read_csv("data/simu_data_4000/Heart1/Heart1_SimuData_4000_200Cropped.csv", header=None).to_numpy()
    labels = pd.read_csv("data/simu_data_4000/Heart1/Coord1_4000.csv", header=None).to_numpy() / 1000
    uvc = pd.read_csv("data/simu_data_4000/UVC/Coord1_UVC_4000.csv", header=None).to_numpy()
    #####only left ventricular#####
    idx=lv_rv(uvc,-1)
    uvc=uvc[idx]
    uvc = uvc[:,0:3]
    ecgs = ecgs[idx]
    labels = labels[idx]
    ##################################
    bounds = get_heart_bounds(labels) # getting bounds 
    init = 10 #initial random sample
    steps = 50 # total AL step
    af = "ucb" 
    for n in range(92,100):
        print("experiment:",n+1)
        tidx = np.random.randint(0, labels.shape[0])    # Pick out a sample to use as a target
        target, target_ecg,target_uvc = labels[tidx], ecgs[tidx], uvc[tidx]
        optimizer = BayesianOptimization(f=black_box,pbounds=bounds,random_state=None,cc_thres=0.99, real_set=labels,ecgs = ecgs,target_ecg = target_ecg)
        gp,lead = optimizer.gpfit(init_points=init, n_iter=steps,  acq=af, kappa = 2.5,kappa_decay=0.90,kappa_decay_delay=5,xi=0,lead_condition=True)
        #     heart(target,labels)
        best,max_cc, al,pass_lead,loc = result_metric_xyz(optimizer,target,labels,lead_condition = True)
        f= open("./exp_res/xyz_check/200ms_cc_95.txt","a")
        f.write(str(n)+","+str(target[0]) + "," + str(target[1]) + "," + str(target[2])+","+str(best[0]) + "," + str(best[1]) + "," + str(best[2])+","+str(max_cc)+","+str(al)+","+str(pass_lead)+","+str(loc) +"\n")
        f.close()
