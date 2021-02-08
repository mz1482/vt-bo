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
matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
# from prettytable import PrettyTable
import plotly.graph_objects as go

ecgs = pd.read_csv("data/simu_data_4000/Heart1/Heart1_SimuData_4000_200Cropped.csv", header=None).to_numpy()
labels = pd.read_csv("data/simu_data_4000/Heart1/Coord1_4000.csv", header=None).to_numpy() / 1000
uvc = pd.read_csv("data/simu_data_4000/UVC/Coord1_UVC_4000.csv", header=None).to_numpy()

idx=lv_rv(uvc,-1)
uvc_lv=uvc[idx]
uvc_lv = uvc_lv[:,0:3]
ecgs_lv = ecgs[idx]
labels_lv = labels[idx]

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

def black_box_uvc_12_lead(x, y, z):
    """
    Represents a black box function to maximize for CC of two ECGs given an XYZ coordinate
    :param x:
    :param y: coordinates of the prediction
    :param z:
    :return: CC of that point and the target
    """
    sample_ecg = ecgs_lv[get_index(np.array([x, y, z]),uvc_lv)]
    sample_ecg = np.reshape(sample_ecg, [12, -1])
    nums = 0
    for i in range(12):
#         ccs = np.corrcoef(sample_ecg[i], target_ecg.reshape(12,-1)[i])[0, 1]
        ccs = abs(correlation_coef(target_ecg.reshape(12,-1)[i], sample_ecg[i]))
        if ccs > .9:
            nums += 1

#     return True if nums == 12 else False 
    return nums

if __name__ == '__main__':
    init = 10 #initial random sample
    steps = 50 # total AL step
    af = "ucb" 
    bounds = get_heart_bounds(uvc_lv)
    total_error,total_step,cc = [],[],[]
    for n in range(200):
        print("experiment:",n+1)
        tidx = np.random.randint(0, uvc_lv.shape[0])    # Pick out a sample to use as a target
        target_xyz, target_ecg,target_uvc = labels_lv[tidx], ecgs_lv[tidx], uvc_lv[tidx]
        print("target location",target_uvc)
        optimizer = BayesianOptimization(f=black_box_uvc,pbounds=bounds,random_state=None, cc_thres=0.95,real_set=uvc_lv,ecgs = ecgs_lv,target_ecg = target_ecg)
        gp,lead = optimizer.gpfit(init_points=init, n_iter=steps,  acq=af, kappa = 2.5,kappa_decay=0.90,kappa_decay_delay=5,xi=0,lead_condition=False)
        best,max_cc, al,pass_lead,loc=result_metric_uvc(optimizer,target_xyz,target_ecg,uvc_lv,labels_lv,lead_condition = False)
        f= open("./exp_res/exp_200ms/uvc_lv_max_cc_200.txt","a")
        f.write(str(n)+","+str(target_uvc[0]) + "," + str(target_uvc[1]) + "," + str(target_uvc[2])+","+str(best[0]) + "," + str(best[1]) + "," + str(best[2])+","+str(max_cc)+","+str(al)+","+str(pass_lead)+","+str(loc) +"\n")
        f.close()

