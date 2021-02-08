from models.ccmodel import CCModel
from models.rsmodel import RSModel
from models.bomodel import BOModel
from models.confi import *
from models.utilfuncs import *
from graph import *
from data_analysis import *
import pandas as pd
from tqdm import tqdm
import random

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')
# Reading in the ECGs and labels
aucs = pd.read_csv("data/simu_data_4000/Heart1/Heart1_AUCS_4000.csv", header=None).to_numpy()
ecgs = pd.read_csv("data/simu_data_4000/Heart1/Heart1_SimuData_4000_200Cropped.csv", header=None).to_numpy()
labels = pd.read_csv("data/simu_data_4000/Heart1/Coord1_4000.csv", header=None).to_numpy() / 1000
uvc = pd.read_csv("data/simu_data_4000/UVC/Coord1_UVC_4000.csv", header=None).to_numpy()
###taking only left ventricular#####
idx=lv_rv(uvc,-1)
uvc_lv=uvc[idx]
uvc_lv = uvc_lv[:,0:3]
ecgs = ecgs[idx]
labels = labels[idx]
aucs = aucs[idx]
##################
total_cases = 0
alle, all_points = [], []
cc_euclids = []                # Random init CC arrays
cc_successes, cc_avg_sites = [], []
cc_drop = []
# Set mm threshold for finding nearest
mm_thres = 200000
def get_random_dataset(data, coords):
    """
    Handles getting the random dataset for the randomly initialized models
    :param target_coord: target coordinate
    :param data: full dataset for a patient
    :param coords: full labels for a patient
    :return: x, y of size 4
    """
    indices = np.random.choice(range(0, data.shape[0]), NUM_POINTS_START, replace=False)
    return data[indices], coords[indices]
def model_run(model, x, y, train, labels, target, target_coord, target_raw, successes, avg_sites, all_euclids):
    """
    Handles running a single runthrough of a given model on a target site
    Train and labels are the initialized sets to start with (either random or by segment prediction)
    """
    # Run the training loop for the model
    euclids, preds, sites, success, nsites = model.run(x, y, train, labels, target, target_coord, target_raw)
    sites = np.asarray(sites)

    # Add to arrays if successful and add the per steps error
    if success:
        successes.append(euclids[-1])
        avg_sites.append(nsites)
    for j in range(len(euclids)):
        all_euclids[j].append(euclids[j])
        # if j != 0:
        #     drop.append(euclids[j] - euclids[j - 1])
    return successes, avg_sites, all_euclids, sites

cc_model = CCModel(leads=LEADS, steps=NUM_STEPS, svr_c=SVR_C, cc=CC_THRES, cc_succ=CC_SUCC,
                   mm=mm_thres, samp_raw=ecgs, samp_coords=uvc_lv)
exp = np.random.randint(0,len(labels),200)
for n in range(len(exp)):
    print("experiemnt:",n+1)
    idx = exp[n]
    target = aucs[idx]
#     target_coord = labels[idx]
    target_coord = uvc_lv[idx]
    target_raw = ecgs[idx]

    # Drop the target from the training set
    if idx == 0:
        x, y, raw = aucs[idx + 1:, :], uvc_lv[idx + 1:], ecgs[idx + 1:, :]
    else:
        x = np.concatenate((aucs[:idx, :], aucs[idx + 1:, :]))
        y = np.concatenate((uvc_lv[:idx], uvc_lv[idx + 1:]))
        raw = np.concatenate((ecgs[:idx], ecgs[idx + 1:]))


    random_x, random_y = get_random_dataset(x, y)

    cc_euclids, cc_preds, cc_sites, success, num_sites,no_of_lead = cc_model.run(x, y, random_x, random_y,
                                                             target, target_coord, target_raw)
    f= open("./exp_res/exp_200ms/cc_model_uvc_200.txt","a")
    f.write(str(n)+","+str(target_coord[0]) + "," + str(target_coord[1]) + "," + str(target_coord[2])+","+str(num_sites)+","+str(no_of_lead) +"\n")
    f.close()

# cc_model_graph(target_coord,target_raw,ecgs,labels,cc_sites,cc_preds)