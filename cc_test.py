from models.ccmodel import CCModel
from models.rsmodel import RSModel
from models.bomodel import BOModel
from models.confi import *
from models.utilfuncs import *
from graph import *
import pandas as pd
from tqdm import tqdm
import random

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')
# Reading in the ECGs and labels
aucs = pd.read_csv("data/simu_data_4000/Heart1/Heart1_AUCS.csv", header=None).to_numpy()
ecgs = pd.read_csv("data/simu_data_4000/Heart1/Heart1_SimuData_4000.csv", header=None).to_numpy()
labels = pd.read_csv("data/simu_data_4000/Heart1/Coord1_4000.csv", header=None).to_numpy() / 1000
total_cases = 0
alle, all_points = [], []
cc_euclids = [[] for _ in range(NUM_STEPS + 21)]                    # Random init CC arrays
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
cc_model = CCModel(leads=LEADS, steps=10, svr_c=SVR_C, cc=CC_THRES, cc_succ=CC_SUCC,
                   mm=mm_thres, samp_raw=ecgs, samp_coords=labels)
success_list=[]
exp = np.random.randint(0,3999,10)
for n in range(len(exp)):
    idx = exp[n]
    target = aucs[idx]
    target_coord = labels[idx]
    target_raw = ecgs[idx]

    # Drop the target from the training set
    if idx == 0:
        x, y, raw = aucs[idx + 1:, :], labels[idx + 1:], ecgs[idx + 1:, :]
    else:
        x = np.concatenate((aucs[:idx, :], aucs[idx + 1:, :]))
        y = np.concatenate((labels[:idx], labels[idx + 1:]))
        raw = np.concatenate((ecgs[:idx], ecgs[idx + 1:]))

#     x,y,raw = narrow_cc(target_coord,y,raw,x,30)


    random_x, random_y = get_random_dataset(x, y)

    cc_euclids, cc_preds, cc_sites, success, num_sites = cc_model.run(x, y, random_x, random_y,
                                                             target, target_coord, target_raw)
    print(success)
    success_list = np.append(success_list,success)
print(success_list)
# cc_model_graph(target_coord,target_raw,ecgs,labels,cc_sites,cc_preds)