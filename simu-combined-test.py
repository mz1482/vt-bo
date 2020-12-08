"""
@file: simu-combined-test.py
@author: Ryan Missel
Combined script that handles running all of the different patient-specific models on the same random initialization
within the segments predicted by the population model. Runs it over every unique pacing site as the target site for a
number of repeated trials.
Prints out averaged metrics at the end for each model, e.g. number of successful runs, average error distance per time
step, etc.
"""
from models.ccmodel import CCModel
from models.rsmodel import RSModel
from models.bomodel import BOModel
from models.confi import *
from models.utilfuncs import *
import pandas as pd
from tqdm import tqdm
import random

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')


def get_seg_dataset(row, segs, data, coords):
    """
    Handles grabbing all of the points within the predicted segment and its neighbor
    """
    # Get the labels of the segments right next to the predicted segment
    counts = get_neighbors(row[0])

    # Getting converted segs from 16 segments to the 7 segment model we are using
    nums = []
    for num in counts:
        for seg in convert_1016(num):
            nums.append([seg, row[1]])

    # Grab all of the points within the predicted segment and its neighbors. If there aren't enough to initialize the
    # model with the NUM_POINTS_START param, return and skip it
    x, y = get_segment_dataset(nums, segs, data, coords)
    if x.shape[0] < NUM_POINTS_START + 1:
        return None, None

    # Sample from the segment dataset the number of starting points desired
    samples = random.sample(range(x.shape[0]), NUM_POINTS_START)
    train, labels = x[samples], y[samples]
    return train, labels


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


def print_model_stats(models, names):
    """
    Simple formatting function for the per step average accuracy of all the models 
    """
    for model, name in zip(models, names):
        print("Model {} ---".format(name))
        for i in range(NUM_STEPS):
            if len(model[i]) == 0:
                continue

            # Step # | Num Active Mean STD | Min Max
            print("Set len %2d | N: %5d %02.2f %02.2f | Min: %3d Max: %3d" %
                  (i + NUM_POINTS_START, len(model[i]), np.mean(model[i]), np.std(model[i]), np.min(model[i]), np.max(model[i])))
        print(" ")


def model_run(model, x, y, train, labels, target, target_coord, target_raw, successes, avg_sites, all_euclids):
    """
    Handles running a single runthrough of a given model on a target site
    Train and labels are the initialized sets to start with (either random or by segment prediction)
    """
    # Run the training loop for the model
    euclids, _, _, success, nsites = model.run(x, y, train, labels, target, target_coord, target_raw)

    # Add to arrays if successful and add the per steps error
    if success:
        successes.append(euclids[-1])
        avg_sites.append(nsites)
    for j in range(len(euclids)):
        all_euclids[j].append(euclids[j])
        # if j != 0:
        #     drop.append(euclids[j] - euclids[j - 1])
    return successes, avg_sites, all_euclids


def main():
    """
    The loop throughout this function is kind of cumbersome, but needed to make it easy to test all models on the
    same given dataset/initialization for all patients
    It loops through each patient a number of times for different initializations and tests each model's runtime on it,
    gathering run statistics and aggregating them into the arrays at the top here for printing at the end
    """
    # Arrays for all metrics
    total_cases = 0
    alle, all_points = [], []

    # bo_euclids = [[] for _ in range(NUM_STEPS + NUM_POINTS_START)]                     # Random init BO arrays
    # bo_successes, bo_avg_sites, bo_drop = [], [], []

    cc_euclids = [[] for _ in range(NUM_STEPS + 21)]                    # Random init CC arrays
    cc_successes, cc_avg_sites = [], []
    cc_drop = []

    rs_euclids = [[] for _ in range(NUM_STEPS + 21)]                     # Random init RS arrays
    rs_successes, rr_avg_sites = [], []
    rr_drop = []

    # Set mm threshold for finding nearest
    mm_thres = 200000

    # Reading in the ECGs and labels
    aucs = pd.read_csv("simu-data/Heart3_AUCS.csv", header=None).to_numpy()
    ecgs = pd.read_csv("simu-data/Heart3_SimuData.csv", header=None).to_numpy()
    labels = pd.read_csv("simu-data/Heart3_XYZsub.csv", header=None).to_numpy()[:, :3] / 1000

    # Initializing the models to test for patient
#     bo_model = BOModel(leads=LEADS, steps=NUM_STEPS, svr_c=SVR_C, cc=CC_THRES, cc_succ=CC_SUCC, mm=mm_thres, samp_raw=ecgs, samp_coords=labels)
    cc_model = CCModel(leads=LEADS, steps=NUM_STEPS, svr_c=SVR_C, cc=CC_THRES, cc_succ=CC_SUCC,
                       mm=mm_thres, samp_raw=ecgs, samp_coords=labels)
    rs_model = RSModel(steps=NUM_STEPS, svr_c=SVR_C, samp_raw=ecgs, samp_coords=labels, cc_succ=CC_SUCC)

    # Loop through each patient, performing a number of trials
    for target, target_coord, target_raw, idx in tqdm(zip(aucs, labels, ecgs, range(len(labels)))):
        if idx > 5:
            break

        # Drop the target from the training set
        if idx == 0:
            x, y, raw = aucs[idx + 1:, :], labels[idx + 1:], ecgs[idx + 1:, :]
        else:
            x = np.concatenate((aucs[:idx, :], aucs[idx + 1:, :]))
            y = np.concatenate((labels[:idx], labels[idx + 1:]))
            raw = np.concatenate((ecgs[:idx], ecgs[idx + 1:]))

        # Looping through every point to test, n number of times for variance in initialization
        for _ in range(NUM_TRIALS):
            # Get initial datasets for random and segment initializations
            random_x, random_y = get_random_dataset(x, y)

            # All model testing
            all_euclid, all_pred = build_target_model(SVR_C, x, y, target, target_coord)
            pred_check = get_closest(10000, all_pred, raw, y)[1]
            if pred_check is not None and check_cc_success(target_raw, pred_check) >= CC_SUCC:
                all_points.append(len(x))
                alle.append(all_euclid)

            # BO Model
#             bo_successes, bo_avg_sites, bo_euclids = model_run(bo_model, raw, y, random_x, random_y, target,
#                                                                target_coord, target_raw, bo_successes,
#                                                                bo_avg_sites, bo_euclids)

            # CCRI Model
            cc_successes, cc_avg_sites, cc_euclids = model_run(cc_model, x, y, random_x, random_y,
                                                                     target, target_coord, target_raw,
                                                                     cc_successes, cc_avg_sites, cc_euclids)

            # RSRI Model
            rs_successes, rr_avg_sites, rs_euclids = model_run(rs_model, x, y, random_x, random_y,
                                                                     target, target_coord, target_raw,
                                                                     rs_successes, rr_avg_sites, rs_euclids)

        total_cases += 1

    # Print the file names used
    print("File set used: ", DATA_PATH)

    # Metric section here
    print("--- Average Success errors ---")
    print("Total number of cases: %d"                            %  total_cases)
    print("All Points: %.2f +- %.2f for %d cases" % (np.mean(alle), np.std(alle), len(alle)))
    # print("Bayes: %.2f +- %.2f for %d cases"      % (np.mean(bo_successes), np.std(bo_successes), len(bo_successes)))
    print("CC RI: %.2f +- %.2f for %d cases"      % (np.mean(cc_successes), np.std(cc_successes), len(cc_successes)))
    print("RS RI: %.2f +- %.2f for %d cases"      % (np.mean(rs_successes), np.std(rs_successes), len(rs_successes)))

    print(" ")

    print("--- Average Num of Sites ---")
    print("All  : %.2f +- %.2f" % (np.mean(all_points), np.std(all_points)))
    # print("Bayes: %.2f +- %.2f" % (np.mean(bo_avg_sites), np.std(bo_avg_sites)))
    print("CCRI: %.2f +- %.2f"  % (np.mean(cc_avg_sites), np.std(cc_avg_sites)))
    print("RSRI: %.2f +- %.2f"  % (np.mean(rr_avg_sites), np.std(rr_avg_sites)))

    print(" ")

    print("--- Average Drop ---")
    # print("Bayes: %.2f +- %.2f" % (np.mean(bo_drop), np.std(bo_drop)))
    print("CCRI: %.2f +- %.2f"  % (np.mean(cc_drop), np.std(cc_drop)))
    print("RSRI: %.2f +- %.2f"  % (np.mean(rr_drop), np.std(rr_drop)))

    print(" ")

    print("--- Success Rates ---")
    print("CCRI: %.2f" % (len(cc_successes) / total_cases))
    print("RSRI: %.2f" % (len(rs_successes) / total_cases))

    print(" ")

    print("--- Per Step Averages ---")
    # print_model_stats([bo_euclids, cc_euclids, rs_euclids], ["Bayes", "CC RI", "RS RI"])
    print_model_stats([cc_euclids, rs_euclids], ["CC RI", "RS RI"])

if __name__ == '__main__':
    main()