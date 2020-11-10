"""
@file full-pipeline.py
@author Ryan Missel

Handles the full pipeline of creating the patient datasets for the R15 data, by first getting the integrals of
the raw data, sampling each pacing site for one unique beat, and then splitting the data into their relevant patients

Requires a folder structure of:
r15/
-> r15Data.csv (raw data)
-> r15Coords.csv (cartesian coordinates)
-> r15Points.csv (output from population model on top-3 segment prediction)
-> r15Segs.csv (segment labels on a heart mesh)
"""
import pandas as pd
import numpy as np
import scipy.stats as sp
import os


""" 
Generating 12-Lead Integrals for all data points 
"""
# Dataset and resulting matrix
dataset = pd.read_csv("R15/r15Data.csv", header=None).to_numpy()

# Handles getting the AUCs for the new ECG data, for only the first 120ms
output_matrix = np.zeros([len(dataset), 12])
length = dataset.shape[1] // 12
print(length)
for i in range(len(dataset)):
    index = 0
    for j in range(12):
        sample = dataset[i][index:index + 30]
        auc = np.trapz(sample, dx=1)
        output_matrix[i][j] = auc
        index += length

print(output_matrix)

# Save to CSV file
np.savetxt("r15/auc.csv", output_matrix, delimiter=",")


""" 
Performing sampling on all of the unique pacing sites 
"""
dataset = pd.read_csv("r15/auc.csv", header=None).to_numpy()
raw = pd.read_csv("r15/r15Data.csv", header=None).to_numpy()
coords = pd.read_csv("r15/r15Coords.csv", header=None).to_numpy()
segments = pd.read_csv("r15/r15Segments.csv", header=None).to_numpy()
top3 = pd.read_csv("r15/r15Points.csv", header=None).to_numpy()


# Test to check for number of unique elements
current = [None, None, None]
unique = 0
for line in coords:
    if np.array_equal(line, current):
        continue
    else:
        unique += 1
        current = line

print(unique)

# Loop to generate random index values within select ranges
loop = 0

selected = []
current = coords[0]
start = 0
counter = 0

pop_seg_preds = []      # Used to get the majority vote for a pacing site and grabbing a beat from it
for line in coords:
    if np.array_equal(line, current) and loop != len(coords) - 1:
        pop_seg_preds.append(top3[counter][3])
        counter += 1
    else:
        print(sp.mode(pop_seg_preds))

        while True:
            rand = np.random.randint(start, counter)
            if top3[rand][3] == sp.mode(pop_seg_preds)[0]:
                break

        pop_seg_preds = []
        selected.append(rand)
        counter += 1
        current = line
        start = counter
    loop += 1

print(selected)

for i in selected:
    print("Index: ", i, "Val: ", coords[i])

print("Len of Selected: ", len(selected))


# Check to make sure values are all unique
current = [None, None, None]
unique = 0
for i in selected:
    line = coords[i]
    if np.array_equal(line, current):
        print("UNIQUES: ", current, line)
        continue
    else:
        unique += 1
        current = line

print("Calculated # Uniques: ", unique)
print("---")


# Create the dataset and label csvs
# selected = top3[:, 0]
print(selected)
examples = [dataset[i] for i in selected]
raw_data = [raw[i] for i in selected]
labels = [coords[i] for i in selected]
segs = [segments[i] for i in selected]
top3 = [top3[i] for i in selected]

# Save samples to folder
if not os.path.isdir("r15/sampled/"):
    os.mkdir("r15/sampled")
np.savetxt("r15/sampled/samp_auc.csv", examples, delimiter=",")
np.savetxt("r15/sampled/samp_raw.csv", raw_data, delimiter=",")
np.savetxt("r15/sampled/samp_coords.csv", labels, delimiter=",")
np.savetxt("r15/sampled/samp_segments.csv", segs, delimiter=",")
np.savetxt("r15/sampled/samp_top3.csv", top3, delimiter=",")


"""
Perform patient separation based on beats
"""


def get_intervals(ids):
    intervals = []
    start = 0
    counter = 0
    for i in np.unique(ids):
        print(i)
        while counter < len(ids) and ids[counter] == i:
            counter += 1

        intervals.append([start, counter])
        start = counter
    return intervals


# Making sure directories are made before population
if not os.path.isdir("r15/patient-datasets/"):
    os.mkdir("r15/patient-datasets")

for i in range(41, 57):
    if os.path.isdir("r15/patient-datasets/{}".format(i)):
        continue
    else:
        os.mkdir("r15/patient-datasets/{}".format(i))

""" Training """
ints = get_intervals(np.array(segs)[:, [1]])
print(ints)

patient_count = 41
for intv in ints:
    start, end = intv[0], intv[1]

    # Calculating pat-spec r15-padded data/AUCs
    rawd = np.array(raw_data[start:end])
    rawd = rawd[:, ~np.all(rawd == 0, axis=0)]
    final = np.zeros([len(rawd), 12])

    length = rawd.shape[1] // 12
    for i in range(len(rawd)):
        index = 0
        for j in range(12):
            sample = rawd[i][index:index + 120]
            auc = np.trapz(sample, dx=1)
            final[i][j] = auc
            index += length

    print(length, final.shape)

    # Grabbing the specific data in interval
    sgs = segs[start:end]
    cs = labels[start:end]
    tp = top3[start:end]

    # Saving to relevant location
    np.savetxt("r15/patient-datasets/{}/x.csv".format(patient_count), final, delimiter=",")
    np.savetxt("r15/patient-datasets/{}/y.csv".format(patient_count), sgs, delimiter=",")
    np.savetxt("r15/patient-datasets/{}/raw.csv".format(patient_count), rawd, delimiter=",")
    np.savetxt("r15/patient-datasets/{}/coord.csv".format(patient_count), cs, delimiter=",")
    np.savetxt("r15/patient-datasets/{}/top.csv".format(patient_count), tp, delimiter=",")
    patient_count += 1
