"""
@file auc_script.py
@author Ryan Missel
Script to generate time interval integration values for R15 ECG data
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# User input for parameters
# dataset_name = input("Enter dataset file: ")
csv_name = input("Enter csv file to save: ")

# Dataset and resulting matrix
dataset = pd.read_csv("new_simu-data/Heart1/Heart1_SimuData_4000.csv", header=None).to_numpy()


def new_twelve_lead():
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

    return output_matrix


# Choose which function to use here
output_matrix = new_twelve_lead()
print(output_matrix)

# Save to CSV file
np.savetxt(csv_name, output_matrix, delimiter=",")