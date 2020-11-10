"""
@file bomodel.py
@author Ryan Missel

Holds the model for the Bayesian Optimization algorithm
"""
from models.utilfuncs import *
import pandas as pd
import numpy as np
from BayesOptLib.bayes_opt.bayesian_optimization import BayesianOptimization

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')


class BOModel:
    def __init__(self, leads=None, svr_c=5, steps=20, mm=15, cc=.75, cc_succ=.9, samp_coords=None, samp_raw=None):
        # Which leads to use
        self.leads = leads if leads is not None else [i for i in range(12)]

        # Hyper params
        self.cc_thres = cc
        self.cc_succ = cc_succ
        self.mm_thres = mm
        self.num_steps = steps
        self.svr_c = svr_c

        # Sampled data to pull from
        self.samp_coords = samp_coords
        self.samp_raw = samp_raw

        # Given dataset for a patient
        self.target_ecg = None
        self.ecgs = None
        self.coords = None

    def euclidean_distance(self, one, two):
        """ Computes the euclidean distance between two coordinates """
        return np.sqrt(np.sum((one - two) ** 2))

    def get_index(self, label):
        """
        Gets the idx of a label in the labels array
        :param label: label to check for
        :return: idx
        """
        idx = 0
        for coord in self.coords:
            if np.array_equal(label, coord):
                break
            idx += 1
        return idx

    def black_box(self, x, y, z):
        """
        Represents a black box function to maximize for CC of two ECGs given an XYZ coordinate
        :param x:
        :param y: coordinates of the prediction
        :param z:
        :return: CC of that point and the target
        """
        sample_ecg = self.ecgs[self.get_index(np.array([x, y, z]))]
        return np.corrcoef(self.target_ecg, sample_ecg)[0, 1]

    def optimize_point(self, labels, given_set):
        # Build the optimizer with the heart bounds
        optimizer = BayesianOptimization(
            f=self.black_box,
            pbounds=self.bounds(labels),
            random_state=None, real_set=labels
        )

        # Maximize over x number of points
        success = optimizer.maximize(init_points=4, given_set=given_set, n_iter=16, kappa=1.5)
        return optimizer, success

    def bounds(self, labels):
        """
        Handles getting the bounds of the heart mesh for constrained Bayesian Optimization
        :return: bounds in a list [(low x, high x), (low y, high y), (low z, high z)]
        """
        low_x, high_x = np.min(labels[:, 0]), np.max(labels[:, 0])
        low_y, high_y = np.min(labels[:, 1]), np.max(labels[:, 1])
        low_z, high_z = np.min(labels[:, 2]), np.max(labels[:, 2])
        return {'x': (low_x, high_x),
                'y': (low_y, high_y),
                'z': (low_z, high_z)}

    def run(self, x, y, train, labels, target, target_coord, target_raw):
        """
        Handles doing a full run of the CC model on a single target site
        :param x: raw ecg dataset for a patient
        :param y: labels for a patient
        :param train: initial training set derived from x
        :param labels: initial label set derived from y
        :param target: AUC of the target site
        :param target_coord: label for the target site
        :return: euclid error, model predictions, and added sites during run
        """
        cc_euclids = list()
        cc_preds = list()
        cc_sites = list()

        self.ecgs = x
        self.coords = y
        self.target_ecg = target_raw

        optimizer, success = self.optimize_point(y, labels)
        num_sites = len(optimizer.visited)

        print(optimizer.visited)
        print(optimizer.predicted)

        for site in optimizer.visited:
            cc_euclids.append(self.euclidean_distance(site, target_coord))

        print(cc_euclids)
        return cc_euclids, cc_preds, cc_sites, success, num_sites
