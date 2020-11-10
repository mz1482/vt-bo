"""
@file RandomSampler
@author Ryan Missel

Holds a class that handles building a Gaussian Process using randomly selected points
"""
import numpy as np
from data_analysis import correlation_coef, get_closest_point
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class RandomSampler:
    def __init__(self, ecgs, labels):
        self.ecgs = ecgs
        self.labels = labels

    def optimize(self, target, target_ecg, num_points, n_iter=50):
        # Best CC found
        highest_cc = -1000
        points_taken = 0

        # Convert to np array
        target = np.reshape(target, [1, -1])
        target_ecg = np.reshape(target_ecg, [1, -1])

        # Internal GP regressor
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=None,
        )

        # Get initial random set
        init_idx = np.random.choice(num_points, n_iter + 4)
        train_x, train_y = self.ecgs[init_idx], self.labels[init_idx]

        # Print out first points found
        for point in train_y:
            print(point)

        # General loop
        for i in range(4, n_iter + 4):
            # Build GP with data
            gp.fit(train_x[:i], train_y[:i])

            # Get pred and calculate changes
            pred = gp.predict(target_ecg)
            _, idx = get_closest_point(pred)
            cc = correlation_coef(target_ecg, self.ecgs[idx])

            if cc > 0.9:
                highest_cc = cc
                break
            if cc > highest_cc:
                highest_cc = cc
            points_taken = i
        return points_taken, highest_cc

    def get_index(self, label):
        """
        Gets the idx of a label in the labels array
        :param label: label to check for
        :return: idx
        """
        idx = 0
        for coord in self.labels:
            if np.array_equal(label, coord):
                break
            idx += 1
        return idx