"""
@file data_manipulation
@author Ryan Missel

Holds scripts that modify the data for usage, like a perspective projection on the endo LV mesh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def perspective_proj(points):
    """
    Performs a perspective projection on a pair of 3d points
    :param points: points to perform proj on
    :return: final matrix of points
    """
    mat = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 1, 0]])

    for point in points:
        point = np.concatenate((point, [1]))

        point = np.matmul(mat, point.T)
        point /= point[3]
        print(point)

    # Plot points
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("simu-data/endoLVy.csv", header=None).to_numpy()
    perspective_proj(data)