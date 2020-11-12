"""
@file data_analysis
@author Ryan Missel
create obj mesh from point cloud
Doing basic analysis on the simulated heart mesh to check for correctness
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# ecgs = pd.read_csv("simu-data/Heart3_SimuData.csv", header=None).to_numpy()
# labels = pd.read_csv("simu-data/UVC3_Corresp2pacingSite.csv", header=None).to_numpy()[:, :3]

ecgs = pd.read_csv("simu-data/Heart3_SimuData.csv", header=None).to_numpy()
labels = pd.read_csv("simu-data/Heart3_XYZsub.csv", header=None).to_numpy() / 1000


def euclidean_distance(one, two):
    """ Computes the euclidean distance between two coordinates """
    return np.sqrt(np.sum((one - two)**2))


def correlation_coef(one, two):
    """ Returns the CC between two ECG signals"""
    return np.corrcoef(one, two)[0, 1]


def get_closest_point(label):
    """
    Finds the closest point within the dataset to the given point
    :param label: starting point
    :return: idx of closest point
    """
    closest, closest_idx = np.inf, None

    for idx in range(len(labels)):
        # Check for starting array
        if np.array_equal(label, labels[idx]):
            continue

        # Calculate l2 norm and compare
        dist = euclidean_distance(label, labels[idx])
        if dist < closest:
            closest = dist
            closest_idx = idx

    return closest, closest_idx


def data_resolution():
    """
    Function that handles getting the resolution of the data in terms of average closest distance and CC between ECGS
    Plots a point cloud mesh with labels colored by closest CC value
    """
    # Arrays for metrics and point cloud plotting
    avg_dist, avg_cc = [], []
    high_cc, low_cc = [], []

    # Loop through all the labels to find the closest point and append stats
    for cur_idx in range(len(labels)):
        # Get cur label
        label = labels[cur_idx]

        # Get closest point and add metrics
        dist, idx = get_closest_point(label)
        avg_dist.append(dist)
        avg_cc.append(correlation_coef(ecgs[cur_idx], ecgs[idx]))

        # Add label to right bin
        high_cc.append(label) if correlation_coef(ecgs[cur_idx], ecgs[idx]) > .5 else low_cc.append(label)

    # Print out metrics
    print("--- Euclid Dist ---")
    print("Mean: %.3f +- %.3f" % (np.mean(avg_dist), np.std(avg_dist)))
    print("Max/Min: %.1f | %.1f" % (np.max(avg_dist), np.min(avg_dist)))
    print("")
    print("--- CC ---")
    print("Mean: %.3f +- %.3f" % (np.mean(avg_cc), np.std(avg_cc)))
    print("Max/Min: %.1f | %.1f" % (np.max(avg_cc), np.min(avg_cc)))
    print("")

    # Plot point cloud mesh by colors
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    high_cc, low_cc = np.asarray(high_cc), np.asarray(low_cc)
    ax.scatter(xs=high_cc[:, 0], ys=high_cc[:, 1], zs=high_cc[:, 2], color='blue')
    ax.scatter(xs=low_cc[:, 0], ys=low_cc[:, 1], zs=low_cc[:, 2], color='red')
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    plt.show()


def get_heart_bounds():
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


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def graph_3d(x, y, z, xname, yname, zname):
    # Building and plotting the point mesh cloud
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    ax.scatter(xs=x, ys=y, zs=z, zdir='z', alpha=0.75)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_zlabel(zname)
    plt.show()


def graph_cc_distribution(target):
    """
    Function that handles color coating the 3d mesh to see the distribution of CC over distance from a given target
    coordinate. Useful to see the general function in 3D
    """
    # Lists to hold the points with the color coating
    true, blue, green, yellow, red = None, [], [], [], []
    color_gradient = []
    # Loop through all points to get CC with that point
    for ecg, coord in zip(ecgs, labels):
        if np.array_equal(target, ecg):
            true = coord
            color_gradient.append(1)
            continue

        cc = correlation_coef(target, ecg)
        color_gradient.append(cc)
        if cc >= .9:
            blue.append(coord)
        elif .9 > cc >= .75:
            green.append(coord)
        elif .75 > cc > .3:
            yellow.append(coord)
        else:
            red.append(coord)

    # Plot out the points according to color
    fig = plt.figure(55)
    ax = fig.gca(projection='3d')

    # blue, green, yellow, red = np.array(blue), np.array(green), np.array(yellow), np.array(red)
    ax.scatter(true[0], true[1], true[2], color='black')
    # ax.scatter(xs=blue[:, 0], ys=blue[:, 1], zs=blue[:, 2], color='blue')
    # ax.scatter(xs=green[:, 0], ys=green[:, 1], zs=green[:, 2], color='green')
    # ax.scatter(xs=yellow[:, 0], ys=yellow[:, 1], zs=yellow[:, 2], color='yellow')
    # ax.scatter(xs=red[:, 0], ys=red[:, 1], zs=red[:, 2], color='gray')
    ax.scatter(xs=labels[:, 0], ys=labels[:, 1], zs=labels[:, 2], c=color_gradient, cmap = plt.cm.rainbow)
    ax.scatter(true[0], true[1], true[2], color='black', marker = "X", s = 100)
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    fig.show()
    
    
def narrow(target,target_ecg,ecgs,labels,limit):
    """
    Function that handles color coating the 3d mesh to see the distribution of CC over distance from a given target
    coordinate. Useful to see the general function in 3D
    """
    # Lists to hold the points with the color coating
    true, blue, green, yellow, red = None, [], [], [], []
    color_gradient = []
    # Loop through all points to get CC with that point
    n_ecgs = np.empty((0, 7212))
    n_labels = np.empty((0, 3))
    for i in range(len(labels)):
        d = np.sqrt(np.sum((target - labels[i])**2))
        if d < limit:
            n_labels = np.append(n_labels,labels[i].reshape(1,3),axis=0)
            n_ecgs = np.append(n_ecgs,ecgs[i].reshape(1,7212),axis=0)
    for ecg, coord in zip(n_ecgs, n_labels):
        if np.array_equal(target_ecg, ecg):
            true = coord
            color_gradient.append(1)
            continue

        cc = correlation_coef(target_ecg, ecg)
        color_gradient.append(cc)
        if cc >= .9:
            blue.append(coord)
        elif .9 > cc >= .75:
            green.append(coord)
        elif .75 > cc > .3:
            yellow.append(coord)
        else:
            red.append(coord)

    # Plot out the points according to color
    fig = plt.figure(3)
    ax = fig.gca(projection='3d')

    # blue, green, yellow, red = np.array(blue), np.array(green), np.array(yellow), np.array(red)
    ax.scatter(true[0], true[1], true[2], color='black')
    # ax.scatter(xs=blue[:, 0], ys=blue[:, 1], zs=blue[:, 2], color='blue')
    # ax.scatter(xs=green[:, 0], ys=green[:, 1], zs=green[:, 2], color='green')
    # ax.scatter(xs=yellow[:, 0], ys=yellow[:, 1], zs=yellow[:, 2], color='yellow')
    # ax.scatter(xs=red[:, 0], ys=red[:, 1], zs=red[:, 2], color='gray')
    ax.scatter(xs=n_labels[:, 0], ys=n_labels[:, 1], zs=n_labels[:, 2], c=color_gradient, cmap = plt.cm.rainbow)
    ax.scatter(true[0], true[1], true[2], color='black', marker = "X", s = 100)
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    fig.show()

    
    
    
    
def corrplot3axes(tidx,labels,ecgs,dis_limit):
    target_loc = labels[tidx]
    target_ecg = ecgs[tidx]
    target_x,target_y, target_z = target_loc[0],target_loc[1],target_loc[2]
    cc = np.array([[1]])
    dis = np.array([[0]])
    first_row=np.concatenate((target_loc.reshape(1,3), dis,cc),axis =1)
    nn_loc = np.empty((0, 3))
    nn_cc = np.empty ((0,1))
    nn_dis = np.empty((0,1))
    for i in range(len(labels)):
        d = np.sqrt(np.sum((target_loc - labels[i])**2))
        if d < dis_limit:
            nn_loc = np.append(nn_loc,labels[i].reshape(1,3),axis=0)
            nn_cc = np.append(nn_cc,np.corrcoef(target_ecg, ecgs[i])[0, 1])
            nn_dis = np.append(nn_dis,d)
    nn_dis = nn_dis.reshape(-1,1)
    nn_cc = nn_cc.reshape(-1,1)
    near_points = np.concatenate((nn_loc, nn_dis, nn_cc),axis =1)
    x,y,z=[],[],[]
    for j in range(len(near_points)):
        xx=abs(near_points[j,0] - target_loc[0])
        yy=abs(near_points[j,1] - target_loc[1])
        zz=abs(near_points[j,2] - target_loc[2])
        x = np.append(x,xx)
        y = np.append(y,yy)
        z = np.append(z,zz)
    fig, axs = plt.subplots(3)
    axs[0].scatter(x,nn_cc,color = 'green',label = 'x axis')
    axs[0].legend()
    axs[1].scatter(y,nn_cc,color = 'red',label = 'y axis')
    axs[1].legend()
    axs[2].scatter(z,nn_cc,color = 'blue',label = 'z axis')
    axs[2].legend()
    plt.show
    return x,y,z,nn_cc

def graph_dist_over_axis(target):
    ccs = []

    # Loop through all points to get CC with that point
    for ecg, coord in zip(ecgs, labels):
        if np.array_equal(target, ecg):
            true = coord
            ccs.append(1.0)
            continue

        ccs.append(correlation_coef(target, ecg))

    # Plot out the points according to color
    fig = plt.figure(88)
    ax = fig.gca(projection='3d')

    print(labels[:, 0].shape, np.asarray(ccs).shape)
    ax.scatter(xs=labels[:, 0], ys=labels[:, 1], zs=ccs)
    plt.xlabel('x')
    plt.ylabel('y')
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
#     fig.show()


if __name__ == '__main__':
    data_resolution()