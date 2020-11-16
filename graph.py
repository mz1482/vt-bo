import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec





def correlation_coef(one, two):
    """ Returns the CC between two ECG signals"""
    return np.corrcoef(one, two)[0, 1]

def euclidean_distance(one, two):
    return np.sqrt(np.sum((one - two)**2))

def approx(c1,labels):
    dist = []
    for i in range(len(labels)):
        d = euclidean_distance(c1, labels[i])
        dist = np.append(dist,d)
    for j in range(len(dist)):
        if dist[j]==np.amin(dist):
            break
    return labels[j]

def corners(labels):
    minx,maxx = np.amin(labels[:,0]),np.amax(labels[:,0])
    miny,maxy = np.amin(labels[:,1]),np.amax(labels[:,1])
    minz,maxz = np.amin(labels[:,2]),np.amax(labels[:,2])
    c1 = np.array([minx,miny,minz])
    c2 = np.array([maxx,miny,minz])
    c3 = np.array([maxx,maxy,minz])
    c4 = np.array([minx,maxy,minz])
    c5 = np.array([minx,miny,maxz])
    c6 = np.array([maxx,miny,maxz])
    c7 = np.array([maxx,maxy,maxz])
    c8 = np.array([minx,maxy,maxz])
    c = np.asarray([c1,c2,c3,c4,c5,c6,c7,c8])   
    c9 = np.asarray([np.mean(c[:,0]),np.mean(c[:,1]),np.mean(c[:,2])]).reshape(1,3)
    c = np.concatenate((c, c9),axis =0)
    for i in range(len(c)):
        c[i] = approx(c[i],labels)
    return c


def trend(target,visited,actual):
    visited = np.asarray(visited)
    actual = np.asarray(actual)
    visited = visited[len(visited)-len(actual):len(visited)]
    dis_tar_act = []
    dis_tar_vis = []
    dis_dif = []
    for i in range(len(actual)):
        d1 = np.linalg.norm(target-visited[i,:])
        dis_tar_vis = np.append(dis_tar_vis,d1)
        d2 = np.linalg.norm(target-actual[i,:])
        dis_tar_act = np.append(dis_tar_act,d2)
        d3 = np.linalg.norm(actual[i,:]-visited[i,:])
        dis_dif = np.append(dis_dif,d3)
    plt.figure(5)
    plt.plot(dis_tar_vis,label = 'between target and visited')
    plt.plot(dis_tar_act,label = 'between target and predicted')
    plt.plot(dis_dif, label = 'between predicted and visited')
    plt.xlabel("iteration")
    plt.ylabel("distance")
    plt.legend(loc='upper right')
    plt.show

    
def nearest(tidx,labels,ecgs,dis_limit):
    target_loc = labels[tidx]
    target_ecg = ecgs[tidx]
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
    table = np.concatenate((first_row,near_points), axis = 0)
    table = np.around(table,2)
    plt.figure(2)
    plt.scatter(table[:,3],table[:,4])
    plt.xlabel("distance")
    plt.ylabel("correlation")
    plt.show
    return table




def graph_cc_distribution(target,ecgs,labels):
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
            red.append(coord)
        elif .9 > cc >= .75:
            yellow.append(coord)
        elif .75 > cc > .3:
            green.append(coord)
        else:
            blue.append(coord)

    # Plot out the points according to color
    fig = plt.figure(55)
    ax = fig.gca(projection='3d')

    # blue, green, yellow, red = np.array(blue), np.array(green), np.array(yellow), np.array(red)
    ax.scatter(true[0], true[1], true[2], color='black')
#     ax.scatter(xs=blue[:, 0], ys=blue[:, 1], zs=blue[:, 2], color='blue')
#     ax.scatter(xs=green[:, 0], ys=green[:, 1], zs=green[:, 2], color='green')
#     ax.scatter(xs=yellow[:, 0], ys=yellow[:, 1], zs=yellow[:, 2], color='yellow')
#     ax.scatter(xs=red[:, 0], ys=red[:, 1], zs=red[:, 2], color='gray')
    ax.scatter(xs=labels[:, 0], ys=labels[:, 1], zs=labels[:, 2], c=color_gradient, cmap = plt.cm.rainbow)
    ax.scatter(true[0], true[1], true[2], color='black', marker = "X", s = 100)
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    plt.show()
    
    
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
            red.append(coord)
        elif .9 > cc >= .75:
            yellow.append(coord)
        elif .75 > cc > .3:
            green.append(coord)
        else:
            blue.append(coord)

    # Plot out the points according to color
    fig = plt.figure(3)
    ax = fig.gca(projection='3d')

    # blue, green, yellow, red = np.array(blue), np.array(green), np.array(yellow), np.array(red)
    ax.scatter(true[0], true[1], true[2], color='black')
    # ax.scatter(xs=blue[:, 0], ys=blue[:, 1], zs=blue[:, 2], color='blue')
    # ax.scatter(xs=green[:, 0], ys=green[:, 1], zs=green[:, 2], color='green')
    # ax.scatter(xs=yellow[:, 0], ys=yellow[:, 1], zs=yellow[:, 2], color='yellow')
    # ax.scatter(xs=red[:, 0], ys=red[:, 1], zs=red[:, 2], color='gray')
    ax.scatter(xs=n_labels[:, 0], ys=n_labels[:, 1], zs=n_labels[:, 2], s=50,c=color_gradient, cmap = plt.cm.rainbow)
    ax.scatter(true[0], true[1], true[2], color='black', marker = "X", s = 100)
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
#     ax.set_xlim3d(np.amin(n_labels[:,0]), np.amax(n_labels[:,0]))
#     ax.set_ylim3d(np.amin(n_labels[:,1]),np.amax(n_labels[:,1]))
#     ax.set_zlim3d(np.amin(n_labels[:,2]),np.amax(n_labels[:,2]))
    fig.suptitle('Narrowing the space around the target', fontsize=16)
    plt.show()

    
    
    
    
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
    fig.suptitle('correlation in different axis wrt distance', fontsize=16)
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
    ax.scatter(xs=labels[:, 0], ys=labels[:, 1], zs=labels[:,2])
    plt.xlabel('x')
    plt.ylabel('y')
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
#     fig.show()



def cube(target,ecgs,labels):
    c = corners(labels)
    true = None
    color_gradient = []
    for ecg, coord in zip(ecgs, labels):
        if np.array_equal(target, ecg):
            true = coord
            color_gradient.append(1)
            continue
        cc = correlation_coef(target, ecg)
        color_gradient.append(cc)
    fig = plt.figure(20)
    ax = fig.gca(projection='3d')
    ax.scatter(xs=labels[:, 0], ys=labels[:, 1], zs=labels[:, 2], c=color_gradient, cmap = plt.cm.rainbow)
    ax.scatter(true[0], true[1], true[2], color='black', marker = "X", s = 100)
    ax.scatter(c[:,0], c[:,1], c[:,2], color='red', marker = "*", s = 150)
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    plt.show()

def plot_exploration(target,labels,visited, color_gradient):
    """
    Handles plotting the predictions of the network over time
    :param visited:
    :return:
    """
    path = np.array(visited)
    color_gradient = np.array(color_gradient)
    # for i in range(len(path)):
    #     cur = np.array(path[:i])
    #     rest = np.delete(labels, np.where(np.isin(labels, cur)), axis=0)
    #
    #     fig = plt.figure(0)
    #     ax = fig.gca(projection='3d')
    #
    #     ax.scatter(xs=rest[:, 0], ys=rest[:, 1], zs=rest[:, 2], zdir='z', alpha=0.75, color='gray')
    #     ax.scatter(xs=cur[:, 0], ys=cur[:, 1], zs=cur[:, 2], zdir='z', color='blue')
    #     ax.scatter(xs=target[0], ys=target[1], zs=target[2], color='black')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #     plt.pause(1)
    #     fig.clear()
    # Plot final for viewing
    rest = np.delete(labels, np.where(np.isin(labels, path)), axis=0)
    color_gradient = np.delete(color_gradient, np.where(np.isin(labels, path)), axis=0)
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')

    ax.scatter(xs=rest[:, 0], ys=rest[:, 1], zs=rest[:, 2], zdir='z', alpha=0.75, c=color_gradient, cmap = plt.cm.rainbow)
    ax.scatter(xs=path[:, 0], ys=path[:, 1], zs=path[:, 2], zdir='z', color='blue')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color = 'blue')

    m = path
    for i in range(len(m)):
        ax.text(m[i, 0], m[i, 1], m[i, 2], '%s' % (str(i)), size=10, zorder=1, color='k')
    ax.scatter(xs=target[0], ys=target[1], zs=target[2], color='black', s = 100)
    fig.suptitle('Path of BO to target', fontsize=16)
    plt.show()