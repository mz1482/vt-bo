"""
create obj mesh from point cloud
Doing basic analysis on the simulated heart mesh to check for correctness
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from BayesOptLib.bayes_opt.bayesian_optimization import BayesianOptimization

# ecgs = pd.read_csv("simu-data/Heart3_SimuData.csv", header=None).to_numpy()
# labels = pd.read_csv("simu-data/UVC3_Corresp2pacingSite.csv", header=None).to_numpy()[:, :3]
data_path = "/home/mz1482/project/vt-bayesian-opt-bopt_debug/data/simu_data_4000/"

# ecgs = pd.read_csv(data_path+"Heart1/Heart1_SimuData_4000.csv", header=None).to_numpy()
labels = pd.read_csv(data_path+"Heart1/Coord1_4000.csv", header=None).to_numpy() / 1000
# uvc = pd.read_csv(data_path+"UVC/Coord1_UVC_4000.csv", header=None).to_numpy()

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


def get_index(label,labels):
    """
    Gets the idx of a label in the labels array
    :param label: label to check for
    :return: idx
    """
    idx = 0
    for coord in labels:
        if np.array_equal(label, coord):
            break
        idx += 1
    return idx

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


    
def corr_check(x,labels,ecgs,target_ecg):
    """
    Check the cc from the points selected from GP with the target ecg
    """
    for i in range(len(labels)):
        if np.array_equal(x, labels[i]):
            e = ecgs[i]
            break
    return np.corrcoef(e,target_ecg)
    

def get_heart_bounds(labels):
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


def lv_rv(uvc,v):
    idx=[]
    for i in range(len(uvc)):
        if (uvc[i,3]==v):
            idx.append(i)
    return idx

def graph_3d(x, y, z, xname, yname, zname):
    # Building and plotting the point mesh cloud
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    ax.scatter(xs=x, ys=y, zs=z, zdir='z', alpha=0.75)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_zlabel(zname)
    plt.show()

def uvc_xyz(optimizer,uvc_lv,labels_lv,target_xyz):
    '''
    convert the visited uvc to visited xyz
    '''
    visited = np.asarray(optimizer.visited)
    path_xyz=np.empty((0, 3))
    for i in range(len(visited)):
        t=get_index(visited[i],uvc_lv)
        b=labels_lv[t]
        path_xyz = np.append(path_xyz,b.reshape(1,-1),axis=0)
    return path_xyz

def result_metric(optimizer,target_xyz,target_ecg,labels,ecgs):
    '''
    it calculates location error, AL step and the max cc. It helped to analyze large experiment when 
    we consider CC of ECG as a concatenated version
    '''
    init = len(optimizer.visited)-len(optimizer.predicted)
#     Y = optimizer._space.target
#     X = optimizer.visited
    max_cc,best = list(optimizer.max.values())[0], np.fromiter(list(optimizer.max.values())[1].values(),dtype=float)
    
#     max_cc = np.amax(Y)
#     for i in range(len(Y)):
#         if Y[i]==max_cc:
#             best = X[i]
#             break
    print("best value",best)
    for i in range(len(optimizer.visited)):
        if np.array_equal(best, optimizer.visited[i]):
            break
    al_step = i-init+1
    if al_step<0:
        al_step=0
    loc_error = euclidean_distance(best, target_xyz)
    for j in range(len(labels)):
        if np.array_equal(best, labels[j]):
            idx = j
            break
    best_ecg = ecgs[idx]
    best_ecg = np.reshape(best_ecg, [12, -1])
    nums = 0
    for k in range(12):
        ccs = abs(correlation_coef(target_ecg.reshape(12,-1)[k], best_ecg[k]))
        if ccs > .90:
            nums += 1
    loc_error,max_cc = np.around(loc_error,2),  np.around(max_cc,2)
    return max_cc,al_step,nums,loc_error

def result_table_xyz(target,optimizer):
    '''
    for 1 experiment it prints/return the table with distance from target at every iteration.
    This function is for xyz coordinates
    '''
    print("The target location (in 3d) is:",target)
    visited = np.asarray(optimizer.visited)
    f = optimizer._space.target
    nn_dis = np.empty((0,1))
    for i in range(len(visited)):
        d = np.sqrt(np.sum((target - visited[i])**2))
        nn_dis = np.append(nn_dis,d)
    nn_dis = nn_dis.reshape(-1,1)
    nn_dis = np.around(nn_dis,2)
    f = f.reshape(-1,1).astype(int)
    it = (np.arange(len(visited))+1).reshape(-1,1).astype(int)
    floating_part = np.around(np.concatenate((visited,nn_dis), axis = 1),2)
    int_part = np.concatenate((it,f), axis = 1)
    result = np.concatenate((int_part,floating_part), axis = 1)
    table = PrettyTable(result.dtype.names)
    table.field_names = ['iteration','passing_lead', 'x','y','z','dis(mm) from target']
    for row in result:
        table.add_row(row)
    return table

def result_table_uvc(target_xyz,target_uvc,optimizer,uvc_lv,labels_lv):
    '''
    for 1 experiment it prints/return the table with distance from target at every iteration.
    This function is for uvc coordinates
    '''
    print("The target location (in uvc) is:",target_uvc)
    visited = np.asarray(optimizer.visited)
    visited_xyz = uvc_xyz(optimizer,uvc_lv,labels_lv,target_xyz)
    f = optimizer._space.target
    nn_dis = np.empty((0,1))
    for i in range(len(visited)):
        d = np.sqrt(np.sum((target_xyz - visited_xyz[i])**2))
        nn_dis = np.append(nn_dis,d)
    nn_dis = nn_dis.reshape(-1,1)
    nn_dis = np.around(nn_dis,2)
    f = f.reshape(-1,1).astype(int)
    it = (np.arange(len(visited))+1).reshape(-1,1).astype(int)
    floating_part = np.around(np.concatenate((visited,nn_dis), axis = 1),2)
    int_part = np.concatenate((it,f), axis = 1)
    result = np.concatenate((int_part,floating_part), axis = 1)
    table = PrettyTable(result.dtype.names)
    table.field_names = ['iteration','passing_lead', 'apicobasal','transmural','rotational','dis(mm) from target']
    for row in result:
        table.add_row(row)
    return table


if __name__ == '__main__':
    data_resolution()