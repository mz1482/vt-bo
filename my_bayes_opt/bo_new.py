from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from gprmy import GaussianProcessRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared
from my_helpers import my_utility_function, unique_rows, acq_max,PrintLog


def check(test, array):
    """
    Simple generator to check whether a 1d array lies within a multidimensional array
    :param test: value to test for
    :param array: multidimensional array to check through
    :return: true iff test in array
    """
    return any(np.array_equal(x, test) for x in array)


def euclidean_distance(one, two):
    """ Computes the euclidean distance between two coordinates """
    return np.sqrt(np.sum((one - two)**2))


def approximate_point(suggestion, real_set, visited, threshold):
    """
    Handles approximating the given suggestion from the acquisition function to a discrete value within the
    input space, ie a real XYZ
    :param visited: visited set of the BayOpt class to avoid dups
    :param real_set: set of real labels to use
    :param suggestion: XYZ point to approximate
    :param threshold: how close the closest point has to be to be considered
    :return: closest real point
    """
    closest, closest_point = np.inf, None
    for coord in real_set:
        # Skip if already visited
        if check(coord, visited):
            continue

        # Get euclid dist and check for closeness
        dist = euclidean_distance(coord, suggestion)
        if dist < closest:
            closest = dist
            closest_point = coord

    # Check if nearest point under threshold
    if closest <= threshold:
        return closest_point
    else:
        return None

    
    
def approx(c1,labels):
    dist = []
    for i in range(len(labels)):
        d = euclidean_distance(c1, labels[i])
        dist = np.append(dist,d)
    for j in range(len(dist)):
        if dist[j]==np.amin(dist):
            break
    return labels[j]
        

class mybo(object):
    def __init__(self, f, pbounds, real_set=None,verbose=2,cc_thres=0.99, mm_thres=15):
        # Store the original dictionary
        self.pbounds = pbounds
        self.real_set = real_set
        self.keys = list(pbounds.keys())
        #no of parameters
        self.dim = len(pbounds)
        # Create an array with parameters bounds
        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)
        #function to be optimized
        self.f = f
        # Initialization flag
        self.initialized = False
 
        # Initialization lists --- stores starting points before process begins
        self.visited = []
        self.predicted = []
        self.init_points = []
        self.x_init = []
        self.y_init = []
        # Target thres to hit between suggestion and pred
        self.cc_thres = cc_thres

        # Distance threshold for grabbing the next point
        self.mm_thres = mm_thres
        
        # Numpy array place holders
        self.X = None
        self.Y = None
 
        # Counter of iterations
        self.i = 0
        kernels = [None, Matern(nu=2.5), ExpSineSquared()]
        # Internal GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=kernels[1],
            n_restarts_optimizer=25,
        )
        # PrintLog object
        self.plog = PrintLog(self.keys)
        # Utility Function placeholder
        self.util = None
        self.res = {}
        # Output dictionary

        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [], 'params': []}
        # Verbose
        self.verbose = verbose
        
    def init(self,init_points):
        y_init = []
        idx= np.random.choice(len(self.real_set), size=init_points, replace=False)
        self.init_points = self.real_set[idx,:]
        for x in self.init_points:
            y_init.append(self.f(**dict(zip(self.keys, x))))
            if self.verbose:
                self.plog.print_step(x, y_init[-1])
#         self.init_points +=self.x_init
#         y_init += self.y_init
        # Turn it into np array and store.
        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)
        self.initialized = True
        
    def explore(self, points_dict):
        """
        Method to explore user defined points
 
        :param points_dict:
        :return:
        """
 
        # Consistency check
        param_tup_lens = []
 
        for key in self.keys:
            param_tup_lens.append(len(list(points_dict[key])))
 
        if all([e == param_tup_lens[0] for e in param_tup_lens]):
            pass
        else:
            raise ValueError('The same number of initialization points '
                             'must be entered for every parameter.')
 
        # Turn into list of lists
        all_points = []
        for key in self.keys:
            all_points.append(points_dict[key])
 
        # Take transpose of list
        self.init_points = list(map(list, zip(*all_points)))
        
    def initialize(self, points_dict):
        """
        Method to introduce point for which the target function
        value is known
 
        :param points_dict:
        :return:
        """
 
        for target in points_dict:
 
            self.y_init.append(target)
 
            all_points = []
            for key in self.keys:
                all_points.append(points_dict[target][key])
 
            self.x_init.append(all_points)
    
    
    def initialize_df(self, points_df):
        """
        Method to introduce point for which the target function
        value is known from pandas dataframe file
 
        :param points_df: pandas dataframe with columns (target, {list of columns matching self.keys})
 
        ex:
              target        alpha      colsample_bytree        gamma
        -1166.19102       7.0034                0.6849       8.3673
        -1142.71370       6.6186                0.7314       3.5455
        -1138.68293       6.0798                0.9540       2.3281
        -1146.65974       2.4566                0.9290       0.3456
        -1160.32854       1.9821                0.5298       8.7863
 
        :return:
        """
 
        for i in points_df.index:
            self.y_init.append(points_df.loc[i, 'target'])
 
            all_points = []
            for key in self.keys:
                all_points.append(points_df.loc[i, key])
 
            self.x_init.append(all_points)
    
    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds
 
        :param new_bounds:
            A dictionary with the parameter name and its new bounds
 
        """
 
        # Update the internal object stored dict
        self.pbounds.update(new_bounds)
 
        # Loop through the all bounds and reset the min-max bound matrix
        for row, key in enumerate(self.pbounds.keys()):
 
            # Reset all entries, even if the same.
            self.bounds[row] = self.pbounds[key]
        
        
    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """
        Main optimization method.
 
        Parameters
        ----------
        :param init_points:
            Number of randomly chosen points to sample the
            target function before fitting the gp.
 
        :param n_iter:
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.
 
        :param acq:
            Acquisition function to be used, defaults to Expected Improvement.
 
        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object
 
        Returns
        -------
        :return: Nothing
        """
        # Reset timer
        self.plog.reset_timer()
 
        # Set acquisition function
        self.util = my_utility_function(kind=acq, kappa=kappa, xi=xi)
 
        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)
 
        y_max = self.Y.max()
 
        # Set parameters if any was passed
        self.gp.set_params(**gp_params)
 
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
 
        # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds)
        x_max = approx(x_max,self.real_set)
#         x_max = approximate_point(x_max, self.real_set, self.X, self.mm_thres)
#         print(x_max)
 
        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            if np.any((self.X - x_max).sum(axis=1) == 0):
 
                x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])
                x_max = approx(x_max,self.real_set)
#                 x_max = approximate_point(x_max, self.real_set, self.X, self.mm_thres)
 
                pwarning = True
 
            # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))
 
            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
 
            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.bounds)
            x_max = approx(x_max,self.real_set)
#             x_max = approximate_point(x_max, self.real_set, self.X, self.mm_thres)
 
            # Print stuff
            if self.verbose:
                self.plog.print_step(self.X[-1], self.Y[-1], warning=pwarning)
 
            # Keep track of total number of iterations
            self.i += 1
 
            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))
 
        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()
        return self.gp, self.X
            
    
    
    def gpfit(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):

        print("This is GP")

        self.util = my_utility_function(kind=acq,kappa=kappa,xi=xi)
        dim=self.dim
        dim_limit = self.bounds
        if not self.initialized:
            self.init(init_points)
        y_max = self.Y.max()
        #Set parameters if any was passed
        self.gp.set_params(**gp_params)
 
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
                # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds)
#         dif = []
#         xx = np.linspace(-5,5,1000)
#         xx=np.reshape((xx),(1000,1))
#         yy1 = self.gp.predict(xx,return_std=False, return_cov=False)
#         k1_inv = self.gp.fit2(self.X[ur], self.Y[ur])
#         f1 = self.Y
#         a=np.sum(np.dot(k1_inv,f1))
        for i in range(n_iter):
            print(i)
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            if np.any((self.X - x_max).sum(axis=1) == 0):
 
                x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])
 
                pwarning = True
    # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))
 
            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
#             if i == 0:
#                 yy1 = self.gp.predict(xx,return_std=False, return_cov=False)
#                 k1_inv = self.gp.fit2(self.X[ur], self.Y[ur])
#                 f1 = self.Y
 
            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
 
            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.bounds)
        # Keep track of total number of iterations
            self.i += 1
 
            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))
        return self.gp, self.X
    
    def gpfit2(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):

        print("This is testing for new AF")

        self.util = my_utility_function(kind=acq,kappa=kappa,xi=xi)
        dim=self.dim
        dim_limit = self.bounds
        if not self.initialized:
            self.init(init_points)
        y_max = self.Y.max()
        #Set parameters if any was passed
        self.gp.set_params(**gp_params)
 
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
                # Finding argmax of the acquisition function.
#         x_max = acq_max(ac=self.util.utility,
#                         gp=self.gp,
#                         y_max=y_max,
#                         bounds=self.bounds)
#         print(x_max)
#         xx = np.linspace(-5,5,1000)
#         xx=np.reshape((xx),(1000,1))
#         yy1 = self.gp.predict(xx,return_std=False, return_cov=False)
        k1_inv = self.gp.fit2(self.X[ur], self.Y[ur])
        a=np.sum(np.dot(k1_inv,self.Y))

        for i in range(n_iter):
            print(i)
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
#             pwarning = False
#             if np.any((self.X - x_max).sum(axis=1) == 0):
 
#                 x_max = np.random.uniform(self.bounds[:, 0],
#                                           self.bounds[:, 1],
#                                           size=self.bounds.shape[0])
 
#                 pwarning = True
    # Append most recently generated values to X and Y arrays
            temp_y = []
            temp_x = []
            d=np.empty([100,3])
#             rs = np.random.multivariate_normal(np.zeros(self.dim), 1.0/10*np.diag(np.ones(self.dim)),100)
            rs = np.random.uniform(low=-4, high=4, size=(100,self.dim))
#             rs_new = rs
            for p in range(100):
                temp_x =  np.vstack((self.X, rs[p,:].reshape((1, -1))))
                y2 = self.gp.predict(rs[p,:].reshape((1, -1)),return_std=False, return_cov=False)
                temp_y = np.append(self.Y, y2)
                # Updating the GP.
#                 temp_gp = self.gp.fit(temp_x, temp_y)
                k2_inv = self.gp.fit2(temp_x, temp_y)
                b=np.sum(np.dot(k2_inv,temp_y))
                d[p,:] = rs[p,:],b,np.abs(a-b)
#             sam = np.amax(d[], axis=1) 
        
            selected = d[d[:,2]==np.amax(d[:,2])]
            x_max = np.array([selected[0,0]])
            a = np.array([selected[0,1]])
    
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))
 
            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
#             yy2 = self.gp.predict(xx,return_std=False, return_cov=False)
#             plt.plot(xx,yy1,c = 'red',label = "previous")
#             plt.plot(xx,yy2,c='green',label = 'current')
#             plt.legend(loc='lower left')
#             plt.show()
#             yy1 = yy2
            # Update maximum value to search for next probe point.
#             if self.Y[-1] > y_max:
#                 y_max = self.Y[-1]
 
#             # Maximize acquisition function to find next probing point
#             x_max = acq_max(ac=self.util.utility,
#                             gp=self.gp,
#                             y_max=y_max,
#                             bounds=self.bounds)
#         # Keep track of total number of iterations
            self.i += 1
 
            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))
        return self.gp
    
    
    
    
    def gpfit3(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):

        print("This is GP")

        self.util = my_utility_function(kind=acq,kappa=kappa,xi=xi)
        dim=self.dim
        dim_limit = self.bounds
        if not self.initialized:
            self.init(init_points)
        y_max = self.Y.max()
        #Set parameters if any was passed
        self.gp.set_params(**gp_params)
 
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
                # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds)
        dif = []
        xx = np.linspace(-5,5,1000)
        xx=np.reshape((xx),(1000,1))
        yy1 = self.gp.predict(xx,return_std=False, return_cov=False)
        k1_inv = self.gp.fit2(self.X[ur], self.Y[ur])
        f1 = self.Y
        a=np.sum(np.dot(k1_inv,f1))
        for i in range(n_iter):
            print(i)
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            if np.any((self.X - x_max).sum(axis=1) == 0):
 
                x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])
 
                pwarning = True
    # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))
 
            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
#             if i == 0:
#                 yy1 = self.gp.predict(xx,return_std=False, return_cov=False)
#                 k1_inv = self.gp.fit2(self.X[ur], self.Y[ur])
#                 f1 = self.Y
 
            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
 
            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.bounds)
        # Keep track of total number of iterations
            self.i += 1
 
            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))
            yy2 = self.gp.predict(xx,return_std=False, return_cov=False)
            k2_inv = self.gp.fit2(self.X[ur], self.Y[ur])
            f2 = self.Y
            b=np.sum(np.dot(k2_inv,f2))
            c = a-b
            dif = np.append(dif,c)
            plt.plot(xx,yy1,c = 'red',label = "previous")
            plt.plot(xx,yy2,c='green',label = 'current')
            plt.legend(loc='lower left')
            plt.show()
            yy1 = yy2
            a = b
            trainx= self.X
            


        return xx,k1_inv,k2_inv,f1,f2,yy1,yy2,trainx,np.abs(dif),self.gp
    
    def gp_shape(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 mh_size = 1000,
                 **gp_params):

        print("This is GP based shape")

        self.util = my_utility_function(kind=acq,kappa=kappa,xi=xi)
        dim=self.dim
        dim_limit = self.bounds
        if not self.initialized:
            self.init(init_points)
        y_max = self.Y.max()
        #Set parameters if any was passed
        self.gp.set_params(**gp_params)
 
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
                # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds)
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            if np.any((self.X - x_max).sum(axis=1) == 0):
 
                x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])
 
                pwarning = True
    # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))
 
            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
 
            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
 
            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.bounds)
        # Keep track of total number of iterations
            self.i += 1
 
            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))


            if i==n_iter - 1:
#                 chain = []
                mcmc_chain = 2
                z_sample = np.empty((0,dim))
                for c in range(mcmc_chain):
                    ux=np.zeros(dim)
                    for k in range(dim):
                        ux[k]= np.random.uniform(dim_limit[k,0],dim_limit[k,1])
                    zfix = ux
                    for j in range(mh_size):
                        x_0=np.reshape(zfix,(1,-1))
                        param = self.gp.predict(x_0, return_std= True)
                        m=param[0].ravel() # mean of gp
                        v=param[1].ravel() # sigma of gp
                        z_star = zfix + np.random.normal(loc = 0, scale = .5,size=dim)
                        z_star2 = np.reshape(z_star,(1,-1))
                        inside = 0
                        for pp in range(dim):
                            if (dim_limit[pp,0]<z_star[pp]<dim_limit[pp,1]):
                                inside = inside + 1
                        if inside == dim:
                            param2 = self.gp.predict(z_star2, return_std= True)
                            m2=param2[0].ravel() # mean of gp
                            v2=param2[1].ravel() # sigma of gp
                            rho = min(1, np.exp(m2)/np.exp(m))
                            r=np.random.rand()
                            if r < rho:
                                zfix = z_star
                                z_sample = np.append(z_sample,z_star2,axis=0)
#                 print("iteration no is ",i+1)
#                 print("acceptance rate", ar)
        return z_sample

    def exp_gp_shape(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 mh_size = 1000,
                 **gp_params):

        print("This is exp(GP) based shape")

        self.util = my_utility_function(kind=acq,kappa=kappa,xi=xi)
        dim=self.dim
        dim_limit = self.bounds
        if not self.initialized:
            self.init(init_points)
        y_max = self.Y.max()
        #Set parameters if any was passed
        self.gp.set_params(**gp_params)
 
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
                # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds)
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            if np.any((self.X - x_max).sum(axis=1) == 0):
 
                x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])
 
                pwarning = True
    # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))
 
            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
 
            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
 
            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.bounds)
        # Keep track of total number of iterations
            self.i += 1
 
            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))


            if i==n_iter - 1:
#                 chain = []
                mcmc_chain = 2
                z_sample = np.empty((0,dim))
                for c in range(mcmc_chain):
                    ux=np.zeros(dim)
                    for k in range(dim):
                        ux[k]= np.random.uniform(dim_limit[k,0],dim_limit[k,1])
                    zfix = ux
                    for j in range(mh_size):
                        x_0=np.reshape(zfix,(1,-1))
                        param = self.gp.predict(x_0, return_std= True)
                        m=param[0].ravel() # mean of gp
                        v=param[1].ravel() # sigma of gp
                        log_mean_0 = np.exp(m+v**2/2)
                        z_star = zfix + np.random.normal(loc = 0, scale = .5,size=dim)
                        z_star2 = np.reshape(z_star,(1,-1))
                        inside = 0
                        for pp in range(dim):
                            if (dim_limit[pp,0]<z_star[pp]<dim_limit[pp,1]):
                                inside = inside + 1
                        if inside == dim:
                            param2 = self.gp.predict(z_star2, return_std= True)
                            m2=param2[0].ravel() # mean of gp
                            v2=param2[1].ravel() # sigma of gp
                            log_mean_2 = np.exp(m2+v2**2/2)
                            rho = min(1, log_mean_2/log_mean_0)
                            r=np.random.rand()
                            if r < rho:
                                zfix = z_star
                                z_sample = np.append(z_sample,z_star2,axis=0)
#                 ar=len(accepted_f)/(j+1)
#                 print("iteration no is ",i+1)
#                 print("acceptance rate", ar)
        return z_sample