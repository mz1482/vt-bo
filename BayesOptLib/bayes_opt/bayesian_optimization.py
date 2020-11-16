import warnings
import numpy as np
import random

from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, acq_max, ensure_rng, acq_max_dis

from sklearn.gaussian_process.kernels import Matern, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor


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


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback == None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, random_state=None, verbose=2, real_set=None, cc_thres=0.99, mm_thres=15):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state, real_set=real_set)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        kernels = [None, Matern(nu=2.5), ExpSineSquared()]
        self._gp = GaussianProcessRegressor(
            kernel=kernels[0],
            alpha=.001,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=self._random_state,
        )

        self._verbose = verbose

        # Holds all of the discrete points within the set to approximate suggestions to
        self._real_set = real_set

        # Holds set of visited and predicted points for approximation
        self.visited = []
        self.predicted = []

        # Target thres to hit between suggestion and pred
        self.cc_thres = cc_thres

        # Distance threshold for grabbing the next point
        self.mm_thres = mm_thres

        # Super call
        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            cc = self._space.probe(params)
            self.dispatch(Events.OPTMIZATION_STEP)
            return cc

    def suggest(self, utility_function):
        labels = self._real_set
        """Most promising point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            n_warmup=1000,
            random_state=self._random_state
        )
#         suggestion = acq_max_dis(
#             ac=utility_function.utility,
#             gp=self._gp,
#             y_max=self._space.target.max(),
#             bounds=self._space.bounds,
#             labels=labels,
#             n_warmup=1000,
#             random_state=self._random_state
#         )


        # Add suggestion to predicted list
        self.predicted.append(suggestion)
#         cont_suggestion  = suggestion

        # Approximate the suggestion to the nearest real point
        suggestion = approximate_point(suggestion, self._real_set, self.visited, self.mm_thres)

        # If no suggestion within mm threshold, return None
        if suggestion is None:
            return None
        else:
            self.visited.append(suggestion)
            return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points, given_set=None):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        # Initialize points within a given segment set if populated
        if given_set is not None:
            for sample in given_set:
                self.visited.append(sample)
                self._queue.add(sample)

        # Otherwise just simply sample random points from the space
        else:
            for _ in range(init_points):
                point = self._space.random_sample()
                self.visited.append(point)
                self._queue.add(point)

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTMIZATION_START, _logger)
            self.subscribe(Events.OPTMIZATION_STEP, _logger)
            self.subscribe(Events.OPTMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 given_set=None,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTMIZATION_START)
        success = False

        # If giving a segment initialization, need to pull samples from that given set only
        if given_set is not None:
            self._prime_queue(init_points, given_set)
        else:
            self._prime_queue(init_points)

        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                x_probe = self.suggest(util)
                iteration += 1

            # Check for none
            if x_probe is None:
                print("No points within threshold!")
                break

            # Probe for Y value of suggestion, check for success
            cc = self.probe(x_probe, lazy=False)
            if cc > self.cc_thres:
                self.suggest(util)
                success = True
                break

        self.dispatch(Events.OPTMIZATION_END)

        # Returns whether the runtime were successful in finding the last site
        return success

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)
