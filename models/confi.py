# Global Vars
DATA_PATH = "r15/patient-datasets/"
LEADS = [i for i in range(12)]

# How many random initializations of the models to perform
NUM_TRIALS = 5

# Number of steps to take in a single run
NUM_STEPS = 20

# How many randomly selected points to start with (within the predicted segments)
NUM_POINTS_START = 10

# Hyperparameter of the SVR
SVR_C = 50

# Checks whether to use the full set of data or whether to constrain the testing set to only the successful
# cases of the CCSI model
FULL_SET = True

# Correlation Coefficient Hyperparams
CC_THRES = .75
CC_SUCC = .95

# Force n neighbors within area an area of 15mm around the target site
# This is used to skip out on unrealistic tests
NNEIGHBORS = True