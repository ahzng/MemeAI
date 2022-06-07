from numberAI import *

################################################## INPUT ##################################################

np.set_printoptions(threshold=sys.maxsize) # print entire array
#np.set_printoptions(threshold = False) # print truncated array

# Iterations
numIters = 300

# Learning rate
alpha = 0.3

# Lambda
lamb = 1

# Load matlab data
#mat1 = loadmat("ex3data1.mat")
#mat2= loadmat("ex3weights.mat")
mat1 = loadmat("ex4data1.mat")
mat2= loadmat("ex4weights.mat")

################################################## DEFINE ##################################################

# 5000x400 matrix (5000 training examples with 400 features each)
# Each feature is a grayscale pixel value in a 20x20 image
X = mat1["X"]
m = X.shape[0] # Number of training examples
n = X.shape[1] # Number of features (excluding bias)
X = np.insert(X, 0, 1, axis=1) # Insert bias feature to make 5000x401 2d numpy

# 5000x1 matrix (5000 headers associated with each training example)
# Values range from 1-10 (10 means 0 bc indexing starts at 1 in matlab)
y = mat1["y"] # 5000x1 2d numpy

# Parameters
Theta1 = mat2["Theta1"] # 25x401 matrix
Theta2 = mat2["Theta2"] # 10x26 matrix
numHidden = Theta1.shape[0] # Number of units in hidden layer
numClasses = Theta2.shape[0] # Number of classes

# Take a random sample of X and y to reduce computing time
# Can take multiple different samples and use them one at a time, results should be similar
sampSize = 100
X_samp, y_samp = sample(X,y,sampSize) # X_samp is 100x401, Y_samp is 100x1

################################################## EXECUTE ##################################################
# Initial guess for theta
# Num of entries must equal num of features (inc bias)
#theta = np.zeros((X.shape[1],1))


#pred = listOfPredictions(X, Theta1, Theta2)
#print(checkAcc(pred,y))
