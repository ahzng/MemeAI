from numberAI import *

################################################## INPUT ##################################################

np.set_printoptions(threshold=sys.maxsize) # print entire array
#np.set_printoptions(threshold = False) # print truncated array

# Neural network structure
numInputs = 400 # Number of features/inputs (exc bias)
numHidden = 25 # Number of hidden (exc bias)
numClasses = 10 # Number of classes

# Iterations
numIters = 1000

# Learning rate
alpha = 0.8

# Lambda
lamb = 1

# Read data
mat1 = loadmat("ex4data1.mat")
mat2 = loadmat("ex4weights.mat")
df1 = pd.read_csv('Theta1.csv') # Learned weights
df2 = pd.read_csv('Theta2.csv') # Learned weights

################################################## DEFINE ##################################################

# Parameters
#Theta1 = mat2["Theta1"] # 25x401 matrix
#Theta2 = mat2["Theta2"] # 10x26 matrix

# Convert the appropriate data in dataframe to numpy 2d array
Theta1 = df1.loc[:,:].to_numpy()
Theta2 = df2.loc[:,:].to_numpy()

# 5000x400 matrix (5000 training examples with 400 features each)
# Each feature is a grayscale pixel value in a 20x20 image
X = mat1["X"]
X = np.insert(X, 0, 1, axis=1) # Insert bias feature to make 5000x401 2d numpy
m = X.shape[0] # Number of training examples

# 5000x1 matrix (5000 headers associated with each training example)
# Values range from 1-10 (10 means 0 bc indexing starts at 1 in matlab)
y = mat1["y"] # 5000x1 2d numpy

# Take a random sample of X and y to reduce computing time
# Can take multiple different samples and use them one at a time, results should be similar
#sampSize = 1000
#X_samp, y_samp = sample(X,y,sampSize) # X_samp is 100x401, Y_samp is 100x1

# Initial guess for theta
#Theta1 = initWeights(numInputs,numHidden) # Returns numHidden by numInputs+1 matrix of weights (inc bias)
#Theta2 = initWeights(numHidden,numClasses) # Returns numClasses by numHidden+1 matrix of weights (inc blias)

################################################## EXECUTE ##################################################

# Optimize weights
#Theta1, Theta2, J_hist = regGradDesc(X, y, Theta1, Theta2, alpha, lamb, numIters)


# Save data for future use
#np.savetxt("trainingSet1.csv", X, delimiter=",") # X includes bias
#np.savetxt("classes1.csv", y, delimiter=",")
#np.savetxt("Theta1.csv", Theta1, delimiter=",")
#np.savetxt("Theta2.csv", Theta2, delimiter=",")


# Check accuracy
pred = listOfPredictions(X, Theta1, Theta2)
print(checkAcc(pred,y))

# Uncomment to plot speed of convergence
'''
x = np.arange(0, numIters + 1)
plt.plot(x,J_hist, label='0.8')

plt.style.use('seaborn-whitegrid')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Speed of convergence')
plt.xlim(0, numIters + 1)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
'''
