import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import pandas as pd
import math
from scipy.io import loadmat
import matplotlib.image as mpimg
import sys
import random

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient of sigmoid function
# Works with both scalars and arrays
def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Calculate hypothesis/prediction for logistic regression
def predictLog(x, theta):
    return sigmoid( np.dot(x, theta) )

# Computes vectorized unregularized cost function for logistic regression with theta (theta_0 to theta_n) as input value
# Using '*' between two col vectors means multiply corresponding entries, not dot product
# Postcondition: Returns float value
def cost_log(X, y, theta):
    m = X.shape[0]

    # Vector of all h(xi)
    h_vals = sigmoid(X @ theta)

    # Put each term (from 1 to m) in summation into vector
    term1 = -y * np.log(h_vals) - (1 - y) * np.log(1 - h_vals)

    # Add up all the terms in vector and multiply it by 1/m
    res = (1/m) * sum(term1)

    return res

# Computes vectorized regularized cost function for logistic regression with theta (theta_0 to theta_n) as input value
# Using '*' between two col vectors means multiply corresponding entries, not dot product
# Combats overfitting
# Postcondition: Returns float value
def regCost_log(X, y, theta, lamb):
    m = X.shape[0]

    # Vector of all h(xi)
    h_vals = sigmoid(X @ theta)

    # Put each term (from 1 to m) in summation into vector
    term1 = -y * np.log(h_vals) - (1 - y) * np.log(1 - h_vals)

    res = (1/m) * sum(term1) + (lamb/(2*m)) * sum(theta[1:,0]**2)

    return res[0]

# Implement nonregularized gradient descent to find the optimal parameters for minimizing the cost function
# in both logistic reg and linear reg (you would just need to redefine the h function)
# Precondition:
#   X: Matrix where rows equal num of training ex and columns equal num of features (inc bias)
#   y: Column vector with length equal to num of training examples
#   theta:
# Returns tuple of optimized theta and cost history
def gradDesc_log(X, y, theta, alpha, num_iters):
    m = y.shape[0]

    # Create space to store values of error with each iteration
    J_hist = np.zeros(num_iters+1) # 1d numpy
    J_hist[0] = cost_log(X, y, theta)

    # Run this for a specified number of iterations
    for k in range(0, num_iters):

        # Initialize delta (aka the summation term)
        delta = np.zeros((theta.shape[0], 1), dtype='float64') # 2d numpy w/ shape (n+1,1)

        # Define delta
        for i in range(0, m):

            # Define the particular training example to be used
            x = np.array( [X[i, :]] ) # 2d numpy w/ shape (1,n+1)

            # Update delta
            delta = delta + ( predictLog(x, theta) - y[i, 0] ) * x.T # x.T is 2d numpy w/ shape (n+1,1), which matches delta

        # Update theta
        theta = theta - alpha * (1/m) * delta

        # Update cost history
        J_hist[k+1] = cost_log(X, y, theta)

    # To access theta: gradientDescent(...)[0]
    # To access J_hist: gradientDescent(...)[1]
    return theta, J_hist

# Implement regularized gradient descent to find the optimal parameters for minimizing the cost function
# in logistic regression only
# Precondition:
#   X: Matrix where rows equal num of training ex and columns equal num of features (inc bias)
#   y: Column vector with length equal to num of training examples
#   theta:
# Returns tuple of optimized theta and cost history
def regGradDesc_log(X, y, theta, alpha, lamb, num_iters):
    m = y.shape[0]

    # Create space to store values of error with each iteration
    J_hist = np.zeros(num_iters + 1)  # 1d numpy
    J_hist[0] = cost_log(X, y, theta)

    # Run this for a specified number of iterations
    for k in range(0, num_iters):

        # Initialize delta (aka the summation term)
        delta = np.zeros((theta.shape[0], 1), dtype='float64')  # 2d numpy w/ shape (n+1,1)

        # Define delta
        for i in range(0, m):
            # Define the particular training example to be used
            x = np.array([X[i, :]])  # 2d numpy w/ shape (1,n+1)

            # Update delta
            delta = delta + (predictLog(x, theta) - y[i, 0]) * x.T  # x.T is 2d numpy w/ shape (n+1,1), which matches delta

        # Update theta
        # Note: Returning part of a 2d numpy array such as theta[1:,0] gives the values but in shape of a 1d array
        # In this case, it doesn't really matter bc you're just changing the values
        theta[0,0] = theta[0,0] - alpha * (1 / m) * delta[0,0]
        theta[1:,0] = theta[1:,0] * (1 - alpha * (lamb/m)) - alpha * (1 / m) * delta[1:,0]

        # Update cost history
        J_hist[k + 1] = regCost_log(X, y, theta, lamb)

    # To access theta: gradientDescent(...)[0]
    # To access J_hist: gradientDescent(...)[1]
    return theta, J_hist

# Take a random sample of training examples to represent entire training set population
# This can significantly reduce computing time
# Precondition
#   X: includes bias
#   samp_size is smaller than total number of training examples
def sample(X, y, samp_size):
    # Number of features (excluding bias), n is defined this way to match notation consistency of the mathematical formulas
    # Include -1 term bc this function assumes X already includes the bias
    n = X.shape[1] - 1

    # Initialize samples
    X_samp = np.zeros((samp_size, n + 1))  # Includes space for bias feature
    y_samp = np.zeros((samp_size, 1))

    # Choose random examples
    samp = random.sample(range(0, 5000), samp_size)
    samp = np.array(samp)  # 1d numpy

    # Create samples
    for i in range(0, samp_size):
        X_samp[i] = X[samp[i]]
        y_samp[i] = y[samp[i]]

    # tuple
    return X_samp, y_samp

# Calculate hypothesis for a neural net with 1 hidden layer
# Can be improved by implementing batches of inputs instead of one training example at a time, which removes the
# need for a for loop
# Precondition:
#   X has bias unit
#   i must be in the range 0 to m, excluding m
#   Theta1 and Theta2 are the appropriate dims depending on
# Postcondition:
def predict(X, i, Theta1, Theta2):
    # Input layer
    a1 = X[i] # 1d numpy that already includes bias unit

    # Hidden layer
    a2 = sigmoid(Theta1 @  a1) # 1d numpy
    a2 = np.reshape(a2, (a2.shape[0],1)) # Change from 1d to 2d numpy
    a2 = np.insert(a2, 0, 1, axis=0) # Insert bias a2_0

    # Output layer
    # NOTE: The a2 we initially obtain does not include bias unit. However, when calculating a3, we must define a2 as
    # including a bias unit.
    a3 = sigmoid(Theta2 @ a2) # 2d numpy (num of classes by 1)

    # Return index + 1 of the largest valued class. This number is equivalent to the actual predicted number.
    # For ex, if pred = 10, the predicted number is 0. The number 0 is associated with the 10th entry or index #9
    pred = np.argmax(a3) + 1 # Scalar

    return a3, pred

# Calculate predictions for a neural net with 1 hidden layer
# Can be improved by implementing batches of inputs instead of one training example at a time, which removes the
# need for a for loop
# Precondition:
#   X has bias unit
#   Theta1 and Theta2 are the appropriate dims depending on
# Postcondition:
#   preds:
def listOfPredictions(X, Theta1, Theta2):
    m = X.shape[0]

    # Initialize 1d numpy of predictions for each training example
    preds = np.array([])

    for i in range(0,m):

        a3 = predict(X, i, Theta1, Theta2)[0] # 2d numpy (num of classes by 1)

        # Return index + 1 of the largest valued class. This number is equivalent to the actual predicted number.
        # For ex, if pred = 10, the predicted number is 0. The number 0 is asssociated with the 10th entry or index #9
        preds = np.append(preds, np.argmax(a3) + 1)

    preds = np.reshape(preds, (preds.shape[0],1)) # Reshape from 1d to 2d array

    return preds

# Keep for now
# Calculate predictions for a neural net with 1 hidden layer
# Can be improved by implementing batches of inputs instead of one training example at a time, which removes the
# need for a for loop
# Precondition:
#   X has bias unit
#   Theta1 and Theta2 are the appropriate dims depending on
# Postcondition:
#   preds:
def listOfPredictions2(X, Theta1, Theta2):
    m = X.shape[0]

    # Initialize 1d numpy of predictions for each training example
    preds = np.array([])

    # For each training example
    for i in range(0,m):

        # Input layer
        a1 = X[i] # 1d numpy that already includes bias unit

        # Hidden layer
        a2 = sigmoid(Theta1 @  a1) # 1d numpy
        a2 = np.reshape(a2, (a2.shape[0],1)) # Change from 1d to 2d numpy
        a2 = np.insert(a2, 0, 1, axis=0) # Insert bias a2_0

        # Output layer
        # NOTE: The a2 we initially obtain does not include bias unit. However, when calculating a3, we must define a2 as
        # including a bias unit.
        a3 = sigmoid(Theta2 @ a2) # 2d numpy (num of classes by 1)

        # Returns predicted class (aka class with highest value). Keep in mind +1 makes up for difference in matlab indexing
        # and python indexing. Future data sets take the +1 off maybe.
        preds = np.append(preds, np.argmax(a3) + 1) # Append index of the largest valued class

    preds = np.reshape(preds, (preds.shape[0],1)) # Reshape from 1d to 2d array

    return preds

# Check accuracy of prediction by comparing predicted values with actual values
# pred: column vector in form of numpy 2d array
# y: column vector in form of numpy 2d array
def checkAcc(pred,y):
    m = y.shape[0]
    numCorrect = 0
    for i in range(0,pred.shape[0]):
        if pred[i,0] == y[i,0]:
            numCorrect += 1
    return numCorrect / m

# Randomly initialize weights
def initWeights(lay, nextLay):
    eps = (6 ** (1/2)) / (lay + nextLay) ** (1/2)
    weights = np.random.rand(nextLay, lay+1) * (2 * eps) - eps
    return weights

# Computes unregularized cost for a neural network with 1 hidden layer
# Postcondition: Returns float value
def calcCost(X, y, Theta1, Theta2):
    m = X.shape[0] # Number of training examples
    numClasses = Theta2.shape[0] # Number of classes

    term = 0.0

    # For each training example
    for i in range(0,m):

            # Vector h(xi), which consists of h(xi)_1, h(xi)_2, ...,h(xi)_K
            hxi = predict(X, i, Theta1, Theta2)[0] # 2d numpy (num of classes by 1)

            # Vector yi,  where one entry is 1 and other entries are 0
            yi = np.zeros((numClasses,1)) # 2d numpy (num of classes by 1)
            yi[y[i]-1] = 1 # Set the index corresponding with the actual number with one, other indices remain 0

            # Put each term (from 1 to K) in summation into vector
            temp = -yi * np.log(hxi) - (1 - yi) * np.log(1 - hxi)

            term += sum(temp)

    res = (1/m) * term

    return res[0]

# Computes regularized cost for a neural network with 1 hidden layer
# Postcondition: Returns float value
def calcRegCost(X, y, Theta1, Theta2, lamb):
    m = X.shape[0]

    # Account for cost of each class in each training example (unregularized cost function)
    term1 = calcCost(X,y,Theta1,Theta2)

    # Regularize parameters in Theta1 (except biases)
    subTheta1 = np.delete(Theta1, 0, axis=1) # Remove biases bc they don't need to be regularized
    subTheta1 = np.square(subTheta1) # Square all entries
    sum1 = subTheta1.sum() # Sum of all entries

    # Regularize parameters in Theta2 (except biases)
    subTheta2 = np.delete(Theta2, 0, axis=1) # Remove biases bc they don't need to be regularized
    subTheta2 = np.square(subTheta2) # Square all entries
    sum2 = subTheta2.sum()  # Sum of all entries

    # Regularization term
    term2 = sum1 + sum2

    res = term1 + (lamb/(2*m)) * term2

    return res

# Computes unregularized cost function and its gradient for a neural network with 1 hidden layer
# Postcondition:
#   cost: Cost of neural network given the inputted parameters
def costFunction(X, y, Theta1, Theta2):
    m = X.shape[0]  # Number of training examples
    numInputs = X.shape[1] - 1  # Number of features/inputs (exc bias)
    numHidden = Theta1.shape[0]  # Number of hidden (exc. bias)
    numClasses = Theta2.shape[0]  # Number of classes

    # Initialize
    costTerm = 0.0  # Will be used for computing cost
    Delta1 = np.zeros(Theta1.shape) # Will be used for computing gradient of Theta1
    Delta2 = np.zeros(Theta2.shape) # Will be used for computing gradient of Theta2

    # For each training example
    for i in range(0, m):
        ############ FORWARD PROP ###########

        # Input layer
        a1 = X[i]  # 1d numpy that already includes bias unit

        # Hidden layer
        a2 = sigmoid(Theta1 @ a1)  # 1d numpy
        a2 = np.reshape(a2, (a2.shape[0], 1))  # Change from 1d to 2d numpy
        a2 = np.insert(a2, 0, 1, axis=0)  # Insert bias a2_0

        # Output layer
        # NOTE: The a2 we initially obtain does not include bias unit. However, when calculating a3, we must define a2 as
        # including a bias unit.
        a3 = sigmoid(Theta2 @ a2)  # 2d numpy (num of classes by 1)

        # Vector yi,  where one entry is 1 and other entries are 0
        yi = np.zeros((numClasses, 1))  # 2d numpy (num of classes by 1)
        yi[y[i] - 1] = 1  # Set the index corresponding with the actual number with one, other indices remain 0

        # Will be used for computing cost
        # Put each term (from 1 to K) in summation into vector
        temp = -yi * np.log(a3) - (1 - yi) * np.log(1 - a3)
        costTerm += sum(temp)

        ########### BACK PROP ###########

        # Output layer
        d3 = a3 - yi  # 10x1 2d numpy

        # Hidden layer
        d2 = (Theta2.T @ d3) * (a2 * (1 - a2))  # 26x1 2d numpy
        # d2 = (Theta2.T @ d3) * sigmoidGradient(Theta1 @ a1)
        d2 = np.delete(d2, 0, axis=0)  # Delete bias d2_0 to make 25x1 2d numpy

        ########## Update Delta ##########
        a1 = np.reshape(a1, (a1.shape[0], 1))  # Convert a1 from 1d to 2d numpy in order for matmul to work
        Delta1 = Delta1 + d2 @ a1.T
        Delta2 = Delta2 + d3 @ a2.T

    cost = (1 / m) * costTerm
    grad1 = (1 / m) * Delta1
    grad2 = (1 / m) * Delta2

    return cost[0], grad1, grad2

# Computes everything that is useful for gradient descent, including:
# Unregularized and regularized cost and all of the gradients for a neural network with 1 hidden layer
# Postcondition:
# Note: while this code includes a function specifically designed for only calculating the cost (costFunction), there is no such
# code written specifically for regularized cost bc such a code would be nearly identical to this function, ultimateCostFunction
#   cost and regCost are floats
#   Gradients are the same size with their respectiv Thetas
def ultimateCostFunction(X, y, Theta1, Theta2, lamb):
    m = X.shape[0]  # Number of training examples
    numInputs = X.shape[1] - 1  # Number of features/inputs (exc bias)
    numHidden = Theta1.shape[0]  # Number of hidden (exc. bias)
    numClasses = Theta2.shape[0]  # Number of classes

    # Initialize
    costTerm = 0.0  # Will be used for computing cost
    Delta1 = np.zeros(Theta1.shape) # Will be used for computing gradient of Theta1
    Delta2 = np.zeros(Theta2.shape) # Will be used for computing gradient of Theta2

    # For each training example
    for i in range(0, m):
        ############ FORWARD PROP ###########

        # Input layer
        a1 = X[i]  # 1d numpy that already includes bias unit

        # Hidden layer
        a2 = sigmoid(Theta1 @ a1)  # 1d numpy
        a2 = np.reshape(a2, (a2.shape[0], 1))  # Change from 1d to 2d numpy
        a2 = np.insert(a2, 0, 1, axis=0)  # Insert bias a2_0

        # Output layer
        # NOTE: The a2 we initially obtain does not include bias unit. However, when calculating a3, we must define a2 as
        # including a bias unit.
        a3 = sigmoid(Theta2 @ a2)  # 2d numpy (num of classes by 1)

        # Vector yi,  where one entry is 1 and other entries are 0
        yi = np.zeros((numClasses, 1))  # 2d numpy (num of classes by 1)
        yi[y[i] - 1] = 1  # Set the index corresponding with the actual number with one, other indices remain 0

        # Will be used for computing cost
        # Put each term (from 1 to K) in summation into vector
        temp = -yi * np.log(a3) - (1 - yi) * np.log(1 - a3)
        costTerm += sum(temp)

        ########### BACK PROP ###########

        # Output layer
        d3 = a3 - yi  # 10x1 2d numpy

        # Hidden layer
        d2 = (Theta2.T @ d3) * (a2 * (1 - a2))  # 26x1 2d numpy
        # d2 = (Theta2.T @ d3) * sigmoidGradient(Theta1 @ a1)
        d2 = np.delete(d2, 0, axis=0)  # Delete bias d2_0 to make 25x1 2d numpy

        ########## UPDATE DELTA ##########
        a1 = np.reshape(a1, (a1.shape[0], 1))  # Convert a1 from 1d to 2d numpy in order for matmul to work
        Delta1 = Delta1 + d2 @ a1.T
        Delta2 = Delta2 + d3 @ a2.T

    ########## REGULARIZE PARAMETERS ##########
    # Regularize parameters in Theta1 (except biases)
    subTheta1 = np.delete(Theta1, 0, axis=1)  # Remove biases bc they don't need to be regularized
    subTheta1 = np.square(subTheta1)  # Square all entries
    sum1 = subTheta1.sum()  # Sum of all entries

    # Regularize parameters in Theta2 (except biases)
    subTheta2 = np.delete(Theta2, 0, axis=1)  # Remove biases bc they don't need to be regularized
    subTheta2 = np.square(subTheta2)  # Square all entries
    sum2 = subTheta2.sum()  # Sum of all entries

    # Regularization term
    regCostTerm = sum1 + sum2

    ########## RETURN VALUES ##########

    # Cost
    cost = (1 / m) * costTerm

    # Gradient matrix for Theta1
    grad1 = (1 / m) * Delta1

    # Gradient matrix for Theta2
    grad2 = (1 / m) * Delta2

    # Regularized cost
    regCost = cost + (lamb / (2 * m)) * regCostTerm

    # Same as grad1 and grad2 except add a regularization term to every entry but keep the biases unchanged
    # Reg gradient matrix for Theta1
    regGrad1 = (1 / m) * Delta1 + (lamb/m) * Theta1
    regGrad1[:, 0] = grad1[:, 0]

    # Reg gradient matrix for Theta2
    regGrad2 = (1 / m) * Delta2 + (lamb / m) * Theta2
    regGrad2[:, 0] = grad2[:, 0]

    return cost[0], grad1, grad2, regCost[0], regGrad1, regGrad2

# Implement nonregularized gradient descent to find the optimal weights for minimizing cost
# Precondition:
#   X: Matrix where rows equal num of training ex and columns equal num of features (inc bias)
#   y: Column vector with length equal to num of training examples
def gradDesc(X, y, Theta1, Theta2, alpha, numIters):

    # Create space to store values of error with each iteration
    J_hist = np.zeros(numIters + 1)  # 1d numpy
    J_hist[0] = calcCost(X, y, Theta1, Theta2)

    # Run this for a specified number of iterations
    for k in range(0, numIters):

        # Define cost and gradients
        cost, grad1, grad2 = costFunction(X, y, Theta1, Theta2)[0:3]

        # Update weights
        Theta1 = Theta1 - alpha * grad1
        Theta2 = Theta2 - alpha * grad2

        # Update cost history
        J_hist[k + 1] = cost

    return Theta1, Theta2, J_hist

# Implement regularized gradient descent to find the optimal weights for minimizing cost
# Precondition:
#   X: Matrix where rows equal num of training ex and columns equal num of features (inc bias)
#   y: Column vector with length equal to num of training examples
def regGradDesc(X, y, Theta1, Theta2, alpha, lamb, numIters):

    # Create space to store values of error with each iteration
    J_hist = np.zeros(numIters + 1)  # 1d numpy
    J_hist[0] = calcRegCost(X, y, Theta1, Theta2, lamb)

    # Run this for a specified number of iterations
    for k in range(0, numIters):

        # Define cost and gradients
        regCost, regGrad1, regGrad2 = ultimateCostFunction(X, y, Theta1, Theta2, lamb)[3:]


        # Update weights
        Theta1 = Theta1 - alpha * regGrad1
        Theta2 = Theta2 - alpha * regGrad2

        # Update cost history
        J_hist[k + 1] = regCost

    return Theta1, Theta2, J_hist
