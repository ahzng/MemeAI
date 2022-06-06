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

# Defined hypothesis for logistic regression
def h(x, theta):
    return sigmoid( np.dot(x, theta) )

# Computes vectorized unregularized cost function for logistic regression with theta (theta_0 to theta_n) as input value
# Using '*' between two col vectors means multiply corresponding entries, not dot product
# Postcondition: Returns float value
def computeCost(X, y, theta):
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
def computeRegCost(X, y, theta, lamb):
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
def gradDesc(X, y, theta, alpha, num_iters):
    m = y.shape[0]

    # Create space to store values of error with each iteration
    J_hist = np.zeros(num_iters+1) # 1d numpy
    J_hist[0] = computeCost(X, y, theta)

    # Run this for a specified number of iterations
    for k in range(0, num_iters):

        # Initialize delta (aka the summation term)
        delta = np.zeros((theta.shape[0], 1), dtype='float64') # 2d numpy w/ shape (n+1,1)

        # Define delta
        for i in range(0, m):

            # Define the particular training example to be used
            x = np.array( [X[i, :]] ) # 2d numpy w/ shape (1,n+1)

            # Update delta
            delta = delta + ( h(x, theta) - y[i, 0] ) * x.T # x.T is 2d numpy w/ shape (n+1,1), which matches delta

        # Update theta
        theta = theta - alpha * (1/m) * delta

        # Update cost history
        J_hist[k+1] = computeCost(X, y, theta)

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
def regGradDesc(X, y, theta, alpha, lamb, num_iters):
    m = y.shape[0]

    # Create space to store values of error with each iteration
    J_hist = np.zeros(num_iters + 1)  # 1d numpy
    J_hist[0] = computeCost(X, y, theta)

    # Run this for a specified number of iterations
    for k in range(0, num_iters):

        # Initialize delta (aka the summation term)
        delta = np.zeros((theta.shape[0], 1), dtype='float64')  # 2d numpy w/ shape (n+1,1)

        # Define delta
        for i in range(0, m):
            # Define the particular training example to be used
            x = np.array([X[i, :]])  # 2d numpy w/ shape (1,n+1)

            # Update delta
            delta = delta + (h(x, theta) - y[i, 0]) * x.T  # x.T is 2d numpy w/ shape (n+1,1), which matches delta

        # Update theta
        # Note: Returning part of a 2d numpy array such as theta[1:,0] gives the values but in shape of a 1d array
        # In this case, it doesn't really matter bc you're just changing the values
        theta[0,0] = theta[0,0] - alpha * (1 / m) * delta[0,0]
        theta[1:,0] = theta[1:,0] * (1 - alpha * (lamb/m)) - alpha * (1 / m) * delta[1:,0]

        # Update cost history
        J_hist[k + 1] = computeRegCost(X, y, theta, lamb)

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
