## Introduction

:wave:Hey there! My name is SymbolAI. I have been trained to recognize grayscale images of human symbols, such as numbers, letters, and mathematical notation.

On average, I have a ***95% success rate***. Yeah, I tend to struggle with my 5's and S's.

"insert some images of test data here"

## How It Works

I designed all algorithms from scratch using just the NumPy module in Python. To make the process more efficient, I have developed several cases of vectorization and matrix multiplication where applicable.

First, I implemented an artificial neural network capable of both forward and back propogation to make classification predictions. It takes input data (the greyscale values of every single pixel in the 20x20 image), passes through a hidden layer, then gives an output about the probability of each symbol being the actual symbol in the image. The output unit with the highest probability is deemed as the prediction.

"image of neural network structure"

Initially, SymbolAI was absolutely clueless, and the guesses were chaotic and random. To teach SymbolAI how to learn, I implemented a gradient descent algorithm to minimize the cost function. Over time, I took more and more steps in the direction of the negative gradient to find the optimal weights and biases that minimized error of each output unit in my neural network.

"image of speed of convergence"

Using the optimized parameters, SymbolAI is now able to venture past the training data and make pretty accurate predctions on totally unique images that it has never encountered before!

## Putting It Into Practice

Here are the results from using a training set of 5000 training examples with 400 features each (obtained from data provided by Andrew Ng's Machine Learning course on Coursera):

![Figure_10](https://user-images.githubusercontent.com/106856325/172764862-041f9e4f-55d0-497e-90b5-0dbaf7dac64e.png)

## Conclusion/Future Work

- interactive feature
- rgb
