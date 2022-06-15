## Introduction

SymbolAI is an AI trained to recognize 20x20 pixel images of human symbols.

On average, SymbolAI has a ***95% success rate***.

*Examples of images that it can read.*

<img src="https://user-images.githubusercontent.com/106856325/173853045-bdb886f9-eea0-445f-b35e-8f72d841cabf.png" width="800">

## How It Works

I designed all algorithms from scratch using only the NumPy module in Python. To make the process more efficient, I have developed several cases of vectorization and matrix multiplication where applicable.

We implement an artificial neural network capable of both forward and back propagation to make classification predictions. It takes input data (the values of every single pixel in the 20x20 image), passes through a hidden layer, then gives an output about the probability of each symbol being the actual symbol in the image. The output unit with the highest probability is deemed as the prediction. Forward propagation acts as the thought process of SymbolAI in making a prediction, and back propagation acts as feedback by contributing to computing the gradient of the cost.

*Neural network structure for single digit recognition training session.*

<img width="360" alt="image" src="https://user-images.githubusercontent.com/106856325/173197002-c5241e06-d84e-4c4c-8853-ced0647d48da.png">

Initially, SymbolAI was absolutely clueless, and the guesses were chaotic and random. To teach SymbolAI how to learn, we implement a gradient descent algorithm to minimize the cost function. Over time, it takes more and more steps in the direction of the negative gradient to find the optimal weights and biases that minimized error of each output unit in the neural network. The larger the learning rate, the bigger (but less precise) steps it takes. To prevent the issue of overfitting, we add a regularization term to the cost function.

*Error decreases over time due to gradient descent for single digit recognition training session (learning rate alpha = 0.8).*

<img src="https://user-images.githubusercontent.com/106856325/173171486-c9810d2d-65ea-4da1-83fa-682cc5561540.png" width="500">

Using the optimized parameters, SymbolAI is now able to venture past the training data and make pretty accurate predictions on totally unique images that it has never encountered before!

## Putting It Into Practice

For each training session, we use a training set with thousands of training examples and 400 features each (to represent the 400 pixels that make up the image).

### Session 1: Training single digit image recognition

The training set we use is obtained from MATLAB data provided by Stanford's "Machine Learning" course on Coursera. Then, implementing our gradient descent algorithm, the resulting learned parameters can be referenced in Theta1.csv and Theta2.csv.

- 5000 training examples with 400 features each
- 10 classes (to represent digits 0,1,2,3,4,5,6,7,8,9)
- Learning rate alpha = 0.8
- Regularization parameter lambda = 1
- Ran gradient descent for 1000 iterations

***FINAL TRAINING ACCURACY: 95.3%***

## Conclusion/Future Work

SymbolAI, while primitive in the grand scheme of things, demonstrates the incredible potential that AI computer vision can have.

Future plans include implementing an interactive feature where the user can draw their own images, and extending image complexity by introducing RGB pixels.
