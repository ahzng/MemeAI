## Introduction

:wave:Hey there! My name is SymbolAI. I have been trained to recognize grayscale images of human symbols, such as numbers, letters, and mathematical notation.

On average, I have a ***95% success rate***. Yeah, I tend to struggle with my 5's and S's.

"insert some images of test data here"

## How It Works

I designed all algorithms from scratch using just the NumPy module in Python. To make the process more efficient, I have developed several cases of vectorization and matrix multiplication where applicable.

First, I implemented an artificial neural network capable of both forward and back propogation to make classification predictions. It takes input data (the greyscale values of every single pixel in the 20x20 image), passes through a hidden layer, then gives an output about the probability of each symbol being the actual symbol in the image. The output unit with the highest probability is deemed as the prediction.

"image of neural network structure"

Initially, SymbolAI was absolutely clueless, and the guesses were chaotic and random. To teach SymbolAI how to learn, I implemented a gradient descent algorithm to minimize the cost function. Over time, I took more and more steps in the direction of the negative gradient to find the optimal weights and biases that minimized error of each output unit in my neural network.

Using the optimized parameters, SymbolAI is now able to venture past the training data and make pretty accurate predctions on totally unique images that it has never encountered before!

## Putting It Into Practice

### Training single digit image recognition:

 from using a training set of 5000 training examples of single digit numbers with 400 features each (obtained from data provided by Andrew Ng's Machine Learning course on Coursera):

- Learned parameters can be referenced in Theta1.csv and Theta2.csv
- Training accuracy: 95.3%
- ![Figure_10](https://user-images.githubusercontent.com/106856325/172764862-041f9e4f-55d0-497e-90b5-0dbaf7dac64e.png | width=100)


### Training letter recognition:

- coming soon

### Training mathematical symbol recognition:

- coming soon

## Conclusion/Future Work

SymbolAI demonstrates the incredible potential that AI computer vision can have. If humanity wants to create embodied autonomous systems that change the world, visual sensors will be an absolutely crucial component.

Future plans include implementing an interactive feature where the user can draw their own images, and extending image complexity by introducing RGB pixels.
