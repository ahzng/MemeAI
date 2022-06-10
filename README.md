## Introduction

:wave:Hey there! My name is SymbolAI. I have been trained to recognize grayscale images of human symbols, such as numbers, letters, and mathematical notation.

On average, I have a ***95% success rate***. Yeah, I tend to struggle with my 5's and S's :(

"insert some images here"

## How It Works

I designed all algorithms from scratch using just the NumPy module in Python.

First, I implemented an artificial neural network capable of both forward and back propogation to make classification predictions. It takes in input data (the greyscale values of every single pixel in the image), passes through a hidden layer, then gives an output about the probability of each symbol being the actual symbol in the image. The output unit with the highest probability is deemed as the prediction.

"image of neural network structure"

Then a gradient descent algorithm to learn the optimal weights and biases in my neural network to minimize error.


![Figure_10](https://user-images.githubusercontent.com/106856325/172764862-041f9e4f-55d0-497e-90b5-0dbaf7dac64e.png)
