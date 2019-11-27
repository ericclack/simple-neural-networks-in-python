# Neural Network for XOR

import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, x, y):
        # Matrix 4 rows by 3 cols in this example:
        self.input      = x
        # Matrix 3 rows by 4 cols
        self.weights1   = np.random.rand(self.input.shape[1],4)
        # Matrix 4 rows by 1 col
        self.weights2   = np.random.rand(4,1)
        # The output we want: a matrix w 4 rows and 1 column
        self.y          = y
        # The output so far
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        # Layer 1 is inputs x weights, fit to range 0..1 by sigmoid
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        # Output is layer1 x weights, again fit with sigmoid
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self, debug=False):
        # application of the chain rule to find derivative of the loss
        # function with respect to weights2 and weights1

        # Difference between desired outputs (y) and what we have so far:
        error = self.y - self.output

        if debug:
            print("Output is", self.output)
            print("Error is", error)

        # Delta weights2 (the change we'll make) is scaled by error
        # and gradient of sigmoid for current outputs
        d_weights2 = np.dot(self.layer1.T, (2 * error * sigmoid_derivative(self.output)))

        d_weights1 = np.dot(self.input.T,  (np.dot(2 * error * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # When are our changes zero?
        # When output == y, or sigmod_derivaive is zero
        # When are they nearly zero?
        # When output is almost y, or output is close to zero or one.

        if debug:
            print(d_weights1)
            print(d_weights2)
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

        if debug: print()


def testit(a,b):
    # Test out the neural network with A and B inputs
    nn.input = np.array([a,b,1])
    nn.feedforward()
    print(nn.output)


if __name__ == "__main__":
    # Why do we have a 1 in 3rd col?
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    # These are the results we're trying to generate, the
    # training data
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    # Now do the training
    for i in range(10000):
        nn.feedforward()
        if debug:
        debug = (i % 1000 == 0)
            print("Iteration", i)
        nn.backprop(debug)

    print(nn.output)


# Inspired by:
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6