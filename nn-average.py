# Can we use a neural network to compute an average?

import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    """ A neural network that takes multiple rows of input, passes them
    through 4 nodes in hidden layer and maps to a set of outputs (one for
    each row of input.
    """

    def __init__(self, x, y):
        self.input      = x
        # 4 nodes in hidden layer with a weight for each connection to input width
        self.weights1   = np.random.rand(self.input.shape[1],4)
        # Weights to map 4 nodes to single output node
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        # Each row of input maps to a row in the hidden layer (layer1)
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        # Map each row in layer1 to output row
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        # To start with these outputs will simply be random
        # based on random weights

    def backprop(self):
        # application of the chain rule to find derivative of the loss
        # function with respect to weights2 and weights1

        # errors are defined as the difference between what we want
        # and what we have at the moment: self.y - self.output
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))

        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


def testit(a,b):
    # Test out the neural network with A and B inputs
    nn.input = np.array([a,b,1])
    nn.feedforward()
    print(nn.output)

# Inputs with bias, our training data 
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# These are the results we're trying to generate, the training data
y = np.array([[0],[0.5],[0.5],[1]])
nn = NeuralNetwork(X,y)

# Now do the training
for i in range(1000): #15000
    nn.feedforward()
    nn.backprop()

# The result
print(nn.output)
    
# Now for some testing... especially some values we didn't
# train with
print("Testing...")
testit(1,1)
testit(0,0)
testit(0.5,0.5) # 0.5?
testit(1,0.5) # 0.75?

# Inspired by and initial code from:
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
