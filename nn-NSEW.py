# Neural Network for recognising shaded squares, N, E, S, W

import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, x, y):
        # Matrix 4 rows by 17 cols (4x4 bits + a bias) in this example:
        self.input      = x
        # Matrix 17 rows by 4 cols
        self.weights1   = np.random.rand(self.input.shape[1],16)
        # Matrix 4 rows by 2 col
        self.weights2   = np.random.rand(16,2)
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


def clarify_bit(fuzzy_input):
    if fuzzy_input < 0.2:
        return 0
    if fuzzy_input > 0.8:
        return 1
    else:
        return None

def testit(pixels):
    # Test out the neural network with A and B inputs
    nn.input = np.array(pixels + [1])
    nn.feedforward()
    bits = tuple(map(clarify_bit, nn.output))
    return { (0,0): "N",
             (0,1): "S",
             (1,0): "E",
             (1,1): "W" }.get(bits, "?")

    # E.g.
    # testit([0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1]) #S
    # testit([1,1,1,1, 1,1,1,1, 0,0,0,0, 0,0,0,0]) #N
    # testit([0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1]) #E
    # testit([1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0]) #W

def pretty_print(bits):
    if len(bits):
        print(bits[:4])
        pretty_print(bits[4:])

if __name__ == "__main__":

    # Inputs with bias, our training data
    X = np.array([
        [1,1,1,1,
         1,1,1,1,
         0,0,0,0,
         0,0,0,0,1],

        [0,0,0,0,
         0,0,0,0,
         1,1,1,1,
         1,1,1,1,1],

        [0,0,1,1,
         0,0,1,1,
         0,0,1,1,
         0,0,1,1,1],

        [1,1,0,0,
         1,1,0,0,
         1,1,0,0,
         1,1,0,0,1]
    ])

    # These are the results we're trying to generate, the
    # results for the training data
    #              N      S      E      W
    y = np.array([[0,0], [0,1], [1,0], [1,1]])
    nn = NeuralNetwork(X,y)

    # Now do the training
    for i in range(10000):
        nn.feedforward()
        debug = False # (i % 1000 == 0)
        if debug:
            print("Iteration", i)
        nn.backprop(debug)

    print("Output for the training data...")
    print(nn.output)
    print()

    # Now test each input -- add more if you like:
    for bits in [[0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1],
                 [1,1,1,1, 1,1,1,1, 0,0,0,0, 0,0,0,0],
                 [0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1],
                 [1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0],

                 [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
                 [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
                 [1,0,1,0, 0,1,0,1, 0,0,0,0, 0,0,0,0],
                 [1,0,0,0, 1,1,0,0, 0,1,0,0, 0,1,0,0],
                 ]:
        print("Input")
        pretty_print(bits)
        print("produces output", testit(bits))

