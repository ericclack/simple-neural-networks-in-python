# Exploring sigmoid and sigmoid_derivative functions with some simple
# plots drawn with Pygame Zero.

import numpy as np

TITLE = "Sigmoid graph"
WIDTH = 500
HEIGHT = 500

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
GRAY = (100, 100, 100)

def sigmoid(x):
    """Scale x to be between zero and 1. Most active range between -4 and 4."""
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    """The gradient of the sigmoid function at x???"""
    return sigmoid(x) * (1.0 - sigmoid(x))

def plot(x, y, xr, yr, colour):
    """Scale x and y and draw a point"""
    sx = WIDTH/2 + WIDTH * x/xr
    sy = HEIGHT/2 - HEIGHT * y/yr
    screen.draw.rect(Rect((sx,sy), (2,2)), colour)
    if colour != GRAY:
        tabs = "\t" * 2 if colour == GREEN else 0
        print("%s (%.2f, %.2f)" % (tabs, x,y))

def draw():

    # Grid lines
    for x in range(-10, 10):
        for fraction in range(0, 10):
            x2 = x + fraction/10
            plot(x2, 0, 15, 1, GRAY)
            plot(0, x2, 1, 15, GRAY)

    for x in range(-10, 10):
        for fraction in range(1, 10):
            x2 = x + fraction/10
            plot(x2, sigmoid(x2), 15, 10, WHITE)
            plot(x2, sigmoid_derivative(x2), 15, 10, GREEN)