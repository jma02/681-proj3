import numpy as np
def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))
def relu(x: np.ndarray):
    return x * (x > 0)

def dsigmoid(x: np.ndarray):
    return sigmoid(x)*(1-sigmoid(x))
def drelu(x: np.ndarray):
    return 1. * (x > 0)
