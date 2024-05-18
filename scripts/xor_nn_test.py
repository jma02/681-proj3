from neural_network import NeuralNetwork
import numpy as np
from activation import sigmoid, relu

def main():
    X = np.array([
        [[0],[0]], 
        [[0],[1]],
        [[1],[0]],
        [[1],[1]]
    ])
    Y = np.array([
        [[0]],
        [[1]],
        [[1]],
        [[0]]
    ])
    nn = NeuralNetwork([2,10,5,1])
    nn.set_verbose(False)
    nn.SGD(X, Y, learning_rate=.1, epochs=10000, batch_size=1)
    for x, y in zip(X, Y):
        print(f"Input: {x}, Target: {y}, Prediction: {nn.classify(x)}")

main()