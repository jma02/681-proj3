import numpy as np
from activation_functions.activation import sigmoid, dsigmoid
from typing import List, Callable, Tuple
class NeuralNetwork:
    def __init__(self, shape: List[int]):
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(shape[:-1], shape[1:])]
        self.biases = list(map(lambda x : np.zeros((x, 1)), shape[1:]))
        self.verbose = False
    
    def set_verbose(self, on: bool = True):
        self.verbose = on

    
    def classify(self, 
                 X: np.ndarray, 
                 activation: Callable[[np.ndarray], np.ndarray]=sigmoid
                 ) -> np.ndarray:
        A = X.copy()
        for b,w in zip(self.biases, self.weights):
            A = activation(np.add(np.dot(w, A), b))
        return A
    
    def backpropagate(self,
                      X: np.ndarray,
                      Y: np.ndarray,
                      activation: Callable[[np.ndarray], np.ndarray]=sigmoid,
                      derivative: Callable[[np.ndarray], np.ndarray]=dsigmoid,
                    ) -> Tuple[np.ndarray, np.ndarray]:
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        # Activations
        a_last = X
        A = [X]
        # pre non-linearity values of linear transformation
        Z = []
        for b,w in zip(self.biases, self.weights):
            z = np.add(np.dot(w, a_last), b)
            Z.append(z)
            a_last = activation(z)
            A.append(a_last)

        if self.verbose:
            print("Loss: ", np.mean((Y - A[-1])**2))
        
        delta = np.multiply(np.multiply((A[-1]-Y), 2),derivative(Z[-1]))
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, A[-2].T)
        for l in range(2, len(self.weights)):
            z = Z[-l]
            ap = derivative(z)
            delta = np.multiply(np.dot(self.weights[-l+1].T, delta),ap)
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, A[-l-1].T)
        
        return (grad_w, grad_b)

    def SGD(self, 
            X_train: np.ndarray, 
            Y_train: np.ndarray, 
            batch_size: int = 32,
            learning_rate: float = 0.001, 
            epochs: int = 1000,
            activation: Callable[[np.ndarray], np.ndarray]=sigmoid,
            derivative: Callable[[np.ndarray], np.ndarray]=dsigmoid
            ):
        n = X_train.shape[0]
        shuffled_indices = np.random.permutation(n)

        for _ in range(epochs):
            for batch_start in range(0, n, batch_size):
                batch_indices = shuffled_indices[batch_start:batch_start + batch_size]
                grad_w_batch = [np.zeros_like(w) for w in self.weights]
                grad_b_batch = [np.zeros_like(b) for b in self.biases]

                for idx in batch_indices:
                    X_batch = X_train[idx]
                    Y_batch = Y_train[idx]

                    grad_w, grad_b = self.backpropagate(
                        X_batch,
                        Y_batch, 
                        activation=activation,
                        derivative=derivative
                        )
                    for i in range(len(self.weights)):
                        grad_w_batch[i] = np.add(grad_w[i], grad_w_batch[i])
                        grad_b_batch[i] = np.add(grad_b[i], grad_b_batch[i])

                batch_size_actual = len(batch_indices)
                self.weights = [w-(learning_rate/batch_size_actual)*nw
                        for w, nw in zip(self.weights, grad_w)]
                self.biases = [b-(learning_rate/batch_size_actual)*nb
                       for b, nb in zip(self.biases, grad_b)]