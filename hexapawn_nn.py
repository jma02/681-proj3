import pickle
from neural_network import NeuralNetwork
from activation_functions.activation import relu, drelu

with open('hexapawn_dataset_X.pickle', 'rb') as file:
    X = pickle.load(file)

with open('hexapawn_dataset_Y.pickle', 'rb') as file:
    Y = pickle.load(file)

"""
# I'm a little short on time, so I won't experiment
# the result into a pickle file.
hexapawn_nn = NeuralNetwork([10, 64, 32, 16, 9])
hexapawn_nn.SGD(X, Y, learning_rate=.001, batch_size=1, epochs = 10000)

 
for x, y in zip(X, Y):
    print(f"Input: {x}, \
            Target: {y}, \
            Prediction: {hexapawn_nn.classify(x)}")

with open('hexapawn_trained_model_sigmoid.pickle', 'wb') as file:
    pickle.dump(hexapawn_nn, file)
"""
hexapawn_nn = NeuralNetwork([10,20,9])

hexapawn_nn.SGD(X,Y,learning_rate=.00001, batch_size=1, epochs=10000, activation=relu, derivative=drelu)
with open('hexapawn_trained_model_relu.pickle', 'wb') as file:
    pickle.dump(hexapawn_nn, file)

for x, y in zip(X, Y):
    print(f"Input: {x}, \
            Target: {y}, \
            Prediction: {hexapawn_nn.classify(x)}")