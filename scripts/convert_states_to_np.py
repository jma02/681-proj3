import numpy as np
import pickle
from hexapawn_game_states import states

X = list(states.keys())
Y = states.values()

X = list(map(lambda x : x.strip('[]').split(), X))
X = list(map(lambda x : list(map(lambda y : [int(y)], x)), X))

def helper(x: list, y: dict) -> list:
    ret = [0] * 9
    optimal = None
    if(x[0] == 1):
        optimal = y[1] if 1 in y else y[-1]
    else:
        optimal = y[-1] if -1 in y else y[1]
    for a in optimal:
        _, _, to = a
        ret[to-1] = 1
    return [ret]

Y = list(map(
    lambda y: helper(*y)
, zip(X, Y)))
X = np.array([np.array(elem) for elem in X])
Y = np.array([np.array(elem).T for elem in Y])
print(X[1].shape)
print(Y[1].shape)
with open('hexapawn_dataset_X.pickle', 'wb') as file:
    pickle.dump(X, file)

# Save Y to a pickle file
with open('hexapawn_dataset_Y.pickle', 'wb') as file:
    pickle.dump(Y, file)