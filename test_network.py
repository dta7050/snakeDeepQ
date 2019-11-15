import numpy as np
from deepq import *

state = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
pre_state = np.array(state)
nn = NeuralNetwork(1, [24, 12], 'adam')
nn.train_on_timestep(pre_state, pre_state, 0, 10)
"""
print("Epsilon is " + str(0.5))

for i in range(11):
    action = epsilon_greedy_policy(nn, 0.5, i/10, 0)
    print("Random number is " + str(i/10))
    print("Action taken is " + str(action))
"""
