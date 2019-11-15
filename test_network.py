import numpy as np
from deepq import *

state = [0,1,0,1, 1,0,0,1, 0,1,0,1]
state = np.array(state)
nn = NeuralNetwork(2, [16, 32], 'adam')
q_learning(nn.model, nn.model, state)
"""
print("Epsilon is " + str(0.5))

for i in range(11):
    action = epsilon_greedy_policy(nn, 0.5, i/10, 0)
    print("Random number is " + str(i/10))
    print("Action taken is " + str(action))
"""
