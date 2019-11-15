from deepq import *

nn = NeuralNetwork(2, [16, 32], 'adam')

print("Epsilon is " + str(0.5))

for i in range(11):
    action = epsilon_greedy_policy(nn, 0.5, i/10, 0)
    print("Random number is " + str(i/10))
    print("Action taken is " + str(action))

