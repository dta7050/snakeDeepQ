"""
File for implementing Deep Q Learning.

Contents:
    Neural Network class:
        Input Arguments:
            Number of layers
            Type of layers (dense, LSTM, convolution, ...)
            Number of Neurons in each layer
            Optimizer type
            ...
        Methods:
            init(): defines important parameters for the model creation
            save_model(): saves network weights
            load_model(): loads network weights
    Function(s) for implementing Q Learning algorithm
    Function(s) for training network
    Functions(s) for updating policy and target networks
"""
import snake_game
import state

import random
import tensorflow
import keras
import numpy as np

from typing import List
from keras.models import Sequential
from keras.layers import Dense

STATE_SIZE = 12
ACTION_SET_SIZE = 4

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

DIRECTIONS = [LEFT, DOWN, RIGHT, UP]


class NeuralNetwork:

    def __init__(self, num_layers: int, num_neurons: List[int], opt: str):
        """
        For now, network is just simple, vanilla network
        :param num_layers: number of input layers
        :param num_neurons: number of neurons in each hidden layer
        :param opt: optimizer to use in network
        """
        self.model = Sequential()  # initialize the model

        self.model.add(Dense(STATE_SIZE, input_shape=(12,)))  # input layer

        for i in range(num_layers):
            self.model.add(Dense(num_neurons[i], activation='relu'))  # initialize hidden layers

        self.model.add(Dense(ACTION_SET_SIZE, activation='linear'))  # output layer
        self.model.compile(loss='mse', optimizer=opt)
        print(self.model.summary())

    def save_model(self, name: str):
        """
        :param name: Name of the save file
        :return:
        """
        self.model.save(name)
        
    def load_model(self, name: str):
        """
        :param name: Name of file to load
        :return:
        """
        self.model.load(name)


def epsilon_greedy_policy(epsilon, rand_num, action):
    if rand_num <= epsilon:
        action = random.randrange(ACTION_SET_SIZE)

    return action


def q_learning(policy_net, target_net):
    game = snake_game()

    while True:
        game.start()
        
