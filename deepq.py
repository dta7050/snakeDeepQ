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
from snake_game import *
from state import *
from constants import *

import random
import tensorflow
import keras
import numpy as np

from typing import List
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


class NeuralNetwork:

    def __init__(self, num_layers: int, num_neurons: List[int], opt: str):
        """
        For now, network is just simple, vanilla network
        :param num_layers: number of input layers
        :param num_neurons: number of neurons in each hidden layer
        :param opt: optimizer to use in network
        """
        self.model = Sequential()  # initialize the model

        self.model.add(Dense(STATE_SIZE*2, input_dim=12))  # input layer

        # for i in range(num_layers):
        #     self.model.add(Dense(num_neurons[i], activation='relu'))  # initialize hidden layers

        self.model.add(Dense(ACTION_SET_SIZE, activation='softmax'))  # output layer
        self.model.compile(loss='mse', optimizer=opt)
        # print(self.model.summary())

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

    def train_on_timestep(self, pre_state, post_state, action, reward):
        # Use Bellman Equation to calculate the target q value for the post action state
        target_post_state_q = reward + GAMMA * np.amax(self.model.predict(np.reshape(post_state, (1, 12)))[0])
        # Use the model to predict the q vector of the pre action state
        predicted_pre_state_q_vector = self.model.predict(np.reshape(pre_state, (1, 12)))
        # Place the target post state q into its corresponding spot in the predicted pre state q vector
        predicted_pre_state_q_vector[0][np.argmax(action)] = target_post_state_q
        # This is now the target q vector to fit the model against
        target_q_vector = predicted_pre_state_q_vector
        # Use the target q vector to fit the model
        self.model.fit(np.reshape(pre_state, (1, 12)), target_q_vector, epochs=1, verbose=0)


def epsilon_greedy_policy(epsilon, action):
    """

    :param epsilon:
    :param action:
    :return:
    """
    rand_num = np.random.rand()  # produces random number between 0 and 1
    if rand_num <= epsilon:
        action = random.randrange(ACTION_SET_SIZE)
    return action
        

def run_deep_q():
    """

    :return:
    """
    # Initializations
    num_games = 1000
    nn = NeuralNetwork(1, [12], "adam")
    scores = []

    for game_counter in range(num_games):
        # Create game object
        game = SnakeGame(gui=False)
        # start game
        game.start()

        while not game.done:
            # update game state
            _, _, snake, food, _ = game.generate_observations()

            # get pre action snake state
            pre_state = State(snake, food).state

            # predict action to take using model
            prediction = nn.model.predict(np.reshape(pre_state, (1, 12)))
            action = np.argmax(prediction[0])

            # Use epsilon greedy action to either take predicted action or random action
            # Takes random actions less often as more games are played
            epsilon = np.random.rand() - (MODEL_BIAS_FACTOR * game_counter)
            action = epsilon_greedy_policy(epsilon, action)

            # take action and get updated game state
            _, _, snake, food, reward = game.step(action)

            # Get post action snake state
            post_state = State(snake, food).state

            # Train neural network
            nn.train_on_timestep(pre_state, post_state, action, reward)

        # end while game.done()
        scores.append(game.score)
        print('Game:' + str(game_counter + 1) + '       ' + 'Score: ' + str(game.score))


run_deep_q()
