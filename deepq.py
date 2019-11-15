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
            train_on_timestep(): trains the model using Q learning
    model_loader(): loads a previously saved model
    epsilon_greedy_policy(): picks action for snake to take
    run_deep_q(): runs the deep q learning algorithm
"""

from snake_game import *
from state import *
from constants import *

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model


class NeuralNetwork:

    def __init__(self, num_layers: int, num_neurons: List[int], opt: str) -> None:
        """
        For now, network is just simple, vanilla network
        :param num_layers: number of input layers
        :param num_neurons: number of neurons in each hidden layer
        :param opt: optimizer to use in network
        """
        self.model = Sequential()  # initialize the model

        self.model.add(Dense(100, input_dim=12))  # input layer

        for i in range(num_layers):
            self.model.add(Dense(num_neurons[i], activation='relu'))  # initialize hidden layers

        self.model.add(Dense(ACTION_SET_SIZE, activation='softmax'))  # output layer
        self.model.compile(loss='mse', optimizer=opt)
        # print(self.model.summary())

    def save_model(self) -> None:
        """
        Saves the model in the predefined path
        :return: None
        """
        # make model directory if it doesnt exist
        if not os.path.isdir('model/'):
            os.makedirs('model/')
        # save model
        self.model.save('model/model1')

    def train_on_timestep(self, pre_state: np.ndarray, post_state: np.ndarray,
                          action: int, reward: int, done: bool) -> None:
        """
        This function takes in the pre and post action states of the snake, the action taken,
        the reward received for taking the action and if the game is over.
        Using this information, the Bellman Equation is used to calculate the target Q value.
        This target Q value is used to train the neural network model.
        :param pre_state: the snakes state prior to action
        :param post_state: the snakes state after the action
        :param action: the action taken by the snake
        :param reward: the reward received for taking the action
        :param done: tells if game is over. True = over, False = not over
        :return: None
        """
        # Use the model to predict the q vector of the pre action state
        predicted_pre_state_q_vector = self.model.predict(np.reshape(pre_state, (1, 12)))
        # Current q value is the index in the predicted pre state q vector corresponding to the action
        current_q_value = predicted_pre_state_q_vector[0][np.argmax(action)]

        if done:
            # Use Bellman Equation to calculate the target q value.
            # Here we consider the maximum predicted Q value of the new state as zero,
            # since this new value is an invalid state deemed by the reules of the game
            target_post_state_q = current_q_value + LEARNING_RATE*(reward - current_q_value)
        else:
            # Use Bellman Equation to calculate the target q value for the post action state
            target_post_state_q = current_q_value + \
                                  LEARNING_RATE*(reward + (GAMMA *
                                                 np.amax(self.model.predict(np.reshape(post_state, (1, 12)))[0]))
                                                 - current_q_value)

        # Place the target post state q into its corresponding spot in the predicted pre state q vector
        predicted_pre_state_q_vector[0][np.argmax(action)] = target_post_state_q
        # This is now the target q vector to fit the model against
        target_q_vector = predicted_pre_state_q_vector
        # Use the target q vector to fit the model
        self.model.fit(np.reshape(pre_state, (1, 12)), target_q_vector, epochs=1, verbose=0)


def model_loader() -> None:
    """
    loads the model from the predefined path
    :return:
    """
    return load_model('model/model1')


def epsilon_greedy_policy(epsilon: float, action: int, permissible_actions: List[int]) -> int:
    """
    Compares epsilon to a random number to choose whether to take the best q action or a random permissible action
    :param epsilon: Factor used to decide if random action is used or best q action is used
    :param action: best q action
    :param permissible_actions: the actions the snake can take that will not result in it hitting itself
    :return: the action to take (int)
    """
    rand_num = np.random.rand()  # produces random number between 0 and 1
    if rand_num <= epsilon:
        action = random.choice(permissible_actions)
    return action


def run_deep_q(com: str) -> str:
    """
    This is the main function of the Deep Q Learning algorithm. It preforms Q learning and uses a Deep Neural Network
    to approximate the Q-matrix values.
    This function has two options that are chosen using the input parameter 'com'.
        1) com = 'train' will train on a specified number of games (See 'num_games') and then save the model
        2) com = 'load' will load a previously trained model and then visually play a game in the terminal
    :param com: (str) input parameter to choose option, either 'train' or 'load'
    :return: str
    """
    # If specified, load model from directory
    if com == 'load':
        # load model
        model = model_loader()

        # Create game object
        game = SnakeGame(gui=True)

        # start game
        game.start()

        # Get game state
        _, _, snake, food, _ = game.generate_observations()

        while not game.done:

            # get snake state
            state = State(snake, food).state

            # predict action to take using model
            prediction = model.predict(np.reshape(state, (1, 12)))
            action = np.argmax(prediction[0])

            # take action and get updated game state
            done, _, snake, food, reward = game.step(action)

        return "Score: " + str(game.score)

    # Train Procedure
    elif com == 'train':

        # Initializations
        nn = NeuralNetwork(2, [100, 50], "adam")
        scores = []

        for game_counter in range(NUM_GAMES):
            # Create game object
            game = SnakeGame(gui=False)
            # start game
            game.start()

            while not game.done:
                # print("Score: " + str(game.score))
                # update game state
                _, _, snake, food, _ = game.generate_observations()

                # print("Snake: " + str(snake))

                # get pre action snake state
                pre_state = State(snake, food).state

                # predict action to take using model
                prediction = nn.model.predict(np.reshape(pre_state, (1, 12)))
                action = np.argmax(prediction[0])

                # Use epsilon greedy action to either take predicted action or random action
                # Takes random actions less often as more games are played
                epsilon = np.random.rand() - (MODEL_BIAS_FACTOR * game_counter)
                permissible_actions = State(snake, food).get_permissible_actions()
                action = epsilon_greedy_policy(epsilon, action, permissible_actions)

                # take action and get updated game state
                done, _, snake, food, reward = game.step(action)

                # Get post action snake state
                post_state = State(snake, food).state

                # Train neural network
                nn.train_on_timestep(pre_state, post_state, action, reward, done)

            # end while game.done()
            scores.append(game.score)
            print('Game:' + str(game_counter + 1) + ' | ' + 'Score: ' + str(game.score))

        # save model
        nn.save_model()

        # Plot score set
        plt.plot(scores)
        plt.ylabel('Score')
        plt.xlabel('Game Number')
        plt.title("Score per Game")
        plt.show()

        return "Successfully trained on " + str(NUM_GAMES) + ' games.'

    else:
        return "Incorrect Command Line Argument. Enter eiher 'train' or 'simulate'."


if __name__ == '__main__':
    run_deep_q(sys.argv[1])
