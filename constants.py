"""
Constants.py

Contains global constants for the Snake Game and Deep Q Learning Algorithm
"""

from math import pow, sqrt

# PLAYABLE AREA BOARD DIMENSIONS. CANT GO SMALLER THAN 5!
WIDTH = HEIGHT = 10

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

LEFT_DIR = [-1, 0]
DOWN_DIR = [0, 1]
RIGHT_DIR = [1,0]
UP_DIR = [0, -1]

DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

SNAKE_SIZE = 3

MAX_DIST = sqrt(pow(WIDTH-1, 2) + pow(HEIGHT-1, 2))  # Maximum distance a snake head point can be from food

STATE_SIZE = 14
ACTION_SET_SIZE = 3

NUM_GAMES = 5000

# MODEL PARAMETERS
# The greater this number, the less likely to choose a random action as opposed to the models predicted action
MODEL_BIAS_FACTOR = 0.01
GAMMA = 0.9  # Discount factor
LEARNING_RATE = 0.001
