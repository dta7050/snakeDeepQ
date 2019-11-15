LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

LEFT_DIR = [-1, 0]
DOWN_DIR = [0, 1]
RIGHT_DIR = [1,0]
UP_DIR = [0, -1]

DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

STATE_SIZE = 12
ACTION_SET_SIZE = 4

# MODEL PARAMETERS
# The greater this number, the less likely to choose a random action as opposed to the models predicted action
MODEL_BIAS_FACTOR = 0.01
GAMMA = 0.9  # Discount factor
