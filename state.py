"""
Development of the state variable inspired by Mauro Comi from towardsdatascience.com
"""
# ---------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM ---------------------------------------------------------------------------------------------------
#
# (x,y)
#
# y
#  ^
#  |
#  |
#  |
#  -------->
#           x
#
# GAME WORLD ----------------------------------------------------------------------------------------------------------
#
#   (0,0) ---------- (WIDTH, 0)
#         |        |
#         |        |
#         |        |
#         |        |
# (0, HEIGHT) ---------- (WIDTH, HEIGHT)
#
# ---------------------------------------------------------------------------------------------------------------------

from constants import *
from typing import List
from math import sqrt, pow, atan2, pi
import numpy as np


class State:
    def __init__(self, snake: List[List[int]], food: List[int]):
        """
        Initializes the State object, given a snake

        The State is a numpy array containing 11 boolean integers ( 0 or 1 ) and 2 float values:
            state = np.array(blocked_dirs[], motion_dirs[], food_dirs[], distance, angle)

            an example state looks like np.ndarray(0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0.54, 0.33)

        The State is comprised of:

            blocked_dirs; List[int]:
                blocked_dirs[0] = LEFT BLOCKED?  (1 = YES, 0 = NO)
                blocked_dirs[1] = FORWARD BLOCKED?  (1 = YES, 0 = NO)
                blocked_dirs[2] = RIGHT BLOCKED? (1 = YES, 0 = NO)

            motion_dirs; List[int]:
                motion_dirs[0] = LEFT TRAVELING?  (1 = YES, 0 = NO)
                motion_dirs[1] = DOWN TRAVELING?  (1 = YES, 0 = NO)
                motion_dirs[2] = RIGHT TRAVELING? (1 = YES, 0 = NO)
                motion_dirs[3] = UP TRAVELING ?   (1 = YES, 0 = NO)

            food_dirs; direction of food relative to snake head (Snake direction not taken into account):
                food_dirs[0] = FOOD TO LEFT OF SNAKE HEAD?   (1 = YES, 0 = NO)
                food_dirs[1] = FOOD DOWN FROM SNAKE HEAD?    (1 = YES, 0 = NO)
                food_dirs[2] = FOOD TO RIGHT OF SNAKE HEAD?  (1 = YES, 0 = NO)
                food_dirs[3] = FOOD UP FROM SNAKE HEAD?      (1 = YES, 0 = NO)

            distance; (float) the normalized distance [0, 1] between the snake's head and the food point
            angle; (float) the normalized angle [0, 1] between the snake's head and the food point

        :param snake: List of points [x,y] of snake
        :param food: [x,y] coordinate of food point
        """
        # Initialize state variables
        self.blocked_dirs = [0] * 3  # type: List[int]
        self.motion_dirs = [0] * 4  # type: List[int]
        self.food_dirs = [0] * 4  # type: List[int]

        # Calculate state variables
        self.get_blocked_dirs(snake)
        self.get_motion_dirs(snake)
        self.get_food_dirs(snake, food)
        # self.distance = self.get_distance(snake, food)
        # self.angle = self.get_angle(snake, food)

        # Create full state list
        # self.state = np.array(self.blocked_dirs + self.motion_dirs + self.food_dirs + [self.distance, self.angle])
        self.state = np.array(self.blocked_dirs + self.motion_dirs + self.food_dirs)

    def get_blocked_dirs(self, snake: List[List[int]]) -> None:
        """
        Given a snake, calculate the next action directions that are blocked.
        Directions can be blocked either by the snakes body or by the boundaries of the game world.

        blocked_dirs[0] = LEFT BLOCKED?     (1 = YES, 0 = NO)
        blocked_dirs[1] = FORWARD BLOCKED?  (1 = YES, 0 = NO)
        blocked_dirs[2] = RIGHT BLOCKED?    (1 = YES, 0 = NO)

        :param snake: List of points [x,y] of snake
        :return: None
        """

        # Get direction of snake
        snake_dir = get_snake_direction(snake)

        # Find direction and set motion dir list
        # LEFT TRAVELING
        if np.array_equal(snake_dir, np.array(LEFT_DIR)):
            # LEFT (DOWN)
            point = np.array(snake[0]) + np.array(DOWN_DIR)
            self.blocked_dirs[0] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
            # FORWARD (LEFT)
            point = np.array(snake[0]) + np.array(LEFT_DIR)
            self.blocked_dirs[1] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
            # RIGHT (UP)
            point = np.array(snake[0]) + np.array(UP_DIR)
            self.blocked_dirs[2] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
        # DOWN
        elif np.array_equal(snake_dir, np.array(DOWN_DIR)):
            # LEFT (RIGHT)
            point = np.array(snake[0]) + np.array(RIGHT_DIR)
            self.blocked_dirs[0] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
            # FORWARD (DOWN)
            point = np.array(snake[0]) + np.array(DOWN_DIR)
            self.blocked_dirs[1] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
            # RIGHT (LEFT)
            point = np.array(snake[0]) + np.array(LEFT_DIR)
            self.blocked_dirs[2] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
        # RIGHT
        elif np.array_equal(snake_dir, np.array(RIGHT_DIR)):
            # LEFT (UP)
            point = np.array(snake[0]) + np.array(UP_DIR)
            self.blocked_dirs[0] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
            # FORWARD (RIGHT)
            point = np.array(snake[0]) + np.array(RIGHT_DIR)
            self.blocked_dirs[1] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
            # RIGHT (DOWN)
            point = np.array(snake[0]) + np.array(DOWN_DIR)
            self.blocked_dirs[2] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
        # UP
        elif np.array_equal(snake_dir, np.array(UP_DIR)):
            # LEFT (LEFT)
            point = np.array(snake[0]) + np.array(LEFT_DIR)
            self.blocked_dirs[0] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
            # FORWARD (UP)
            point = np.array(snake[0]) + np.array(UP_DIR)
            self.blocked_dirs[1] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))
            # RIGHT (RIGHT)
            point = np.array(snake[0]) + np.array(RIGHT_DIR)
            self.blocked_dirs[2] = int(point.tolist() in snake or
                                   point[0] == 0 or point[1] == 0 or point[0] >= (WIDTH + 1) or point[1] >= (WIDTH + 1))

    def get_motion_dirs(self, snake: List[List[int]]) -> None:
        """
        Uses the direction of the snake to set up the motion_dirs list of the State.
        motion_dirs is a one-hot array, with the hot index corresponding to the direction the snake is traveling in.

        motion_dirs[0] = LEFT TRAVELING?  (1 = YES, 0 = NO)
        motion_dirs[1] = DOWN TRAVELING?  (1 = YES, 0 = NO)
        motion_dirs[2] = RIGHT TRAVELING? (1 = YES, 0 = NO)
        motion_dirs[3] = UP TRAVELING ?   (1 = YES, 0 = NO)

        :param snake: List of points [x,y] of snake
        :return: None
        """
        # Get direction of snake
        snake_dir = get_snake_direction(snake)

        # Find direction and set motion dir list
        # LEFT
        if np.array_equal(snake_dir, np.array(LEFT_DIR)):
            self.motion_dirs[LEFT] = 1
        # DOWN
        elif np.array_equal(snake_dir, np.array(DOWN_DIR)):
            self.motion_dirs[DOWN] = 1
        # RIGHT
        elif np.array_equal(snake_dir, np.array(RIGHT_DIR)):
            self.motion_dirs[RIGHT] = 1
        # UP
        elif np.array_equal(snake_dir, np.array(UP_DIR)):
            self.motion_dirs[UP] = 1

    def get_food_dirs(self, snake: List[List[int]], food: List[int]) -> None:
        """
        Calculates which directions the snake has to travel to get to the food.
        Can be up to two directions simultaneously.

        food_dirs[0] = FOOD TO LEFT OF SNAKE HEAD?   (1 = YES, 0 = NO)
        food_dirs[1] = FOOD DOWN FROM SNAKE HEAD?    (1 = YES, 0 = NO)
        food_dirs[2] = FOOD TO RIGHT OF SNAKE HEAD?  (1 = YES, 0 = NO)
        food_dirs[3] = FOOD UP FROM SNAKE HEAD?      (1 = YES, 0 = NO)
        :param snake: List of points [x,y] of snake
        :param food: [x,y] coordinates of food point
        :return: None
        """
        # Get array describing x and y displacement of snake from food (food point - head point)
        displacement = np.array(food) - np.array(snake[0])

        # normalize the displacement to [+- 1, +- 1] to extract direction vectors
        if displacement[0] == 0 and displacement[1] == 0:
            return  # Snake is on top of food point, keeps food_dirs as initialized ([0, 0, 0, 0])
        elif displacement[0] == 0:
            norm_displacement = [0, displacement[1]/abs(displacement[1])]
        elif displacement[1] == 0:
            norm_displacement = [displacement[0]/abs(displacement[0]), 0]
        else:
            norm_displacement = [displacement[0]/abs(displacement[0]), displacement[1]/abs(displacement[1])]

        # Get food_dirs from normalized displacement
        # LEFT
        if norm_displacement[0] == LEFT_DIR[0]:
            self.food_dirs[LEFT] = 1
        # DOWN
        if norm_displacement[1] == DOWN_DIR[1]:
            self.food_dirs[DOWN] = 1
        # RIGHT
        if norm_displacement[0] == RIGHT_DIR[0]:
            self.food_dirs[RIGHT] = 1
        # UP
        if norm_displacement[1] == UP_DIR[1]:
            self.food_dirs[UP] = 1

    def get_permissible_actions(self) -> List[int]:
        """
        Uses the direction of the snake to return a list of actions that the snake can take that will
        result in the snake not colliding with itself
        :return: list of the actions that snake can take
        """
        # TRAVELING LEFT
        if self.motion_dirs[0]:
            return [DOWN, LEFT, UP]
        # TRAVELING DOWN
        elif self.motion_dirs[1]:
            return [RIGHT, DOWN, LEFT]
        # TRAVELING RIGHT
        elif self.motion_dirs[2]:
            return [UP, RIGHT, DOWN]
        # TRAVELING UP
        return [LEFT, UP, RIGHT]

    def get_distance(self, snake: List[List[int]], food: List[int]) -> int:
        """
        Calculates the distance between the snake head point and the food point using the Pythagorean Theorem.
        Normalizes the distance to a value between 0 and 1 by dividing the
        :param snake: List of points [x,y] of snake
        :param food: [x,y] coordinates of food point
        :return: int
        """

        # Get array describing x and y components of distance of snake head from food (food point - head point)
        distance_comps = np.array(food) - np.array(snake[0])
        # Compute distance using components, using the pythagorean theorem
        dist = sqrt(pow(abs(distance_comps[0]), 2) + pow(abs(distance_comps[1]), 2))
        # Normalize distance by dividing it by the maximum possible distance (MAX_DIST in constants.py)
        norm_dist = dist/MAX_DIST

        return norm_dist

    def get_angle(self, snake: List[List[int]], food: List[int]) -> int:
        """
        Gets a value between 0 and 1 corresponding to the angle between the head of the snake and the food point.
        :param snake: List of points [x,y] of snake
        :param food: [x,y] coordinates of food point
        :return: int
        """
        # Get array describing x and y components of distance of snake head from food (food point - head point)
        distance_comps = np.array(food) - np.array(snake[0])
        # Caluclate the inverse tangent to get an angle in radians, using atan2() to get the correct quadrant
        angle = atan2(distance_comps[1],distance_comps[0])
        # Map the output of the atan2 function to a value between 0 and 1
        norm_angle = np.interp(angle, [-pi, pi], [0, 1])

        return norm_angle


def get_snake_direction(snake: List[List[int]]) -> np.ndarray:
    """
    Returns an np array describing the direction of a snake
    LEFT = [-1, 0]
    DOWN = [0, 1]
    RIGHT = [1, 0]
    UP = [0, -1]
    :param snake: List of points [x,y] of snake
    :return: np.array describing direction
    """
    return np.array(snake[0]) - np.array(snake[1])

