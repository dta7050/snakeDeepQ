from typing import List
import enum
import snake_game


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

class State:
    def __init__(self):
        self.blocked_dirs = [0] * 4  # type: List[int]
        self.motion_dirs = [0] * 4  # type: List[int]
        self.food_angle = 0.0  # type: float

    def get_blocked_dirs(self):
        for dir in DIRECTIONS:
            # CHECK UP
            point = np.array(snake[0]) + np.array([0,1])
            return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def get_motion_dirs(self):
        return

    def get_food_angle(self):
        return

