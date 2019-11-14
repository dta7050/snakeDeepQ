from typing import List
import enum
import snake_game


LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

Directions = [LEFT, RIGHT, UP, DOWN]

class State:
    def __init__(self):
        self.blocked_dirs = [0] * 4  # type: List[int]
        self.motion_dirs = [0] * 4  # type: List[int]
        self.food_angle = 0.0  # type: float

    def get_blocked_dirs(self):
        for
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def get_motion_dirs(self):
        return

    def get_food_angle(self):
        return

