from typing import List


class State:
    def __init__(self):
        self.blocked_dirs = [0] * 4  # type: List[int]
        self.motion_dirs = [0] * 4  # type: List[int]
        self.food_angle = 0.0  # type: float

    def get_blocked_dirs(self):
        return

    def get_motion_dirs(self):
        return

    def get_food_angle(self):
        return

