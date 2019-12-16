"""
This file contains code developed by Github user, korolvs.
Changes Made:
- Most of the comments
- Addition of rewards in 'step' method
- Modified how the snake is initialized
- Replaced some hard coded values with variables to generalize code and make it easier to use
"""
# -*- coding: utf-8 -*-
import curses
from constants import *
from random import randint
from typing import List


class SnakeGame:
    def __init__(self, board_width: int = WIDTH, board_height: int = HEIGHT, gui: bool = False):
        """
        Initializes the snake game class. Sets the game score to zero,
        the game over bool to false, the board dimensions to the specified
        values and the gui bool to the specified value.
        :param board_width: Width of the game window
        :param board_height: Height of the game window
        :param gui: Whether or not to show the game
        """
        self.score = 0  # initialize the score (increments after eating food)
        self.reward = 0
        self.done = False  # bool indicating if the game is over
        self.board = {'width': board_width, 'height': board_height}  # dict containing game window dimensions
        self.gui = gui  # bool indicating whether or not to display the game

    def start(self) -> (bool, int, List[List[int]], List[int]):
        """
        Creates the snake, creates a food point
        in a random location, and if the gui variable
        is true, generates the game window
        :return: The snake, the score, whether the game is done, and the food location
        """
        self.snake_init()  # creates snake
        self.generate_food()  # creates food point
        if self.gui:
            self.render_init()  # if gui == true, create game window
        return self.generate_observations()

    def snake_init(self) -> None:
        """
        Creates the snake at a random point on
        the screen, at least five points away
        from each wall. The snake can start
        being vertical or horizontal and initially
        consists of three points
        :return: None
        """
        # NEW, NOT RANDOM
        x = self.board["width"] // 2
        y = x

        # x = randint(SNAKE_SIZE, self.board["width"] - SNAKE_SIZE + 1)  # generate random x coordinate
        # y = randint(SNAKE_SIZE, self.board["height"] - SNAKE_SIZE + 1)  # generate random y coordinate
        self.snake = []  # empty list to store snake points
        vertical = randint(0, 1) == 0  # 50% chance of snake starting vertical
        for i in range(SNAKE_SIZE):
            point = [x + i, y] if vertical else [x, y + i]  # creates the points of the snake
            self.snake.append(point)  # adds it to the snake list

    def generate_food(self) -> None:
        """
        Creates food point in a random spot on the board
        :return: None
        """
        food = []  # empty list to store food point coordinates
        while food == []:
            food = [randint(1, self.board["width"]), randint(1, self.board["height"])]  # gets random coordinates
            if food in self.snake:  # if the coordinates for the food are in the snake coordinate list
                food = []  # empty the food list
        self.food = food  # snake game food coordinate equals generated food coordinate

    def render_init(self) -> None:
        """
        Initializes the game window.
        :return: None
        """
        curses.initscr()  # initializes curses library
        win = curses.newwin(self.board["width"] + 2, self.board["height"] + 2, 0, 0)  # sets window corner points
        curses.curs_set(0)  # sets the visibility to invisible
        win.nodelay(1)  # sets window to no delay mode (if no button is pressed, win.getch() returns -1)
        win.timeout(200)  # calls win.getch() every 200ms
        self.win = win  # set the snake game variable, win, to the initialized window
        self.render()  # calls render function

    def render(self) -> None:
        """
        Updates the game window, waits for button press
        :return: None
        """
        self.win.clear()  # clears the game window
        self.win.border(0)  # draws the walls
        self.win.addstr(0, 2, 'S: ' + str(self.score) + ' ')  # displays the score
        self.win.addch(self.food[1], self.food[0], 'F')  # displays the food point
        for i, point in enumerate(self.snake):  # adds the snake points
            if i == 0:
                self.win.addch(point[1], point[0], 'H')  # starts with the head
            else:
                self.win.addch(point[1], point[0], 'b')
        self.win.getch()  # checks for button press

    def step(self, key: int) -> (bool, int, List[List[int]], List[int]):
        """
        Moves the snake, checks if it ate food, checks
        if it hit the wall or itself, and then updates
        the GUI
        :param key: The snake's direction of motion
        :return: The snake, the score, whether the game is done, and the food location
        """
        # 0 - LEFT
        # 1 - DOWN
        # 2 - RIGHT
        # 3 - UP
        # 'Moves' the snake
        self.create_new_point(key)
        # Sees if snake hit wall or itself
        self.check_collisions()
        # Game over check and reward assignment
        if self.done:  # if the snake dies, end the game
            self.reward = -10
            self.score -= 0
            self.end_game()
            return self.generate_observations()
        if self.food_eaten():
            self.score += 1
            self.reward = 10
            self.generate_food()
        else:
            self.reward = 0
            self.score += 0
            self.remove_last_point()
        if self.gui:
            self.render()
        return self.generate_observations()

    def create_new_point(self, key: int) -> None:
        """
        Creates a new head point in the direction
        of the snakes motion (i.e. moves the snake).
        :param key: The snake's direction of motion
        :return: None
        """
        # Get current head
        new_point = [self.snake[0][0], self.snake[0][1]]
        if key == 0:  # moving left
            new_point[0] -= 1
        elif key == 1:  # Moving down
            new_point[1] += 1
        elif key == 2:  # Moving right
            new_point[0] += 1
        elif key == 3:  # Moving up
            new_point[1] -= 1
        self.snake.insert(0, new_point)  # add the new point to the snake

    def remove_last_point(self) -> None:
        """
        Removes the endpoint of the snake
        :return: None
        """
        self.snake.pop()  # remove the last point in the snake list

    def food_eaten(self) -> bool:
        """
        Check if the head of the snake is at
        the same coordinate point as the food
        :return: a boolean stating whether or not snake ate food
        """
        return self.snake[0] == self.food

    def check_collisions(self) -> None:
        """
        Check if the snake is in the wall or
        collides with itself
        :return: None
        """
        if (self.snake[0][0] == 0 or  # if the snake is in the right wall
            self.snake[0][0] == self.board["width"] + 1 or  # if the snake is in the left wall
            self.snake[0][1] == 0 or  # if the snake is in the top wall
            self.snake[0][1] == self.board["height"] + 1 or  # if the snake is in the bottom wall
            self.snake[0] in self.snake[1:]):  # if the snake collides with itself
            self.done = True  # sets bool to end game

    def generate_observations(self) -> (bool, int, List[List[int]], List[int]):
        """
        Simply returns whether the game is running,
        the snake's score, the list of snake body
        points, and the food point
        :return:
        """
        return self.done, self.score, self.snake, self.food, self.reward

    def render_destroy(self) -> None:
        """
        Disables the display window
        :return: None
        """
        curses.endwin()

    def end_game(self) -> None:
        """
        Calls render_destroy to disable the game window
        :return: None
        """
        if self.gui:
            self.render_destroy()
        # raise Exception("Game over")


if __name__ == "__main__":
    game = SnakeGame(gui=True)
    game.start()
    for _ in range(20):
        game.step(randint(0, 3))  # snake takes random movements

