# -*- coding: utf-8 -*- 
import curses
from random import randint

class SnakeGame:
    def __init__(self, board_width = 20, board_height = 20, gui = False):
        """
        Initializes the snake game class. Sets the game score to zero,
        the game over bool to false, the board dimensions to the specified
        values and the gui bool to the specified value.
        :param board_width: Width of the game window
        :param board_height: Height of the game window
        :param gui: Whether or not to show the game
        """
        self.score = 0  # initialize the score (increments after eating food)
        self.done = False  # bool indicating if the game is over
        self.board = {'width': board_width, 'height': board_height}  # dict containing game window dimensions
        self.gui = gui  # bool indicating whether or not to display the game

    def start(self):
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

    def snake_init(self):
        """
        Creates the snake at a random point on
        the screen, at least five points away
        from each wall. The snake can start
        being vertical or horizontal and initially
        consists of three points
        :return:
        """
        x = randint(5, self.board["width"] - 5)  # generate random x coordinate
        y = randint(5, self.board["height"] - 5)  # generate random y coordinate
        self.snake = []  # empty list to store snake points
        vertical = randint(0, 1) == 0  # 50% chance of snake starting vertical
        for i in range(3):
            point = [x + i, y] if vertical else [x, y + i]  # creates the points of the snake
            self.snake.insert(0, point)  # adds it to the snake list

    def generate_food(self):
        """
        Creates food point in a random spot on the board
        :return:
        """
        food = []  # empty list to store food point coordinates
        while food == []:
            food = [randint(1, self.board["width"]), randint(1, self.board["height"])]  # gets random coordinates
            if food in self.snake:  # if the coordinates for the food are in the snake coordinate list
                food = []  # empty the food list
        self.food = food  # snake game food coordinate equals generated food coordinate

    def render_init(self):
        """
        Initializes the game window.
        :return:
        """
        curses.initscr()  # initializes curses library
        win = curses.newwin(self.board["width"] + 2, self.board["height"] + 2, 0, 0)  # sets window corner points
        curses.curs_set(0)  # sets the visibility to invisible
        win.nodelay(1)  # sets window to no delay mode (if no button is pressed, win.getch() returns -1)
        win.timeout(200)  # calls win.getch() every 200ms
        self.win = win  # set the snake game variable, win, to the initialized window
        self.render()  # calls render function

    def render(self):
        """
        Updates the game window, waits for button press
        :return:
        """
        self.win.clear()  # clears the game window
        self.win.border(0)  # draws the walls
        self.win.addstr(0, 2, 'Score : ' + str(self.score) + ' ')  # displays the score
        self.win.addch(self.food[0], self.food[1], 'F')  # displays the food point
        for i, point in enumerate(self.snake):  # adds the snake points
            if i == 0:
                self.win.addch(point[0], point[1], 'H')  # starts with the head
            else:
                self.win.addch(point[0], point[1], 'b')
        self.win.getch()  # checks for button press

    def step(self, key):
        """

        :param key: The snake's direction of motion
        :return:
        """
        # 0 - UP
        # 1 - RIGHT
        # 2 - DOWN
        # 3 - LEFT
        if self.done == True:  # if the snake dies, end the game
            self.end_game()
        self.create_new_point(key)
        if self.food_eaten():
            self.score += 1
            self.generate_food()
        else:
            self.remove_last_point()
        self.check_collisions()
        if self.gui: self.render()
        return self.generate_observations()

    def create_new_point(self, key):
        new_point = [self.snake[0][0], self.snake[0][1]]
        if key == 0:
            new_point[0] -= 1
        elif key == 1:
            new_point[1] += 1
        elif key == 2:
            new_point[0] += 1
        elif key == 3:
            new_point[1] -= 1
        self.snake.insert(0, new_point)

    def remove_last_point(self):
        self.snake.pop()

    def food_eaten(self):
        return self.snake[0] == self.food

    def check_collisions(self):
        if (self.snake[0][0] == 0 or
            self.snake[0][0] == self.board["width"] + 1 or
            self.snake[0][1] == 0 or
            self.snake[0][1] == self.board["height"] + 1 or
            self.snake[0] in self.snake[1:-1]):
            self.done = True

    def generate_observations(self):
        return self.done, self.score, self.snake, self.food

    def render_destroy(self):
        curses.endwin()

    def end_game(self):
        if self.gui: self.render_destroy()
        raise Exception("Game over")

if __name__ == "__main__":
    game = SnakeGame(gui = True)
    game.start()
    for _ in range(20):
        game.step(randint(0,3))
