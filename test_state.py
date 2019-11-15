
from state import State

fake_food = [5,5]


# TESTS FOR GET_BLOCKED_DIRS() --------------------------------------------------------------------------------------
print("TESTING GET_BLOCKED_DIRS()")
# LEFT BLOCKED BY SNAKE
snake = [[12,9], [11,9], [10,9]]
s = State(snake, fake_food)
if s.blocked_dirs == [1, 0, 0, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.blocked_dirs)
# LEFT BLOCKED BY WALL, DOWN BLOCKED BY SNAKE
snake = [[1,9], [1,10], [1,11]]
s = State(snake, fake_food)
if s.blocked_dirs == [1, 1, 0, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.blocked_dirs)

# DOWN BLOCKED BY SNAKE
snake = [[10,5], [10,6], [10,7]]
s = State(snake, fake_food)
if s.blocked_dirs == [0, 1, 0, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.blocked_dirs)
# DOWN BLOCKED BY WALL, UP BLOCKED BY SNAKE
snake = [[10,20], [10,19], [10,18]]
s = State(snake, fake_food)
if s.blocked_dirs == [0, 1, 0, 1]:
    print("PASS")
else:
    print("FAIL")
    print(s.blocked_dirs)

# RIGHT BLOCKED BY SNAKE
snake = [[10,9], [11,9], [12,9]]
s = State(snake, fake_food)
if s.blocked_dirs == [0, 0, 1, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.blocked_dirs)

# RIGHT BLOCKED BY WALL, LEFT BLOCKED BY SNAKE
snake = [[20,9], [19,9], [18,9]]
s = State(snake, fake_food)
if s.blocked_dirs == [1, 0, 1, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.blocked_dirs)

# UP BLOCKED BY SNAKE
snake = [[10,9], [10,8], [10,7]]
s = State(snake, fake_food)
if s.blocked_dirs == [0, 0, 0, 1]:
    print("PASS")
else:
    print("FAIL")
    print(s.blocked_dirs)

# UP BLOCKED BY WALL, DOWN BLOCKED BY SNAKE
snake = [[5,1], [5,2], [5,3]]
s = State(snake, fake_food)
if s.blocked_dirs == [0, 1, 0, 1]:
    print("PASS")
else:
    print("FAIL")
    print(s.blocked_dirs)

# UP AND LEFT BLOCKED BY WALL, DOWN BLOCKED BY SNAKE
snake = [[1,1], [1,2], [1,3]]
s = State(snake, fake_food)
if s.blocked_dirs == [1, 1, 0, 1]:
    print("PASS")
else:
    print("FAIL")
    print(s.blocked_dirs)

# DOWN AND RIGHT BLOCKED BY WALL, UP BLOCKED BY SNAKE
snake = [[20,20], [20,19], [20,18]]
s = State(snake, fake_food)
if s.blocked_dirs == [0, 1, 1, 1]:
    print("PASS")
else:
    print("FAIL")
    print(s.blocked_dirs)


# TESTS FOR GET_MOTION_DIRS() --------------------------------------------------------------------------------------
print("TESTING GET_MOTION_DIRS()")
# LEFT MOVING SNAKE
snake = [[10,9], [11,9], [12,9]]
s = State(snake, fake_food)
if s.motion_dirs == [1, 0, 0, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.motion_dirs)

# DOWN MOVING SNAKE
snake = [[10,9], [10,8], [10,7]]
s = State(snake, fake_food)
if s.motion_dirs == [0, 1, 0, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.motion_dirs)

# RIGHT MOVING SNAKE
snake = [[10,9], [9,9], [8,9]]
s = State(snake, fake_food)
if s.motion_dirs == [0, 0, 1, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.motion_dirs)

# UP MOVING SNAKE
snake = [[10,9], [10,10], [10,11]]
s = State(snake, fake_food)
if s.motion_dirs == [0, 0, 0, 1]:
    print("PASS")
else:
    print("FAIL")
    print(s.motion_dirs)

# TESTS FOR GET_FOOD_DIRS() --------------------------------------------------------------------------------------
print("TESTING GET_FOOD_DIRS()")
# FOOD LEFT OF SNAKE
snake = [[10,9], [11,9], [12,9]]
food = [5, 9]
s = State(snake, food)
if s.food_dirs == [1, 0, 0, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.food_dirs)

# FOOD DOWN FROM SNAKE
snake = [[10,9], [10,8], [10,7]]
food = [10, 20]
s = State(snake, food)
if s.food_dirs == [0, 1, 0, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.food_dirs)

# FOOD RIGHT OF SNAKE
snake = [[10,9], [9,9], [8,9]]
food = [15, 9]
s = State(snake, food)
if s.food_dirs == [0, 0, 1, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.food_dirs)

# FOOD UP FROM SNAKE
snake = [[10,9], [10,10], [10,11]]
food = [10, 5]
s = State(snake, food)
if s.food_dirs == [0, 0, 0, 1]:
    print("PASS")
else:
    print("FAIL")
    print(s.food_dirs)

# FOOD LEFT AND UP OF SNAKE
snake = [[10,9], [11,9], [12,9]]
food = [5, 8]
s = State(snake, food)
if s.food_dirs == [1, 0, 0, 1]:
    print("PASS")
else:
    print("FAIL")
    print(s.food_dirs)

# FOOD LEFT AND DOWN OF SNAKE
snake = [[10,9], [11,9], [12,9]]
food = [5, 11]
s = State(snake, food)
if s.food_dirs == [1, 1, 0, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.food_dirs)

# FOOD RIGHT AND DOWN OF SNAKE
snake = [[10,9], [9,9], [8,9]]
food = [15, 20]
s = State(snake, food)
if s.food_dirs == [0, 1, 1, 0]:
    print("PASS")
else:
    print("FAIL")
    print(s.food_dirs)

# FOOD RIGHT AND UP FROM SNAKE
snake = [[10,9], [10,10], [10,11]]
food = [11, 8]
s = State(snake, food)
if s.food_dirs == [0, 0, 1, 1]:
    print("PASS")
else:
    print("FAIL")
    print(s.food_dirs)

