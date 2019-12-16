# snakeDeepQ
Project that trains a machine learning agent to play the 
game, "Snake".  
## How to Run
From the terminal, run deepq.py with an argument of type,
string. To train the agent, the string must be "train". If 
the string is "load", a previously saved model will be
loaded to play a single game that will be displayed
in the terminal. To load a previously trained agent
for re-training use the arguments "train" followed
by "retrain"
## Task Environment
The task environment used was provided by Github user,
korolvs, who created a simple snake game and two neural
networks that trained an agent to play it. We took only
the game part, snake_game.py, and used it as the environment
our agent was to learn.
## Our Code
- state.py:  
File that defines the state of our environment. The state
consists of 11 boolean variables: variables to indicate
an obstacle immediately above, to the right, or to 
the left of of the agent; 4 variables to indicate which 
direction the snake is moving; and 4 variables to indicate
where the food is in relation to the snake's head (above,
below, to the right, or to the left).    
- test_state.py:  
File to test the state.py file.
- deepq.py:  
File that performs the Q-learning and training. Contains
a neural network class which is used to perform the deep
Q-learning. The way it performs the learning is by using
the model to predict Q values for each possible action
given the state of the environment. The action corresponding
to the highest Q value is considered the best action. Then
using an epsilon-greedy policy, that action is either taken
or a random action is taken. Nevertheless, for whichever
action ends up being taken, the expected Q value is calculated
and compared to the calculated "best Q". That best Q value is
then used to train the network. Additionally,
each timestep, the pre-state, action taken, reward
received, and the post-state are stored in a memory
list. At the end of each episode, the agent is trained
on a portion of this list.
- test_network.py  
File to test the deepq.py file.  
## References
- https://github.com/korolvs/snake_nn  
the Github repository containing the snake game used.
- https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a  
Website that helped us learn how to implement the Q-learning
algorithm.
## Libraries Used
- tensorflow
- keras
- windows-curses
- numpy
- matplotlib
- pickle

