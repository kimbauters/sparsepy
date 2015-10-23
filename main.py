from random import choice
import logging as log
import math

from search_structure import Node
from pdo_parser import PDOParser
from mcts import mcts


my_input = """
(define (problem example-maffia)
(:init (and guns riches))

(:goal (and house yacht (not guns)))

(:goal-reward 1)

(:action traffic
 :precondition riches
 :effect (probabilistic 9/10 (and house (not riches))
                        1/10 (not riches)
         )
 )

(:action raid
 :precondition (and guns riches)
 :effect (probabilistic 5/10 (not guns riches)
                        2/10 yacht
                        3/10 (and yacht (not riches))
         )
)

(:action dump
 :precondition yacht
 :effect (probabilistic 0.5 (not guns)
                        0.5 (not yacht)
         )
)

(:action gamble
 :precondition house
 :effect (probabilistic 3/10 guns
                        3/10 riches
                        4/10 (and riches (not house))
         )
)

(:action beg
 :effect (probabilistic 1/4  guns
                        3/20 riches
                        1/20 house
                        1/20 yacht
         )
)

(:action plead
 :precondition (not riches)
 :effect (probabilistic 0.8 yacht
                        0.2 (decrease (reward) 0.6))
)

)"""


# create a PDO parser ...
my_parser = PDOParser()
# ... and parse the input using it.
my_problem = my_parser.process_input(my_input)


# generator function to create functions that will return False after a number of allowed iterations
def iteration_budget(allowed_iterations):

    def inner_iteration_budget(iterations):
        return iterations < allowed_iterations
    return inner_iteration_budget


# generator function to create functions that will return False after a set amount of seconds have passed
def timed_budget(allowed_secs):
    from time import perf_counter
    start_time = None

    def inner_timed_budget(_):
        nonlocal start_time  # use the start time in the closure
        if not start_time:  # if it hasn't been set ...
            start_time = perf_counter()  # set it to now
        if perf_counter() - start_time > allowed_secs:  # check if the allowed have been reached
            start_time = None  # if so, reset the start_time to None for a next run
            return False  # and return False to indicate the iteration should stop
        else:
            return True  # continue until the allotted number of seconds is reached
    return inner_timed_budget


def my_expand_action(node):
    """ Expand one of the untried actions at random. """
    return choice(node.untried_actions)


def my_select_action(node):
    """ Select the best action, by taking the action that has 
        the best average reward vs the least number of explorations.
        This is based on the UCB1 mechanism for selecting the best action. """
    best_action = (None, 0)
    log.info("  choosing 1 action out of " + str(len(node.tried_actions)))
    for action, (action_reward, visits) in node.tried_actions.items():
        ucb1 = (1/math.sqrt(2)) * math.sqrt(math.log(node.visits)/visits)+(action_reward/visits)
        if not best_action[0] or ucb1 > best_action[1]:
            best_action = (action, ucb1)
    return best_action[0]


def my_rollout_action(node):
    """ Select a random action during the rollout phase. """
    return choice(node.untried_actions)


# initialise a basic root, which identifies only the problem space as well as the initial state
root = Node(my_problem, None, None, None, my_problem.init)
# initialise the reward to 0
reward = 0

# trigger the actual MCTS search by setting all desired parameters
while not my_problem.goal_reached(root.state):
    my_action = mcts(root.state,
                     my_problem,
                     timed_budget(0.05),
                     50,
                     select_action=my_select_action,
                     verbose=False)
    print(str(root.state) + " " + my_action.name)  # print information about the current state and best next action
    root = root.perform_action(my_action)  # simulate the execution of this best next action
    reward += root.effect.reward  # for the probabilistic case: keep track of any intermediate rewards
reward += my_problem.goal_reward  # for the probabilistic case: keep track of the reward of the goal
print(root.state)  # display information on the final state to verify we reached the goal
print(reward)  # display information on the reward accumulated during this run

# # used for debugging and illustratign the graphviz system
# my_action = mcts(root.state,
#                  my_problem,
#                  timed_budget(1),  #iteration_budget(350),
#                  50,
#                  select_action=my_select_action,
#                  graphviz=True)
