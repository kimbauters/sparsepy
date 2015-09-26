from random import choice
from math import log, sqrt

from search_structure import Node
from pdo_parser import PDOParser
from mcts import mcts

verbose = False

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


my_parser = PDOParser()
my_problem = my_parser.process_input(my_input)


def iteration_budget(allowed_iterations):

    def inner_iteration_budget(iterations):
        return iterations < allowed_iterations
    return inner_iteration_budget


def timed_budget(allowed_secs):
    from time import perf_counter
    start_time = None

    def inner_timed_budget(_):
        nonlocal start_time
        if not start_time:
            start_time = perf_counter()
        if perf_counter() - start_time > allowed_secs:
            start_time = None
            return False
        else:
            return True
    return inner_timed_budget


def my_expand_action(node):
    """ Expand one of the untried actions at random. """
    return choice(node.untried_actions)


def my_select_action(node):
    """ Select the best action, by taking the action that has 
        the best average reward vs the least number of explorations. """
    best_action = (None, 0)
    if verbose:
        print("  choosing 1 action out of " + str(len(node.tried_actions)))
    for action, (action_reward, visits) in node.tried_actions.items():
        ucb1 = (1/sqrt(2))*sqrt(log(node.visits)/visits)+(action_reward/visits)
        if not best_action[0] or ucb1 > best_action[1]:
            best_action = (action, ucb1)
    return best_action[0]


def my_rollout_action(node):
    """ Select a random action during the rollout phase. """
    return choice(node.untried_actions)


root = Node(my_problem, None, None, None, my_problem.init)
reward = 0

while not my_problem.goal_reached(root.state):
    my_action = mcts(root.state,
                     my_problem,
                     timed_budget(0.05),
                     50,
                     select_action=my_select_action,
                     verbose=False)
    print(str(root.state) + " " + my_action.name)
    root = root.perform_action(my_action)
    reward += root.effect.reward
reward += my_problem.goal_reward
print(root.state)
print(reward)

# my_action = mcts(root.state,
#                  my_problem,
#                  timed_budget(1),  #iteration_budget(350),
#                  50,
#                  select_action=my_select_action,
#                  graphviz=True)
