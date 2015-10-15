"""
This module implements a Monte-Carlo Tree Search (MCTS), enriched with UCB1 and
 techniques from PROST to handle stochastic effects of actions (notably, from sparse UCT).

Intuitively, MCTS works by (1) selecting a node to expand, (2) expanding the node,
 (3) simulating one of the rollouts from that node to a final state/horizon limit, and
 (4) backpropagating the results to guide the search in selecting the next node.

Intuitively, UCB1 works by balancing the exploration/exploitation tradeoff.
 It minimises regret by ensuring that we do explore less promising branches
 to ensure we don't miss out, while ensuring that we sufficiently explore
 more promising branches to achieve the best possible results.

Intuitively, PROST works by dividing the tree in layers, where one of the
 alternating layers consists of states, in which we can select the next action
 to take, and the other layer consists of actions, from which we transition to
 other states pending on the stochastic effect that occurs (which affects (3-4)).
"""
import logging as log
from random import choice
from collections import namedtuple
from search_structure import Node

__author__ = "Kim Bauters"


# provide a named tuple for easy access to all information relevant to available actions after the search finishes
ActInfo = namedtuple('ActInfo', 'action reward visits')


def mcts(root_state, problem, budget, horizon,
         select_action=lambda node: choice(list(node.tried_actions.keys())),
         expand_action=lambda node: choice(node.untried_actions),
         rollout_action=lambda node: choice(node.untried_actions),
         select_best=lambda acts: sorted(acts, key=lambda act: act.reward/act.visits, reverse=True)[0].action,
         *, discounting=0.9, verbose=False, graphviz=False):
    """
    :param root_state: the initial state from which to start the search
    :param problem: a description of the problem in the form of a Problem instance data structure
    :param budget: a function that is called after each cycle to determine
                     whether we should stop (return False) or continue (return True)
    :param horizon: the maximum depth up to which to explore the search tree
    :param select_action:  heuristic used to select an action during step 1 of each MCTS iteration
                           the function should select one argument, which is a list of actions we already tried
    :param expand_action:  heuristic used to select an action to expand to a new node during step 2 of an MCTS iteration
                           the function should accept two arguments: the current node and the untried actions
    :param rollout_action: heuristic used to select which action to simulate during each step of the rollout phase
    :param discounting: can only be given as named parameter; alters the default discounting value
    :param verbose: can only be given as named parameter; provides (very) verbose output while searching
    :param graphviz: can only be given as named parameter; return the DOT graphviz contents associated with the search
    :return: the next best action to take
    """

    log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG if verbose else log.ERROR)
    root = Node(problem, None, None, None, root_state)  # each episode starts from the root node
    iterations = 0  # so far, no iterations as we still have to start

    while budget(iterations):  # continue exploring for as long as we have the computational budget

        node = root  # the node to start from is the root node
        depth = 1  # we are at the start, so a depth of 1

        # (1) select: descend through the search tree to find a node to expand
        log.info("Monte-Carlo Tree Search iteration starting from " + str(node.state))
        log.info("  step (1): selecting node")

        # find a node with untried actions by recursing through the children
        while not node.untried_actions and node.children and depth <= horizon:
            action = select_action(node)  # use heuristics to select the best action to follow
            log.info("  -> " + action.name)
            node = node.simulate_action(action)  # simulate the action to determine its stochastic outcome
            depth += 1
        # stop once we find a node with untried actions, or when the node does not have any children
        log.info("  selected node with the state " + str(node.state))

        # (2) expand: expand the node we just found
        log.info("  step (2): expanding node on depth " + str(depth))
        # check that the node we ended up with has actions we still have to try
        if node.untried_actions and depth <= horizon and not node.is_goal:
            action = expand_action(node)  # use heuristics to pick one of the actions to try
            log.info("  -> " + action.name)
            node = node.perform_action(action)  # execute this action; set the node to the generated child
            log.info("  the new state became " + str(node.state))
            depth += 1

        # (3) rollout: simulate a full run from the expanded node
        log.info("  step (3): performing rollout")
        # perform a rollout from the current node; return final node, and total descend depth
        node, depth = node.rollout_actions(rollout_action, depth, horizon)
        log.info("  ended up in the state " + str(node.state))

        # (4) backpropagate: update the search tree to reflect the results from the rollout
        log.info("  step (4): backpropagating from depth " + str(depth) + " with " +
                 ("success" if node.is_goal else "no success"))

        node.update(discounting)  # perform the update of the values

        iterations += 1

    log.info("search completed\n")
    if graphviz:
        print(root.get_graphviz())

    # collect all the actions available in the root along with any information used for choosing
    actions = [ActInfo(action, reward, visits) for
               action, (reward, visits) in root.tried_actions.items()]

    return select_best(actions)