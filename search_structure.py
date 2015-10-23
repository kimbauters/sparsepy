import textwrap  # used for embellishing the Graphviz DOT file layout


class Node:
    # since we will be using a lot of Node instances, optimise the memory use by relying on slots rather than a dict
    __slots__ = ['problem', 'parent', 'action', 'effect', 'state', 'is_goal', 'children',
                 'visits', 'utility', 'untried_actions', 'tried_actions']

    def __init__(self, problem, parent, action, effect, state):
        self.problem = problem  # the problem space in which this node is relevant
        self.parent = parent  # parent node of this node
        self.action = action  # action that was used to get from the parent node to this node
        self.effect = effect  # effect of the action that resulted in the current node
        self.state = state  # the state of the world in this node
        self.is_goal = problem.goal_reached(self.state)  # whether or not this node represents a goal state
        self.children = dict()  # dictionary of children of this node, key-ed by the action and effect to get to them
        self.visits = 0  # number of times this node has been visited
        self.utility = 0  # cumulative utility from going through this node
        # the available actions for which the current state agrees with their preconditions
        self.untried_actions = [a for a in problem.actions if
                                any(pos <= self.state and not (neg & self.state) for neg, pos in a.preconditions)]
        self.tried_actions = {}  # dictionary with the actions we tried so far as keys,
        # and linked to a tuple consisting of their average reward and number of times we applied them: e.g.
        # a1 -> (15, 2)
        # a2 -> (10, 1)

    def simulate_action(self, action, most_probable=False):
        """ Execute the rollout of an action, *without* taking this action out of the list of untried actions.
           :param action: the action to execute
           :return: a new node obtained by applying the action in the current node """
        if most_probable:
            effect = action.effects[0]
        else:
            effect = action.outcome()  # trigger one of the effects of the action
        if (action, effect) in self.children:  # check whether we already applied this action, and gotten this effect
            child = self.children[(action, effect)]  # we already encountered this state; retrieve it
        else:
            state = self.state - effect.delete | effect.add  # compute the new state by using set operations
            child = Node(self.problem, self, action, effect, state)  # create a new node with state
            self.children[(action, effect)] = child  # add this child to the children of this node
        return child

    def perform_action(self, action):
        """ Execute the rollout of an action, *with* taking this action out of the list of untried actions.
           :param action: the action to execute
           :return: a new node obtained  through action in the current node, and the reward associated with this effect
           :raises: a ValueError if trying to perform an action that is already tried for this node """
        self.untried_actions.remove(action)  # remove the action from the list of untried actions
        self.tried_actions[action] = (0, 0)  # add the action to the sequence of actions we already tried
        return self.simulate_action(action)  # get and return (one of) the child(ren) as a result of applying the action

    def rollout_actions(self, rollout_action, depth, horizon):
        """ Organise a rollout from a given node to either a goal node or a leaf node (e.g. by hitting the horizon).
           :param rollout_action: the heuristic to select the action to use for the rollout
           :param depth: the current depth at which the rollout is requested
           :param horizon: the maximum depth to consider
           :return: a new node obtained  through action in the current node, and the reward associated with this effect
           :raises: a ValueError if trying to perform an action that is already tried for this node """
        if self.is_goal:  # check if we have hit a goal state
            return self, depth
        elif depth < horizon:
            action = rollout_action(self)  # use the heuristic to select the next action to perform
            node = self.simulate_action(action, True)  # simulate the execution of this action
            return node.rollout_actions(rollout_action, depth + 1, horizon)
        else:  # the horizon has been reached; return the current node, reward so far, and the current depth
            return self, depth

    def update(self, discounting):
        """ Traverse back up a branch to collect all rewards and to backpropagate these rewards to successor nodes.
            :param discounting: the discounting factor to use when updating ancestor nodes """
        node = self  # set this node as the current node in the backpropagation
        current_reward = 0  # initialise the reward to 0
        while node is not None:  # continue until we have processed the root node
            current_reward *= discounting  # discount the reward obtained in descendants
            if node.is_goal:  # check if this node is a goal state
                current_reward += self.problem.goal_reward  # if it is, assign to it the goal reward
            if node.effect:
                current_reward += node.effect.reward  # add any rewards obtained associated with the effect
            if not node.parent or node.action in node.parent.tried_actions:  # only update the real non-simulated nodes
                if node.parent:  # check if it is not the root node; continue if not
                    utility, visits = node.parent.tried_actions[node.action]  # get the action info from the parent
                    node.parent.tried_actions[node.action] = (utility + current_reward, visits + 1)  # and update
                node.utility += current_reward  # update the total utility gathered in this node
                node.visits += 1  # update the  number of visits to this node
            node = node.parent  # move to the parent node

    def create_graphviz(self, location="graphviz.dot"):
        """ Produce the contents for a Graphviz DOT file representing the search tree as starting from this node.
        :param location: the location of where to save the generated file.
        :return: the location where the Graphviz DOT file has been saved """
        output = "graph sparsepy {\n"
        output += textwrap.indent(self.__graphviz(), "  ")
        output += "}"
        with open(location, 'w') as file:
            file.write(output)
        return location

    def __graphviz(self, name="0"):
        """ Internal method used in the creation of the Graphviz DOT file. This method will be called recursively,
            and only helps to fill the body specifications of the DOT file. """
        output = 'decision_node' + str(name)  # give a unique name to this node
        output += ' [label="' + ', '.join(self.state) + '\n' + \
                  str('%0.2f' % self.utility) + ',' + str(self.visits) + '"]\n'  # add the label to identify its state
        next_id = 0
        for key, child in self.children.items():
            (action, effect) = key  # extract the action out of the (action, effect) pair
            if action in self.tried_actions:  # if this is an action we actually performed, not just simulated: show it
                output += 'action_node' + str(name) + action.name
                output += '[label="' + action.name + '", shape=box]\n'
                child_node_name = name + '_' + str(next_id)
                output += child.__graphviz(child_node_name)
                output += 'action_node' + str(name) + action.name + ' -- '
                output += 'decision_node' + str(child_node_name) + ' [style=dashed, label="' + str(effect) + '"]\n'
                next_id += 1
        for action, info in self.tried_actions.items():
            reward, visits = info
            output += 'decision_node' + str(name) + ' -- action_node' + str(name) + action.name
            output += ' [label="' + '%0.2f' % reward + ',' + str(visits) + '", penwidth="' + str(visits**(1/4)) + '"]\n'
        return output
