from data_structure import Action, Effect, Problem
from grako.ast import AST
from fractions import Fraction
from parse import PPDDLsubParser


class PDOParser(PPDDLsubParser):
    """ The PDOParser class is used to modify the parser output and to override certain behaviour in the parser. """
    def __init__(self):
        super(PDOParser, self).__init__()  # standard init calling the superclass

    class PDOSemantics(object):

        def effect(self, intercepted_ast):
            """ When encountering an effect rule, verify whether the body is a probabilistic effect.
            If not, convert it into one with a probability of 1 for the outcome. """
            # when the effects of an action does not have probabilistic components ...
            if 'prob_effects' not in intercepted_ast.args:
                # ... retrieve what the effect is ...
                conjunction = intercepted_ast.args
                # ... and convert it into a probabilistic effect with probability 1.
                intercepted_ast = AST({'args': AST({'prob_effects': [AST({'conjunction': conjunction, 'value': 1})]})})
            # push this change into the parse tree
            return intercepted_ast

        def value(self, intercepted_ast):
            """ Process all the values to turn them from strings into either fractions or floats. """
            # verify if the symbol / is used when we except a number
            if '/' in intercepted_ast:
                # if it is, split the value into a numerator and denominator
                split = intercepted_ast.split('/', 1)
                # use the numerator and denominator to make a Fraction, which allows calculations without fps issues
                intercepted_ast = Fraction(int(split[0]), int(split[1]))
            else:
                # otherwise, parse the number as a float
                intercepted_ast = float(intercepted_ast)
            # push this change into the parse tree
            return intercepted_ast

    def _parse_input(self, new_input):
        """ Internal function that will make the parsing start from the desired rule and with the desired semantics
            to modify the behaviour of the parser (as defined above).
        :param new_input: the input to parse
        :return: the parsed AST """
        return super(PDOParser, self).parse(new_input, "start", semantics=self.PDOSemantics())

    def process_input(self, new_input):
        """ The function will take an input and will parse it into a legal Problem class when the input is valid.
        :param new_input: the input to parse
        :return: the parsed input as a Problem class """
        ast = self._parse_input(new_input)

        # construct the initial state ...
        initial_state = []
        # ... by iteration over all the atoms in the conjunction used to denote the initial state.
        for arg in ast.init.conjunction.args:
            initial_state.append(arg.atom)

        # construct the goal states ...
        goal_states = []
        # ... by looking at every possible goal we accept
        for each_goal in ast.goal.subgoals.disjunction:
            # for each goal, prepare a structure to keep track of negative/positive atoms
            sub_goal = (set(), set())
            # go over each element of the conjunction
            for arg in each_goal.args:
                # if negated ...
                if "negation" in arg:
                    for neg in arg.negation:
                        # ... add the atom to the negative atoms set
                        sub_goal[0].add(neg.atom)
                else:
                    # otherwise, add the atom to the positive atoms set
                    sub_goal[1].add(arg.atom)
            # add this sub goal to the list of all potential goals
            goal_states.append(sub_goal)

        # gather the reward for reaching a goal ...
        goal_reward = 0
        if ast.extra and "goal_reward" in ast.extra:
            # ... by converting the value into a float
            goal_reward = float(ast.extra.goal_reward)

        # gather all available actions ...
        actions = []
        # ... by iterating over each action statement in the parse tree
        for action in ast['actions']:
            # an action consist of effects ...
            effects = []
            # ... and of preconditions
            preconditions = []
            # we only need to worry about probabilistic effects as classical effects have been converted already
            if 'prob_effects' in action.effect.args:
                # for each probabilistic effect ...
                for prob_effect in action.effect.args['prob_effects']:
                    # ... prepare its delete/add sets, as well as any potential reward for performing the effect
                    atoms_delete = set()
                    atoms_add = set()
                    reward = 0
                    # go over each element of the conjunction
                    for arg in list(prob_effect['conjunction']['args']):
                        # if negated ...
                        if 'negation' in arg:
                            for neg in arg['negation']:
                                # ... add the atom to the negative atoms set
                                atoms_delete.add(neg['atom'])
                        elif 'atom' in arg:
                            # otherwise, add the atom to the positive atoms set
                            atoms_add.add(arg['atom'])
                        elif 'increase' in arg:
                            # if we find a positive reward for executing this effect parse it as a positive float
                            reward = float(arg['increase'])
                        elif 'decrease' in arg:
                            # if we find a negative reward for executing this effect parse it as a negative float
                            reward = -float(arg['decrease'])
                    # create the new effect based on its delete/add list, its probability, and its reward
                    new_effect = Effect(atoms_delete, atoms_add, prob_effect['value'], reward)
                    # add this effect to the list of effects for this action
                    effects.append(new_effect)

            # verify whether any preconditions are satisfied
            if action.precondition:
                # if so, iterate over each precondition
                for condition in action.precondition.args.disjunction:
                    # a precondition is defined by its delete/add list
                    atoms_delete = set()
                    atoms_add = set()
                    # go over each element of the conjunction
                    for arg in condition.args:
                        # if negated ...
                        if 'negation' in arg:
                            for neg in arg['negation']:
                                # ... add the atom to the negative atoms set
                                atoms_delete.add(neg['atom'])
                        else:
                            # otherwise, add the atom to the positive atoms set
                            atoms_add.add(arg['atom'])
                    # add this precondition to the list of preconditions for this effect
                    preconditions.append((atoms_delete, atoms_add))

            # create the new action based on its name, its preconditions, and its effects
            new_action = Action(action.name, preconditions, effects)
            # append this action to the list of actions available for our problem domain
            actions.append(new_action)

        # finally, create the problem domain using the name for the problem domain,
        # the initial state (which is still a list, so convert it),
        # the goal states, the reward for reaching any goal state, and the actions
        my_problem = Problem(ast.problem.problem_name, set(initial_state), goal_states, goal_reward, actions)

        return my_problem
