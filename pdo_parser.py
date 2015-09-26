from data_structure import Action, Effect, Problem
from grako.ast import AST
from fractions import Fraction
from parse import PPDDLsubParser


class PDOParser(PPDDLsubParser):
    def __init__(self):
        super(PDOParser, self).__init__()

    class PDOSemantics(object):
        """ When encountering an effect rule, verify whether the body is a probabilistic effect.
            If not, convert it into one with a probability of 1 for the outcome. """
        def effect(self, intercepted_ast):
            if 'prob_effects' not in intercepted_ast.args:
                conjunction = intercepted_ast.args
                intercepted_ast = AST({'args': AST({'prob_effects': [AST({'conjunction': conjunction, 'value': 1})]})})
            return intercepted_ast

        """ Process all the values to turn them from strings into either fractions or floats/"""
        def value(self, intercepted_ast):
            if '/' in intercepted_ast:
                split = intercepted_ast.split('/', 1)
                intercepted_ast = Fraction(int(split[0]), int(split[1]))
            else:
                intercepted_ast = float(intercepted_ast)
            return intercepted_ast

    def _parse_input(self, new_input):
        return super(PDOParser, self).parse(new_input, "start", semantics=self.PDOSemantics())

    def process_input(self, new_input):
        ast = self._parse_input(new_input)

        initial_state = []
        for arg in ast.init.conjunction.args:
            initial_state.append(arg.atom)

        goal_states = []
        for each_goal in ast.goal.subgoals.disjunction:
            sub_goal = (set(), set())
            for arg in each_goal.args:
                if "negation" in arg:
                    for neg in arg.negation:
                        sub_goal[0].add(neg.atom)
                else:
                    sub_goal[1].add(arg.atom)
            goal_states.append(sub_goal)

        goal_reward = 0
        if ast.extra and "goal_reward" in ast.extra:
            goal_reward = float(ast.extra.goal_reward)

        actions = []
        for action in ast['actions']:
            effects = []
            preconditions = []
            if 'prob_effects' in action.effect.args:
                for prob_effect in action.effect.args['prob_effects']:
                    atoms_delete = set()
                    atoms_add = set()
                    reward = 0
                    for arg in list(prob_effect['conjunction']['args']):
                        if 'negation' in arg:
                            for neg in arg['negation']:
                                atoms_delete.add(neg['atom'])
                        elif 'atom' in arg:
                            atoms_add.add(arg['atom'])
                        elif 'increase' in arg:
                            reward = float(arg['increase'])
                        elif 'decrease' in arg:
                            reward = -float(arg['decrease'])
                    new_effect = Effect(atoms_delete, atoms_add, prob_effect['value'], reward)
                    effects.append(new_effect)

            if action.precondition:
                for condition in action.precondition.args.disjunction:
                    atoms_delete = set()
                    atoms_add = set()
                    for arg in condition.args:
                        if 'negation' in arg:
                            for neg in arg['negation']:
                                atoms_delete.add(neg['atom'])
                        else:
                            atoms_add.add(arg['atom'])
                    preconditions.append((atoms_delete, atoms_add))

            new_action = Action(action.name, preconditions, effects)
            actions.append(new_action)

        my_problem = Problem(ast.problem.problem_name, set(initial_state), goal_states, goal_reward, actions)

        return my_problem
