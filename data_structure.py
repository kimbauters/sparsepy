from vose import Vose
import textwrap


class Problem:
    def __init__(self, name, initial, goals, goal_reward, actions):
        self.name = name
        self.init = initial
        self.goals = goals
        self.goal_reward = goal_reward
        self.actions = actions

    def goal_reached(self, state):
        """ Verify if a given state satisfies at least one of the goals as defined for this problem.
        :param state: the state to verify, of the form (negative_atoms, positive_atoms)
        :return: True if the state satisfies one of the goals defined for this problem; False otherwise """
        return any(subgoal[1] <= state and not(subgoal[0] & state) for subgoal in self.goals)

    def __str__(self):
        output = "Problem description of " + self.name + ":"
        output += "\n init conditions:\n"
        output += textwrap.indent(textwrap.fill(", ".join(self.init), 68), "  ") + "\n"

        output += "\n goal conditions:\n"
        for subgoal in self.goals:
            atoms = []
            for pos in subgoal[1]:
                atoms.append(pos)
            for neg in subgoal[0]:
                atoms.append("-" + neg)
            output += textwrap.indent(textwrap.fill(", ".join(atoms), 65), "  -> ") + "\n"
        output += "\n"

        output += "\n " + str(len(self.actions)) + " actions:\n"
        for action in self.actions:
            output += textwrap.indent(str(action), "  ")

        return output


class Effect:
    """ Provide a way to define effects, including their delete and add sets, as well as their probability of occurring.
    """
    def __init__(self, delete, add, probability, reward=0):
        self.delete = delete
        self.add = add
        self.probability = probability
        self.reward = reward

    @property
    def delete(self):
        """ Getter for the delete set of atoms.
        :return: the delete set of atoms
        """
        return self._delete

    @delete.setter
    def delete(self, value):
        """ Setter for the delete set of atoms.
        :param value: an iterable containing the atoms to delete
        """
        self._delete = set(value)  # convert to set for easy interaction

    @property
    def add(self):
        """ Getter for the add set of atoms.
        :return: the add set of atoms
        """
        return self._add

    @add.setter
    def add(self, value):
        """ Setter for the add set of atoms.
        :param value: an iterable containing the atoms to add
        """
        self._add = set(value)  # convert to set for easy interaction

    @property
    def probability(self):
        """ Getter for probability.
        :return: the probability of this effect.
        """
        return self._probability

    @probability.setter
    def probability(self, value):
        """ Setter for probability, verifying that the value is greater or equal than 0 and smaller or equal to 1
        :param value: the value to change the probability into
        """
        if 0 < value > 1:
            raise AttributeError("The value should be between 0 and 1 (inclusive).")
        else:
            self._probability = value

    def __repr__(self):
        return "Effect(" + str(self.delete) + ", " + str(self.add) + ", " + str(self.probability) + ")"

    def __str__(self):
        atoms = []
        for pos in self.add:
            atoms.append(pos)
        for neg in self.delete:
            atoms.append("-" + neg)
        output = "%0.2f" % self.probability + "  "  # pretty print the probability with 2 digits in the fractional part

        if atoms:
            output += textwrap.fill(", ".join(atoms), 63).replace("\n", "\n" + " "*len(str(self.probability))) + "  "

        output += "(" + ("+" if self.reward >= 0 else "") + "%0.2f" % self.reward + ")"
        return output


class Action:
    def __init__(self, name, preconditions=None, effects=None):
        # excellent resource on weighted selection, i.e. loaded dice (i.e. PDF):
        # http://www.keithschwarz.com/darts-dice-coins/
        self.name = name
        self.preconditions = [(set(), set())] if not preconditions else preconditions  # multiple conditions may apply
        self.effects = [] if effects is None else effects
        total_probability = sum([effect.probability for effect in effects])
        if total_probability > 1:  # test whether the total probability does not exceed 1
            raise AttributeError("The probability of the effects of an action must sum up to 1 or less than 1.")
        elif total_probability < 1:  # if the probability is less than 1, add the default effect where nothing changes
            self.effects.append(Effect(set(), set(), 1 - total_probability))

        # sort the effects so that the most likely effect is on top; use for efficient rollouts along most probable path
        self.effects = sorted(self.effects, key=lambda effect: effect.probability, reverse=True)
        self._vose = Vose([(effect.probability, effect) for effect in self.effects])

    def outcome(self):
        """ Determine one of the effects of this action, according to the underlying probability distribution.
            :return: one of the effects of the action. """
        return self._vose.random()

    def __repr__(self):
        return "Action(" + self.name + ", " + str(self.preconditions) + ", " + str(self.effects) + ")"

    def __str__(self):
        output = "name: " + self.name + "\n"
        output += "  preconditions:\n"
        for precondition in self.preconditions:
            atoms = []
            for pos in precondition[1]:
                atoms.append(pos)
            for neg in precondition[0]:
                atoms.append("-" + neg)
            output += textwrap.indent(textwrap.fill(", ".join(atoms), 63), "    -> ") + "\n"
        if not self.preconditions:
            output += "    (none)\n"
        output += "\n"

        output += "  effects:\n"

        for effect in self.effects:
            output += textwrap.indent(str(effect), " "*4) + "\n"
        if not self.effects:
            output += "    (none)\n"

        output += "\n\n"
        return output
