#!/usr/bin/env python
# This module implements the very efficient Vose's algorithm for sampling a weighted list, or loaded die.
# The algorithm works by creating an alias table, which can be sampled using only two calls to a RNG.
# An excellent description of Vose's algorithm can be found on http://www.keithschwarz.com/darts-dice-coins/ .
# The original paper is titled "A Linear Algorithm For Generating Random Numbers With a Given Distribution".

from random import random

# __author__ = "Kim Bauters"
# __copyright__ = "Copyright 2015"
# __credits__ = ["Kim Bauters"]
# __license__ = "BSD"
# __version__ = "1.0.0"
# __maintainer__ = "Kim Bauters"
# __status__ = "Production"


class Vose:
    def __init__(self, elements):
        """ Implementation of the Michael Vose algorithm to efficiently - O(n) - construct an alias table
            to allow very fast - O(1) - random selection of an element in a weighted list.
        :param elements: A list of pairs consisting of the probability and the element to be drawn.
                         For example, [(0.1, 'a'), (0.2, 'b'), (0.3, 'c'), (0.4, 'd')]"""
        self._alias = []
        self._prob = []

        if [element for element in elements if element[0] < 0]:  # raise an error in case of offensive elements
            raise AttributeError("The probability/frequency of each element should be 0 or strictly greater than 0.")
        total_probability = sum([element[0] for element in elements])  # calculate the total probability/frequency
        if total_probability > 0:  # verify this is greater than 0, and use it to normalise the elements
            elements = [(element[0]/total_probability, element[1]) for element in elements]
        else:  # raise an error in case of an empty list or a list equivalent to empty
            raise AttributeError("The sum of the probability/frequency of all elements is not greater than 0.")

        # update the probability of all the elements by normalising them to the average probability of 1/n
        elements = [(element[0] * len(elements), element[1]) for element in elements]
        small = [element for element in elements if element[0] < 1]  # put all elements with p < 1 in small
        large = [element for element in elements if element[0] >= 1]  # and put all the others in large
        while large and small:  # continue as long as both the small and large list are non-empty
            small_element = small.pop()
            large_element = large.pop()
            self._prob.append(small_element[0])  # associate the correct probability with the slot
            self._alias.append((small_element[1], large_element[1]))  # put the elements in their slot
            # update the large element to determine its remaining probability
            large_element = ((large_element[0] + small_element[0]) - 1, large_element[1])
            # if it falls below 1, move it to the list with small elements
            if large_element[0] < 1:
                small.append(large_element)
            else:
                large.append(large_element)
        while large or small:  # continue as long as one list has elements
            element = large.pop() if large else small.pop()  # pop an element from this list
            self._prob.append(1)  # set the probability to 1, as the element will occupy the entire slot
            self._alias.append((element[1], element[1]))  # set the element in both the upper and lower part of the slot

    def random(self):
        """ Randomly draw an element from the weighted list.
        :return: a random element, drawn according to the weighted list """
        i = int(random() * len(self._prob))
        # use the probability to select one part of the slot to return
        return self._alias[i][0] if self._prob[i] >= random() else self._alias[i][1]
