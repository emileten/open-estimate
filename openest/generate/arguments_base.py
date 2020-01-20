"""Argument self-documentation system.

Calculations can report their arguments, for use in interfaces that
allow calculations to be pieced together. The baseline class defined
here is instantiated in `arguments.py`.
"""

import copy

class ArgumentType(object):
    """A description of a given argument.

    Attributes
    ----------
    name : str
        The short name of the argument (typically, its function argument variable name).
    description : str
        A longer description of the argument.
    types : sequence of type
        The possible types that are accepted for this argument.
    examplemaker : function(dict) -> one of types
        Takes a variable dictionary environment, and produces a possible example argument.
    parent : ArgumentType
        If not None, defines this as a sub-type of another ArgumentType.
    is_optional : bool
        If true, the argument can be absent, and the keyword attribute should be set.
    keyword : str
        The name for this argument in the keyword arguments dictionary.
    """

    def __init__(self, name, description, types, examplemaker):
        self.name = name
        self.description = description
        self.types = types
        self.examplemaker = examplemaker
        self.parent = None
        self.is_optional = False
        self.keyword = None

    def __str__(self):
        return "%s argument: %s" % (self.name, self.description)

    def optional(self, keyword=None):
        """Record that the argument can be absent and set the keyword."""
        child = copy.copy(self)
        child.is_optional = True
        child.keyword = keyword
        child.parent = self
        return child

    def describe(self, description):
        """Change the description of an argumet."""
        child = copy.copy(self)
        child.description = description
        child.parent = self
        return child

    def rename(self, name):
        """Change the name of an argument."""
        child = copy.copy(self)
        child.name = name
        child.parent = self
        return child

    def isa(self, parent):
        """Check if the argument type is a sub-type of (or equal to) the given parent."""
        return self == parent or self.parent == parent

# Foundational argument types

numeric = ArgumentType('numeric', "A numeric value.",
                       [float], lambda context: np.pi)
output_unit = ArgumentType("output_unit", "Units for the result.",
                           [str], lambda context: 'widgets')
