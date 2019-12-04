import copy

class ArgumentType(object):
    def __init__(self, name, description, types, examplemaker):
        self.name = name
        self.description = description
        self.types = types
        self.examplemaker = examplemaker
        self.parent = None

    def __str__(self):
        return "%s argument: %s" % (self.name, self.description)

    def optional(self, keyword=None):
        child = copy.copy(self)
        child.is_optional = True
        child.keyword = keyword
        child.parent = self
        return child

    def describe(self, description):
        child = copy.copy(self)
        child.description = description
        child.parent = self
        return child

    def rename(self, name):
        child = copy.copy(self)
        child.name = name
        child.parent = self
        return child

    def isa(self, parent):
        return self == parent or self.parent == parent

numeric = ArgumentType('numeric', "A numeric value.",
                       [float], lambda context: np.pi)
output_unit = ArgumentType("output_unit", "Units for the result.",
                           [str], lambda context: 'widgets')
