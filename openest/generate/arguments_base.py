import copy

class ArgumentType(object):
    def __init__(self, name, types, description, examplemaker):
        self.name = name
        self.types = types
        self.description = description
        self.examplemaker = examplemaker

    def optional(self, keyword=None):
        child = copy.copy(self)
        child.optional = True
        child.keyword = keyword
        return child

    def describe(self, description):
        child = copy.copy(self)
        child.description = description
        return child

    def rename(self, name):
        child = copy.copy(self)
        child.name = name
        return child

numeric = ArgumentType('numeric', "A numeric value.",
                       [float], lambda context: np.pi)
output_unit = ArgumentType("output_unit", "Units for the result.",
                           [str], lambda context: 'widgets')
