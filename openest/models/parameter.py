class ParameterBase(object):
    def __init__(self, title, units):
        self.title = title
        self.units = units

    def derive(self, subtitle):
        return ParameterBase(title=self.title + ' ' + subtitle, units=self.units)
