from .calculation import Calculation, ApplicationByYear
from .formatting import FormatElement

class Constant(Calculation):
    def __init__(self, value, units):
        super(Constant, self).__init__([units])
        self.value = value

    def format(self, lang):
        return {'main': FormatElement(str(self.value))}

    def apply(self, region):
        def generate(region, year, temps, **kw):
            yield year, self.value

        return ApplicationByYear(region, generate)

    def column_info(self):
        return [dict(name='response', title="Constant value", description="Always equal to " + str(self.value))]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='year',
                    arguments=[arguments.numeric, arguments.output_unit],
                    description="Return a given constant for each year.")
