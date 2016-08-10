import numpy as np
from calculation import Calculation, ApplicationEach
from curvegen import CurveGenerator

class SingleWeatherApply(Calculation):
    def __init__(self, units, curve, curve_description, weather_change=lambda x: x):
        super(SingleWeatherApply, self).__init__([units])
        if isinstance(curve, CurveGenerator):
            assert curve.depenunits == units

        self.curve = curve
        self.curve_description = curve_description
        self.weather_change = weather_change

    def latex(self):
        raise NotImplementedError()

    def apply(self, region, *args):
        if isinstance(self.curve, CurveGenerator):
            curve = self.curve.get_curve(region, *args)
        else:
            curve = self.curve

        def generate(region, time, weather, **kw):
            weather = self.weather_change(weather)
            result = curve(weather)

            if not np.isnan(result):
                yield time, result

        return ApplicationEach(region, generate)

    def column_info(self):
        description = "Single value applied to " + self.curve_description
        return [dict(name='response', title='Direct marginal response', description=description)]
