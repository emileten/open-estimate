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

class InstaZScoreApply(CustomFunctionalCalculation):
    def __init__(self, units, curve, curve_description, lasttime, weather_change=lambda x: x):
        super(SingleWeatherApply, self).__init__([units])
        if isinstance(curve, CurveGenerator):
            assert curve.depenunits == units

        self.curve = curve
        self.curve_description = curve_description
        self.lasttime = lasttime
        self.weather_change = weather_change

        self.mean = None # The mean to subtract off
        self.sdev = None # The sdev to divide by
        self.pastresults = [] # results before lasttime

    def latex(self):
        raise NotImplementedError()

    def init_apply(self):
        if isinstance(self.curve, CurveGenerator):
            self.curve = self.curve.get_curve(region, *args)

        self.pastweathers = [] # don't copy this across instances!

    def pushhandler(self, time, weather, lasttime):
        weather = self.weather_change(weather)

        # Have we collected all the data?
        if year == lasttime or (lasttime is None and self.mean is None):
            self.mean = np.mean(self.pastweathers)
            self.sdev = np.std(self.pastweathers)

            # Print out all past weathers, now that we have them
            for pastweather in self.pastweathers:
                yield [time, curve((pastweather - self.mean) / self.sdev)]

        if self.mean is None:
            # Keep track of this until we have a base
            self.pastweathers.append(weather)
        else:
            # calculate this and tack it on
            yield [time, curve((weather - self.mean) / self.sdev)]

    def column_info(self):
        description = "Single value applied to " + self.curve_description
        return [dict(name='response', title='Direct marginal response', description=description)]
