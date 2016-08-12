import copy
import numpy as np
from calculation import Calculation, Application, ApplicationEach
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

class InstaZScoreApply(Calculation, Application):
    def __init__(self, units, curve, curve_description, lasttime, weather_change=lambda x: x):
        super(InstaZScoreApply, self).__init__([units])
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

    def apply(self, region, *args, **kwargs):
        app = copy.copy(self)

        if isinstance(self.curve, CurveGenerator):
            app.curve = self.curve.get_curve(region, *args)
        else:
            app.curve = self.curve

        app.pastweathers = [] # don't copy this across instances!

        return app

    def push(self, time, weather):
        time = time[0]
        weather = self.weather_change(weather[0])

        # Have we collected all the data?
        if time == self.lasttime or (self.lasttime is None and self.mean is None):
            self.mean = np.mean(self.pastweathers)
            self.sdev = np.std(self.pastweathers)

            # Print out all past weathers, now that we have them
            for pastweather in self.pastweathers:
                yield [time, self.curve((pastweather - self.mean) / self.sdev)]

        if self.mean is None:
            # Keep track of this until we have a base
            self.pastweathers.append(weather)
        else:
            # calculate this and tack it on
            yield [time, self.curve((weather - self.mean) / self.sdev)]

    def column_info(self):
        description = "Single value applied to " + self.curve_description
        return [dict(name='response', title='Direct marginal response', description=description)]