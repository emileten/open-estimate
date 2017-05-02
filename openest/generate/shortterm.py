import copy
import numpy as np
from calculation import Calculation, Application, ApplicationEach
from curvegen import CurveGenerator

class SingleWeatherApply(Calculation):
    def __init__(self, units, curve, curve_description, weather_change=lambda x: x):
        super(SingleWeatherApply, self).__init__([units])
        if isinstance(curve, CurveGenerator):
            assert curve.depenunit == units

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

class MonthlyClimateApply(Calculation):
    def __init__(self, units, curve, curve_description, monthmeans, regions, weather_change=lambda x: x):
        super(MonthlyClimateApply, self).__init__([units])
        if isinstance(curve, CurveGenerator):
            assert curve.depenunit == units

        self.curve = curve
        self.curve_description = curve_description
        self.monthmeans = monthmeans
        self.regions = regions
        self.weather_change = weather_change

    def latex(self):
        raise NotImplementedError()

    def apply(self, region, *args):
        if isinstance(self.curve, CurveGenerator):
            curve = self.curve.get_curve(region, *args)
        else:
            curve = self.curve

        region_index = self.regions.index(region)

        def generate(region, time, weather, **kw):
            # Ignores weather
            weather = self.weather_change(self.monthmeans[int(time - 1.5) % 12][region_index])
            result = curve(weather)

            if not np.isnan(result):
                yield time, result

        return ApplicationEach(region, generate)

    def column_info(self):
        description = "Single monthly value from climate applied to " + self.curve_description
        return [dict(name='climatic', title='Direct climatic response', description=description)]

class InstaZScoreApply(Calculation, Application):
    def __init__(self, units, curve, curve_description, lasttime, weather_change=lambda x: x):
        super(InstaZScoreApply, self).__init__([units])
        if isinstance(curve, CurveGenerator):
            assert curve.depenunit == units

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
            self.mean = np.mean(map(lambda x: x[1], self.pastweathers))
            self.sdev = np.std(map(lambda x: x[1], self.pastweathers))

            # Print out all past weathers, now that we have them
            for pastweather in self.pastweathers:
                yield pastweather[0], self.curve((pastweather[1] - self.mean) / self.sdev)

        if self.mean is None:
            # Keep track of this until we have a base
            self.pastweathers.append((time, weather))
        else:
            # calculate this and tack it on
            yield [time, self.curve((weather - self.mean) / self.sdev)]

    def column_info(self):
        description = "Single value applied to " + self.curve_description
        return [dict(name='response', title='Direct marginal response', description=description)]

class MonthlyZScoreApply(Calculation, Application):
    def __init__(self, units, curve, curve_description, monthmeans, monthsdevs, regions, weather_change=lambda x: x):
        super(MonthlyZScoreApply, self).__init__([units])
        if isinstance(curve, CurveGenerator):
            assert curve.depenunit == units

        self.curve = curve
        self.curve_description = curve_description
        self.monthmeans = monthmeans
        self.monthsdevs = monthsdevs
        self.regions = regions
        self.weather_change = weather_change

    def latex(self):
        raise NotImplementedError()

    def apply(self, region, *args, **kwargs):
        app = copy.copy(self)

        if isinstance(self.curve, CurveGenerator):
            app.curve = self.curve.get_curve(region, *args)
        else:
            app.curve = self.curve

        app.region_index = self.regions.index(region)

        return app

    def push(self, time, weather):
        time = time[0]
        weather = self.weather_change(weather[0])

        yield (time, self.curve((weather - self.monthmeans[time % 12][self.region_index]) / self.monthsdevs[time % 12][self.region_index]))

    def column_info(self):
        description = "Single value as z-score applied to " + self.curve_description
        return [dict(name='response', title='Direct marginal response', description=description)]

class SplitByMonth(Calculation):
    def __init__(self, subcalc):
        super(SplitByMonth, self).__init__([subcalc.unitses[0]])
        self.subcalc = subcalc

    def latex(self):
        raise NotImplementedError()

    def apply(self, region, *args, **kwargs):
        bymonth = [self.subcalc.apply(region + '-' + str(month), *args, **kwargs) for month in range(1, 13)]

        timeminmax = [np.inf, -np.inf]
        valuebytime = {}
        def generate(region, time, weather, *args, **kwargs):
            timeminmax[0] = min(time, timeminmax[0])
            timeminmax[1] = max(time, timeminmax[1])
            for yearresult in bymonth[time % 12].push([time], [weather]):
                valuebytime[yearresult[0]] = yearresult[1:]
            return []

        def finish():
            for time in range(mintime, maxtime + 1):
                yield [time] + valuebytime.get(time, [np.nan])

        return ApplicationEach(region, generate, finish)

    def column_info(self):
        description = "Separately applied by month"
        return [dict(name='split', title='Month split', description=description)]
