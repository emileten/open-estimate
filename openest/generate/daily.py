import os, csv, random
import numpy as np
import latextools
from calculation import Calculation, Application, ApplicationByYear
from ..models.model import Model
from ..models.spline_model import SplineModel
from ..models.memoizable import MemoizedUnivariate
from ..models.curve import UnivariateCurve

# Generate integral over daily temperature
class MonthlyDayBins(Calculation):
    def __init__(self, model, units, pval=.5, weather_change=lambda temps: temps):
        super(MonthlyDayBins, self).__init__([units])
        if isinstance(model, UnivariateCurve):
            spline = model
        else:
            model = MemoizedUnivariate(model)
            model.set_x_cache_decimals(1)
            spline = model.get_eval_pval_spline(pval, (-40, 80), threshold=1e-2)

        self.spline = spline
        self.weather_change = weather_change

    def latex(self):
        funcvar = latextools.get_function()
        yield ("Equation", r"\frac{1}{12} \sum_{d \in y(t)} %s(T_d)" % (funcvar), self.unitses[0])
        yield ("T_d", "Temperature", "deg. C")
        yield ("%s(\cdot)" % (funcvar), str(self.model), self.unitses[0])

    def apply(self, region):
        def generate(region, year, temps, **kw):
            temps = self.weather_change(temps)
            results = self.spline(temps)

            result = np.sum(results) / 12

            if not np.isnan(result):
                yield (year, result)

        return ApplicationByYear(region, generate)

    def column_info(self):
        description = "The combined result of daily temperatures, organized into bins according to %s, divided by 12 to describe monthly effects." % (str(self.model))
        return [dict(name='response', title='Direct marginal response', description=description)]

class YearlyDayBins(Calculation):
    def __init__(self, model, units, pval=.5):
        super(YearlyDayBins, self).__init__([units])
        self.model = model

        if isinstance(model, UnivariateCurve):
            self.spline = model
        else:
            memomodel = MemoizedUnivariate(model)
            memomodel.set_x_cache_decimals(1)
            self.spline = memomodel.get_eval_pval_spline(pval, (-40, 80), threshold=1e-2)

    def latex(self):
        funcvar = latextools.get_function()
        yield ("Equation", r"\sum_{d \in y(t)} %s(T_d)" % (funcvar), self.unitses[0])
        yield ("T_d", "Temperature", "deg. C")
        yield ("%s(\cdot)" % (funcvar), str(self.model), self.unitses[0])

    def apply(self, region, *args):
        if isinstance(self.spline, AdaptableCurve):
            spline = self.spline.create(region, *args)
        else:
            spline = self.spline
            
        def generate(region, year, temps, **kw):
            result = np.sum(spline(temps))

            if not np.isnan(result):
                yield (year, result)

        return ApplicationByYear(region, generate)

    def column_info(self):
        description = "The combined result of daily temperatures, organized into bins according to %s." % (str(self.model))
        return [dict(name='response', title='Direct marginal response', description=description)]

class AverageByMonth(Calculation):
    def __init__(self, model, units, func=lambda x: x, pval=.5):
        super(AverageMonthToYear, self).__init__([units])
        self.days_bymonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.transitions = np.cumsum(self.days_bymonth)

        if isinstance(model, UnivariateCurve):
            self.spline = model
        else:
            model = MemoizedUnivariate(model)
            model.set_x_cache_decimals(1)
            self.spline = model.get_eval_pval_spline(pval, (-40, 80), threshold=1e-2)

    def latex(self):
        funcvar = latextools.get_function()
        yield ("Equation", r"mean(\{mean_{d \in m(t)} %s(T_d)\})" % (funcvar), self.unitses[0])
        yield ("T_d", "Temperature", "deg. C")
        yield ("%s(\cdot)" % (funcvar), str(self.model), self.unitses[0])

    def apply(self, region):
        def generate(region, year, temps, **kw):
            bymonth = []
            for mm in range(12):
                avgmonth = np.mean(temps[self.transitions[mm]-self.days_bymonth[mm]:self.transitions[mm]])
                bymonth.append(self.spline(avgmonth))

            result = np.mean(bymonth)
            if not np.isnan(result):
                yield (year, func(result))

        return ApplicationByYear(region, generate)

    def column_info(self):
        description = "The effects of monthly average temperatures, organized into bins according to %s, averaged over months." % (str(self.model))
        return [dict(name='response', title='Direct marginal response', description=description)]

class PercentWithin(Calculation):
    def __init__(self, endpoints):
        super(PercentWithin, self).__init__(['portion'])
        self.endpoints = endpoints

    def latex(self):
        pass

    def apply(self, region):
        def generate(region, year, temps, **kw):
            results = []
            for ii in range(len(self.endpoints)-1):
                result = np.sum(temps > self.endpoints[ii]) - np.sum(temps > self.endpoints[ii+1])
                results.append(result)

            results = list(np.array(results) / float(len(temps)))

            yield tuple([year] + results)

        return ApplicationByYear(region, generate)

    def column_info(self):
        return [dict(name='bin' + str(ii), title="Portion in bin " + str(ii), description="The portion of each year falling between %f and %f" % (self.endpoints[ii], self.endpoints[ii+1])) for ii in range(len(self.endpoints)-1)]

class Constant(Calculation):
    def __init__(self, value, units):
        super(Constant, self).__init__([units])
        self.value = value

    def latex(self):
        yield ("Equation", str(self.value), self.unitses[0])

    def apply(self, region):
        def generate(region, year, temps, **kw):
            yield (year, self.value)

        return ApplicationByYear(region, generate)

    def column_info(self):
        return [dict(name='response', title="Constant value", description="Always equal to " + str(self.value))]
