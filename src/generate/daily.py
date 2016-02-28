import os, csv, random
import numpy as np
import weather, latextools
from calculation import Calculation, Application, ApplicationByYear
from ..models.model import Model
from ..models.spline_model import SplineModel
from ..models.memoizable import MemoizedUnivariate

# Generate integral over daily temperature

class MonthlyDayBins(Calculation):
    def __init__(self, model, units, pval=.5, weather_change=lambda temps: temps):
        super(MonthlyDayBins, self).__init__([units])
        model = MemoizedUnivariate(model)
        model.set_x_cache_decimals(1)
        spline = model.get_eval_pval_spline(pval, (-40, 80), threshold=1e-2)

        self.spline = spline
        self.weather_change = weather_change

    def apply(self, fips):
        def generate(fips, year, temps, **kw):
            temps = self.weather_change(temps)
            results = self.spline(temps)

            result = np.sum(results) / 12

            if not np.isnan(result):
                yield (year, result)

        return ApplicationByYear(fips, generate)

    def column_info(self):
        return [dict(name='response', title='Direct impulse response', source="From " + str(self.model))]

class YearlyDayBins(Calculation):
    def __init__(self, model, units, pval=.5):
        super(YearlyDayBins, self).__init__([units])
        self.model = model
        memomodel = MemoizedUnivariate(model)
        memomodel.set_x_cache_decimals(1)
        self.spline = memomodel.get_eval_pval_spline(pval, (-40, 80), threshold=1e-2)

    def latex(self):
        funcvar = latextools.get_function()
        yield ("Equation", r"\sum_{d \in y(t)} %s(T_d)" % (funcvar), self.unitses[0])
        yield ("T_d", "Temperature", "deg. C")
        yield ("%s(\cdot)" % (funcvar), str(self.model), self.unitses[0])

    def apply(self, fips):
        def generate(fips, year, temps, **kw):
            result = np.sum(self.spline(temps))

            if not np.isnan(result):
                yield (year, result)

        return ApplicationByYear(fips, generate)

    def column_info(self):
        description = "The combined result of daily temperatures, organizaed into bins according to %s." % (str(self.model))
        return [dict(name='response', title='Direct marginal response', description=description)]

def make_daily_averagemonth(model, func=lambda x: x, pval=.5):
    days_bymonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    transitions = np.cumsum(days_bymonth)

    model = MemoizedUnivariate(model)
    model.set_x_cache_decimals(1)
    spline = model.get_eval_pval_spline(pval, (-40, 80), threshold=1e-2)

    def generate(fips, yyyyddd, temps, **kw):
        if fips == effectset.FIPS_COMPLETE:
            return

        for (year, temps) in weather.yearly_daily_ncdf(yyyyddd, temps):
            bymonth = []
            for mm in range(12):
                avgmonth = np.mean(temps[transitions[mm]-days_bymonth[mm]:transitions[mm]])
                bymonth.append(spline(avgmonth))
                #bymonth.append(model.eval_pval(avgmonth, pval, threshold=1e-2))

            result = np.mean(bymonth)
            if not np.isnan(result):
                yield (year, func(result))

    return generate

def make_daily_percentwithin(endpoints):
    def generate(fips, yyyyddd, temps, **kw):
        if fips == effectset.FIPS_COMPLETE:
            return

        for (year, temps) in weather.yearly_daily_ncdf(yyyyddd, temps):
            results = []
            for ii in range(len(endpoints)-1):
                result = np.sum(temps > endpoints[ii]) - np.sum(temps > endpoints[ii+1])
                results.append(result)

            results = list(np.array(results) / float(len(temps)))

            yield tuple([year] + results)

    return generate
