import os, csv, random
import numpy as np
import latextools
from calculation import Calculation, Application, ApplicationByYear
from ..models.model import Model
from ..models.spline_model import SplineModel
from ..models.bin_model import BinModel
from ..models.memoizable import MemoizedUnivariate
from ..models.curve import UnivariateCurve, AdaptableCurve, StepCurve
from curvegen import CurveGenerator

# Generate integral over daily temperature
class MonthlyDayBins(Calculation):
    def __init__(self, model, units, pval=.5, weather_change=lambda temps: temps):
        super(MonthlyDayBins, self).__init__([units])
        self.model = model
        if isinstance(model, UnivariateCurve):
            spline = model
        elif isinstance(model, BinModel):
            spline = StepCurve(model.get_xx(), [model.eval_pval(x, pval) for x in model.get_xx_centers()])
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

            result = np.nansum(results) / 12

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
            self.xx = model.get_xx()
            self.spline = model
        elif isinstance(model, BinModel):
            self.xx = model.get_xx_centers()
            self.spline = StepCurve(model.get_xx(), [model.eval_pval(x, pval) for x in model.get_xx_centers()])
        else:
            self.xx = model.get_xx()
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
            if len(temps.shape) == 2:
                if temps.shape[0] == 12 and temps.shape[1] == len(self.xx):
                    yy = spline(self.xx)
                    yy[np.isnan(yy)] = 0
                    result = np.sum(temps.dot(yy))
                else:
                    raise RuntimeError("Unknown format for temps: " + str(temps.shape[0]) + " x " + str(temps.shape[1]) + " <> len " + str(self.xx))
            else:
                result = np.nansum(spline(temps))

            if not np.isnan(result):
                yield (year, result)

            if isinstance(self.spline, AdaptableCurve):
                spline.update(year, temps)

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

class YearlyAverageDay(Calculation):
    def __init__(self, units, curvegen, curve_description, weather_change=lambda x: x):
        super(YearlyAverageDay, self).__init__([units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description
        self.weather_change = weather_change

    def latex(self):
        funcvar = latextools.get_function()
        yield ("Equation", r"\frac{1}{365} \sum_{d \in y(t)} %s(T_d)" % (funcvar), self.unitses[0])
        yield ("T_d", "Temperature", "deg. C")
        yield ("%s(\cdot)" % (funcvar), self.curve_description, self.unitses[0])

    def apply(self, region, *args):
        curve = self.curvegen.get_curve(region, *args)

        def generate(region, year, temps, **kw):
            temps = self.weather_change(temps)
            result = np.nansum(curve(temps)) / len(temps)

            if not np.isnan(result):
                yield (year, result)

            if isinstance(self.curve, AdaptableCurve):
                curve.update(year, temps)

        return ApplicationByYear(region, generate)

    def column_info(self):
        description = "The average result across a year of daily temperatures applied to " + self.curve_desciption
        return [dict(name='response', title='Direct marginal response', description=description)]

class YearlyDividedPolynomialAverageDay(Calculation):
    def __init__(self, units, curvegen, curve_description, weather_change=lambda x: x):
        super(YearlyAverageDay, self).__init__([units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description
        self.weather_change = weather_change

    def latex(self):
        raise NotImplementedError

    def apply(self, region, *args):
        curve = self.curvegen.get_curve(region, *args)

        def generate(region, year, temps, **kw):
            temps = self.weather_change(temps)
            assert temps.shape[1] == len(curve.ccs)
            result = np.nansum(np.dot(curve.ccs, temps)) / len(temps)

            if not np.isnan(result):
                yield (year, result)

            if isinstance(self.curve, AdaptableCurve):
                curve.update(year, temps)

        return ApplicationByYear(region, generate)

    def column_info(self):
        description = "The average result across a year of daily temperatures applied to a polynomial."
        return [dict(name='response', title='Direct marginal response', description=description)]
