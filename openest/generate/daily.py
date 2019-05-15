import os, csv, random
import numpy as np
import xarray as xr
import formatting, arguments, diagnostic
from calculation import Calculation, Application, ApplicationByYear
from formatting import FormatElement
from ..models.model import Model
from ..models.spline_model import SplineModel
from ..models.bin_model import BinModel
from ..models.memoizable import MemoizedUnivariate
from ..models.curve import UnivariateCurve, StepCurve
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

    def format(self, lang):
        funcvar = formatting.get_function()
        if lang == 'latex':
            return {'main': FormatElement(r"\frac{1}{12} \sum_{m \in y(t)} %s(T_m)",
                                          ['T_m', "%s(\cdot)" % (funcvar)]),
                    'T_m': FormatElement("Temperature", is_abstract=True),
                    "%s(\cdot)" % (funcvar): FormatElement(str(self.model))}
        elif lang == 'julia':
            return {'main': FormatElement(r"sum(%s(Tbymonth)) / 12",
                                          ['Tbymonth', "%s(T)" % (funcvar)]),
                    'Tbymonth': FormatElement("# Days of within each bin"),
                    "%s(T)" % (funcvar): FormatElement(str(self.model))}            

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

    @staticmethod
    def describe():
        return dict(input_timerate='month', output_timerate='year',
                    arguments=[arguments.model, arguments.output_unit, arguments.qval.optional(),
                               arguments.input_change.optional()],
                    description="Evaluate a curve in each month, and take the yearly average.")

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

    def format(self, lang):
        funcvar = formatting.get_function()
        if lang == 'latex':
            return {'main': FormatElement(r"\sum_{d \in y(t)} %s(T_d)" % (funcvar),
                                          ['T_d', "%s(\cdot)" % (funcvar)]),
                    'T_d': FormatElement("Temperature", is_abstract=True),
                    "%s(\cdot)" % (funcvar): FormatElement(str(self.model))}
        elif lang == 'julia':
            return {'main': FormatElement(r"sum(%s(Tbins))",
                                          ['Tbins', "%s(T)" % (funcvar)]),
                    'Tbins': FormatElement("# Temperature in bins"),
                    "%s(T)" % (funcvar): FormatElement(str(self.model))}


    def apply(self, region, *args):
        def generate(region, year, temps, **kw):
            if isinstance(curvegen, CurveGenerator):
                spline = self.spline.get_spline.get_curve(region, year, *args, weather=temps)
            else:
                spline = self.spline

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

        return ApplicationByYear(region, generate)

    def column_info(self):
        description = "The combined result of daily temperatures, organized into bins according to %s." % (str(self.model))
        return [dict(name='response', title='Direct marginal response', description=description)]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='year',
                    arguments=[arguments.model, arguments.output_unit, arguments.qval.optional()],
                    description="Evaluate a binned curve, and sum over the year.")
    
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

    def format(self, lang):
        funcvar = formatting.get_function()
        if lang == 'latex':
            return {'main': FormatElement(r"mean(\{mean_{d \in m(t)} %s(T_d)\})" % (funcvar),
                                          ['T_d', "%s(\cdot)" % (funcvar)]),
                    'T_d': FormatElement("Temperature", is_abstract=True),
                    "%s(\cdot)" % (funcvar): FormatElement(str(self.spline))}
        elif lang == 'julia':
            return {'main': FormatElement("mean([mean(%s(Tbyday[monthday[ii]:monthday[ii+1]-1])) for ii in 1:12])" % (funcvar),
                                          ['Tbyday', "monthday", "%s(T)" % (funcvar)]),
                    'Tbyday': FormatElement("# Temperature by day"),
                    'monthday': FormatElement("cumsum([1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])"),
                    "%s(T)" % (funcvar): FormatElement(str(self.spline))}

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
        description = "The effects of monthly average temperatures, organized into bins according to %s, averaged over months." % (str(self.spline))
        return [dict(name='response', title='Direct marginal response', description=description)]

    @staticmethod
    def describe():
        return dict(input_timerate='day', output_timerate='year',
                    arguments=[arguments.model, arguments.output_unit, arguments.input_change.optional(),
                               arguments.qval.optional()],
                    description="Apply a curve to the average of each month, and average over the year.")

class PercentWithin(Calculation):
    def __init__(self, endpoints):
        super(PercentWithin, self).__init__(['portion'])
        self.endpoints = endpoints

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

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='year',
                    arguments=[arguments.ordered_list],
                    description="Determine the portion of days that fall between pairs of points.")

class YearlyAverageDay(Calculation):
    def __init__(self, units, curvegen, curve_description, weather_change=lambda region, x: x, norecord=False):
        super(YearlyAverageDay, self).__init__([units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description
        self.weather_change = weather_change
        self.norecord = norecord

    def format(self, lang):
        if lang == 'latex':
            result = self.curvegen.format_call(lang, "T_d")
            result.update({'main': FormatElement(r"\frac{1}{365} \sum_{d \in y(t)} %s" % result['main'].repstr,
                                                 ['T_d'] + result['main'].dependencies),
                           'T_d': FormatElement("Temperature", is_abstract=True)})
        elif lang == 'julia':
            result = self.curvegen.format_call(lang, "Tbyday")
            result.update({'main': FormatElement(r"sum(%s) / 365" % result['main'].repstr,
                                                 ['Tbyday'] + result['main'].dependencies),
                           'Tbyday': FormatElement("Daily temperature", is_abstract=True)})
        return result

    def apply(self, region, *args):
        checks = dict(lastyear=-np.inf)

        def generate(region, year, temps, **kw):
            # Ensure that we aren't called with a year twice
            assert year > checks['lastyear'], "Push of %d, but already did %d." % (year, checks['lastyear'])
            checks['lastyear'] = year

            curve = self.curvegen.get_curve(region, year, *args, weather=temps) # Passing in original (not weather-changed) data

            temps2 = self.weather_change(region, temps)
            result = np.nansum(curve(temps2)) / len(temps2)

            if not self.norecord and diagnostic.is_recording():
                if isinstance(temps2, xr.Dataset):
                    for var in temps2._variables:
                        if var not in ['time', 'year']:
                            diagnostic.record(region, year, var, float(np.nansum(temps2._variables[var])) / len(temps2._variables[var]))
                else:
                    diagnostic.record(region, year, 'avgv', float(np.nansum(temps2)) / len(temps2))
                diagnostic.record(region, year, 'zero', float(np.nansum(curve(temps2) == 0)) / len(temps2))

            if not np.isnan(result):
                yield (year, result)

        return ApplicationByYear(region, generate)

    def column_info(self):
        description = "The average result across a year of daily temperatures applied to " + self.curve_description
        return [dict(name='response', title='Direct marginal response', description=description)]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='year',
                    arguments=[arguments.output_unit.rename('units'),
                               arguments.curvegen, arguments.curve_description,
                               arguments.input_change.optional(), arguments.debugging.optional()],
                    description="Apply a curve to values and take the average over each year.")

class YearlySumDay(YearlyAverageDay):
    def __init__(self, units, curvegen, curve_description, weather_change=lambda region, x: x, norecord=False):
        super(YearlySumDay, self).__init__(units, curvegen, curve_description, weather_change=weather_change, norecord=norecord)

    def format(self, lang):
        if lang == 'latex':
            result = self.curvegen.format_call(lang, "T_d")
            result = formatting.build_format(r"\sum_{d \in y(t)} %s", result)
            formatting.build_adddepend(result, 'T_d', FormatElement("Temperature", is_abstract=True))
        elif lang == 'julia':
            result = self.curvegen.format_call(lang, "Tbyday")
            result = formatting.build_format(r"sum(%s)", result)
            formatting.build_adddepend(result, 'Tbyday', FormatElement("Daily temperature", is_abstract=True))
        return result

    def apply(self, region, *args):
        checks = dict(lastyear=-np.inf)

        def generate(region, year, temps, **kw):
            # Ensure that we aren't called with a year twice
            assert year > checks['lastyear'], "Push of %d, but already did %d." % (year, checks['lastyear'])
            checks['lastyear'] = year

            temps2 = self.weather_change(region, temps)

            if self.deltamethod:
                terms = self.curvegen.get_lincom_terms(region, year, temps2.sum(), temps2)
                yield (year, terms)
                return
            
            curve = self.curvegen.get_curve(region, year, *args, weather=temps) # Passing in original (not weather-changed) data
            result = np.nansum(curve(temps2))

            if not self.norecord and diagnostic.is_recording():
                if isinstance(temps2, xr.Dataset):
                    for var in temps2._variables:
                        if var not in ['time', 'year']:
                            diagnostic.record(region, year, var, float(np.nansum(temps2._variables[var])))
                else:
                    diagnostic.record(region, year, 'avgv', float(np.nansum(temps2)))

            if not np.isnan(result):
                yield (year, result)

        return ApplicationByYear(region, generate)

    def column_info(self):
        description = "The summed result across a year of daily temperatures applied to " + self.curve_description
        return [dict(name='response', title='Direct marginal response', description=description)]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='year',
                    arguments=[arguments.output_unit.rename('units'),
                               arguments.curvegen, arguments.curve_description,
                               arguments.input_change.optional(), arguments.debugging.optional()],
                    description="Apply a curve to values and take the sum over each year.")

class YearlyDividedPolynomialAverageDay(Calculation):
    def __init__(self, units, curvegen, curve_description, weather_change=lambda x: x):
        super(YearlyDividedPolynomialAverageDay, self).__init__([units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description
        self.weather_change = weather_change

    def apply(self, region, *args):
        def generate(region, year, temps, **kw):
            temps = self.weather_change(temps)
            curve = self.curvegen.get_curve(region, year, *args, weather=temps) # Passing in weather-changed data

            assert temps.shape[1] == len(curve.curr_curve.ccs), "%d <> %d" % (temps.shape[1], len(curve.curr_curve.ccs))

            #result = np.nansum(np.dot(temps, curve.curr_curve.ccs)) / len(temps)
            result = np.dot(np.sum(temps, axis=0), curve.curr_curve.ccs) / len(temps)

            if diagnostic.is_recording():
                sumtemps = np.sum(temps, axis=0) / len(temps)
                for ii in range(temps.shape[1]):
                    diagnostic.record(region, year, 'avgtk_' + str(ii+1), sumtemps[ii])

            if not np.isnan(result):
                yield (year, result)

        return ApplicationByYear(region, generate)

    def column_info(self):
        description = "The average result across a year of daily temperatures applied to a polynomial."
        return [dict(name='response', title='Direct marginal response', description=description)]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='year',
                    arguments=[arguments.output_unit, arguments.curvgen, arguments.curve_description,
                               arugments.input_change],
                    description="Apply a curve to values and take the average over each year.")

class ApplyCurve(Calculation):
    def __init__(self, curvegen, unitses, names, titles, descriptions):
        super(ApplyCurve, self).__init__(unitses)
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen

        self.names = names
        self.titles = titles
        self.descriptions = descriptions

    def apply(self, region, *args):
        def generate(region, year, temps, **kw):
            curve = self.curvegen.get_curve(region, year, *args, weather=temps)

            yield [year] + curve(temps)

        return ApplicationByYear(region, generate)

    def column_info(self):
        return [{'name': self.names[ii], 'title': self.titles[ii],
                 'description': self.descriptions[ii]} for ii in range(len(self.names))]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='year',
                    arguments=[arguments.curvegen, arguments.output_unitss, arguments.column_names,
                               arguments.column_titles, arguments.column_descriptions],
                    description="Apply a curve to values.")
