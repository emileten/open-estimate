import numpy as np
from calculation import Calculation, ApplicationEach
from curvegen import CurveGenerator
import formatting, arguments, diagnostic
from formatting import FormatElement

class YearlyBins(Calculation):
    def __init__(self, units, curvegen, curve_description):
        super(YearlyBins, self).__init__([units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description

    def format(self, lang):
        funcvar = formatting.get_function()
        if lang == 'latex':
            return {'main': FormatElement(r"\sum_{d \in y(t)} %s(T_d)" % (funcvar),
                                          self.unitses[0], ['T_d', "%s(\cdot)" % (funcvar)]),
                    'T_d': FormatElement("Temperature", "days"),
                    "%s(\cdot)" % (funcvar): FormatElement(str(self.model), self.unitses[0])}
        elif lang == 'julia':
            return {'main': FormatElement(r"sum(%s(Tbins))" % (funcvar),
                                          self.unitses[0], ['Tbins', "%s(T)" % (funcvar)]),
                    'Tbins': FormatElement("# Temperature in bins", "days"),
                    "%s(T)" % (funcvar): FormatElement(str(self.model), self.unitses[0])}            

    def apply(self, region, *args):
        def generate(region, year, temps, **kw):
            curve = self.curvegen.get_curve(region, year, *args, weather=temps)

            if len(temps) == len(curve.xx):
                yy = curve(curve.xx)
                yy[np.isnan(yy)] = 0
                result = np.sum(temps.dot(yy))
            else:
                raise RuntimeError("Unknown format for temps: " + str(temps.shape) + " <> len " + str(curve.xx))

            if not np.isnan(result):
                yield (year, result)

        return ApplicationEach(region, generate)

    def column_info(self):
        description = "The combined result of daily temperatures, organized into bins according to %s." % (str(self.curve_description))
        return [dict(name='response', title='Direct marginal response', description=description)]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.output_unit, arguments.curvegen, arguments.curve_description],
                    description="Apply the results of a previous calculation to a curve.")

class YearlyCoefficients(Calculation):
    def __init__(self, units, curvegen, curve_description, getter=lambda curve: curve.yy, weather_change=lambda region, x: x):
        super(YearlyCoefficients, self).__init__([units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description
        self.getter = getter
        self.weather_change = weather_change

    def apply(self, region, *args):
        def generate(region, year, temps, **kw):
            curve = self.curvegen.get_curve(region, year, *args, weather=temps) # Passing in original (not weather-changed) data

            coeffs = self.getter(region, year, temps, curve)
            if len(temps) == len(coeffs):
                result = np.sum(self.weather_change(region, temps).dot(coeffs))
            else:
                raise RuntimeError("Unknown format for temps: " + str(temps.shape) + " <> len " + str(coeffs))

            if diagnostic.is_recording():
                for ii in range(temps.shape[0]):
                    diagnostic.record(region, year, 'var-' + str(ii), temps[ii])

            if not np.isnan(result):
                yield (year, result)

        return ApplicationEach(region, generate)

    def column_info(self):
        description = "The combined result of yearly values, with coefficients from %s." % (str(self.curve_description))
        return [dict(name='response', title='Direct marginal response', description=description)]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.output_unit, arguments.curvegen, arguments.curve_description,
                               arguments.parameter_getter, arguments.input_change],
                    description="Apply the results of a previous calculation to a curve, as a dot product on that curve's coefficients.")
