import numpy as np
from calculation import Calculation, ApplicationEach
from curvegen import CurveGenerator
import diagnostic

class YearlyBins(Calculation):
    def __init__(self, units, curvegen, curve_description, weather_change=lambda x: x):
        super(YearlyBins, self).__init__([units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description
        self.weather_change = weather_change

    def latex(self):
        funcvar = latextools.get_function()
        yield ("Equation", r"\sum_{d \in y(t)} %s(T_d)" % (funcvar), self.unitses[0])
        yield ("T_d", "Temperature", "deg. C")
        yield ("%s(\cdot)" % (funcvar), str(self.curve), self.unitses[0])

    def apply(self, region, *args):
        def generate(region, year, weather, **kw):
            curve = self.curvegen.get_curve(region, year, *args, weather=weather)

            weather2 = self.weather_change(weather)
            if len(weather2) == len(curve.xx):
                yy = curve(curve.xx)
                yy[np.isnan(yy)] = 0
                result = np.sum(weather2.dot(yy))
            else:
                raise RuntimeError("Unknown format for weather: " + str(weather2.shape) + " <> len " + str(curve.xx))

            if not np.isnan(result):
                yield (year, result)

        return ApplicationEach(region, generate)

    def column_info(self):
        description = "The combined result of daily temperatures, organized into bins according to %s." % (str(self.curve_description))
        return [dict(name='response', title='Direct marginal response', description=description)]

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
