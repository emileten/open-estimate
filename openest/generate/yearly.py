import numpy as np
from calculation import Calculation, ApplicationEach
from ..models.curve import AdaptableCurve
from curvegen import CurveGenerator

class YearlyBins(Calculation):
    def __init__(self, units, curvegen, curve_description):
        super(YearlyBins, self).__init__([units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description

    def latex(self):
        funcvar = latextools.get_function()
        yield ("Equation", r"\sum_{d \in y(t)} %s(T_d)" % (funcvar), self.unitses[0])
        yield ("T_d", "Temperature", "deg. C")
        yield ("%s(\cdot)" % (funcvar), str(self.curve), self.unitses[0])

    def apply(self, region, *args):
        curve = self.curvegen.get_curve(region, *args)

        def generate(region, year, temps, **kw):
            if len(temps) == len(curve.xx):
                yy = curve(curve.xx)
                yy[np.isnan(yy)] = 0
                result = np.sum(temps.dot(yy))
            else:
                raise RuntimeError("Unknown format for temps: " + str(temps.shape) + " <> len " + str(curve.xx))

            if not np.isnan(result):
                yield (year, result)

            if isinstance(curve, AdaptableCurve):
                curve.update(year, temps)

        return ApplicationEach(region, generate)

    def column_info(self):
        description = "The combined result of daily temperatures, organized into bins according to %s." % (str(self.curve_description))
        return [dict(name='response', title='Direct marginal response', description=description)]
