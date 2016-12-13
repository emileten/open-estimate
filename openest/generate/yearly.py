import numpy as np
from calculation import Calculation, ApplicationEach
from ..models.curve import AdaptableCurve

class YearlyBins(Calculation):
    def __init__(self, curve, units):
        super(YearlyBins, self).__init__([units])
        self.curve = curve # Instance of UnivariateCurve
        self.xx = curve.get_xx()

    def latex(self):
        funcvar = latextools.get_function()
        yield ("Equation", r"\sum_{d \in y(t)} %s(T_d)" % (funcvar), self.unitses[0])
        yield ("T_d", "Temperature", "deg. C")
        yield ("%s(\cdot)" % (funcvar), str(self.curve), self.unitses[0])

    def apply(self, region, *args):
        if isinstance(self.curve, AdaptableCurve):
            curve = self.curve.create(region, *args)
        else:
            curve = self.curve

        def generate(region, year, temps, **kw):
            if len(temps.shape) == 2:
                if temps.shape[0] == 12 and temps.shape[1] == len(self.xx):
                    yy = curve(self.xx)
                    yy[np.isnan(yy)] = 0
                    result = np.sum(temps.dot(yy))
                else:
                    raise RuntimeError("Unknown format for temps: " + str(temps.shape[0]) + " x " + str(temps.shape[1]) + " <> len " + str(self.xx))
            else:
                result = np.nansum(curve(temps))

            if not np.isnan(result):
                yield (year, result)

            if isinstance(self.curve, AdaptableCurve):
                curve.update(year, temps)

        return ApplicationEach(region, generate)

    def column_info(self):
        description = "The combined result of daily temperatures, organized into bins according to %s." % (str(self.curve))
        return [dict(name='response', title='Direct marginal response', description=description)]
