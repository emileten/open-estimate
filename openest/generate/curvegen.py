from calculation import Calculation
from ..models.curve import AdaptableCurve

## Top-level class
class CurveGenerator(object):
    def __init__(self, indepunits, depenunit):
        self.indepunits = indepunits
        self.depenunit = depenunit

    def get_curve(self, *args, **kw):
        """Returns an object of type Curve."""
        raise NotImplementedError()

class ConstantCurveGenerator(CurveGenerator):
    def __init__(self, indepunits, depenunit, curve):
        super(ConstantCurveGenerator, self).__init__(indepunits, depenunit)
        self.curve = curve

    def get_curve(self, region, *args):
        return self.curve

## Labor-style recursive curve
class RecursiveInstantaneousCurve(AdaptableCurve):
    def __init__(self, region, curvegen, curr_curve):
        self.region = region
        self.curvegen = curvegen
        self.curr_curve = curr_curve

    def update(self, year, weather):
        self.curr_curve = self.curvegen.get_updated_curve(self.region, year, weather)

    def __call__(self, x):
        return self.curr_curve(x)

class RecursiveInstantaneousCurveGenerator(CurveGenerator):
    def __init__(self, indepunits, depenunits, predgen, curvegenfunc):
        super(RecursiveInstantaneousCurveGenerator, self).__init__(indepunits, depenunits)
        self.predgen = predgen
        self.curvegenfunc = curvegenfunc

    def get_updated_curve(self, region, year, weather):
        predictors = self.predgen.get_update(region, year, weather)
        return self.get_curve(region, predictors)

    def get_curve(self, region, predictors={}):
        if not predictors:
            predictors = self.predgen.get_baseline(region)
        return RecursiveInstantaneousCurve(region, self, self.curvegenfunc(predictors))
