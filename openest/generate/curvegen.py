from calculation import Calculation

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
    def __init__(self, region, predgen, curr_curve):
        self.region = region
        self.predgen = predgen
        self.curr_curve = curr_curve

    def update(self, year, weather):
        self.predictors = self.predgen.get_update(self.region, year, weather)
        self.curr_curve = self.curvegen.get_curve(self.region, *self.predictors)

    def __call__(self, x):
        return self.curr_curve(x)

class RecursiveInstantaneousCurveGenerator(CurveGenerator):
    def __init__(self, indepunits, depenunits, curvegenfunc):
        super(RecursiveInstantaneousCurveGenerator, self).__init__(indepunits, depenunits)
        self.curvegenfunc = curvegenfunc

    def get_curve(self, region, *predictors):
        return RecursiveInstantaneousCurve(region, self, self.curvegenfunc(predictors))
