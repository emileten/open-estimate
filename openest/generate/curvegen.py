from calculation import Calculation

class CurveGenerator(object):
    def __init__(self, indepunits, depenunits):
        self.indepunits = indepunits
        self.depenunits = depenunits

    def get_curve(self, *args, **kw):
        """Returns an object of type Curve."""
        raise NotImplementedError()

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
