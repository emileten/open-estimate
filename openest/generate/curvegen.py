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

class TransformCurveGenerator(CurveGenerator):
    def __init__(self, curvegen, transform):
        super(TransformCurveGenerator, self).__init__(curvegen.indepunits, curvegen.depenunit)
        self.curvegen = curvegen
        self.transform = transform

    def get_curve(self, region, *args):
        return self.transform(region, self.curvegen.get_curve(region, *args))

class WeatherDelayedCurveGenerator(CurveGenerator):
    def get_curve(self, region, *args, **kwargs):
        if self.curr_curve is None:
            # Calculate no-weather before update covariates by calling with weather
            weather = kwargs['weather']
            del kwargs['weather']
            curve = self.curvegen.get_curve(region, *args, **kwargs)
            kwargs['weather'] = weather
        else:
            curve = self.last_curve

        self.last_curve = self.curvegen.get_curve(region, *args, **kwargs)
        return curve

    def get_baseline_curve(self, region, *args, **kwargs):
        raise NotImplementedError()

    def get_next_curve(self, region, *args, **kwargs):
        raise NotImplementedError()
