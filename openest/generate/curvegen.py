from calculation import Calculation

class CurveGenerator(object):
    def __init__(self, indepunits, depenunits):
        self.indepunits = indepunits
        self.depenunits = depenunits

    def get_curve(self, *args, **kw):
        """Returns an object of type Curve."""
        raise NotImplementedError()
