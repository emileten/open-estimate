import traceback
from calculation import Calculation
import latextools, juliatools

## Top-level class
class CurveGenerator(object):
    def __init__(self, indepunits, depenunit):
        self.indepunits = indepunits
        self.depenunit = depenunit

    def get_curve(self, region, year, *args, **kw):
        """Returns an object of type Curve."""
        raise NotImplementedError()

    def format_call(self, lang, *args):
        raise NotImplementedError()        

class ConstantCurveGenerator(CurveGenerator):
    def __init__(self, indepunits, depenunit, curve):
        super(ConstantCurveGenerator, self).__init__(indepunits, depenunit)
        self.curve = curve

    def get_curve(self, region, year, *args, **kw):
        return self.curve

    def format_call(self, lang, *args):
        result = self.curve.format_call(lang, args[0])
        result['main'].unit = self.depenunit
        return result

class TransformCurveGenerator(CurveGenerator):
    def __init__(self, transform, description, *curvegens):
        super(TransformCurveGenerator, self).__init__(curvegens[0].indepunits, curvegens[0].depenunit)
        self.curvegens = curvegens
        self.description = description
        self.transform = transform

    def get_curve(self, region, year, *args, **kw):
        try:
            return self.transform(region, *tuple([curvegen.get_curve(region, year, *args, **kw) for curvegen in self.curvegens]))
        except Exception as ex:
            print self.curvegens
            print args, kw
            traceback.print_exc()
            raise ex

    def format_call(self, lang, *args):
        try:
            result = {}
            curveargs = []
            for curvegen in self.curvegens:
                equation = curvegen.format_call(lang, *args)
                result.update(equation)
                curveargs.append(equation['main'])

            if lang == 'latex':
                result.update(latextools.call(self.transform, self.depenunit, self.description, *tuple(curveargs)))
            elif lang == 'julia':
                result.update(juliatools.call(self.transform, self.depenunit, self.description, *tuple(curveargs)))
            
            return result
        except Exception as ex:
            print self.curvegens
            raise ex

class DelayedCurveGenerator(CurveGenerator):
    def __init__(self, curvegen):
        super(DelayedCurveGenerator, self).__init__(curvegen.indepunits, curvegen.depenunit)
        self.curvegen = curvegen
        self.last_curves = {}
        self.last_years = {}
        
    def get_curve(self, region, year, *args, **kwargs):
        if self.last_years.get(region, None) == year:
            return self.last_curves[region]
        
        if region not in self.last_curves:
            # Calculate no-weather before update covariates by calling with weather
            weather = kwargs['weather']
            del kwargs['weather']
            curve = self.get_next_curve(region, year, *args, **kwargs)
            kwargs['weather'] = weather
        else:
            curve = self.last_curves[region]

        self.last_curves[region] = self.get_next_curve(region, year, *args, **kwargs)
        self.last_years[region] = year
        return curve

    def get_next_curve(self, region, year, *args, **kwargs):
        return self.curvegen.get_curve(region, year, *args, **kwargs)

    def format_call(self, lang, *args):
        try:
            result = self.curvegen.format_call(lang, *tuple(map(lambda x: x + '[t-1]', args)))
            result['main'].unit = self.depenunit
            return result
        except Exception as ex:
            print self.curvegen
            raise ex
