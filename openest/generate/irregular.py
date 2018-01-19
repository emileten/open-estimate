import numpy as np
import xarray as xr
from calculation import Calculation, ApplicationByIrregular
import formatting, arguments, diagnostic
from formatting import FormatElement
from curvegen import CurveGenerator

class YearlySumIrregular(Calculation):
    def __init__(self, units, curvegen, curve_description):
        super(YearlySumIrregular, self).__init__([units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description

    def format(self, lang):
        if lang == 'latex':
            result = self.curvegen.format_call(lang, "X_t")
            result.update({'main': FormatElement(r"\sum_{s \in y(t)} %s" % result['main'].repstr,
                                                 self.unitses[0], ['X_t'] + result['main'].dependencies),
                           'X_d': FormatElement("Weather", "unknown", is_abstract=True)})
        elif lang == 'julia':
            result = self.curvegen.format_call(lang, "Xbyt")
            result.update({'main': FormatElement(r"sum(%s)" % result['main'].repstr,
                                                 self.unitses[0], ['Xbyt'] + result['main'].dependencies),
                           'Xbyt': FormatElement("Weather", "unknown", is_abstract=True)})
        return result

    def apply(self, region, *args):
        checks = dict(lastyear=-np.inf)

        def generate(region, year, weather, **kw):
            # Ensure that we aren't called with a year twice
            assert year > checks['lastyear'], "Push of %d, but already did %d." % (year, checks['lastyear'])
            checks['lastyear'] = year

            curve = self.curvegen.get_curve(region, year, *args, weather=weather) # Passing in original (not weather-changed) data

            result = np.nansum(curve(weather))

            if isinstance(weather, xr.Dataset):
                for var in weather._variables:
                    if var not in ['time', 'year'] and var not in weather.coords:
                        diagnostic.record(region, year, var, float(np.nansum(weather._variables[var]._data)))
            else:
                diagnostic.record(region, year, 'avgv', float(np.nansum(weather)))
                        
            if not np.isnan(result):
                yield (year, result)

        return ApplicationByIrregular(region, generate)

    def column_info(self):
        description = "The total result across a year of weather applied to " + self.curve_description
        return [dict(name='response', title='Direct marginal response', description=description)]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='year',
                    arguments=[arguments.output_unit.rename('units'),
                               arguments.curvegen, arguments.curve_description],
                    description="Apply a curve to values and take the sum over each year.")
