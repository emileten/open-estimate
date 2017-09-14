import numpy as np
from curvegen import CurveGenerator
from calculation import Calculation, ApplicationByYear

class LincomDay2Year(Calculation):
    def __init__(self, units, curvegen, curve_description):
        super(LincomDay2Year, self).__init__([units, units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description

    def apply(self, region, *args):
        checks = dict(lastyear=-np.inf)

        def generate(region, year, temps, **kw):
            # Ensure that we aren't called with a year twice
            assert year > checks['lastyear']
            checks['lastyear'] = year

            terms = self.curvegen.get_lincom_terms(region, year, temps.sum())

            result = np.dot(terms, self.curvegen.get_csvv_coeff())

            vcv = self.curvegen.get_csvv_vcv()

            variance = 0
            for ii in range(len(terms)):
                for jj in range(len(terms)):
                    variance += vcv[ii, jj] * terms[ii] * terms[jj]
            
            if not np.isnan(result):
                yield (year, result, np.sqrt(variance))

        return ApplicationByYear(region, generate)

    def column_info(self):
        description = "The cummulative result across a year of daily temperatures applied to " + self.curve_description
        return [dict(name='response', title='Direct marginal response', description=description),
                dict(name='stddev', title='Linear combination std. err.', description="Standard deviation of `response`.")]
