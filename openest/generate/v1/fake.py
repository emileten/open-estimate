import os, csv, random
import numpy as np
from scipy.interpolate import UnivariateSpline
import effect_bundle, weather
from ..models.memoizable import MemoizedUnivariate

def make_generator_linear(zero, baseline, minslope, maxslope):
    slope = minslope + (maxslope - minslope) * random.random()

    def generate_linear(fips, yyyyddd, values):
        yearly = weather.yearly_daily_ncdf(yyyyddd, values)
        for (year, daily) in yearly:
            result = baseline + np.sum((np.array(daily) - zero) * slope)
            if not np.isnan(result):
                yield year, result

    return generate_linear

def make_generator_bilinear(zero, baseline, minslope_neg, maxslope_neg, minslope_pos, maxslope_pos):
    pval = random.random()
    slope_neg = minslope_neg + (maxslope_neg - minslope_neg) * pval
    slope_pos = minslope_pos + (maxslope_pos - minslope_pos) * pval

    def generate_bilinear(fips, yyyyddd, values):
        yearly = weather.yearly_daily_ncdf(yyyyddd, values)
        for (year, daily) in yearly:
            relative = np.array(daily) - zero
            relative[relative > 0] = relative[relative > 0] * slope_pos
            relative[relative < 0] = relative[relative < 0] * slope_neg
            result = baseline + np.sum(relative)
            if not np.isnan(result):
                yield year, result

    return generate_bilinear

def make_print_bymonthdaybins(model, func=lambda x: x, weather_change=lambda temps: temps - 273.15, limits=(-40, 100), pval=.5):
    model = MemoizedUnivariate(model)
    model.set_x_cache_decimals(1)
    spline_orig = model.get_eval_pval_spline(pval, (-40, 80), threshold=1e-2)

    xx = model.get_xx()
    yy = list(range(len(xx)))

    xxlm = np.concatenate(([limits[0]], xx, [limits[1]]))
    yylm = np.concatenate(([yy[0]], yy, [yy[-1]]))
    spline = UnivariateSpline(xxlm, yylm, s=0, k=1)

    def generate(fips, yyyyddd, temps, **kw):
        for (year, temps) in weather.yearly_daily_ncdf(yyyyddd, temps):
            temps = weather_change(temps)

            results_orig = spline_orig(temps)

            results = spline(temps)
            results = np.around(results)

            bybin = []
            avgval = []
            for ii in range(len(xx)):
                which = results == ii
                bybin.append(sum(which))
                avgval.append(np.mean(results_orig[which]))

            print(bybin)
            print(avgval)

            yield year, func(np.sum(results_orig) / 12)

    return generate
