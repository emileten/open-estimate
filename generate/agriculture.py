import os, csv, random
import numpy as np
import aggregator
from aggregator.lib.acra.iam import effect_bundle, weather
from aggregator.model.meta_model import MetaModel
from aggregator.lib.bayes.model import Model
from aggregator.lib.bayes.integral_model import IntegralModel
from aggregator.lib.bayes.spline_model import SplineModel
from aggregator.lib.bayes.memoizable import MemoizedUnivariate
from aggregator.lib.acra.adaptation.adapting_curve import SimpleAdaptingCurve

# Seasonal Temperature Impacts

def make_generator_single_crop(crop, id, pval):
    calendar = weather.get_crop_calendar(aggregator.__path__[0] + "/lib/acra/iam/cropdata/" + crop + ".csv")

    if isinstance(id, Model):
        model = id
    else:
        meta = MetaModel.get(id, False)
        model = meta.model()
        model = MemoizedUnivariate(model)
        model.set_x_cache_decimals(1)

    def generate(fips, yyyyddd, dailys, lat=None, lon=None):
        if fips not in calendar:
            return

        if isinstance(model, SimpleAdaptingCurve):
            model.setup(yyyyddd, dailys['tas'])
            
        seasons = weather.growing_seasons_mean_ncdf(yyyyddd, dailys['tas'], calendar[fips][0], calendar[fips][1])
        for (year, temp) in seasons:
            result = model.eval_pval(temp - 273.15, pval, 1e-2)
            if not np.isnan(result):
                yield (year, result)

            if isinstance(model, SimpleAdaptingCurve):
                model.update()

    return generate

def make_generator_combo_crops(generators, scale_files, scale_factors, dont_divide=False):
    scales = []
    for ii in range(len(generators)):
        generator = weather.read_scale_file(aggregator.__path__[0] + "/lib/acra/iam/cropdata/" + scale_files[ii] + ".csv", scale_factors[ii])
        scales.append({fips: scale for (fips, scale) in generator})

    def generate(fips, yyyyddd, dailys, **kw):
        if fips == effect_bundle.FIPS_COMPLETE:
            print "completing combo"
            for generator in generators:
                try:
                    generator(fips, yyyyddd, dailys).next()
                except:
                    pass
            return

        lat = kw['lat']
        lon = kw['lon']
        singles = [generator(fips, yyyyddd, dailys, lat, lon) for generator in generators]

        currvals = [(-1, None) for ii in range(len(generators))]
        minyear = -1
        while True:
            # Go next for all with the lowest year
            for ii in range(len(generators)):
                if currvals[ii][0] == minyear:
                    try:
                        currvals[ii] = singles[ii].next()
                    except StopIteration:
                        currvals[ii] = (float('inf'), None)

            # Find the new lowest year
            minyear = float('inf')
            for ii in range(len(generators)):
                minyear = min(minyear, currvals[ii][0])

            if minyear == float('inf'):
                raise StopIteration()

            # Generate a result with all from this lowest year
            numer = 0
            denom = 0
            for ii in range(len(generators)):
                if currvals[ii][0] == minyear:
                    if fips in scales[ii]:
                        numer += currvals[ii][1] * scales[ii][fips]
                        denom += scales[ii][fips]

            if denom > 0:
                if dont_divide:
                    yield (minyear, numer)
                else:
                    yield (minyear, numer / denom)

    return generate

# Generate integral over daily temperature

def make_daily_degreedaybinslog(crop, id_temp, id_precip, scaling, pvals):
    calendar = weather.get_crop_calendar(aggregator.__path__[0] + "/lib/acra/iam/cropdata/" + crop + ".csv")

    if isinstance(id_temp, Model):
        model_temp = id_temp
    else:
        model_temp = MetaModel.get_model(id_temp, False, True)
        model_temp = MemoizedUnivariate(model_temp)
        model_temp.set_x_cache_decimals(1)
        
    model_precip = MetaModel.get_model(id_precip, False, True)
    model_precip = MemoizedUnivariate(model_precip)
    model_precip.set_x_cache_decimals(1)

    def generate(fips, yyyyddd, dailys, *args, **kw):
        if fips not in calendar:
            return

        if isinstance(model_temp, SimpleAdaptingCurve):
            model_temp.setup(yyyyddd, dailys['tas'])

        seasons = weather.growing_seasons_daily_ncdf(yyyyddd, dailys, calendar[fips][0], calendar[fips][1])
        for (year, result) in degreedaybinslog_result(model_temp, model_precip, seasons, pvals, scaling):
            yield (year, result)

    return generate

def make_daily_degreedaybinslog_conditional(crop, ids_temp, ids_precip, conditional, scaling, pvals):
    calendar = weather.get_crop_calendar(aggregator.__path__[0] + "/lib/acra/iam/cropdata/" + crop + ".csv")

    models_temp = []
    for ii in range(len(ids_temp)):
        if isinstance(ids_temp[ii], Model):
            models_temp.append(ids_temp[ii])
        else:
            models_temp.append(MemoizedUnivariate(MetaModel.get_model(ids_temp[ii], False, True)))
            models_temp[ii].set_x_cache_decimals(1)

    models_precip = map(lambda id: MemoizedUnivariate(MetaModel.get_model(id, False, True)), ids_precip)
    for model in models_precip:
        model.set_x_cache_decimals(1)

    def generate(fips, yyyyddd, dailys, lat=None, lon=None):
        if fips not in calendar:
            return

        condition = conditional(fips, lat, lon)
        model_temp = models_temp[condition]
        model_precip = models_precip[condition]

        if isinstance(model_temp, SimpleAdaptingCurve):
            model_temp.setup(yyyyddd, dailys['tas'])

        seasons = weather.growing_seasons_daily_ncdf(yyyyddd, dailys, calendar[fips][0], calendar[fips][1])
        for (year, result) in degreedaybinslog_result(model_temp, model_precip, seasons, [pvals[condition], pvals[condition + 2]], scaling):
            yield (year, result)

    return generate

def degreedaybinslog_result(model_temp, model_precip, seasons, pvals, scaling=1):
    assert(isinstance(model_temp, SimpleAdaptingCurve) or len(model_temp.xx) == 3)

    xxs = np.array(model_temp.xx) if not isinstance(model_temp, SimpleAdaptingCurve) else np.array([10, 29, 50]) # XXX: for maize
    midpoints = (xxs[0:len(xxs)-1] + xxs[1:len(xxs)]) / 2
    multiple = np.array(model_temp.eval_pvals(midpoints, pvals[0], 1e-2))

    for (year, weather) in seasons:
        tasmin = weather['tasmin'] - 273.15
        tasmax = weather['tasmax'] - 273.15
        dd_lowup = above_threshold(tasmin, tasmax, xxs[0])
        dd_above = above_threshold(tasmin, tasmax, xxs[1])
        dd_lower = dd_lowup - dd_above
        
        result = (multiple[0] * dd_lower + multiple[1] * dd_above) * scaling

        prpos = weather['pr']
        prpos = prpos * (prpos > 0)
        precip = sum(prpos) / 1000.0
        result += model_precip.eval_pval(precip, pvals[1], 1e-2)

        if not np.isnan(result):
            yield (year, np.exp(result))

        if isinstance(model_temp, SimpleAdaptingCurve):
            model_temp.update()
            multiple = np.array(model_temp.eval_pvals(midpoints, pvals[0], 1e-2))

def above_threshold(mins, maxs, threshold):
    # Determine crossing points
    aboves = mins > threshold
    belows = maxs < threshold
    plus_over_2 = (mins + maxs)/2
    minus_over_2 = (maxs - mins)/2
    two_pi = 2*np.pi
    d0s = np.arcsin((threshold - plus_over_2) / minus_over_2) / two_pi
    d1s = .5 - d0s

    d0s[aboves] = 0
    d1s[aboves] = 1
    d0s[belows] = 0
    d1s[belows] = 0

    # Integral
    F1s = -minus_over_2 * np.cos(2*np.pi*d1s) / two_pi + plus_over_2 * d1s
    F0s = -minus_over_2 * np.cos(2*np.pi*d0s) / two_pi + plus_over_2 * d0s
    return np.sum(F1s - F0s - threshold * (d1s - d0s))

def make_modelscale_byyear(make_generator, id, pval, func=lambda x, y, year: x*y):
    if isinstance(id, Model):
        model = id
    else:
        model = MetaModel.get_model(id, False, True)
    factor = model.eval_pval(None, pval, 1e-2)

    def generate(fips, yyyyddd, temps, lat=None, lon=None):
        generator = make_generator(fips, yyyyddd, temps, lat, lon)
        for (year, result) in generator:
            yield (year, func(result, factor, year))

    return generate

# Combine counties to states

def aggregate_tar_with_scale_file(name, scale_files, scale_factors, targetdir=None, get_region=None, collabel="fraction", return_it=False):
    scales = {}
    for ii in range(len(scale_files)):
        generator = weather.read_scale_file(aggregator.__path__[0] + "/lib/acra/iam/cropdata/" + scale_files[ii] + ".csv", scale_factors[ii])
        for (fips, scale) in generator:
            if fips in scales:
                scales[fips] += scale
            else:
                scales[fips] = scale

    if return_it:
        return scales

    effect_bundle.aggregate_tar(name, scales, targetdir, collabel=collabel, get_region=get_region, report_all=True)
