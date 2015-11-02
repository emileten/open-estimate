import os, csv, random
import numpy as np
import aggregator
from aggregator.lib.acra.iam import effect_bundle, weather
from aggregator.model.meta_model import MetaModel
from aggregator.lib.bayes.model import Model
from aggregator.lib.bayes.spline_model import SplineModel
from aggregator.lib.bayes.memoizable import MemoizedUnivariate
from aggregator.lib.acra.adaptation.adapting_curve import AdaptingCurve
import config

# Generate integral over daily temperature

def make_daily_bymonthdaybins(id, func=lambda x: x, pval=None, weather_change=lambda temps: temps - 273.15):
    if isinstance(id, AdaptingCurve):
        spline = id
    else:
        if isinstance(id, Model):
            model = id
        else:
            model = MetaModel.get_model(id, False, True)

        model = MemoizedUnivariate(model)
        model.set_x_cache_decimals(1)
        spline = model.get_eval_pval_spline(pval, (-40, 80), threshold=1e-2, linextrap=config.linear_extrapolation)

    def generate(fips, yyyyddd, temps, **kw):
        if fips == effect_bundle.FIPS_COMPLETE:
            return

        if isinstance(spline, AdaptingCurve):
            spline.setup(yyyyddd, temps)

        for (year, temps) in weather.yearly_daily_ncdf(yyyyddd, temps):
            temps = weather_change(temps)
            results = spline(temps)

            result = np.sum(results) / 12

            if not np.isnan(result):
                yield (year, func(result))

            if isinstance(spline, AdaptingCurve):
                spline.update()

    return generate

def make_daily_yearlydaybins(id, func=lambda x: x, pval=None):
    if isinstance(id, AdaptingCurve):
        spline = id
    else:
        if isinstance(id, Model):
            model = id
        else:
            model = MetaModel.get_model(id, False, True)

        model = MemoizedUnivariate(model)
        model.set_x_cache_decimals(1)
        spline = model.get_eval_pval_spline(pval, (-40, 80), threshold=1e-2, linextrap=config.linear_extrapolation)

    def generate(fips, yyyyddd, temps, **kw):
        if fips == effect_bundle.FIPS_COMPLETE:
            return

        if isinstance(spline, AdaptingCurve):
            spline.setup(yyyyddd, temps)

        for (year, temps) in weather.yearly_daily_ncdf(yyyyddd, temps):
            result = np.sum(spline(temps - 273.15))

            if not np.isnan(result):
                yield (year, func(result))

            if isinstance(spline, AdaptingCurve):
                spline.update()

    return generate

def make_daily_averagemonth(id, func=lambda x: x, pval=None):
    days_bymonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    transitions = np.cumsum(days_bymonth)

    if isinstance(id, AdaptingCurve):
        spline = id
    else:
        if isinstance(id, Model):
            model = id
        else:
            model = MetaModel.get_model(id, False, True)

        model = MemoizedUnivariate(model)
        model.set_x_cache_decimals(1)
        spline = model.get_eval_pval_spline(pval, (-40, 80), threshold=1e-2, linextrap=config.linear_extrapolation)

    def generate(fips, yyyyddd, temps, **kw):
        if fips == effect_bundle.FIPS_COMPLETE:
            return

        if isinstance(spline, AdaptingCurve):
            spline.setup(yyyyddd, temps)

        for (year, temps) in weather.yearly_daily_ncdf(yyyyddd, temps):
            bymonth = []
            for mm in range(12):
                avgmonth = np.mean(temps[transitions[mm]-days_bymonth[mm]:transitions[mm]])
                bymonth.append(spline(avgmonth - 273.15))
                #bymonth.append(model.eval_pval(avgmonth - 273.15, pval, threshold=1e-2))

            result = np.mean(bymonth)
            if not np.isnan(result):
                yield (year, func(result))

            if isinstance(spline, AdaptingCurve):
                spline.update()

    return generate

def make_daily_percentwithin(endpoints):
    def generate(fips, yyyyddd, temps, **kw):
        if fips == effect_bundle.FIPS_COMPLETE:
            return

        for (year, temps) in weather.yearly_daily_ncdf(yyyyddd, temps):
            results = []
            for ii in range(len(endpoints)-1):
                result = np.sum(temps - 273.15 > endpoints[ii]) - np.sum(temps - 273.15 > endpoints[ii+1])
                results.append(result)

            results = list(np.array(results) / float(len(temps)))

            yield tuple([year] + results)

    return generate

# Combine counties to states

def aggregate_tar_with_scale_file(name, scale_files, scale_factors):
    scales = {}
    for ii in range(len(scale_files)):
        generator = weather.read_scale_file(aggregator.__path__[0] + "/lib/acra/iam/cropdata/" + scale_files[ii] + ".csv", scale_factors[ii])
        for (fips, scale) in generator:
            if fips in scales:
                scales[fips] += scale
            else:
                scales[fips] = scale

    effect_bundle.aggregate_tar(name, scales)
