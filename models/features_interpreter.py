# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, The Open Aggregator
#   GNU General Public License, Ver. 3 (see docs/license.txt)
################################################################################

"""Probability Features File

The probability features file has the following format::

  dpc1,<p-header-1>,<p-header-2>,...
  <x-value-1>,g_1(y | x_1),g_2(y | x_1),...
  <x-value-2>,g_1(y | x_2),g_2(y | x_2),...
  ...

``<p-header>`` headers can be any of the following, with the
corresponding values in their rows (``<p-value-ij>``).

  * ``mean``: E y|x_i
  * ``var``: E (y|x_i - E y|x_i)^2
  * ``sdev``: \sqrt{E (y|x_i - E y|x_i)^2}
  * ``skew``: E ((y|x_i - E y|x_i) / sqrt{E (y|x_i - E y|x_i)^2})^3
  * ``mode``: \max f(y | x_i)
  * numeric (0 - 1): F^{-1}(p_j|x_i)

The row headers (``<x-value>``) can be numeric, in which case a
continuous spline bridges them, or categorical strings.

Below is a sample features file::

  dpc1,mean,var
  treated,0,1
  control,4,4
"""
__copyright__ = "Copyright 2014, The Open Aggregator"
__license__ = "GPL"

__author__ = "James Rising"
__credits__ = ["James Rising", "Solomon Hsiang", "Bob Kopp"]
__maintainer__ = "James Rising"
__email__ = "jar2234@columbia.edu"

__status__ = "Production"
__version__ = "$Revision$"
# $Source$

import csv, math, random, copy
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm, expon
from scipy.optimize import brentq, minimize

from spline_model import SplineModel, SplineModelConditional

class FeaturesInterpreter:
    @staticmethod
    def init_from_feature_file(spline, file, delimiter, limits, status_callback=None):
        reader = csv.reader(file, delimiter=delimiter)
        header = reader.next()
        if header[0] != "dpc1":
            raise ValueError("Unknown format for %s" % (header[0]))

        spline.scaled = True

        while header[-1] == '':
            header = header[:-1]

        # Convert header p-values into numerics
        for ii in range(1, len(header)):
            try:
                val = float(header[ii])
                header[ii] = val
            except ValueError:
                pass        

        spline.xx = []
        spline.xx_text = []
        spline.xx_is_categorical = False
        spline.conditionals = []

        last_row = None
        last_conditional = None
        for row in reader:
            if last_row and last_row[1:] == row[1:]:
                conditional = last_conditional.copy()
            else:
                conditional = FeaturesInterpreter.make_conditional_respecting(header, row, limits)
                last_row = row
                last_conditional = conditional
            
            spline.add_conditional(row[0], conditional)

            if status_callback:
                status_callback("Parsing...", reader.line_num / (reader.line_num + 3.0))

        return spline

    # limits is tuple of (low, high)
    @staticmethod
    def features_to_gaussian(header, row, limits):
        # Does this look like a mean-variance feature file?
        if len(header) == 3:
            mean = None
            if 'mean' in header:
                mean = float(row[header.index('mean')])
            if 'mode' in header:
                mean = float(row[header.index('mode')])
            if .5 in header:
                mean = float(row[header.index(.5)])
            if mean is None:
                return None
            
            if 'var' in header:
                var = float(row[header.index('var')])
            elif 'sdev' in header:
                var = float(row[header.index('sdev')]) * float(row[header.index('sdev')])
            else:
                return None

            if np.isnan(var) or var == 0:
                return SplineModelConditional.make_single(mean, mean, [])

            # This might be uniform
            if mean - 2*var < limits[0] or mean + 2*var > limits[1]:
                return None

            return SplineModelConditional.make_gaussian(limits[0], limits[1], mean, var)
        elif len(header) == 4:
            # Does this look like a mean and evenly spaced p-values?
            header = header[1:] # Make a copy of the list
            row = row[1:]
            mean = None
            if 'mean' in header:
                mean = float(row.pop(header.index('mean')))
                header.remove('mean')
                
            elif 'mode' in header:
                mean = float(row.pop(header.index('mode')))
                header.remove('mode')
            elif .5 in header:
                mean = float(row.pop(header.index(.5)))
                header.remove(.5)
            else:
                return None

            # Check that the two other values are evenly spaced p-values
            row = map(float, row[0:2])
            if np.all(np.isnan(row)):
                return SplineModelConditional.make_single(mean, mean, [])
                
            if header[1] == 1 - header[0] and abs(row[1] - mean - (mean - row[0])) < abs(row[1] - row[0]) / 1000.0:
                lowp = min(header)
                lowv = np.array(row)[np.array(header) == lowp][0]

                if lowv == mean:
                    return SplineModelConditional.make_single(mean, mean, [])

                lowerbound = 1e-4 * (mean - lowv)
                upperbound = np.sqrt((mean - lowv) / lowp)

                sdev = brentq(lambda sdev: norm.cdf(lowv, mean, sdev) - lowp, lowerbound, upperbound)
                if float(limits[0]) < mean - 3*sdev and float(limits[1]) > mean + 3*sdev:
                    return SplineModelConditional.make_gaussian(limits[0], limits[1], mean, sdev*sdev)
                else:
                    return None
            else:
                # Heuristic best curve: known tails, fit to mean
                lowp = min(header)
                lowv = np.array(row)[np.array(header) == lowp][0]

                lowerbound = 1e-4 * (mean - lowv)
                upperbound = np.log((mean - lowv) / lowp)

                low_sdev = brentq(lambda sdev: norm.cdf(lowv, mean, sdev) - lowp, lowerbound, upperbound)
                if float(limits[0]) > mean - 3*low_sdev:
                    return None
                
                low_segment = SplineModelConditional.make_gaussian(float(limits[0]), lowv, mean, low_sdev*low_sdev)

                highp = max(header)
                highv = np.array(row)[np.array(header) == highp][0]

                lowerbound = 1e-4 * (highv - mean)
                upperbound = np.log((highv - mean) / (1 - highp))

                high_scale = brentq(lambda scale: .5 + expon.cdf(highv, mean, scale) / 2 - highp, lowerbound, upperbound)
                if float(limits[1]) < mean + 3*high_scale:
                    return None

                # Construct exponential, starting at mean, with full cdf of .5
                high_segment = SplineModelConditional.make_single(highv, float(limits[1]), [np.log(1/high_scale) + np.log(.5) + mean / high_scale, -1 / high_scale])

                sevenys = np.linspace(lowv, highv, 7)
                ys = np.append(sevenys[0:2], [mean, sevenys[-2], sevenys[-1]])

                lps0 = norm.logpdf(ys[0:2], mean, low_sdev)
                lps1 = expon.logpdf([ys[-2], ys[-1]], mean, high_scale) + np.log(.5)

                #bounds = [norm.logpdf(mean, mean, low_sdev), norm.logpdf(mean, mean, high_sdev)]

                result = minimize(lambda lpmean: FeaturesInterpreter.skew_gaussian_evaluate(ys, np.append(np.append(lps0, [lpmean]), lps1), low_segment, high_segment, mean, lowp, highp), .5, method='Nelder-Mead')
                print np.append(np.append(lps0, result.x), lps1)
                return FeaturesInterpreter.skew_gaussian_construct(ys, np.append(np.append(lps0, result.x), lps1), low_segment, high_segment)

    @staticmethod
    def skew_gaussian_construct(ys, lps, low_segment, high_segment):
        mid_segment = SplineModelConditional.make_conditional_from_spline(InterpolatedUnivariateSpline(ys, lps, k=2), (low_segment.y1s[0], high_segment.y0s[0]))
        conditional = SplineModelConditional()
        conditional.add_segment(low_segment.y0s[0], low_segment.y1s[0], copy.copy(low_segment.coeffs[0]))
        for ii in range(mid_segment.size()):
            conditional.add_segment(mid_segment.y0s[ii], mid_segment.y1s[ii], mid_segment.coeffs[ii])
        conditional.add_segment(high_segment.y0s[0], high_segment.y1s[0], copy.copy(high_segment.coeffs[0]))

        try:
            conditional.rescale()
        except:
            return None

        return conditional

    @staticmethod
    def skew_gaussian_evaluate(ys, lps, low_segment, high_segment, mean, lowp, highp):
        conditional = FeaturesInterpreter.skew_gaussian_construct(ys, lps, low_segment, high_segment)
        if conditional is None:
            return SplineModel.posinf

        error = 0
        # Discontinuities:
        error += np.square(conditional.evaluate(0, low_segment.y1s[0]) - conditional.evaluate(1, low_segment.y1s[0]))
        error += np.square(conditional.evaluate(conditional.size() - 2, high_segment.y0s[0]) - conditional.evaluate(conditional.size() - 1, high_segment.y0s[0]))
        
        # Mean:
        error += np.square(mean - conditional.approximate_mean((low_segment.y0s[0], high_segment.y1s[0])))

        # lps
        error += np.square(conditional.cdf(low_segment.y1s[0]) - lowp)
        error += np.square(conditional.cdf(high_segment.y0s[0]) - highp)

        return error

    @staticmethod
    def features_to_exponential(header, row, limits):
        if len(header) != 2:
            return None

        if 'mean' not in header:
            return None

        mean = float(row[header.index('mean')])

        # Is it one-sided?
        if mean > limits[0] and limits[0] + (mean - limits[0]) * 3 < limits[1]:
            # positive exponential
            return SplineModelConditional.make_single(limits[0], limits[1], [limits[0] / (mean - limits[0]), -1/(mean - limits[0])]).rescale()
        if mean < limits[1] or limits[1] - (limits[1] - mean) * 3 > limits[0]:
            # negative exponential
            return SplineModelConditional.make_single(limits[0], limits[1], [-limits[1] / (limits[1] - mean), 1/(limits[1] - mean)]).rescale()
        else:
            return None

    @staticmethod
    def features_to_uniform(header, row, limits):
        if len(header) != 1:
            return None

        return SplineModelConditional.make_single(limits[0], limits[1], [1/(limits[1] - limits[0])])

    # Only for scaled distributions
    @staticmethod
    def make_conditional_respecting(header, row, limits):
        low = high = None
        if 0 in header:
            low = float(row[header.index(0)])
            header.remove(header.index(0))
        if 1 in header:
            high = float(row[header.index(1)])
            header.remove(header.index(1))

        if low is not None and high is not None:
            model = FeaturesInterpreter.make_conditional(header, row, (low, high))
            model.add_segment(limits[0], low, [SplineModel.neginf])
            model.add_segment(high, limits[1], [SplineModel.neginf])

            return model

        if low is not None:
            model = FeaturesInterpreter.make_conditional(header, row, (low, limits[1]))
            model.add_segment(limits[0], low, [SplineModel.neginf])

            return model
        
        if high is not None:
            model = FeaturesInterpreter.make_conditional(header, row, (limits[0], high))
            model.add_segment(high, limits[1], [SplineModel.neginf])

            return model

        return FeaturesInterpreter.make_conditional(header, row, limits)

    # Only for scaled distributions
    @staticmethod
    def make_conditional(header, row, limits):
        # Look for a special case
        conditional = FeaturesInterpreter.features_to_gaussian(header, row, limits)
        if conditional is not None:
            return conditional

        conditional = FeaturesInterpreter.features_to_exponential(header, row, limits)
        if conditional is not None:
            return conditional

        conditional = FeaturesInterpreter.features_to_uniform(header, row, limits)
        if conditional is not None:
            return conditional

        spline = FeaturesInterpreter.best_spline(header, row, limits)
        conditional = SplineModelConditional.make_conditional_from_spline(spline, limits)
        return conditional.rescale()

    @staticmethod
    def best_knot(knots, newknots):
        """Find the knot furthest from existing knots"""
        if len(knots) == 0:
            return newknots[0]
        
        scores = np.zeros(len(newknots))
        for ii in range(len(newknots)):
            scores[ii] = min(abs(np.array(knots) - newknots[0]))
        return newknots[np.argmax(scores)]

    @staticmethod
    def best_spline(header, row, limits):
        print "Best Spline"
        # Do general solution
        best_ys = []
        best_lps = []
        for ii in range(1, len(header)):
            if isinstance(header[ii], float):
                best_ys.append(float(row[ii]))
                best_lps.append(-6*abs(.5 - header[ii]))
            elif header[ii] == 'mode':
                best_ys.append(float(row[ii]))
                best_lps.append(0)
            elif header[ii] == 'var':
                if 'mean' in header:
                    best_ys.append(FeaturesInterpreter.best_knot(best_ys, [float(row[header.index('mean')]) + math.sqrt(float(row[ii])),
                                                       float(row[header.index('mean')]) - math.sqrt(float(row[ii]))]))
                else:
                    best_ys.append(FeaturesInterpreter.best_knot(best_ys, [math.sqrt(float(row[ii])),
                                                       -math.sqrt(float(row[ii]))]))
                best_lps.append(-1)
            elif header[ii] == 'skew':
                if 'var' in header and 'mean' in header:
                    best_ys.append(FeaturesInterpreter.best_knot(best_ys, [float(row[header.index('mean')]) +
                                                       math.sqrt(float(row[header.index('var')])) * math.pow(float(row[ii]), 1.0/3)]))
                else:
                    best_ys.append(FeaturesInterpreter.best_knot(best_ys, [math.pow(float(row[ii]), 1.0/3)]))
                best_lps.append(-1.5)
            elif header[ii] == 'mean' and 'var' not in header and 'skew' not in header:
                best_ys.append(FeaturesInterpreter.best_knot(best_ys, [float(row[ii])]))
                best_lps.append(0)

        indexes = [elt[0] for elt in sorted(enumerate(best_ys), key=lambda elt: elt[1])]
        best_ys = [best_ys[index] for index in indexes]
        best_lps = [best_lps[index] for index in indexes]

        best_spline = InterpolatedUnivariateSpline(best_ys, best_lps, k=2)
        best_error = FeaturesInterpreter.evaluate_spline(header, row, best_spline, limits)

        # Search space for spline that fits criteria
        print "Searching..."
        for attempt in range(100):
            ys = best_ys + np.random.normal(0, max(best_ys) - min(best_ys), len(best_ys))
            lps = best_lps + np.random.normal(0, max(best_lps) - min(best_lps) + 1, len(best_lps))
            ys = ys.tolist()
            lps = lps.tolist()

            # I use 1 - random() here because random() range is [0, 1).
            if limits[0] == SplineModel.neginf:
                ys.insert(0, ys[0] - 1/(1 - random.random()))
                lps.insert(0, -7)
            elif limits[0] < ys[0]:
                ys.insert(0, limits[0])
                lps.insert(0, -6 - 1/(1 - random.random()))
            
            if limits[1] == SplineModel.posinf:
                ys.insert(0, ys[-1] + 1/(1 - random.random()))
                lps.insert(0, -7)
            elif limits[1] > ys[-1]:
                ys.insert(0, limits[1])
                lps.insert(0, -6 - 1/(1 - random.random()))

            spline = InterpolatedUnivariateSpline(ys, lps, k=2)
            error = FeaturesInterpreter.evaluate_spline(header, row, spline, limits)

            if error < best_error:
                best_spline = spline
                best_error = error

        return best_spline

    @staticmethod
    def evaluate_spline(header, row, spline, limits):
        limits = (max(min(spline.get_knots()), float(limits[0])), min(max(spline.get_knots()), float(limits[1])))

        ys = np.linspace(limits[0], limits[1], len(header) * SplineModel.samples)
        ps = np.exp(spline(ys)) * (limits[1] - limits[0]) / (len(header) * SplineModel.samples)
        ps = ps / sum(ps)
        cfs = np.cumsum(ps)

        if 'mean' in header or 'var' in header or 'skew' in header:
            mean = sum(ps * ys)
        if 'var' in header or 'skew' in header:
            var = sum(ps * np.square(ys - mean))
        
        error = 0
        for ii in range(1, len(header)):
            if isinstance(header[ii], float):
                error = error + np.abs(SplineModelConditional.find_nearest(cfs, header[ii], ys) - float(row[ii]))
            elif header[ii] == 'mean':
                error = error + np.abs(mean - float(row[ii]))
            elif header[ii] == 'mode':
                mode = ys[ps.argmax()]
                error = error + np.abs(mode - float(row[ii]))
            elif header[ii] == 'var':
                error = error + np.sqrt(np.abs(var - float(row[ii])))
            elif header[ii] == 'skew':
                skew = sum(ps * np.pow((ys - mean) / sqrt(var), 3))
                error = error + np.pow(np.abs(skew - float(row[ii])), 1.0/3)

        return error
