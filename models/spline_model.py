# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, The Open Aggregator
#   GNU General Public License, Ver. 3 (see docs/license.txt)
################################################################################

"""Model Spline File

Each line in a model spline file represents a polynomial segment in
log-probability space.  The format is as follows::

  spp1
  <x>,<y0>,<y1>,<a0>,<a1>,<a2>
  ...

Each line describes a segment of a probability distribution of y,
conditional on x = ``<x>``.  The segment spans from ``<y0>`` to
``<y1>``, where the lowest value of ``<y0>`` may be ``-inf``, and the
highest value of ``<y1>`` may be ``inf``.  The ``<x>`` values may also
be categorical or numerical.  If they are numerical, it is assumed
that these values represent samples of a smoothly varying function (a
cubic spline in every y).

The values ``<a0>``, ``<a1>`` and ``<a2>`` are the polynomial
coefficients in y (with quadratic coefficients, only normal or
exponential tails are possible).  The final segment of the probability
function is:
  exp(a0 + a1 y + a2 y2)
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

import csv, math, string
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from scipy.special import erf

from model import Model
from univariate_model import UnivariateModel

class SplineModel(UnivariateModel):
    posinf = float('inf')
    neginf = float('-inf')
    samples = 1000

    def __init__(self, xx_is_categorical=False, xx=None, conditionals=None, scaled=True):
        super(SplineModel, self).__init__(xx_is_categorical, xx, scaled)
        
        if conditionals is not None:
            self.conditionals = conditionals
        else:
            self.conditionals = []

    def kind(self):
        return 'spline_model'

    def copy(self):
        conditionals = []
        for conditional in self.conditionals:
            conditionals.append(conditional.copy())
            
        return SplineModel(self.xx_is_categorical, list(self.get_xx()), conditionals, scaled=self.scaled)

    def get_xx(self):
        if self.xx_is_categorical:
            return self.xx_text
        else:
            return self.xx

    def eval_pval(self, x, p, threshold=1e-3):
        conditional = self.get_conditional(x)
        return conditional.get_pval(p, threshold)

    def scale_y(self, a):
        for conditional in self.conditionals:
            conditional.scale_y(a)
            if self.scaled:
                conditional.rescale()

        return self

    def filter_x(self, xx):
        conditionals = []
        for x in xx:
            conditionals.append(self.get_conditional(x))
        
        return SplineModel(self.xx_is_categorical, xx, conditionals, scaled=self.scaled)

    def interpolate_x(self, newxx):
        (limits, ys) = SplineModelConditional.propose_grid(self.conditionals)
        ddp_model = self.to_ddp(ys).interpolate_x(newxx)
        
        return SplineModel.from_ddp(ddp_model, limits)

    # Only for categorical models
    def recategorize_x(self, oldxx, newxx):
        conditionals = []
        for ii in range(len(oldxx)):
            conditionals.append(self.get_conditional(oldxx[ii]))

        return SplineModel(True, newxx, conditionals, scaled=self.scaled)

    def add_conditional(self, x, conditional):
        if not self.xx_is_categorical:
            try:
                self.xx.append(float(x))
                self.xx_text.append(str(x))
            except ValueError:
                self.xx_is_categorical = True

        if self.xx_is_categorical:
            self.xx_text.append(x)
            self.xx.append(len(self.xx))
            
        self.conditionals.append(conditional)

    def get_conditional(self, x):
        if x is None or x == '':
            return self.conditionals[0]
        
        try:
            return self.conditionals[self.xx_text.index(str(x))]
        except Exception as e:
            return SplineModelConditional.find_nearest(self.xx, x, self.conditionals)

    def write_file(self, filename, delimiter):
        with open(filename, 'w') as fp:
            self.write(fp, delimiter)

    def write(self, file, delimiter):
        if self.scaled:
            file.write("spp1\n")
        else:
            file.write("spv1\n")
        
        writer = csv.writer(file, delimiter=delimiter)

        for ii in range(len(self.xx)):
            for jj in range(len(self.conditionals[ii].y0s)):
                row = [self.xx_text[ii], self.conditionals[ii].y0s[jj], self.conditionals[ii].y1s[jj]]
                row.extend(self.conditionals[ii].coeffs[jj])
                writer.writerow(row)

    def write_gaussian(self, file, delimiter):
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerow(['dpc1', 'mean', 'sdev'])
        for ii in range(len(self.xx)):
            for jj in range(len(self.conditionals[ii].y0s)):
                if len(self.conditionals[ii].coeffs[jj]) == 1 and self.conditionals[ii].coeffs[jj][0] == SplineModel.neginf:
                    continue
                elif len(self.conditionals[ii].coeffs[jj]) == 3:
                    writer.writerow([self.xx_text[ii], self.conditionals[ii].gaussian_mean(jj), self.conditionals[ii].gaussian_sdev(jj)])
                else:
                    writer.writerow([self.xx_text[ii], None, None])

    def write_gaussian_plus(self, file, delimiter):
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerow(['dpc1', 'y0', 'y1', 'mean', 'sdev'])
        for ii in range(len(self.xx)):
            for jj in range(len(self.conditionals[ii].y0s)):
                if len(self.conditionals[ii].coeffs[jj]) == 1 and self.conditionals[ii].coeffs[jj][0] == SplineModel.neginf:
                    continue
                elif len(self.conditionals[ii].coeffs[jj]) == 3:
                    writer.writerow([self.xx_text[ii], self.conditionals[ii].y0s[jj], self.conditionals[ii].y1s[jj], self.conditionals[ii].gaussian_mean(jj), self.conditionals[ii].gaussian_sdev(jj)])
                else:
                    writer.writerow([self.xx_text[ii], self.conditionals[ii].y0s[jj], self.conditionals[ii].y1s[jj], None, None])

    def to_points_at(self, x, ys):
        conditional = self.get_conditional(x)
        return conditional.to_points(ys)

    def cdf(self, xx, yy):
        conditional = self.get_conditional(xx)
        return conditional.cdf(yy)

    def is_gaussian(self, x=None):
        conditional = self.get_conditional(x)
        return len(conditional.y0s) == 1 and len(conditional.coeffs[0]) == 3

    def get_mean(self, x=None):
        conditional = self.get_conditional(x)
        if conditional.is_gaussian():
            return conditional.gaussian_mean(0)

        total = 0
        for ii in range(conditional.size()):
            total += conditional.nongaussian_xpx(ii)

        return total

    def get_sdev(self, x=None):
        conditional = self.get_conditional(x)
        if conditional.is_gaussian():
            return conditional.gaussian_sdev(0)

        total = 0
        for ii in range(conditional.size()):
            total += conditional.nongaussian_x2px(ii)

        mean = self.get_mean(x)

        return math.sqrt(total - mean**2)

    def init_from_spline_file(self, file, delimiter, status_callback=None):
        line = string.strip(file.readline())
        if line == "spp1":
            self.scaled = True
        elif line == 'spv1':
            self.scaled = False
        else:
            raise ValueError("Unknown format: %s" % (line))

        self.xx = []
        self.xx_text = []
        self.xx_is_categorical = False
        self.conditionals = []

        reader = csv.reader(file, delimiter=delimiter)
        x = None
        conditional = None
        for row in reader:
            if row[0] != x:
                x = row[0]
                conditional = SplineModelConditional()
                self.add_conditional(x, conditional)

            conditional.add_segment(float(row[1]), float(row[2]), map(float, row[3:]))

            if status_callback:
                status_callback("Parsing...", reader.line_num / (reader.line_num + 3.0))

        if self.scaled:
            for conditional in self.conditionals:
                conditional.rescale()

        return self

    def to_ddp(self, ys=None):
        if ys is None:
            (limits, ys) = SplineModelConditional.propose_grid(self.conditionals)
            
        pp = np.ones((len(self.xx), len(ys)))
        for ii in range(len(self.xx)):
            pp[ii,] = self.to_points_at(self.xx[ii], ys)

        return DDPModel('ddp1', 'spline_model', self.xx_is_categorical, self.get_xx(), False, ys, pp, scaled=self.scaled)

    ### Class Methods

    @staticmethod
    def create_gaussian(xxs):
        conditionals = []
        xx = []
        for key in xxs:
            xx.append(key)
            mean = xxs[key][0]
            var = xxs[key][1]
            conditional = SplineModelConditional.make_gaussian(SplineModel.neginf, SplineModel.posinf, mean, var)
            conditionals.append(conditional)

        return SplineModel(True, xx, conditionals, True)

    @staticmethod
    def from_ddp(ddp_model, limits):
        lps = ddp_model.log_p()
        
        conditionals = []
        for ii in range(len(ddp_model.xx)):
            spline = UnivariateSpline(ddp_model.yy, lps[ii,], k=2)
            conditionals.append(SplineModelConditional.make_conditional_from_spline(spline, limits))

        return SplineModel(self.xx_is_categorical, self.get_xx(), conditionals, True).rescale()

    @staticmethod
    def merge(models):
        for model in models:
            if not model.scaled:
                raise ValueError("Only scaled distributions can be merged.")

        model = SplineModel()
        
        for ii in range(len(models[0].xx)):
            conditional = SplineModelConditional()
            
            y0 = SplineModel.neginf
            # Loop through each segment
            while y0 != SplineModel.posinf:
                y1 = SplineModel.posinf
                coeffs = np.zeros(3)
                
                for jj in range(len(models)):
                    modcond = models[jj].conditionals[ii]
                    for kk in range(len(modcond.y0s)):
                        if modcond.y0s[kk] <= y0 and modcond.y1s[kk] > y0:
                            if modcond.y1s[kk] < y1:
                                y1 = modcond.y1s[kk]
                            coeffs[0:len(modcond.coeffs[kk])] = np.array(coeffs[0:len(modcond.coeffs[kk])]) + np.array(modcond.coeffs[kk])

                while len(coeffs) > 0 and coeffs[-1] == 0:
                    coeffs = coeffs[0:-1]
                    
                conditional.add_segment(y0, y1, coeffs)
                y0 = y1

            model.add_conditional(models[0].xx_text[ii], conditional.rescale())

        return model

    @staticmethod
    def combine(one, two):
        if one.xx_is_categorical != two.xx_is_categorical:
            raise ValueError("Cannot combine models that do not agree on categoricity")
        if not one.scaled or not two.scaled:
            raise ValueError("Cannot combine unscaled models")

        (one, two, xx) = UnivariateModel.intersect_x(one, two)

        conditionals = []
        for ii in range(len(xx)):
            conditionals.append(one.get_conditional(xx[ii]).convolve(two.get_conditional(xx[ii])).rescale())

        return SplineModel(one.xx_is_categorical, xx, conditionals, True)

class SplineModelConditional():
    # coeffs is ordered low-order to high-order
    def __init__(self, y0s=None, y1s=None, coeffs=None):
        if y0s is None:
            self.y0s = np.array([])
            self.y1s = np.array([])
            self.coeffs = []
        else:
            self.y0s = np.array(y0s)
            self.y1s = np.array(y1s)
            self.coeffs = coeffs

    def size(self):
        return len(self.y0s)

    def copy(self):
        return SplineModelConditional(list(self.y0s), list(self.y1s), list(self.coeffs))

    # Does not maintain scaling
    def scale_y(self, a):
        for ii in range(self.size()):
            self.y0s[ii] *= a
            self.y1s[ii] *= a
            if len(self.coeffs[ii]) > 1:
                self.coeffs[ii][1] /= a
                if len(self.coeffs[ii]) > 2:
                    self.coeffs[ii][2] /= a*a

    # Does not check for overlapping segments
    def add_segment(self, y0, y1, coeffs):
        self.y0s = np.append(self.y0s, [y0])
        self.y1s = np.append(self.y1s, [y1])
        self.coeffs.append(coeffs)

        indexes = np.argsort(self.y0s)
        if indexes[-1] != len(self.y0s) - 1:
            self.y0s = self.y0s[indexes]
            self.y1s = self.y1s[indexes]
            self.coeffs = [self.coeffs[index] for index in indexes]
        
    # Note: after calling, need to set scaled on SplineModel object
    def rescale(self):
        integral = self.cdf(SplineModel.posinf)
        if not np.isnan(integral):
            self.scale(1 / integral)

        return self

    def scale(self, factor):
        if factor == 0:
            self.y0s = [SplineModel.neginf]
            self.y1s = [SplineModel.posinf]
            self.coeffs = [[SplineModel.neginf]]
        else:
            for ii in range(self.size()):
                self.coeffs[ii][0] = self.coeffs[ii][0] + math.log(factor)

    # Similar to to_points
    def evaluate(self, ii, y):
        if y == SplineModel.neginf or y == SplineModel.posinf:
            return 0
        return np.exp(np.polyval(self.coeffs[ii][::-1], y))

    # Similar to evaluate
    def to_points(self, ys):
        result = np.array(ys)
        for ii in range(len(self.y0s)):
            valid = np.logical_and(ys >= self.y0s[ii], ys <= self.y1s[ii])
            result[valid] = np.exp(np.polyval(self.coeffs[ii][::-1], ys[valid]))

        return result

    def cdf(self, yy):
        integral = 0
        for ii in range(len(self.y0s)):
            if self.y1s[ii] >= yy:
                y1 = yy
            else:
                y1 = self.y1s[ii]

            if len(self.coeffs[ii]) == 0:
                return np.nan
            
            if len(self.coeffs[ii]) == 1:
                if self.coeffs[ii][0] == SplineModel.neginf:
                    continue
                
                integral += np.exp(self.coeffs[ii][0]) * (y1 - self.y0s[ii])
            elif len(self.coeffs[ii]) == 2:
                integral += (np.exp(self.coeffs[ii][0]) / self.coeffs[ii][1]) * (np.exp(self.coeffs[ii][1] * y1) - np.exp(self.coeffs[ii][1] * self.y0s[ii]))
            elif self.coeffs[ii][2] > 0:
                if self.y0s[ii] == SplineModel.neginf or self.y1s[ii] == SplineModel.posinf:
                    raise ValueError("Improper area of spline")

                myys = np.linspace(self.y0s[ii], y1, SplineModel.samples)
                integral += sum(np.exp(np.polyval(self.coeffs[ii][::-1], myys))) * (y1 - self.y0s[ii]) / SplineModel.samples
            else:
                var = -.5 / self.coeffs[ii][2]
                mean = self.coeffs[ii][1] * var
                if self.coeffs[ii][0] - (-mean*mean / (2*var) + math.log(1 / math.sqrt(2*math.pi*var))) > 100: # math domain error!
                    continue
                rescale = math.exp(self.coeffs[ii][0] - (-mean*mean / (2*var) + math.log(1 / math.sqrt(2*math.pi*var))))
                below = 0
                if float(self.y0s[ii]) != SplineModel.neginf:
                    below = norm.cdf(float(self.y0s[ii]), loc=mean, scale=math.sqrt(var))
                integral += rescale * (norm.cdf(y1, loc=mean, scale=math.sqrt(var)) - below)

            if self.y1s[ii] >= yy:
                break

        return integral

    def get_pval(self, p, threshold=1e-3):
        y = SplineModelConditional.ascinv(p, self.cdf, SplineModel.neginf, SplineModel.posinf, threshold)
        if np.isnan(y):
            # Let's just give back some value
            return self.y0s[1]

        return y

    # find the x for a given y of an ascending function
    # copied from math.js
    @staticmethod
    def ascinv(y, func, minx, maxx, threshold):
        tries = 0
        while tries < 10000:
            tries += 1
            if (minx == SplineModel.neginf and maxx == SplineModel.posinf) or (minx == SplineModel.neginf and maxx > 0) or (minx < 0 and maxx == SplineModel.posinf):
                midpoint = 0
            elif minx == SplineModel.neginf:
                midpoint = (maxx - 1.0) * 2
            elif maxx == SplineModel.posinf:
                midpoint = (minx + 1.0) * 2
            else:
                midpoint = (minx + maxx) / 2.0
            
            error = func(midpoint) - y
            if abs(error) < threshold:
                return midpoint
            elif np.isnan(error):
                return np.nan
            elif error > 0:
                maxx = midpoint
            elif error < 0:
                minx = midpoint

        return np.nan

    def approximate_mean(self, limits):
        rough_limits = self.rough_limits()
        limits = (max(float(limits[0]), rough_limits[0]), min(float(limits[1]), rough_limits[1]))
        ys = np.linspace(limits[0], limits[1], self.size() * SplineModel.samples)
        ps = self.to_points(ys)
        ps = ps / sum(ps)

        return sum(ps * ys)

    # Allow true gaussian or delta
    def is_gaussian(self):
        return len(self.y0s) == 1 and (len(self.coeffs[0]) == 3 or len(self.coeffs[0]) == 0)

    def gaussian_sdev(self, ii):
        if len(self.coeffs[ii]) == 0:
            return 0
        
        return 1/math.sqrt(-2*self.coeffs[ii][2])

    def gaussian_mean(self, ii):
        if len(self.coeffs[ii]) == 0:
            return (self.y1s[ii] + self.y0s[ii]) / 2
        return -self.coeffs[ii][1] / (2*self.coeffs[ii][2])

    def nongaussian_xpx(self, ii):
        a = self.coeffs[ii][2] if len(self.coeffs[ii]) > 2 else 0
        b = self.coeffs[ii][1] if len(self.coeffs[ii]) > 1 else 0
        c = self.coeffs[ii][0]
        x0 = self.y0s[ii]
        x1 = self.y1s[ii]

        # From Matlab
        if a == 0:
            if x0 == SplineModel.neginf:
                return (math.exp(c + b*x1)*(b*x1 - 1))/b**2
            elif x1 == SplineModel.posinf:
                return -(math.exp(c + b*x0)*(b*x0 - 1))/b**2
            return (math.exp(c + b*x1)*(b*x1 - 1))/b**2 - (math.exp(c + b*x0)*(b*x0 - 1))/b**2

        sqrtpi = math.pi**.5
        na05 = ((-a)**.5)
        na15 = ((-a)**1.5)
        return (math.exp(a*x1**2 + b*x1)*math.exp(c))/(2*a) - (math.exp(a*x0**2 + b*x0)*math.exp(c))/(2*a) + (sqrtpi*b*math.exp(-b**2/(4*a))*math.exp(c)*erf((b + 2*a*x0)/(2*na05)))/(4*na15) - (sqrtpi*b*math.exp(-(b**2)/(4*a))*math.exp(c)*erf((b + 2*a*x1)/(2*na05)))/(4*na15)

    def nongaussian_x2px(self, ii):
        a = self.coeffs[ii][2] if len(self.coeffs[ii]) > 2 else 0
        b = self.coeffs[ii][1] if len(self.coeffs[ii]) > 1 else 0
        c = self.coeffs[ii][0]
        x0 = self.y0s[ii]
        x1 = self.y1s[ii]

        # From Matlab
        if a == 0:
            if x0 == SplineModel.neginf:
                return (math.exp(c + b*x1)*(b**2*x1**2 - 2*b*x1 + 2))/b**3
            elif x1 == SplineModel.posinf:
                return -(math.exp(c + b*x0)*(b**2*x0**2 - 2*b*x0 + 2))/b**3
            return (math.exp(c + b*x1)*(b**2*x1**2 - 2*b*x1 + 2))/b**3 - (math.exp(c + b*x0)*(b**2*x0**2 - 2*b*x0 + 2))/b**3

        sqrtpi = math.pi**.5
        na05 = ((-a)**.5)
        na25 = ((-a)**2.5)
        na35 = ((-a)**3.5)
        return (2*na25*b*math.exp(a*x0**2 + b*x0 + c) - 2*na25*b*math.exp(a*x1**2 + b*x1 + c) + 4*na35*x0*math.exp(a*x0**2 + b*x0 + c) - 4*na35*x1*math.exp(a*x1**2 + b*x1 + c) - 2*(sqrtpi)*a**3*math.exp((- b**2 + 4*a*c)/(4*a))*erf((b + 2*a*x0)/(2*na05)) + 2*(sqrtpi)*a**3*math.exp((- b**2 + 4*a*c)/(4*a))*erf((b + 2*a*x1)/(2*na05)) + (sqrtpi)*a**2*b**2*math.exp((- b**2 + 4*a*c)/(4*a))*erf((b + 2*a*x0)/(2*na05)) - (sqrtpi)*a**2*b**2*math.exp((- b**2 + 4*a*c)/(4*a))*erf((b + 2*a*x1)/(2*na05)))/(8*((-a)**4.5))

    # Duplicated in models.js
    def segment_max(self, jj):
        maxyy = self.y0s[jj]
        maxval = self.evaluate(jj, self.y0s[jj])
        val = self.evaluate(jj, self.y1s[jj])
        if (val > maxval):
            maxval = val
            maxyy = self.y1s[jj]

        coeffs = self.coeffs[jj]
        if len(coeffs) > 2:
            yy = -coeffs[1] / (2*coeffs[2])
            if yy > self.y0s[jj] and yy < self.y1s[jj]:
                val = self.evaluate(jj, yy)
                if val > maxval:
                    maxval = val
                    maxyy = yy

        return (maxyy, maxval)

    # Duplicated in models.js
    # returns (yy, val)
    def find_mode(self):
        maxmax = (None, SplineModel.neginf)
        
        for ii in range(self.size()):
            mymax = self.segment_max(ii)
            if mymax[1] > maxmax[1]:
                maxmax = mymax

        return maxmax

    # Duplicated in models.js
    def rough_span(self):
        span = 0
        for jj in range(self.size()):
            if self.y0s[jj] == SplineModel.neginf or self.y1s[jj] == SplineModel.posinf:
                if len(self.coeffs[jj]) == 3:
                    span += 3 / math.sqrt(abs(2*self.coeffs[jj][2]))
                elif len(self.coeffs[jj]) == 2:
                    span += 5 / abs(self.coeffs[jj][1])
                else:
                    span += 1 / abs(self.coeffs[jj][0]) # improper!
            else:
                span += self.y1s[jj] - self.y0s[jj]

        return span

    # Duplicated in models.js
    def rough_limits(self):
        limits0 = float(self.y0s[0])
        limits1 = float(self.y1s[-1])

        if limits0 == SplineModel.neginf or limits1 == SplineModel.posinf:
            maxmax = self.find_mode()

            span = self.rough_span()
            if limits0 == SplineModel.neginf:
                limits0 = maxmax[0] - span
            if limits1 == SplineModel.posinf:
                limits1 = maxmax[0] + span

        return (limits0, limits1)

    def convolve(self, other):
        # NOTE: below is for a single segment...
        # int_s e^P(s) e^Q(t - s) = int_s e^[P(s) + Q(t - s)] = int_s e^[a1 ss + b1 s + c1 + a2 (tt - 2ts + ss) + b2 t - b2 s + c2]
        # int_s e^[(a1 + a2) ss + (b1 - 2t - b2) s] e^[a2 (tt) + b2 t + c1 + c2]
        # Have to do approximate sum later anyway, so let's just convert to ddp
        (limits, ys) = SplineModelConditional.propose_grid([self, other])
        pp_self = self.to_points(ys)
        pp_other = other.to_points(ys)
        newpp = np.convolve(pp_self, pp_other)
        newpp = newpp / sum(newpp) # Scale

        yy = np.linspace(2*min(ys), 2*max(ys), 2*len(ys) - 1)

        if np.any(newpp == 0):
            conditional = SplineModelConditional()
            # Break into many pieces
            ii = 0
            y0 = min(yy)
            while ii == 0 or (ii < len(newpp) and newpp[ii] == 0):
                if newpp[ii] == 0:
                    while ii < len(newpp) and newpp[ii] == 0:
                        ii += 1
                    if ii < len(newpp):
                        conditional.add_segment(y0, yy[ii], [SplineModel.neginf])
                    else:
                        conditional.add_segment(y0, yy[-1], [SplineModel.neginf])
                        break
                    y0 = yy[ii]
                i0 = ii
                while ii < len(newpp) and newpp[ii] > 0:
                    ii += 1

                print np.log(newpp[i0:ii])                
                spline = UnivariateSpline(yy[i0:ii], np.log(newpp[i0:ii]), k=2, s=(ii - i0) / 1000.0)
                print spline(yy[i0:ii])
                if ii < len(newpp):
                    segments = SplineModelConditional.make_conditional_from_spline(spline, (y0, yy[ii]))
                else:
                    segments = SplineModelConditional.make_conditional_from_spline(spline, (y0, yy[-1]))
                    
                for jj in range(segments.size()):
                    conditional.add_segment(segments.y0s[jj], segments.y1s[jj], segments.coeffs[jj])
                if ii < len(newpp):
                    y0 = yy[ii]
                else:
                    break

            return conditional
        else:
            spline = UnivariateSpline(yy, np.log(newpp), k=2)
        
            return SplineModelConditional.make_conditional_from_spline(spline, (2*limits[0], 2*limits[1]))

    @staticmethod
    def make_single(y0, y1, coeffs):
        return SplineModelConditional(y0s=[y0], y1s=[y1], coeffs=[coeffs])

    @staticmethod
    def make_gaussian(y0, y1, mean, var):
        return SplineModelConditional.make_single(y0, y1, [-mean*mean/(2*var) - np.log(np.sqrt(2*math.pi*var)), mean/var, -1/(2*var)])
    
    @staticmethod
    def make_conditional_from_spline(spline, limits):
        conditional = SplineModelConditional()

        knots = spline.get_knots()
        midpoint = (knots[-1] + knots[1]) / 2

        knots = sorted(knots)
        knots[0] = float(limits[0])
        knots[-1] = float(limits[1])

        for ii in range(1, len(knots)):
            if knots[ii-1] == SplineModel.neginf and knots[ii] == SplineModel.posinf:
                y = midpoint
            elif knots[ii-1] == SplineModel.neginf:
                y = knots[ii]
            elif knots[ii] == SplineModel.posinf:
                y = knots[ii-1]
            else:
                y = (knots[ii-1] + knots[ii]) / 2
            derivs = spline.derivatives(y)
            a = derivs[2] / 2
            b = derivs[1] - derivs[2] * y
            c = derivs[0] - (a*y*y + b*y)
            conditional.add_segment(knots[ii-1], knots[ii], [c, b, a])

        return conditional

    @staticmethod
    def find_nearest(array, value, within):
        idx = (np.abs(np.array(array)-value)).argmin()
        return within[idx]

    @staticmethod
    def approximate_sum(conditionals):
        if len(conditionals) == 1:
            return conditionals[0]

        (limits, ys) = SplineModelConditional.propose_grid(conditionals)
        
        ps = np.zeros(len(ys))
        for ii in range(len(conditionals)):
            ps = ps + conditionals[ii].to_points(ys)
        lps = np.log(ps)

        spline = UnivariateSpline(ys, lps, k=2)
        return SplineModelConditional.make_conditional_from_spline(spline, limits)

    @staticmethod
    def propose_grid(conditionals):
        limits = (SplineModel.neginf, SplineModel.posinf)
        rough_limits = (SplineModel.posinf, SplineModel.neginf)
        max_segments = 0
        for conditional in conditionals:
            limits = (max(limits[0], conditional.y0s[0]), min(limits[1], conditional.y1s[-1]))
            conditional_rough_limits = conditional.rough_limits()
            rough_limits = (min(rough_limits[0], conditional_rough_limits[0]), max(rough_limits[1], conditional_rough_limits[1]))
            
            max_segments = max(max_segments, sum(map(lambda cc: len(cc), conditional.coeffs)))
        
        num_points = 100 * max_segments / (1 + np.log(len(conditionals)))

        ys = np.linspace(rough_limits[0], rough_limits[1], num_points)
        return (limits, ys)

from ddp_model import DDPModel

Model.mergers["spline_model"] = SplineModel.merge
Model.mergers["spline_model+ddp_model"] = lambda models: DDPModel.merge(map(lambda m: m.to_ddp(), models))
Model.combiners['spline_model+spline_model'] = SplineModel.combine
Model.combiners["spline_model+ddp_model"] = lambda one, two: DDPModel.combine(one.to_ddp(), two)
