# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, The Open Aggregator
#   GNU General Public License, Ver. 3 (see docs/license.txt)
################################################################################

"""Discrete-Discrete-Probability (DDP) Format

A DDP file describes a dose-response relationship with a limited
collection of response outcomes.  The dose and response values may be
either categorical or sampled at a collection of numerical levels.
The format of a DDP file is::

  <format>,<y-value-1>,<y-value-2>,...
  <x-value-1>,p(y1|x1),p(y2|x1),...
  <x-value-2>,p(y1|x2),p(y2|x2),...
  ...

<format> may be one of the following values:
  * ``ddp1`` - the p(.) values are simple probabilities (0 < p(.) < 1 and sum p(y|x) = 1)
  * ``ddp2`` - the p(.) values are log probabilities

``<y-value-1>``, ..., ``<y-value-N>`` and ``<x-value-1>``, ...,
``<x-value-N>`` are either strings, for named categories or numerical
values.

Below is a sample categorical DDP file::

  ddp1,live,dead
  control,.5,.5
  treated,.9,.1

Below is a sample numerical DDP file::

  ddp1,-10.0,-.33333333333,3.33333333333,10.0
  0.0,0.5,0.5,0.0,0.0
  13.3333333333,0.0,0.5,0.5,0.0
  26.6666666667,0.0,0.0,0.5,0.5
  40.0,0.0,0.0,0.0,0.5
"""
__copyright__ = "Copyright 2014, The Open Aggregator"
__license__ = "GPL"

__author__ = "James Rising"
__credits__ = ["James Rising", "Solomon Hsiang"]
__maintainer__ = "James Rising"
__email__ = "jar2234@columbia.edu"

__status__ = "Production"
__version__ = "$Revision$"
# $Source$

import csv, random
from numpy import *
from scipy.interpolate import interp1d

from model import Model
from univariate_model import UnivariateModel
from memoizable import MemoizableUnivariate

class DDPModel(UnivariateModel, MemoizableUnivariate):
    def __init__(self, p_format=None, source=None, xx_is_categorical=False, xx=None, yy_is_categorical=False, yy=None, pp=None, unaccounted=None, scaled=True):
        super(DDPModel, self).__init__(xx_is_categorical, xx, scaled)
        
        self.p_format = p_format
        self.source = source
                    
        self.yy_is_categorical = yy_is_categorical
        if yy_is_categorical:
            self.yy = range(len(yy))
            self.yy_text = yy
        elif yy is not None:
            self.yy = yy
            self.yy_text = map(str, yy)

        self.pp = pp
        self.unaccounted = unaccounted

    def kind(self):
        return 'ddp_model'

    def copy(self):
        # Can't use python's copy since could be strange object from ming
        return DDPModel(self.p_format, getattr(self, 'source', 'external'), self.xx_is_categorical, list(self.get_xx()), self.yy_is_categorical, list(self.get_yy()), array(self.pp), unaccounted=getattr(self, 'unaccounted', 0), scaled=self.scaled)

    def get_xx(self):
        if self.xx_is_categorical:
            return self.xx_text
        else:
            return self.xx

    def get_yy(self):
        if self.yy_is_categorical:
            return self.yy_text
        else:
            return self.yy

    # Can rescale non-ddp (that is, as sampling of continuous distribution)
    def rescale(self, as_ddp=True):
        if as_ddp or self.yy_is_categorical:
            newpp = self.lin_p()
            for ii in range(len(self.xx)):
                newpp[ii,] = newpp[ii,] / sum(newpp[ii,])
        else:
            sorts = sorted(self.get_yy())
            if len(sorts) > 0:
                newpp = self.lin_p()
                for ii in range(len(self.xx)):
                    cdf = 0
                    cdf += (sorts[1] - sorts[0]) * newpp[ii, 0]
                    for jj in range(1, len(sorts)-1):
                        cdf += (sorts[jj+1] - sorts[jj-1]) * newpp[ii, jj] / 2
                    cdf += (sorts[-1] - sorts[-2]) * newpp[ii, -1]

                    newpp[ii,] = newpp[ii,] / float(cdf)

        self.pp = newpp
        self.p_format = 'ddp1'
        self.scaled = as_ddp or self.yy_is_categorical

        return self

    def eval_pval(self, x, p, threshold=1e-3):
        return self.eval_pval_index(self.get_closest(x), p, threshold)

    def scale_y(self, a):
        if self.yy_is_categorical:
            raise ValueError("Cannot scale on a categorical y")

        self.yy = [y * a for y in self.yy]
        return self

    def scale_p(self, a):
        self.pp = a * self.log_p()
        self.p_format = 'ddp2'
        return self.rescaled()

    def add_to_y(self, a):
        if self.yy_is_categorical:
            raise ValueError("Cannot add to a categorical y")

        self.yy = [y + a for y in self.yy]
        return self

    def transpose(self):
        other = self.copy()
        other.pp = transpose(other.pp)

        other.xx_is_categorical = self.yy_is_categorical
        other.xx = list(self.yy)
        other.xx_text = self.yy_text

        other.yy_is_categorical = self.xx_is_categorical
        other.yy = list(self.xx)
        other.yy_text = self.xx_text

        return other

    def write_file(self, filename, delimiter):
        with open(filename, 'w') as fp:
            self.write(fp, delimiter)

    def write(self, file, delimiter):
        writer = csv.writer(file, delimiter=delimiter)

        if self.scaled:
            header = [self.p_format]
        elif self.p_format == 'ddp1':
            header = ['ddv1']
        elif self.p_format == 'ddp2':
            header = ['ddv2']
            
        if self.yy_is_categorical:
            header.extend(self.yy_text)
        else:
            header.extend(self.yy)
        writer.writerow(header)
        
        for ii in range(len(self.xx)):
            if self.xx_is_categorical:
                row = [self.xx_text[ii]]
                row.extend(self.pp[ii,])
                writer.writerow(row)
            else:
                row = [self.xx[ii]]
                row.extend(self.pp[ii,])
                writer.writerow(row)

    def lin_p(self):
        if self.p_format == 'ddp1':
            return self.pp
        elif self.p_format == 'ddp2':
            return exp(self.pp)
        else:
            return NotImplementedError("Unknown format in lin_p: " + self.p_format)

    def log_p(self):
        if self.p_format == 'ddp1':
            pp = ones((len(self.xx), len(self.yy))) * float('-inf')
            pp[self.pp > 0] = log(self.pp[self.pp > 0])
            return pp
        elif self.p_format == 'ddp2':
            return self.pp
        else:
            return NotImplementedError("Unknown format in log_p: " + self.p_format)

    def filter_x(self, xx):
        newpp = ones((len(xx), len(self.yy)))
        for ii in range(len(xx)):
            newpp[ii,] = self.pp[self.get_xx() == xx,]
            
        return DDPModel(self.p_format, 'filter_x', self.xx_is_categorical, xx, self.yy_is_categorical, self.get_yy(), newpp, scaled=self.scaled)

    def interpolate_x(self, newxx, kind='quadratic'):
        newpp = zeros((len(newxx), len(self.yy)))

        # Interpolate for each y
        pp = self.lin_p()
        xx = self.xx
        if min(newxx) < min(xx):
            xx = concatenate(([min(newxx)], xx))
            pp = vstack((zeros((1, len(self.yy))), pp))
        if max(newxx) > max(xx):
            xx = concatenate((xx, [max(newxx)]))
            pp = vstack((pp, zeros((1, len(self.yy)))))

        for jj in range(len(self.yy)):
            fx = interp1d(xx, pp[:,jj], kind)
            newpp[:,jj] = fx(newxx)

        # Rescale
        if self.scaled:
            for ii in range(len(newxx)):
                newpp[ii,] = newpp[ii,] / sum(newpp[ii,])

        return DDPModel('ddp1', 'interpolate_x', False, newxx, False, self.yy, newpp, scaled=self.scaled)

    # Only for categorical models
    def recategorize_x(self, oldxx, newxx):
        newpp = zeros((len(newxx), len(self.yy)))
        pp = self.lin_p()

        for ii in range(len(oldxx)):
            newpp[ii,] = self.pp[self.get_xx() == oldxx[ii],]

        return DDPModel('ddp1', 'recategorize_x', True, newxx, self.yy_is_categorical, self.yy, newpp, scaled=self.scaled)

    def interpolate_y(self, newyy, kind='quadratic'):
        newpp = zeros((len(self.xx), len(newyy)))

        # Interpolate for each y
        pp = self.lin_p()

        for ii in range(len(self.xx)):
            if len(self.yy) == 2:
                fx = interp1d(self.yy, pp[ii,:], 'linear')
            else:
                fx = interp1d(self.yy, pp[ii,:], kind)
                
            newpp[ii,:] = fx(newyy)
            if self.scaled:
                newpp[ii,] = newpp[ii,] / sum(newpp[ii,])

        return DDPModel('ddp1', 'interpolate_y', self.xx_is_categorical, self.get_xx(), False, newyy, newpp, scaled=self.scaled)

    def get_closest(self, x=None):
        if x is None:
            return 0

        try:
            return self.xx_text.index(str(x))
        except:
            idx = (abs(array(self.xx)-x)).argmin()
            return idx

    def get_mean(self, x=None):
        if not self.scaled:
            raise ValueError("Cannot take mean of unscaled distribution.")

        ps = self.lin_p()[self.get_closest(x), :]
        return sum(ps * self.yy)

    def get_sdev(self, x=None):
        if not self.scaled:
            raise ValueError("Cannot take sdev of unscaled distribution.")

        ps = self.lin_p()[self.get_closest(x), :]
        mean = sum(ps * self.yy)
        vari = sum(ps * square(self.yy - mean))
        return sqrt(vari)

    def draw_sample(self, x=None):
        if not self.scaled:
            raise ValueError("Cannot draw sample from unscaled distribution.")

        ps = self.lin_p()[self.get_closest(x), :]
        value = random.random()
        total = 0
        for ii in range(len(ps)):
            total += ps[ii]
            if total > value:
                return self.yy[ii]

        return self.yy[-1]
    
    def init_from(self, file, delimiter, status_callback=None):
        reader = csv.reader(file, delimiter=delimiter)
        header = reader.next()
        fmt = header[0]
        if fmt not in ['ddp1', 'ddp2', 'ddv1', 'ddv2']:
            raise ValueError("Unknown format: %s" % (fmt))

        if fmt == 'ddp1' or fmt == 'ddp2':
            self.p_format = fmt
            self.scaled = True
        elif fmt == 'ddv1':
            self.p_format = 'ddp1'
            self.scaled = False
        elif fmt == 'ddv2':
            self.p_format = 'ddp2'
            self.scaled = False

        yy_text = header[1:]
        yy = []
        self.yy_is_categorical = False
        for jj in range(len(yy_text)):
            try:
                yy.append(float(yy_text[jj]))
            except ValueError:
                yy.append(jj)
                self.yy_is_categorical = True

        pp = None
        xx_text = []
        xx = []
        self.xx_is_categorical = False
        for row in reader:
            if pp is None:
                pp = array([map(float, row[1:])])
            else:
                pp = vstack((pp, map(float, row[1:])))
                
            xx_text.append(row[0])
            try:
                xx.append(float(row[0]))
            except ValueError:
                xx.append(len(xx))
                self.xx_is_categorical = True

            if status_callback:
                status_callback("Parsing...", reader.line_num / (reader.line_num + 3.0))

        self.yy = yy
        self.yy_text = yy_text
        self.xx = list(xx)
        self.xx_text = xx_text
        self.pp = pp

        if self.scaled == True:
            print pp
            if self.p_format == 'ddp1':
                sums = sum(pp, axis=1)
            else:
                sums = sum(exp(pp), axis=1)

            if any(sums < .95):
                raise ValueError("Some columns sum to less than .95")

            self.unaccounted = 1 - sums

    def init_from_other(self, ddp):
        self.p_format = ddp.p_format
        self.source = ddp.source

        self.xx_is_categorical = ddp.xx_is_categorical
        self.xx = list(ddp.xx)
        self.xx_text = ddp.xx_text

        self.yy_is_categorical = ddp.yy_is_categorical
        self.yy = list(ddp.yy)
        self.yy_text = ddp.yy_text

        self.pp = ddp.pp
        self.unaccounted = ddp.unaccounted
        self.scaled = ddp.scaled

    def to_ddp(self, ys=None):
        if ys is None:
            return self.copy()

        return self.interpolate_y(ys)

    ### Memoizable

    def eval_pval_index(self, ii, p, threshold=1e-3):
        ps = self.lin_p()[ii, :]
        value = p * sum(ps)
        
        total = 0
        for ii in range(len(ps)):
            total += ps[ii]
            if total > value:
                return self.yy[ii]

        return self.yy[-1]        

    ### Class methods

    @staticmethod
    def from_file(filename, delimiter):
        with open(filename) as fp:
            model = DDPModel()
            model.init_from(fp, delimiter)
            model.source = filename
            return model

    @staticmethod
    def create_lin(yy, xxs):
        pp = None
        xx = []
        for key in xxs:
            xx.append(key)
            if pp is None:
                pp = xxs[key]
            else:
                pp = vstack((pp, xxs[key]))

        return DDPModel('ddp1', 'create_lin', True, xx, False, array(yy), pp)

    @staticmethod
    def merge(models):
        # Decide on master x values
        xx = []
        yy = []

        for model in models:
            if not isinstance(model, DDPModel):
                raise ValueError('Merge only handles ddp models')
            if not model.scaled:
                raise ValueError("Only scaled distributions can be merged.")
        
            xx = concatenate((xx, model.xx))
            yy = concatenate((yy, model.yy))

        xx = unique(xx)
        yy = unique(yy)

        # Bayesian combination of all models
        sumlp = zeros((len(xx), len(yy)))
        for model in models:
            if len(xx) != len(model.xx) or any(xx != model.xx):
                model = model.interpolate_x(xx)
            if len(yy) != len(model.yy) or any(yy != model.yy):
                model = model.interpolate_y(yy)
            sumlp = sumlp + model.log_p()

        # Rescale along each x
        prodp = exp(sumlp)
        sums = sum(prodp, axis=1)
        pp = empty((len(xx), len(yy)))
        for ii in range(len(xx)):
            pp[ii,] = prodp[ii,] / sums[ii]

        return DDPModel('ddp1', 'merge', False, xx, False, yy, pp)

    @staticmethod
    def combine(one, two):
        if one.xx_is_categorical != two.xx_is_categorical:
            raise ValueError("Cannot combine models that do not agree on categoricity")
        if one.yy_is_categorical or two.yy_is_categorical:
            raise ValueError("Cannot combine categorical y models")
        if not one.scaled or not two.scaled:
            raise ValueError("Cannot combine unscaled models")

        (one, two, xx) = UnivariateModel.intersect_x(one, two)

        yy_one_min = min(one.yy)
        yy_one_max = max(one.yy)
        yy_two_min = min(two.yy)
        yy_two_max = max(two.yy)
        yy_step = min(median(diff(sort(one.yy))), median(diff(sort(two.yy))))
        yy_one = arange(yy_one_min, yy_one_max, yy_step)
        if yy_one[-1] + yy_step == yy_one_max:
            yy_one = append(yy_one, [yy_one_max])
        yy_two = arange(yy_two_min, yy_two_max, yy_step)
        if yy_two[-1] + yy_step == yy_two_max:
            yy_two = append(yy_two, [yy_two_max])

        if not array_equal(yy_one, one.yy):
            one = one.interpolate_y(yy_one)
        if not array_equal(yy_two, two.yy):
            two = two.interpolate_y(yy_two)
        pp_one = one.lin_p()
        pp_two = two.lin_p()
        
        newpp = ones((len(xx), len(yy_one) + len(yy_two) - 1))
        for ii in range(len(xx)):
            newpp[ii,] = convolve(pp_one[ii,], pp_two[ii,])
            newpp[ii,] = newpp[ii,] / sum(newpp[ii,]) # Scale

        yy = append(arange(min(yy_one) + min(yy_two), max(yy_one) + max(yy_two), yy_step), [max(yy_one) + max(yy_two)])

        return DDPModel('ddp1', 'combine', one.xx_is_categorical, xx, False, yy, newpp, scaled=True)

Model.mergers["ddp_model"] = DDPModel.merge
Model.combiners['ddp_model+ddp_model'] = DDPModel.combine

#ddp1 = DDPModel.from_file("../test/ddp1.csv", ',')
#ddp2 = DDPModel.from_file("../test/ddp2.csv", ',')
#merged = DDPModel.merge([ddp1, ddp2])
#merged.write("../test/merge.csv", ',')

#ddp2 = DDPModel.from_file("../test/ddp2.csv", ',')
#ddp1 = DDPModel('ddp1', 'test', False, [-5, 0, 7], False, ddp2.yy, ones((3, len(ddp2.yy))))
##ddp1 = DDPModel('ddp1', 'test', False, ddp2.xx, False, ddp2.yy, ones((len(ddp2.xx), len(ddp2.yy))))
#merged = DDPModel.merge([ddp1, ddp2])
#merged.write("../test/merge.csv", ',')
