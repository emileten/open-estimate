# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, The Open Aggregator
#   GNU General Public License, Ver. 3 (see docs/license.txt)
################################################################################

"""Bin Model

A bin model represents bins of different spans, where the distribution
is constant over each bin.  It is a combination of information
describing the bins and an underlying categorical model of one of the
other types.

The underlying model is always categorical, with categories starting
at 1 0 is reserved for a future version that allows an out-of-sample
distribution

The format is::

  bin1
  <x0>,<x1>,<x2>, ...
  <underlying model>
"""
__copyright__ = "Copyright 2014, The Open Aggregator"
__license__ = "GPL"

__author__ = "James Rising"
__credits__ = ["James Rising", "Amir Jina"]
__maintainer__ = "James Rising"
__email__ = "jar2234@columbia.edu"

__status__ = "Production"
__version__ = "$Revision$"
# $Source$

import csv, string
import numpy as np

from model import Model
from univariate_model import UnivariateModel


class BinModel(UnivariateModel):
    def __init__(self, xx=None, model=None):
        super(BinModel, self).__init__(False, xx, model.scaled)
        
        self.model = model

    def kind(self):
        return 'bin_model'

    def copy(self):
        return BinModel(list(self.xx), self.model.copy())

    def get_xx(self):
        return self.xx

    def scale_y(self, a):
        self.model.scale_y(a)
        return self

    def scale_p(self, a):
        self.model.scale_p(a)
        return self

    def filter_x(self, xx):
        bins = []
        for x in xx:
            bins.append(self.xx.index(x) + 1)
        
        model = self.model.filter_x(bins)
        return BinModel(xx, model, scaled=self.model.scaled)

    # Do nothing in interpolation
    def interpolate_x(self, newxx):
        return self.copy()

    def write_file(self, filename, delimiter):
        with open(filename, 'w') as fp:
            self.write(fp, delimiter)

    def write(self, file, delimiter):
        file.write("bin1\n")
        file.write(delimiter.join(map(str, self.xx)) + "\n")
        self.model.write(file, delimiter)

    def get_bin_at(self, x):
        for ii in range(1, len(self.xx)):
            if self.xx[ii-1] <= x and self.xx[ii] > x:
                return ii

        return 0

    def to_points_at(self, x, ys):
        return self.model.to_points_at(self.get_bin_at(x), ys)

    def eval_pval(self, x, p, threshold=1e-3):
        return self.model.eval_pval(x, p, threshold)

    def cdf(self, x, y):
        return self.model.cdf(self.get_bin_at(x), y)

    def get_mean(self, x=None):
        return self.model.get_mean(self.get_bin_at(x))

    def get_sdev(self, x=None):
        return self.model.get_sdev(self.get_bin_at(x))

    def draw_sample(self, x=None):
        return self.model.draw_sample(self.get_bin_at(x))

    def init_from_bin_file(self, file, delimiter, status_callback=None):
        line = string.strip(file.readline())
        if line != "bin1":
            raise ValueError("Unknown format: %s" % (line))

        reader = csv.reader(file, delimiter=delimiter)
        row = reader.next()
        self.xx_text = row
        self.xx = map(float, row)
        self.xx_is_categorical = False

        self.model = None # Need to set this!

        return self

    def to_ddp(self, ys=None):
        newcats = []
        newxx = [self.xx[0]]
        for ii in range(1, len(self.xx)):
            newcats.extend([ii, ii])
            diff = (self.xx[ii] - self.xx[ii-1]) / 100.0
            newxx.extend([self.xx[ii] - diff, self.xx[ii] + diff])
        newxx[-1] = self.xx[-1]

        dupmodel = self.model.recategorize_x(newcats, range(1, len(newcats)+1))
        dupmodel = dupmodel.to_ddp(ys)
        dupmodel.xx = newxx
        dupmodel.xx_text = map(str, newxx)
        dupmodel.xx_is_categorical = False
        
        return dupmodel

    ### Class Methods

    # All models are BinModels
    @staticmethod
    def merge(models):
        allxx = set()
        for model in models:
            if not model.scaled:
                raise ValueError("Only scaled distributions can be merged.")
            allxx.update(set(model.xx))

        allxx = np.array(sorted(allxx))
        midpts = (allxx[1:] + allxx[:-1]) / 2
        newmodels = []
        for model in models:
            newmodels.append(model.model.recategorize_x(map(model.get_bin_at, midpts), range(1, len(allxx))))

        allmodel = Model.merge(newmodels)

        model = BinModel(allxx, allmodel)

        return model

    # Both models are BinModels
    @staticmethod
    def combine(one, two):
        if not one.scaled or not two.scaled:
            raise ValueError("Cannot combine unscaled models")

        (one, two, xx) = UnivariateModel.intersect_x(one, two)

        allxx = set(one.xx) | set(two.xx)
        allxx = np.array(allxx)
        midpts = (allxx[1:] + allxx[:-1]) / 2
        onemodel = one.model.recategorize_x(map(model.get_bin_at, midpts), range(1, len(allxx)))
        twomodel = two.model.recategorize_x(map(model.get_bin_at, midpts), range(1, len(allxx)))

        model = Model.combine([onemodel, twomodel], [1, 1])
        
        return BinModel(allxx, model, True)

from ddp_model import DDPModel

Model.mergers["bin_model"] = BinModel.merge
Model.mergers["bin_model+ddp_model"] = lambda models: DDPModel.merge(map(lambda m: m.to_ddp(), models))
Model.mergers["bin_model+spline_model"] = lambda models: DDPModel.merge(map(lambda m: m.to_ddp(), models))
Model.combiners['bin_model+bin_model'] = BinModel.combine
Model.combiners["bin_model+ddp_model"] = lambda one, two: DDPModel.combine(one.to_ddp(), two)
Model.combiners["bin_model+spline_model"] = lambda one, two: DDPModel.combine(one.to_ddp(), two)
