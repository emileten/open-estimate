# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, The Open Aggregator
#   GNU General Public License, Ver. 3 (see docs/license.txt)
################################################################################
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

from .model import Model
from .univariate_model import UnivariateModel
from .memoizable import MemoizableUnivariate

class BinModel(UnivariateModel, MemoizableUnivariate):
    '''
    Bin Model

    A bin model represents bins of different spans, where the distribution
    is constant over each bin.  It is a combination of information
    describing the bins and an underlying categorical model of one of the
    other types.

    The underlying model is always categorical, with categories starting
    at 1. 0 is reserved for a future version that allows an out-of-sample
    distribution

    The format is::

      bin1
      <x0>,<x1>,<x2>, ...
      <underlying model>

    Parameters
    ----------
    
    xx : list-like
        List-like array of bin edges. `len(xx)` should be one more than the 
        number of bins.

    model : object
        Statistical model used in each bin

    '''

    def __init__(self, xx=None, model=None):
        super(BinModel, self).__init__(False, xx, model.scaled if model is not None else False)

        self.model = model

    
    def kind(self):
        '''
        returns model type ("bin_model")
        '''

        return 'bin_model'

    
    def copy(self):
        '''
        copy data and return BinModel with the same data
        '''
        
        return BinModel(list(self.xx), self.model.copy())

    
    def get_xx(self):
        '''
        returns x axis index
        '''

        return self.xx

    
    def get_xx_centers(self):
        '''
        returns x axis index
        '''

        centers = (np.array(self.xx[:-1]) + np.array(self.xx[1:])) / 2
        if centers[0] == -np.inf:
            centers[0] = self.xx[1] - 10
        if centers[-1] == np.inf:
            centers[-1] = self.xx[-2] + 10
        return centers

    
    def scale_y(self, a):
        '''
        Scales y-axes of underlying bin models

        Interface to `self.model.scale_y(a)`
        '''

        self.model.scale_y(a)
        return self

    
    def scale_p(self, a):
        '''
        Scales p-values of underlying bin models (in log_p format)

        Interface to `self.model.scale_p`.
        '''

        self.model.scale_p(a)
        return self

    
    def filter_x(self, xx):
        '''
        Returns new :py:class:`~.models.bin_model.BinModel` 
        '''

        bins = []
        for x in xx:
            bins.append(self.xx.index(x) + 1)

        model = self.model.filter_x(bins)
        return BinModel(xx, model, scaled=self.model.scaled)

    
    def interpolate_x(self, newxx):
        '''
        Returns a copy of the model. *Does not interpolate.*
        '''

        return self.copy()

    
    def write_file(self, filename, delimiter):
        '''
        Write model as delimited document to filepath

        Wrapper around :py:meth:`~.models.bin_model.BinModel.write` method.

        Parameters
        ----------
        filename : str
            Path to file to be written

        delimiter : str
            Delimiter to use in file (e.g. '\t', ',')
        '''

        with open(filename, 'w') as fp:
            self.write(fp, delimiter)

    
    def write(self, file, delimiter):
        '''
        Write model as delimited document to file-like object

        Prepends model type (``bin1``) and bin borders (:py:attr:`~.models.bin_model.BinModel.xx`) to document written by ``self.model.write``.

        Parameters
        ----------
        file : object
            file-like object

        delimiter : str
            Delimiter to use in file (e.g. '\t', ',')
        '''

        file.write("bin1\n")
        file.write(delimiter.join(map(str, self.xx)) + "\n")
        self.model.write(file, delimiter)

    
    def get_bin_at(self, x):
        '''
        Returns bin containing value *x*

        Parameters
        ----------
        x : numeric
            Value to search for in binned axis

        Returns
        -------
        int
            Returns index of bin containing *x*. If bin is not contained in the bin range, returns ``-1``.
        '''

        for ii in range(len(self.xx)-1):
            if self.xx[ii] <= x and self.xx[ii+1] > x:
                return ii

        return -1

    
    def to_points_at(self, x, ys):
        return self.model.to_points_at(self.get_bin_at(x), ys)

    
    def eval_pval(self, x, p, threshold=1e-3):
        return self.model.eval_pval(self.get_bin_at(x), p, threshold)

    
    def cdf(self, x, y):
        return self.model.cdf(self.get_bin_at(x), y)

    
    def get_mean(self, x=None, index=None):
        if index is None:
            index = self.get_bin_at(x)
            if index == -1:
                return np.nan
        return self.model.get_mean(self.model.get_xx()[index])

    
    def get_sdev(self, x=None, index=None):
        if index is None:
            index = self.get_bin_at(x)
        return self.model.get_sdev(index)

    
    def draw_sample(self, x=None):
        return self.model.draw_sample(self.get_bin_at(x))

    
    def init_from_bin_file(self, file, delimiter, status_callback=None, init_submodel=lambda fp: None):
        line = string.strip(file.readline())
        if line != "bin1":
            raise ValueError("Unknown format: %s" % line)

        reader = csv.reader(file, delimiter=delimiter)
        row = next(reader)
        self.xx_text = row
        self.xx = list(map(float, row))
        self.xx_is_categorical = False

        self.model = init_submodel(file) # Need to set this!
        if self.model is not None:
            self.scaled = self.model.scaled

        return self

    
    def to_ddp(self, ys=None):
        newcats = []
        newxx = [self.xx[0]]
        for ii in range(1, len(self.xx)):
            newcats.extend([ii, ii])
            diff = (self.xx[ii] - self.xx[ii-1]) / 100.0
            newxx.extend([self.xx[ii] - diff, self.xx[ii] + diff])
        newxx[-1] = self.xx[-1]

        dupmodel = self.model.recategorize_x(newcats, list(range(1, len(newcats)+1)))
        dupmodel = dupmodel.to_ddp(ys)
        dupmodel.xx = newxx
        dupmodel.xx_text = list(map(str, newxx))
        dupmodel.xx_is_categorical = False

        return dupmodel

    ### Memoizable

    
    def get_edges(self):
        '''
        Returns bin edges (duplicate of :py:meth:`~.models.bin_model.BinModel.get_xx`)
        '''

        return self.xx

    
    def eval_pval_index(self, ii, p, threshold=1e-3):
        return self.model.eval_pval_index(ii, p, threshold)

    ### Class Methods

    @staticmethod
    def consistent_bins(models):
        '''
        All models are BinModels
        '''

        allxx = set()
        for model in models:
            if not model.scaled:
                raise ValueError("Only scaled distributions can be merged.")
            allxx.update(set(model.xx))

        allxx = np.array(sorted(allxx))
        midpts = (allxx[1:] + allxx[:-1]) / 2
        midpts[midpts == -np.inf] = min(allxx[allxx > -np.inf]) - 10.
        midpts[midpts == np.inf] = max(allxx[allxx < np.inf]) + 10.

        newmodels = []
        for model in models:
            allbins = [model.get_bin_at(x) for x in midpts]
            allxxs = [model.model.get_xx()[bin] if bin >= 0 else np.nan for bin in allbins]
            newmodel = model.model.recategorize_x(allxxs, list(range(0, len(allxx)-1)))
            newmodels.append(BinModel(allxx, newmodel))

        return newmodels

    @staticmethod
    def merge(models):
        '''
        All models are BinModels
        '''

        newmodels = BinModel.consistent_bins(models)

        allmodel = Model.merge([m.model for m in newmodels])

        model = BinModel(newmodels[0].get_xx(), allmodel)

        return model

    @staticmethod
    def combine(one, two):
        '''
        Both models are BinModels
        '''
        
        if not one.scaled or not two.scaled:
            raise ValueError("Cannot combine unscaled models")

        (one, two, xx) = UnivariateModel.intersect_x(one, two)

        allxx = set(one.xx) | set(two.xx)
        allxx = np.array(allxx)
        midpts = (allxx[1:] + allxx[:-1]) / 2
        onemodel = one.model.recategorize_x(list(map(model.get_bin_at, midpts)), list(range(1, len(allxx))))
        twomodel = two.model.recategorize_x(list(map(model.get_bin_at, midpts)), list(range(1, len(allxx))))

        model = Model.combine([onemodel, twomodel], [1, 1])

        return BinModel(allxx, model, True)

from .ddp_model import DDPModel

Model.mergers["bin_model"] = BinModel.merge
Model.mergers["bin_model+ddp_model"] = lambda models: DDPModel.merge([m.to_ddp() for m in models])
Model.mergers["bin_model+spline_model"] = lambda models: DDPModel.merge([m.to_ddp() for m in models])
Model.combiners['bin_model+bin_model'] = BinModel.combine
Model.combiners["bin_model+ddp_model"] = lambda one, two: DDPModel.combine(one.to_ddp(), two)
Model.combiners["bin_model+spline_model"] = lambda one, two: DDPModel.combine(one.to_ddp(), two)
