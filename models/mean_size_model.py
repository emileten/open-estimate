# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, The Open Aggregator
#   GNU General Public License, Ver. 3 (see docs/license.txt)
################################################################################

"""Mean-Size Model

In Mean-Size models, each point is characterized only by a value and the population size that went into estimating that value.  As such, it does not have enough information to generate a full distribution.  It can be safely combined with other mean-size models, or approximated with a Gaussian (with a variance which is equal to the absolute value of the mean for size = 1, and a variance that decreases with the square root of the size, according to the Central Limit Theorem).

The format is::

  msx1,mean,size
  <x0>,<mean0>,<size0>
  <x1>,<mean1>,<size1>
  ...
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

import math
from scipy.interpolate import interp1d
import numpy as np

from model import Model
from univariate_model import UnivariateModel
from attribute import Attribute

class MeanSizeModel(UnivariateModel):
    def __init__(self, xx_is_categorical=False, xx=None, means=None, sizes=None):
        super(MeanSizeModel, self).__init__(xx_is_categorical, xx, True)

        self.means = means
        self.sizes = sizes

    def kind(self):
        return 'mean_size_model'

    def copy(self):
        return MeanSizeModel(self.xx_is_categorical, list(self.get_xx()), list(self.means), list(self.sizes))

    def scale_y(self, a):
        self.means = map(lambda m: m * a, self.means)
        return self

    def scale_p(self, a):
        self.sizes = map(lambda s: s * a, self.sizes)
        return self

    def get_mean(self, x=None):
        return self.means[self.get_xx().index(x)]

    def get_sdev(self, x=None):
        index = self.get_xx().index(x)
        return abs(self.means[index]) / math.sqrt(self.sizes[index])
    
    def filter_x(self, xx):
        return MeanSizeModel(self.xx_is_categorical, xx, map(lambda x: self.means[xx.index(x)], xx), map(lambda x: self.sizes[xx.index(x)], xx))

    def interpolate_x(self, newxx, kind='quadratic'):
        fx = interp1d(self.xx, self.means, kind)
        means = fx(newxx)
        fx = interp1d(self.xx, self.sizes, kind)
        sizes = fx(newxx)
        
        return MeanSizeModel(self.xx_is_categorical, newxx, means, sizes)

    def attribute_list(self):
        return ["sizes"]

    def get_attribute(self, title):
        if title == "sizes":
            if len(self.sizes) == 1:
                return Attribute("Sample Sizes", None, None, None, self.sizes, None, None)
            else:
                return [Attribute("Sample Sizes", None, None, self.get_xx()[ii], self.sizes[ii], None, None) for ii in range(len(self.sizes))]
        
        raise title + " not available"

    def write_file(self, filename, delimiter):
        with open(filename, 'w') as fp:
            self.write(fp, delimiter)

    def write(self, file, delimiter):
        file.write("msx1" + "\n")
        for ii in range(len(self.xx)):
            file.write(delimiter.join(map(str, [self.get_xx()[ii], self.means[ii], self.sizes[ii]])) + "\n")

    def init_from_mean_size_file(self, file, delimiter, status_callback=None):
        reader = csv.reader(file, delimiter=delimiter)
        header = reader.next()
        if header[0] != "msx1":
            raise ValueError("Unknown format: %s" % (fields[0]))

        xx_text = []
        xx = []
        means = []
        sizes = []
        self.xx_is_categorical = False
        for row in reader:
            xx_text.append(row[0])
            try:
                xx.append(float(row[0]))
            except ValueError:
                xx.append(len(xx))
                self.xx_is_categorical = True

            self.means.append(float(row[1]))
            self.sizes.append(float(row[2]))

        self.xx = xx
        self.xx_text = xx_text
        self.means = means
        self.sizes = sizes

    @staticmethod
    def merge(models, treatment="default"):
        if treatment == 'default':
            if 'treated' in models[0].get_xx() and 'control' in models[0].get_xx():
                treatment = 'treated-even'
            else:
                treatment = 'independent'
        
        if treatment == "independent":
            # Collect the union of all x values
            masterxx = set()
            for model in models:
                masterxx.update(model.get_xx())
            masterxx = list(masterxx)
            
            means = []
            sizes = []
            for ii in range(len(masterxx)):
                print masterxx[ii]
                numersum = 0
                denomsum = 0
                for model in models:
                    xx = model.get_xx()
                    try:
                        jj = xx.index(masterxx[ii])
                        numersum += model.means[jj] * model.sizes[jj]
                        denomsum += model.sizes[jj]
                    except:
                        pass

                print numersum, denomsum
                means.append(numersum / float(denomsum))
                sizes.append(denomsum)

            return MeanSizeModel(models[0].xx_is_categorical, masterxx, means, sizes)
        else:
            # All need to have the same x-values
            numersum = 0
            denomsum = 0
            for model in models:
                ii = model.get_xx().index('treated')
                jj = model.get_xx().index('control')
                if treatment == 'treated-even':
                    avgsize = 1
                else:
                    avgsize = (model.sizes[ii] + model.sizes[jj]) / 2
                numersum += (model.means[ii] - model.means[jj]) * avgsize
                denomsum += avgsize

            means = [numersum / float(denomsum)]
            sizes = [denomsum]
            
            return MeanSizeModel(models[0].xx_is_categorical, ["difference"], means, sizes)
    
        
    @staticmethod
    def combine(one, two):
        if one.xx_is_categorical != two.xx_is_categorical:
            raise ValueError("Cannot combine models that do not agree on categoricity")

        (one, two, xx) = UnivariateModel.intersect_x(one, two)

        means = np.array(one.means) + np.array(two.means)
        sizes = 1 / (1 / np.sqrt(np.array(one.sizes)) + 1 / np.sqrt(np.array(two.means)))**2

        return MeanSizeModel(one.xx_is_categorical, xx, means, sizes)

Model.mergers["mean_size_model"] = MeanSizeModel.merge
Model.combiners['mean_size_model+mean_size_model'] = MeanSizeModel.combine
