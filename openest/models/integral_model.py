# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, The Open Aggregator
#   GNU General Public License, Ver. 3 (see docs/license.txt)
################################################################################

"""Integral model

The integral over x of another model.
"""
__copyright__ = "Copyright 2014, The Open Aggregator"
__license__ = "GPL"

__author__ = "James Rising"
__credits__ = ["James Rising"]
__maintainer__ = "James Rising"
__email__ = "jar2234@columbia.edu"

__status__ = "Production"
__version__ = "$Revision$"
# $Source$

import csv, string
import numpy as np

from .model import Model
from .univariate_model import UnivariateModel
from .bin_model import BinModel

class IntegralModel(UnivariateModel):
    def __init__(self, model=None):
        super(IntegralModel, self).__init__(False, model.xx, model.scaled)
        
        self.model = model

    def kind(self):
        return 'integral_model'

    def copy(self):
        return IntegralModel(list(self.xx), self.model.copy())

    def get_xx(self):
        return self.xx

    def scale_y(self, a):
        self.model.scale_y(a)
        return self

    def scale_p(self, a):
        self.model.scale_p(a)
        return self

    # Do nothing in interpolation
    def interpolate_x(self, newxx):
        return self.copy()

    def write_file(self, filename, delimiter):
        with open(filename, 'w') as fp:
            self.write(fp, delimiter)

    def write(self, file, delimiter):
        file.write("int1\n")
        self.model.write(file, delimiter)

    def eval_pval(self, x, p, threshold=1e-3):
        if isinstance(self.model, BinModel):
            if x < self.model.xx[0]:
                return 0
            
            integral = 0
            for ii in range(len(self.model.xx)-1):
                if x < self.model.xx[ii+1]:
                    integral += self.model.eval_pval(x, p, threshold) * (x - self.model.xx[ii])
                    return integral
                else:
                    integral += self.model.eval_pval((self.model.xx[ii+1] + self.model.xx[ii]) / 2, p, threshold) * (self.model.xx[ii+1] - self.model.xx[ii])

            return integral
