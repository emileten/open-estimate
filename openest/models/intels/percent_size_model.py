# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, The Open Aggregator
#   GNU General Public License, Ver. 3 (see docs/license.txt)
################################################################################

"""Percent-Size Model

Like a mean-size model, it's describing a percent change, where both
the hyperdistribution of percent changes is normal, and the initial
value upon which the level changes is normal.
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

from level_size_model import LevelSizeModel

class PercentSizeModel(LevelSizeModel):
    def __init__(self, xx_is_categorical=False, xx=None, means=None, sizes=None):
        super(PercentSizeModel, self).__init(xx_is_categorical, xx, means, sizes)

    def valid_ranges(x=None):
        """Returns a dictionary of {var: (lo, hi)} for knowns and unknowns."""
        ranges = super(PercentSizeModel, self).valid_ranges(x)
        ranges['obs_dlevel[]'] = (-np.inf, np.inf) # Need to predict this
        ranges['mu_level'] = (-np.inf, np.inf)
        ranges['sigma_level'] = (0, np.inf)

        return ranges

    def eval_lp(env, x=None):
        lp = super(PercentSizeModel, self).eval_lp(env, x)
        
        index = self.get_xx().index(x)
        # 100 * dlevel / level = % => (100 * dlevel / %) ~ N(mu_z, sigma_z)
        lp += stats.norm.logpdf(100 * env['obs_dlevel[]'] / self.means[index], env['mu_level'], env['sigma_level'])

        return lp
