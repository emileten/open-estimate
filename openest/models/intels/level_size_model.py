# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, The Open Aggregator
#   GNU General Public License, Ver. 3 (see docs/license.txt)
################################################################################

"""Level-Size Model

Like a mean-size model, it's describing a level change, where both the
hyperdistribution of levels is normal, and a single universal
population is being drawn from.
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

from mean_size_model import MeanSizeModel
from hierarchical_model import HierarchicalModel

class LevelSizeModel(MeanSizeModel, HierarchicalModel):
    def __init__(self, xx_is_categorical=False, xx=None, means=None, sizes=None):
        super(LevelSizeModel, self).__init(xx_is_categorical, xx, means, sizes)

    def valid_ranges(x=None):
        """Returns a dictionary of {var: (lo, hi)} for knowns and unknowns."""
        index = self.get_xx().index(x)
        return {'obs_dlevel[]': self.means[index],
                'true_dlevel[]': (-np.inf, np.inf),
                'sigma_tdl[]': (0, np.inf),
                'obs_count[]': self.sizes[index],
                'mu_dlevel': (-np.inf, np.inf),
                'sigma_dlevel': (0, np.inf),
                'pop_var': (0, np.inf)}

    def eval_lp(env, x=None):
        # Refers to locals through env, in case this is a subclass
        # y_i ~ N(theta_i, tau_i)
        lp = stats.norm.logpdf(env['obs_dlevel[]'], env['true_dlevel[]'], env['sigma_tdl[]'])
        # theta_i ~ N(mu_y, sigma_y)
        lp += stats.norm.logpdf(env['true_dlevel[]'], env['mu_dlevel'], env['sigma_dlevel'])
        # (n - 1) (sqrt(n) tau_i)^2 / sigma_pop ~ Chi^2_n-1
        # From definition of standard error and distribution of sample variance
        n = self.sizes[index]
        lp += stats.chi2.logpdf((n - 1)*n*(env['sigma_tdl[]'] / env['pop_var'])**2, n - 1)

        return lp

