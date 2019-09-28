# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, Distributed Meta-Analysis System
################################################################################

"""Allows models to define a collection of variables which they inform.

Convention: if a variable is "local" to the model (defined only by
this model), it should end in '[]'.
"""

__copyright__ = "Copyright 2014, Distributed Meta-Analysis System"

__author__ = "James Rising"
__credits__ = ["James Rising", "Solomon Hsiang"]
__maintainer__ = "James Rising"
__email__ = "jar2234@columbia.edu"

__status__ = "Production"
__version__ = "$Revision$"
# $Source$

class HierarchicalModel(object):
    def valid_ranges(x=None):
        """Returns a dictionary of {var: (lo, hi)} for knowns and unknowns."""
        return {}

    def eval_lp(env, x=None):
        return 0
