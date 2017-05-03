# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, Distributed Meta-Analysis System
################################################################################
"""Helper functions"""

__copyright__ = "Copyright 2014, Distributed Meta-Analysis System"

__author__ = "James Rising"
__credits__ = ["James Rising"]
__maintainer__ = "James Rising"
__email__ = "jar2234@columbia.edu"

__status__ = "Production"
__version__ = "$Revision$"
# $Source$

import numpy as np
from scipy import sparse

def issparse(portions):
    """Check if an array is sparse."""
    
    return sparse.issparse(portions)

def check_arguments(betas, stderrs, portions):
    """Ensure that the parameters have the right dimensions for calculation."""

    if not issparse(portions):
        portions = np.array(portions) # may already be np.array, but okay

    stdvars = np.array([float(stderr) ** 2 for stderr in stderrs]) # in denom, so make sure float

    assert len(betas) == len(stdvars)
    assert portions.shape[0] == len(betas) # so we can later do betas * portions

    return betas, stdvars, portions
