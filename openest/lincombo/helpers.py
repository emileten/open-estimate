import numpy as np
from scipy import sparse

def issparse(portions):
    return sparse.issparse(portions)

def check_arguments(betas, stderrs, portions):
    if not issparse(portions):
        portions = np.array(portions) # may already be np.array, but okay

    stdvars = np.array([float(stderr) ** 2 for stderr in stderrs]) # in denom, so make sure float

    assert len(betas) == len(stdvars)
    assert portions.shape[0] == len(betas)

    return betas, stdvars, portions
