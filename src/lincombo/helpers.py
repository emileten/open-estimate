import numpy as np

def check_arguments(betas, stderrs, portions):
    portions = np.array(portions)
    stdvars = np.array([float(stderr) ** 2 for stderr in stderrs]) # in denom, so make sure float

    assert len(betas) == len(stdvars)
    assert portions.shape[0] == len(betas)

    return betas, stdvars, portions
