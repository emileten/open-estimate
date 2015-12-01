import numpy as np
from multi_normal import MultivariateNormal
import helpers

def lincombo_pooled(betas, stderrs, portions):
    betas, stdvars, portions = helpers.check_arguments(betas, stderrs, portions)

    numalphas = portions.shape[1]

    ## Fill in the inverse sigma matrix
    invsigma = np.zeros((numalphas, numalphas))
    for jj in range(numalphas):
        for kk in range(numalphas):
            invsigma[jj, kk] = sum(portions[:, jj] * portions[:, kk] / stdvars)

    sigma = np.linalg.inv(invsigma)
    bb = [sum(betas * portions[:, jj] / stdvars) for jj in range(numalphas)]
    alphas = np.dot(sigma, np.transpose(bb))

    return MultivariateNormal(alphas, sigma)

