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

def estimated_maxtau(betas, stderrs, portions):
    """For use with hiernorm by-alpha."""
    mv = lincombo_pooled(betas, stderrs, portions)
    poolbetas = []
    for ii in range(portions.shape[0]):
        poolbetas.append(sum(portions[ii, :] * mv.means))

    # If all of this is attributed to tau, how big would tau be?
    return np.std(np.array(betas) - np.array(poolbetas))

def estimated_maxlintaus(betas, stderrs, portions):
    """For use with hiernorm by-beta."""
    mv = lincombo_pooled(betas, stderrs, portions)
    poolbetas = []
    for ii in range(portions.shape[0]):
        poolbetas.append(sum(portions[ii, :] * mv.means))

    betas = np.array(betas)

    # If all of this is attributed to tau, how big would tau be?
    tau0 = np.std(betas - np.array(poolbetas))

    # If tau0 where 0, how big would this be?
    tau1 = np.std((betas - np.array(poolbetas)) / betas)

    if np.isnan(tau1) or tau1 > tau0:
        tau1 = tau0

    return [tau0, tau1]
