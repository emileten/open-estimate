import numpy as np
from multi_normal import MultivariateNormal
import helpers
import time # XXX

def sum_multiply(sparsecol, densevec):
    indices = sparsecol.nonzero()[0] # just take rows
    total = 0
    for index in indices:
        total += sparsecol[index, 0] * densevec[index]
    return total

def sum_multiply2(sparse, col1, col2, densevec):
    rows1 = sparse.indices[sparse.indptr[col1]:sparse.indptr[col1+1]]
    rows2 = sparse.indices[sparse.indptr[col2]:sparse.indptr[col2+1]]
    # Can improve by construct 'matches': list of (index1, index2) which are the same row, then using to index into data
    rows = set(rows1).intersection(rows2)
    total = 0
    for row in rows:
        total += sparse[row, col1] * sparse[row, col2] * densevec[row]
    return total

def lincombo_pooled(betas, stderrs, portions):
    betas, stdvars, portions = helpers.check_arguments(betas, stderrs, portions)

    numalphas = portions.shape[1]

    ## Fill in the inverse sigma matrix
    overvars = 1.0 / stdvars
    invsigma = np.zeros((numalphas, numalphas))
    for jj in range(numalphas):
        print "Row", jj
        for kk in range(numalphas):
            if helpers.issparse(portions):
                invsigma[jj, kk] = sum_multiply2(portions, jj, kk, overvars)
            else:
                invsigma[jj, kk] = sum(portions[:, jj] * portions[:, kk] * overvars)

    if helpers.issparse(portions):
        bb = [sum_multiply(portions[:, jj], betas * overvars) for jj in range(numalphas)]
    else:
        bb = [sum(betas * portions[:, jj] / stdvars) for jj in range(numalphas)]

    print "Begin inversion...", invsigma.shape
    sigma = np.linalg.inv(invsigma)
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
