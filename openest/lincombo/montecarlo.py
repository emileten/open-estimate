import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from .multi_draws import MultivariateDraws

def regress_draws(means, serrs, XX, count=1000):
    print("Making MC draws...")
    betas = []
    for ii in range(count):
        yy = norm.rvs(means, serrs, 1)
        betas.append(sm.OLS(yy, XX).fit().params)

    return np.array(betas)

def regress_summary(means, serrs, XX, count=1000):
    betas = regress_draws(means, serrs, XX, count=count)
    beta_mean = np.mean(betas, axis=0)
    beta_serr = np.std(betas, axis=0)

    return beta_mean, beta_serr

def regress_distribution(means, serrs, XX, count=1000):
    betas = regress_draws(means, serrs, XX, count=count)
    return MultivariateDraws(betas)

