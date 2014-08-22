import numpy as np
from scipy import stats
import emcee

sizes = [919, 301, 180, 295, 266, 597, 1406, 29315, 2546]
effects = [34.4 - 36, 4.7 - 5, 3.2 - 5.2, 4.1 - 5.2, 4.2 - 4.6, 13.5 - 18, 5.6 - 5.9, 4.9 - 5.7, 5.5 - 5.6]

def lnprob(x, sizes, effects):
    mu = x[0]
    tau = x[1]
    sigma = x[2]
    thetas = x[3:]
    
    if tau <= 0 or sigma <= 0:
        return -np.Inf
    
    lp = np.sum(stats.norm.logpdf(thetas, mu, tau))
    lp += np.sum(stats.t.logpdf(effects, sizes, thetas, sigma / np.sqrt(sizes)))
    
    return lp

ndim, nwalkers = len(sizes) + 3, len(sizes) * 4 + 10

effects_mean = np.mean(effects)
effects_sigma = np.std(effects)

mus = np.random.normal(effects_mean, effects_sigma, (nwalkers, 1))
taus = np.random.rand(nwalkers, 1) * effects_sigma
sigmas = np.random.rand(nwalkers, 1) * effects_sigma
p0 = np.hstack((mus, taus, sigmas))
for ii in range(len(sizes)):
    theta = np.random.normal(effects[ii], effects_sigma, (nwalkers, 1))
    p0 = np.hstack((p0, theta))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[sizes, effects])
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()

sampler.run_mcmc(pos, 1000)

np.percentile(sampler.flatchain[:,1], [25, 50, 75])
