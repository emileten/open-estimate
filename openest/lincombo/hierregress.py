import numpy as np
from scipy.stats import uniform
from multi_normal import MultivariateNormal
from multi_uniform import MultivariateUniform
from continuous_sampled import ContinuousSampled
from multi_delta import MultivariateDelta
import pooling
import helpers

mytaudist = None

def mu_given_tau(yy, stdvars, XX, tau):
    nummus = XX.shape[1]

    ## Fill in the inverse sigma matrix
    invsigma = np.zeros((nummus, nummus))
    for jj in range(nummus):
        for kk in range(nummus):
            invsigma[jj, kk] = sum(XX[:, jj] * XX[:, kk] / (stdvars + tau**2))

    sigma = np.linalg.inv(invsigma)
    bb = [sum(yy * XX[:, jj] / (stdvars + tau**2)) for jj in range(nummus)]
    muhats = np.dot(sigma, np.transpose(bb))

    return MultivariateNormal(muhats, sigma)

def betahat_given_tau(yy, stdvars, XX, tau):
    return MultivariateNormal(yy, np.diag(stdvars + tau**2))

def probability_tau(mus, tau, yy, stdvars, XX, probability_prior_tau):
    mv_betahat_given_tau = betahat_given_tau(yy, stdvars, XX, tau)
    mv_mu_given_tau = mu_given_tau(yy, stdvars, XX, tau)

    betahats = np.dot(XX, np.transpose(np.matrix(mus)))

    return np.exp(probability_prior_tau.logpdf(tau) + mv_betahat_given_tau.logpdf(np.transpose(betahats)) - mv_mu_given_tau.logpdf(mus))

def sample_posterior(yy, stderrs, XX, taudist, draws=100):
    yy, stdvars, XX = helpers.check_arguments(yy, stderrs, XX)

    # Draw samples from posterior
    alltau = []
    allmus = []
    allbetahats = []
    for ii in range(draws):
        tau = taudist.rvs(size=1)
        alltau.append(tau)

        # Sample from p(mus | tau, yy)
        mus = mu_given_tau(yy, stdvars, XX, tau).rvs(size=1)
        allmus.append(mus)

        # Sample from p(betahat | tau, yy)
        betahats = betahat_given_tau(yy, stdvars, XX, tau).rvs(size=1)
        allbetahats.append(betahats)

    return alltau, allmus, allbetahats

def lincombo_hierregress_taubymu(yy, stderrs, XX, maxtau=None, guess_range=False, draws=100):
    yy, stdvars, XX = helpers.check_arguments(yy, stderrs, XX)
    nummus = XX.shape[1]

    print "Sampling tau..."

    if maxtau is None:
        maxtau = pooling.estimated_maxtau(yy, stderrs, XX)
        print "Using maximum tau =", maxtau

    if maxtau > 0:
        probability_prior_tau = uniform(0, maxtau)

        # Prepare to sample from from p(tau | yy)

        # Create pdf for p(tau | yy)
        def pdf(tau):
            # Requires mus, but is invarient to them
            return probability_tau([1] * nummus, tau, yy, stdvars, XX, probability_prior_tau)

        dist = ContinuousSampled(pdf)
        if guess_range:
            mini, maxi = dist.guess_ranges(0, maxtau, draws * 10)
        else:
            mini, maxi = 0, maxtau
        dist.prepare_draws(mini, maxi, count=draws)
    else:
        # maxtau == 0
        dist = MultivariateDelta(np.zeros(nummus))

    print "Sampling mus..."

    return sample_posterior(yy, stderrs, XX, dist, draws)

def lincombo_hierregress_taubybeta(yy, stderrs, XX, maxtau=None, guess_range=False, draws=100):
    yy, stdvars, XX = helpers.check_arguments(yy, stderrs, XX)
    nummus = XX.shape[1]

    print "Sampling tau..."

    if maxtau is None:
        maxtau = pooling.estimated_maxlintau(yy, stderrs, XX)
        print "Using maximum tau =", maxtau

    if maxtau[0] > 0:
        probability_prior_tau = uniform(0, maxtau)

        # Prepare to sample from from p(tau | yy)

        # Create pdf for p(tau | yy)
        def pdf(tau):
            # Requires mus, but is invarient to them
            return probability_tau([np.mean(yy)] * nummus, tau, yy, stdvars, XX, probability_prior_tau)

        dist = ContinuousSampled(pdf, 2)
        if guess_range:
            mini, maxi = dist.guess_ranges([0, 0], maxtau, draws * 10)
        else:
            mini, maxi = 0, maxtau
        dist.prepare_draws(mini, maxi, count=draws)
    else:
        # maxtau == 0
        dist = MultivariateDelta(np.zeros(2))

    print "Sampling mus..."

    return sample_posterior(yy, stderrs, XX, dist, draws)

def get_sampled_column(allvals, col):
    return [allvals[ii][col] for ii in range(len(allvals))]

lincombo_hierregress = lincombo_hierregress_taubymu
