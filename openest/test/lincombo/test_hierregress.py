import numpy as np
from openest.lincombo.hierregress import *
import statsmodels.api as sm
from numpy.testing import *

#lincombo_hierregress = lincombo_hierregress_taubybeta
lincombo_hierregress = lincombo_hierregress_taubymu

def setup_8schools():
    yy = [28,  8, -3,  7, -1,  1, 18, 12]
    sigma = [15, 10, 16, 11,  9, 11, 10, 18]
    XX = np.ones((8, 1))

    return yy, sigma, XX

def test_simple_8schools():
    yy, sigma, XX = setup_8schools()
    draws = 100

    alltau, allmus, allbetahats = lincombo_hierregress(yy, sigma, XX, draws=draws)

    assert(np.mean(allmus) < np.mean(yy) + 2 and np.mean(allmus) > np.mean(yy) - 2)

def test_invarient_assumption():
    yy, sigma, XX = setup_8schools()
    yy, stdvars, XX = helpers.check_arguments(yy, sigma, XX)
    probability_prior_tau = uniform(0, 10.)
    probability_prior_mus = MultivariateUniform([-3], [28])

    meanmu = sm.WLS(yy, XX, weights=1/stdvars).fit().params

    tau = probability_prior_tau.rvs()

    prob_check = None
    for ii in range(10):
        mus = probability_prior_mus.rvs()
        prob = probability_tau(meanmu, tau, yy, stdvars, XX, probability_prior_tau)

        if prob_check is None:
            prob_check = prob
        else:
            assert_almost_equal(prob, prob_check)

def setup_trend(N):
    tt = np.linspace(0, 100, N)
    yy = 3 + 2*tt + np.random.normal(0, 1, N)
    sigma = 10 * np.ones(N)

    XX = np.hstack((np.ones((N, 1)), np.transpose(np.matrix(tt))))

    return yy, sigma, XX

def test_trend():
    yy, sigma, XX = setup_trend(101)
    alltau, allmus, allbetahats = lincombo_hierregress(yy, sigma, XX)

    assert(sum(np.array(alltau)[0] == np.array(alltau)) < len(alltau) / 2)
    assert_almost_equal(np.mean(get_sampled_column(allmus, 0)), 3, 0)
    assert_almost_equal(np.mean(get_sampled_column(allmus, 1)), 2, 1)
