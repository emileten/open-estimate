import numpy as np
from openest.lincombo.continuous_sampled import *
from scipy.stats import norm
from numpy.testing import *

def test_continuous_sampled():
    mean = np.random.normal(1)
    sigma = np.random.uniform(1)

    def pdf(*xxs):
        return norm.pdf(xxs, mean, sigma)

    dist = ContinuousSampled(pdf)
    mini, maxi = dist.guess_ranges(-100, 100)
    print(np.vstack((mean, sigma, mini, maxi)))

    # Check that got 3 stddevs
    assert np.all(mini < mean - sigma * 3)
    assert np.all(maxi > mean + sigma * 3)

    dist.prepare_draws(mini, maxi, count=100000)
    values = dist.rvs(size=10000)

    assert_almost_equal(mean, np.mean(values), 1)
    assert_almost_equal(sigma, np.std(values), 1)


