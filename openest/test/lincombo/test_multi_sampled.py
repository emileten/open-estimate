import numpy as np
from ..lincombo.multi_sampled import *
from scipy.stats import multivariate_normal
from numpy.testing import *

def test_multi_sampled():
    dims = 3
    means = np.random.normal(size=dims)
    sigmas = np.random.uniform(size=dims)

    def pdf(*xxs):
        return multivariate_normal.pdf(np.transpose(xxs), means, np.diag(sigmas ** 2))

    dist = MultivariateSampled(pdf, dims)
    mins, maxs = dist.guess_ranges([-100] * dims, [100] * dims)
    print np.vstack((means, sigmas, mins, maxs))

    # Check that got 3 stddevs
    assert np.all(mins < means - sigmas * 3)
    assert np.all(maxs > means + sigmas * 3)

    dist.prepare_draws(mins, maxs, count=100000)
    values = dist.rvs(size=10000)

    for ii in range(dims):
        assert_almost_equal(means[ii], np.mean(values[ii]), 1)
        assert_almost_equal(sigmas[ii], np.std(values[ii]), 1)


