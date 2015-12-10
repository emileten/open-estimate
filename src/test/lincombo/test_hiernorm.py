from lincombo.hiernorm import *
from numpy.testing import *

lincombo_hiernorm = lincombo_hiernorm_taubybeta
#lincombo_hiernorm = lincombo_hiernorm_taubyalpha

def test_tau_no_alpha1():
    """One requirement of the simplified method we use is that the expression p(tau | y) does not depend on alpha.  Check that this is the case."""
    betas, stdvars, portions = helpers.check_arguments([0, 1, 0], [1, 1, 10], [[1, 0], [.5, .5], [0, 1]])
    probability_prior_taus = MultivariateUniform([0, 0], [1e6, 1e6])
    probability_prior_alphas = MultivariateUniform([-100, 100], [100, 100])

    taus = probability_prior_taus.rvs()

    prob_check = None
    for ii in range(10):
        alphas = probability_prior_alphas.rvs()
        prob = probability_tau(alphas, taus, betas, stdvars, portions, probability_prior_taus)

        if prob_check is None:
            prob_check = prob
        else:
            assert_almost_equal(prob, prob_check)

def test_disagg():
    alltaus, allalphas, allbetahats = lincombo_hiernorm([4, 7, 2], [1, 3, 1], [[1, 0], [0, 1], [1, 0]], draws=200)
    assert_almost_equal(np.mean(get_sampled_column(allalphas, 0)), 3, decimal=0)
    assert_almost_equal(np.mean(get_sampled_column(allalphas, 1)), 7, decimal=0)

def test_single():
    alltaus, allalphas, allbetahats = lincombo_hiernorm([0, 1, 0], [1, 1, 2], [[1], [1], [1]])
    assert_almost_equal(np.mean(allalphas), 4. / 9., decimal=0)

def test_other(draws=100):
    alltaus, allalphas, allbetahats = lincombo_hiernorm([0, 1], [1, 1], [[1, 0], [.5, .5]], draws=draws)
    #alltaus, allalphas, allbetahats = lincombo_hiernorm([0, 1], [1, 1], [[1, 0], [.5, .5]], maxtau=0, draws=draws)
    assert_almost_equal(np.mean(allalphas, axis=0), [0, 2], decimal=0)

    return alltaus, allalphas, allbetahats

#def test_healthlike(draws=100):
# alpha0 = 0, alpha1 = 1
#alltaus, allalphas, allbetahats = lincombo_hiernorm([.5, 1], [.5, 1], [[.5, .5], [0, 1]], draws=draws)
#print np.mean(allalphas, axis=0)
#print np.std(allalphas, axis=0)
#assert_almost_equal(np.mean(allalphas, axis=0), [0, 1], decimal=0)

def test_8schools():
    betas = [28,  8, -3,  7, -1,  1, 18, 12]
    stderrs = [15, 10, 16, 11,  9, 11, 10, 18]
    portions = np.eye(8)
    draws = 100

    alltaus, allalphas, allbetahats = lincombo_hiernorm(betas, stderrs, portions, draws=draws)

    for ii in range(portions.shape[1]):
        print np.mean(get_sampled_column(allalphas, ii))

if __name__ == '__main__':
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()

    alltaus, allalphas, allbetahats = test_other(draws=1000)

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

    print np.mean(allalphas, axis=0)
    print np.mean(alltaus, axis=0)
    print np.mean(allbetahats, axis=0)
