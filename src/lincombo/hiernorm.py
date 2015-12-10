import numpy as np
from multi_normal import MultivariateNormal
from multi_uniform import MultivariateUniform
from multi_sampled import MultivariateSampled
from multi_delta import MultivariateDelta
import pooling
import helpers

def alpha_given_taus(betas, stdvars, portions, obstaus):
    numalphas = portions.shape[1]

    ## Fill in the inverse sigma matrix
    invsigma = np.zeros((numalphas, numalphas))
    for jj in range(numalphas):
        for kk in range(numalphas):
            invsigma[jj, kk] = sum(portions[:, jj] * portions[:, kk] / (stdvars + obstaus**2))

    sigma = np.linalg.inv(invsigma)
    bb = [sum(betas * portions[:, jj] / (stdvars + obstaus**2)) for jj in range(numalphas)]
    alphahats = np.dot(sigma, np.transpose(bb))

    return MultivariateNormal(alphahats, sigma)

def betahat_given_taus(betas, stdvars, portions, obstaus):
    return MultivariateNormal(betas, np.diag(stdvars + obstaus**2))

def probability_tau(alphas, taus, obstaus, betas, stdvars, portions, probability_prior_taus):
    mv_betahat_given_taus = betahat_given_taus(betas, stdvars, portions, obstaus)
    mv_alpha_given_taus = alpha_given_taus(betas, stdvars, portions, obstaus)

    betahats = np.dot(portions, alphas)

    return probability_prior_taus.pdf(taus) * mv_betahat_given_taus.pdf(betahats) / mv_alpha_given_taus.pdf(alphas)

def sample_posterior(betas, stderrs, portions, taudist, taus2obstaus, draws=100):
    betas, stdvars, portions = helpers.check_arguments(betas, stderrs, portions)

    # Draw samples from posterior
    alltaus = []
    allalphas = []
    allbetahats = []
    for ii in range(draws):
        taus = np.ravel(taudist.rvs(size=1))
        alltaus.append(taus)

        obstaus = taus2obstaus(taus)

        # Sample from p(alphas | taus, betas)
        alphas = alpha_given_taus(betas, stdvars, portions, obstaus).rvs(size=1)
        allalphas.append(alphas)

        # Sample from p(betahat | taus, betas)
        betahats = betahat_given_taus(betas, stdvars, portions, obstaus).rvs(size=1)
        allbetahats.append(betahats)

    return alltaus, allalphas, allbetahats

def lincombo_hiernorm_taubyalpha(betas, stderrs, portions, maxtau=None, guess_range=False, draws=100):
    betas, stdvars, portions = helpers.check_arguments(betas, stderrs, portions)
    numalphas = portions.shape[1]

    print "Sampling taus..."

    observed_tau = 2 * np.sqrt(np.var(betas) + max(stderrs)**2)
    if observed_tau == 0:
        return None, None, None

    if maxtau is None:
        maxtau = pooling.estimated_maxtau(betas, stderrs, portions)
        print "Using maximum tau =", maxtau, "vs.", observed_tau

    if maxtau > 0:
        probability_prior_taus = MultivariateUniform([0] * numalphas, [maxtau] * numalphas)

        # Prepare to sample from from p(taus | betas)

        # Create pdf for p(taus | betas)
        def pdf(*taus): # taus is [tau1s, tau2s, ...]
            transtaus = np.transpose(taus) # [[tau1, tau2, ...], ...]
            values = []
            for ii in range(len(taus[0])):
                ## Observation taus
                obstaus = [sum(portions[ii, :] * transtaus[ii]) for ii in range(portions.shape[0])]

                # Requires alphas, but is invarient to them
                values.append(probability_tau([np.mean(betas)] * numalphas, transtaus[ii], obstaus, betas, stdvars, portions, probability_prior_taus))

            return values

        dist = MultivariateSampled(pdf, numalphas)
        if guess_range:
            mins, maxs = dist.guess_ranges([0] * numalphas, [maxtau] * numalphas, draws * 10)
        else:
            mins = [0] * numalphas
            maxs = [maxtau] * numalphas
        dist.prepare_draws(mins, maxs, count=draws)
    else:
        # maxtau == 0
        dist = MultivariateDelta(np.zeros(numalphas))

    print "Sampling alphas..."

    taus2obstaus = lambda taus: [sum(portions[ii, :] * taus) for ii in range(portions.shape[0])]
    return sample_posterior(betas, stderrs, portions, dist, taus2obstaus, draws)

def lincombo_hiernorm_taubybeta(betas, stderrs, portions, maxtaus=None, guess_range=False, draws=100):
    betas, stdvars, portions = helpers.check_arguments(betas, stderrs, portions)
    numalphas = portions.shape[1]

    print "Sampling taus..."

    observed_tau = 2 * np.sqrt(np.var(betas) + max(stderrs)**2)
    if observed_tau == 0:
        return None, None, None

    if maxtaus is None:
        maxtaus = pooling.estimated_maxlintaus(betas, stderrs, portions)
        print "Using maximum tau =", maxtaus, "vs.", observed_tau

    if maxtaus[0] > 0:
        probability_prior_taus = MultivariateUniform([0, 0], maxtaus)

        # Prepare to sample from from p(taus | betas)

        # Create pdf for p(taus | betas)
        def pdf(*taus): # taus is [tau0s, tau1s]
            transtaus = np.transpose(taus) # [[tau0, tau1], ...]
            values = []
            for ii in range(len(taus[0])):
                ## Observation taus
                obstaus = transtaus[ii][0] + transtaus[ii][1] * np.abs(np.array(betas))

                # Requires alphas, but is invarient to them
                values.append(probability_tau([np.mean(betas)] * numalphas, transtaus[ii], obstaus, betas, stdvars, portions, probability_prior_taus))

            return values

        dist = MultivariateSampled(pdf, 2)
        if guess_range:
            mins, maxs = dist.guess_ranges([0, 0], maxtaus, draws * 10)
        else:
            mins = [0, 0]
            maxs = maxtaus
        print mins, maxs
        dist.prepare_draws(mins, maxs, count=draws)
    else:
        # maxtau == 0
        dist = MultivariateDelta(np.zeros(2))

    print "Sampling alphas..."

    taus2obstaus = lambda taus: taus[0] + taus[1] * np.abs(np.array(betas))
    return sample_posterior(betas, stderrs, portions, dist, taus2obstaus, draws)

def get_sampled_column(allvals, col):
    return [allvals[ii][col] for ii in range(len(allvals))]

