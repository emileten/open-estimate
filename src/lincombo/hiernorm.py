import numpy as np
from multi_normal import MultivariateNormal
from multi_uniform import MultivariateUniform
from multi_sampled import MultivariateSampled
import helpers

def alpha_given_taus(betas, stdvars, portions, taus):
    numalphas = portions.shape[1]

    ## Observation taus
    obstaus = [sum(portions[ii, :] * taus) for ii in range(portions.shape[0])]

    ## Fill in the inverse sigma matrix
    invsigma = np.zeros((numalphas, numalphas))
    for jj in range(numalphas):
        for kk in range(numalphas):
            invsigma[jj, kk] = sum(portions[:, jj] * portions[:, kk] / (stdvars + obstaus))

    sigma = np.linalg.inv(invsigma)
    bb = [sum(betas * portions[:, jj] / (stdvars + obstaus)) for jj in range(numalphas)]
    alphahats = np.dot(sigma, np.transpose(bb))

    return MultivariateNormal(alphahats, sigma)

def betahat_given_taus(betas, stdvars, portions, taus):
    ## Observation taus
    obstaus = [sum(portions[ii, :] * taus) for ii in range(portions.shape[0])]

    return MultivariateNormal(betas, obstaus)

def probability_tau(alphas, taus, betas, stdvars, portions, probability_prior_taus):
    mv_betahat_given_taus = betahat_given_taus(betas, stdvars, portions, taus)
    mv_alpha_given_taus = alpha_given_taus(betas, stdvars, portions, taus)

    betahats = np.dot(portions, alphas)

    return probability_prior_taus.pdf(taus) * mv_betahat_given_taus.pdf(betahats) / mv_alpha_given_taus.pdf(alphas)

def lincombo_hiernorm(betas, stderrs, portions, maxtau=None, guess_range=False, draws=100):
    betas, stdvars, portions = helpers.check_arguments(betas, stderrs, portions)
    numalphas = portions.shape[1]

    print "Sampling taus..."

    if maxtau is None:
        maxtau = 2 * np.sqrt(np.var(betas) + max(stderrs)**2)
        print "Using maximum tau =", maxtau

    probability_prior_taus = MultivariateUniform([0] * numalphas, [maxtau] * numalphas)

    # Prepare to sample from from p(taus | betas)

    # Create pdf for p(taus | betas)
    def pdf(*taus): # taus is [tau1s, tau2s, ...]
        transtaus = np.transpose(taus) # [[tau1, tau2, ...], ...]
        values = []
        for ii in range(len(taus[0])):
            # Requires alphas, but is invarient to them
            values.append(probability_tau([np.mean(betas)] * numalphas, transtaus[ii], betas, stdvars, portions, probability_prior_taus))

        return values

    dist = MultivariateSampled(pdf, numalphas)
    if guess_range:
        mins, maxs = dist.guess_ranges([0] * numalphas, [maxtau] * numalphas, draws * 10)
    else:
        mins = [0] * numalphas
        maxs = [maxtau] * numalphas
    dist.prepare_draws(mins, maxs, count=draws)

    print "Sampling alphas..."

    # Draw samples from posterior
    alltaus = []
    allalphas = []
    allbetahats = []
    for ii in range(draws):
        taus = np.ravel(dist.rvs(size=1))
        alltaus.append(taus)

        # Sample from p(alphas | taus, betas)
        alphas = alpha_given_taus(betas, stdvars, portions, taus).rvs(size=1)
        allalphas.append(alphas)

        # Sample from p(betahat | taus, betas)
        betahats = betahat_given_taus(betas, stdvars, portions, taus).rvs(size=1)
        allbetahats.append(betahats)

    return alltaus, allalphas, allbetahats

def get_sampled_column(allvals, col):
    return [allvals[ii][col] for ii in range(len(allvals))]
