import math, random
import numpy as np
from scipy.stats import norm, uniform
from scipy.interpolate import interp1d

def helper_params(means, varis, tau):
    vari_tau_sqrs = varis + tau*tau
    terms = 1.0 / vari_tau_sqrs
    v_mu = 1.0 / sum(terms)
    mu_hat = sum(terms * means) / sum(terms)

    return (vari_tau_sqrs, v_mu, mu_hat)

def p_tau_given_y(tau, means, varis):
    (vari_tau_sqrs, v_mu, mu_hat) = helper_params(means, varis, tau)
    return math.sqrt(v_mu) * np.prod((np.exp(-np.square(means - mu_hat) / (2 * vari_tau_sqrs)) / np.sqrt(vari_tau_sqrs)).astype(np.float_))

#from scipy.spatial import KDTree
#def get_random_index(F_tau, count):
#    kdtree = KDTree([np.transpose(F_tau)])
#    rands = uniform.rvs(count, loc=0, scale=1)
#    (dists, locs) = kdtree.query(rands, k=1)
#    return locs

def get_random(taus, F_tau, count):
    func = interp1d(np.concatenate(([0], F_tau, [1])), np.concatenate(([taus[0]], taus, [taus[-1]])))
    rands = uniform.rvs(size=count, loc=0, scale=1)

    return func(rands)

# Returns Nx(tau, mu, thetas) array
def simulate_normal_model(means, serrs, count, taus=None, do_thetas=False):
    # Check if any deltas, and differ to it
    for ii in range(len(means)):
        if serrs[ii] == 0:
            if do_thetas:
                results = np.zeros((count, 2+len(means)))
                results[:,0] = 0
                results[:,1:] = means[ii]
            else:
                results = np.zeros((count, 2))
                results[:,0] = 0
                results[:,1] = means[ii]

            return results

    means = np.array(means, dtype=np.float_)
    varis = np.square(np.array(serrs, dtype=np.float_))

    if taus is None:
        taus = np.linspace(0, 2*max(serrs), 100)

    p_tau = np.array([p_tau_given_y(tau, means, varis) for tau in taus])
    F_tau = np.cumsum(p_tau)
    F_tau = F_tau / F_tau[-1]

    if do_thetas:
        results = np.zeros((count, 2+len(means)))
    else:
        results = np.zeros((count, 2))

    rands = get_random(taus, F_tau, count)

    for ii in range(count):
        tau = rands[ii]
        (vari_tau_sqrs, v_mu, mu_hat) = helper_params(means, varis, tau)
        mu = norm.rvs(size=1, loc=mu_hat, scale=math.sqrt(v_mu))

        results[ii, 0] = tau
        results[ii, 1] = mu

        if do_thetas:
            if tau == 0:
                vs = np.zeros((1, varis.size))
                theta_hats = mu * ones((1, varis.size))
            else:
                tau_sqr = tau*tau
                denoms = 1.0 / varis + 1.0 / tau_sqr
                vs = 1.0 / denoms
                theta_hats = (means / varis + mu / tau_sqr) / denoms

                thetas = norm.rvs(loc=theta_hats, scale=vs)

            results[ii, 2:] = thetas

    return results

def generate_thetas(mu_counts, mu_range, tau_counts, tau_range, count):
    thetas = []

    for ii in range(count):
        pval = random.uniform(0, 1)
        mu = draw_from_counts(mu_counts, mu_range, pval)
        tau = draw_from_counts(tau_counts, tau_range, pval)
        if tau <= 0:
            thetas.append(mu)
        else:
            thetas.append(float(norm.rvs(size=1, loc=mu, scale=tau)))

    return thetas

def draw_from_counts(x_counts, x_range, pval=None):
    sumcounts = sum(x_counts)
    if sumcounts == 0:
        return np.nan

    if pval is None:
        x = random.uniform(0, sum(x_counts))
    else:
        x = sumcounts * pval

    for ii in range(len(x_counts) - 1):
        if x < x_counts[ii]:
            return x_range[ii] - (x_range[ii+1] - x_range[ii]) / 2 + (x_range[ii+1] - x_range[ii]) * x / x_counts[ii]
        x -= x_counts[ii]

    return x_range[-1]

#results = simulate_normal_model([28, 8, -3, 7, -1, 1, 18, 12], [15, 10, 16, 11, 9, 11, 10, 18], 1000, do_thetas=False)
#print np.mean(results, axis=0)

