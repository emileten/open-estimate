import numpy as np
from scipy.stats import rv_continuous

class ContinuousSampled(rv_continuous):
    def __init__(self, func):
        self.func = func
        self.drawer = None

    def guess_ranges_gridded(self, mini, maxi, count=10000):
        samples = np.random.uniform(mini, maxi, count)
        values = self.pdf(samples) # count values

        cutoff = max(values) / 100 # ignore anything less than this
        aboves = values >= cutoff

        # Draw from this
        mini = min(samples[aboves])
        maxi = max(samples[aboves])

        return mini, maxi

    def prepare_draws_gridded(self, mini, maxi, count=10000):
        values = np.linspace(mini, maxi, count)

        allpps = self.func(values)

        cumpps = allpps
        for ii in range(1):
            cumpps = np.cumsum(cumpps, ii)

        self.drawer = (values, cumpps / max(cumpps))

    def guess_ranges(self, mini, maxi, count=10000):
        values = [0]

        while max(values) < 1e-10: # assume that this isn't intended
            samples = np.random.uniform(mini, maxi, count)
            values = np.array(self.pdf(samples))[:, 0] # count values

        cutoff = max(values) / 1e4 # ignore anything less than this
        aboves = values >= cutoff

        # Draw in each dimension in turn
        post_mini = min(samples[aboves])
        post_maxi = max(samples[aboves])

        pre_span = maxi - mini
        post_span = post_maxi - post_mini

        if post_span / pre_span < .3:
            center = (post_mini + post_maxi) / 2
            post_mini = max(mini, center - (maxi - mini) / 3)
            post_maxi = min(maxi, center + (maxi - mini) / 3)

            # Try it again!
            return self.guess_ranges(post_mini, post_maxi, count=count)

        return post_mini, post_maxi

    def prepare_draws(self, mini, maxi, count=10000):
        samples = np.random.uniform(mini, maxi, count)
        values = self.pdf(samples) # count values

        cdfs = np.cumsum(values)

        self.drawer = (samples, cdfs / cdfs[-1])

    def rvs(self, size=1, random_state=None):
        if not self.drawer:
            raise ValueError("prepare_draws must be called before rvs.")

        samples, cdfs = self.drawer
        indexes = np.searchsorted(cdfs, np.random.uniform(size=size)) # will never be 1, so 0 - N-1

        return samples[indexes]

    def pdf(self, xxs):
        return map(self.func, xxs)

