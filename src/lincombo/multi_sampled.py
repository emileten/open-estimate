import numpy as np
#from models.multivariate_model import MultivariateModel
from scipy.stats._multivariate import multi_rv_frozen

#class MultivariateSampled(MultivariateModel, multi_rv_frozen):
class MultivariateSampled(multi_rv_frozen):
    def __init__(self, func, dims):
        #super(MultivariateSampled, self).__init__([False] * dims, False)
        self.func = func
        self.dims = dims
        self.drawer = None

    def guess_ranges_gridded(self, mins, maxs, count=10000):
        samples = np.transpose(np.random.uniform(mins, maxs, (count, self.dims))) # [xxs, yys, ...]
        values = self.pdf(samples) # count values

        cutoff = max(values) / 100 # ignore anything less than this
        aboves = values >= cutoff

        # Draw in each dimension in turn
        slopes = []
        for ii in range(self.dims):
            mins[ii] = min(samples[ii][aboves])
            maxs[ii] = max(samples[ii][aboves])

            # What is the greatest slope?
            order = np.argsort(samples[ii])
            runs = np.diff(samples[ii][order])
            rises = np.diff(values[order])
            slopes.append(max(np.abs(rises) / runs))

        # Propose dimensions such that 'count' is partitioned across dimensions, so that slope_i / sum(slope_i) \propto log(count_i)
        propconstant = np.log(count)
        lens = map(int, np.exp(propconstant * slopes / sum(slopes)))

        return mins, maxs, lens

    def prepare_draws_gridded(self, mins, maxs, lens):
        ## DO NOT CALL THIS!  We can't use a gridded sampling method
        allvals = []
        for ii in range(self.dims):
            allvals.append(np.linspace(mins[ii], maxs[ii], lens[ii]))

        eachdimvals = np.meshgrid(*allvals)
        allpps = self.func(*eachdimvals)

        cumpps = allpps
        for ii in range(self.dims):
            cumpps = np.cumsum(cumpps, ii)

        self.drawer = (allvals, cumpps / max(cumpps))

    def guess_ranges(self, mins, maxs, count=10000):
        values = [0]

        while max(values) < 1e-10: # assume that this isn't intended
            samples = np.transpose(np.random.uniform(mins, maxs, (count, self.dims))) # [xxs, yys, ...]
            values = self.pdf(samples) # count values

        cutoff = max(values) / 1e4 # ignore anything less than this
        aboves = values >= cutoff

        # Draw in each dimension in turn
        post_mins = [0] * self.dims
        post_maxs = [0] * self.dims
        for ii in range(self.dims):
            post_mins[ii] = min(samples[ii][aboves])
            post_maxs[ii] = max(samples[ii][aboves])

        pre_area = np.array(maxs) - np.array(mins)
        post_area = np.array(post_maxs) - np.array(post_mins)

        if np.any(post_area / pre_area < .3):
            # Just limit one of dimension
            if len(np.where(post_area == 0)[0]) > 0:
                choices = np.where(post_area / pre_area < .3)[0]
                ii = choices[np.argmax(pre_area[choices])] # make sure we go through all axes
            else:
                ii = np.argmin(post_area / pre_area)

            center = (post_mins[ii] + post_maxs[ii]) / 2
            post_mins = mins
            post_maxs = maxs
            post_mins[ii] = max(mins[ii], center - (maxs[ii] - mins[ii]) / 3)
            post_maxs[ii] = min(maxs[ii], center + (maxs[ii] - mins[ii]) / 3)

            # Try it again!
            return self.guess_ranges(post_mins, post_maxs, count=count)

        return post_mins, post_maxs

    def prepare_draws(self, mins, maxs, count=10000):
        samples = np.transpose(np.random.uniform(mins, maxs, (count, self.dims))) # [xxs, yys, ...]
        values = self.pdf(samples) # count values

        cdfs = np.cumsum(values)

        self.drawer = (samples, cdfs / cdfs[-1])

    def rvs(self, size=1, random_state=None):
        if not self.drawer:
            raise ValueError("prepare_draws must be called before rvs.")

        samples, cdfs = self.drawer
        indexes = np.searchsorted(cdfs, np.random.uniform(size=size)) # will never be 1, so 0 - N-1

        values = []
        for ii in range(self.dims):
            values.append(samples[ii][indexes])

        return values

    def pdf(self, xxs):
        # Called with [xxs, yys, ...]
        return self.func(*xxs)

