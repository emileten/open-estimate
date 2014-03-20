import numpy as np
from univariate_model import UnivariateModel
from scipy.interpolate import UnivariateSpline

class MemoizableUnivariate(object):
    #def get_index(self, x):
    #    raise NotImplementedError("get_index not implemented")

    # Only for a non-categorical
    def get_edges(self):
        return (np.concatenate(([float('-inf')], self.get_xx())) +
                np.concatenate((self.get_xx(), [float('inf')]))) / 2

    def eval_pval_index(self, ii, p, threshold=1e-3):
        raise NotImplementedError("eval_pval_index not implemented")

class MemoizedUnivariate(UnivariateModel):
    def __init__(self, model):
        super(MemoizedUnivariate, self).__init__(model.xx_is_categorical, model.xx, model.scaled)

        assert(isinstance(model, MemoizableUnivariate))
        
        self.model = model
        # Only for non-categorical
        self.x_cache_multiple = None
        self.reset_cache()

    def reset_cache(self):
        self.eval_pval_cache = {} # {ii => {p => result}}
        if self.model.xx_is_categorical:
            self.index_cache = {}
        else:
            if self.x_cache_multiple is not None:
                self.index_cache = {}
            else:
                self.index_cache = None

            self.edges = self.model.get_edges()

    def set_x_cache_decimals(self, decimals):
        self.x_cache_multiple = float(10**decimals)
        self.reset_cache()

    def get_index(self, x):
        if self.index_cache is None:
            return np.searchsorted(self.edges, x) - 1
        elif self.model.xx_is_categorical:
            index = self.index_cache.get(x, None)
            if index is None:
                index = self.model.get_xx().index(str(x)) if x is not None else 0
                self.index_cache[x] = index

            return index
        else:
            # Round to the given decimals
            xmul = int(x * self.x_cache_multiple)

            index = self.index_cache.get(xmul, None)
            if index is None:
                index = np.searchsorted(self.edges, xmul / self.x_cache_multiple) - 1
                self.index_cache[xmul] = index

            return index

    def get_indexes(self, xs):
        if self.index_cache is None:
            return np.searchsorted(self.edges, xs) - 1
        elif self.model.xx_is_categorical:
            return [self.get_index(x) for x in xs]
        else:
            # Round to the given decimals
            xmuls = (xs * self.x_cache_multiple).astype(int)

            indexes = []
            for xmul in xmuls:
                index = self.index_cache.get(xmul, None)
                if index is None:
                    index = min(np.searchsorted(self.edges, xmul / self.x_cache_multiple) - 1, len(self.edges) - 2)
                    self.index_cache[xmul] = index

                indexes.append(index)

            return indexes

    def kind(self):
        return self.model.kind

    def copy(self):
        return MemoizedUnivariate(self.model.copy())

    def get_xx(self):
        return self.xx

    def scale_y(self, a):
        self.model.scale_y(a)
        self.reset_cache()
        return self

    def scale_p(self, a):
        self.model.scale_p(a)
        self.reset_cache()
        return self

    def interpolate_x(self, newxx):
        return self.model.interpolate_x(newxx)

    def write_file(self, filename, delimiter):
        self.model.write_file(filename, delimiter)

    def write(self, file, delimiter):
        self.model.write(file, delimiter)

    def eval_pval(self, x, p, threshold=1e-3):
        ii = self.get_index(x)

        cache = self.eval_pval_cache.get(ii, None)
        if cache is None:
            self.eval_pval_cache[ii] = {}
            cache = {}

        cached = cache.get(p, None)
        if cached is not None:
            return cached

        y = self.model.eval_pval_index(ii, p, threshold)
        cache[p] = y

        return y

    def eval_pvals(self, xs, p, threshold=1e-3):
        iis = self.get_indexes(xs)

        ys = []
        for ii in iis:
            cache = self.eval_pval_cache.get(ii, None)
            if cache is None:
                self.eval_pval_cache[ii] = {}
                cache = {}

            y = cache.get(p, None)
            if y is None:
                y = self.model.eval_pval_index(ii, p, threshold)
                cache[p] = y

            ys.append(y)

        return ys
        
    def get_eval_pval_spline(self, p, limits, threshold=1e-3):
        xx = np.array(self.model.get_xx())
        yy = np.array(self.eval_pvals(xx, p, threshold))
        yy[np.isnan(yy)] = 0

        xx = np.concatenate(([limits[0]], xx, [limits[1]]))
        yy = np.concatenate(([yy[0]], yy, [yy[-1]]))
        spline = UnivariateSpline(xx, yy, s=0, k=1)
        return spline

        
