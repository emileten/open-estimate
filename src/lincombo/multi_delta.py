import numpy as np
#from models.multivariate_model import MultivariateModel
from scipy.stats._multivariate import multi_rv_frozen
from scipy.stats import uniform

#class MultivariateDelta(MultivariateModel, multi_rv_frozen):
class MultivariateDelta(multi_rv_frozen):
    def __init__(self, vals):
        #super(MultivariateDelta, self).__init__([False] * len(mins), True)
        self.vals = vals

    def vals(self):
        return self.vals

    # Note: this doesn't handle size properly
    def rvs(self, size=1, random_state=None):
        return np.array(self.vals)

    def pdf(self, xxs):
        if np.all(xxs == self.vals):
            return np.inf
        else:
            return 0
