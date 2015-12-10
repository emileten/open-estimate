import numpy as np
#from models.multivariate_model import MultivariateModel
from scipy.stats._multivariate import multi_rv_frozen
from scipy.stats import uniform

#class MultivariateUniform(MultivariateModel, multi_rv_frozen):
class MultivariateUniform(multi_rv_frozen):
    def __init__(self, mins, maxs):
        #super(MultivariateUniform, self).__init__([False] * len(mins), True)
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

    def mins(self):
        return self.mins

    def maxs(self):
        return self.maxs

    # XXX: This doesn't handle size properly
    def rvs(self, size=1, random_state=None):
        return uniform.rvs(self.mins, self.maxs - self.mins, size=size, random_state=random_state)

    def pdf(self, xxs):
        for ii in range(len(xxs)):
            if xxs[ii] < self.mins[ii] or xxs[ii] > self.maxs[ii]:
                return 0
        return 1

