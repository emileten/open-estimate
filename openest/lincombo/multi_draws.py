import numpy as np
from scipy.stats._multivariate import multi_rv_frozen

class MultivariateDraws(multi_rv_frozen):
    def __init__(self, draws):
        self.draws = draws # np.array with rows of draws and cols of vars

    def rvs(self, size=1, random_state=None):
        if random_state is not None:
            np.random.set_state(random_state)
        return self.draws[np.random.randint(self.draws.shape[0]),]

    def mean(self):
        return np.mean(self.draws, axis=0)

    def std(self):
        return np.std(self.draws, axis=0)

