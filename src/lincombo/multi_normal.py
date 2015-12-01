#from models.multivariate_model import MultivariateModel
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multi_rv_frozen # can't use multivariate_normal_frozen, because can't reliably pass init arguments

#class MultivariateNormal(MultivariateModel, multi_rv_frozen):
class MultivariateNormal(multi_rv_frozen):
    def __init__(self, means, big_sigma):
        #super(MultivariateNormal, self).__init__([False] * len(means), True)
        self.means = means
        self.big_sigma = big_sigma

    def means(self):
        return self.means

    def big_sigma(self):
        """Variance-covariance matrix."""
        return self.big_sigma

    def rvs(self, size=1, random_state=None):
        return multivariate_normal.rvs(self.means, self.big_sigma, size=size, random_state=random_state)

    def pdf(self, xxs):
        return multivariate_normal.pdf(xxs, self.means, self.big_sigma)
