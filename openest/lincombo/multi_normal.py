#from models.multivariate_model import MultivariateModel
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multi_rv_frozen # can't use multivariate_normal_frozen, because can't reliably pass init arguments

#class MultivariateNormal(MultivariateModel, multi_rv_frozen):
class MultivariateNormal(multi_rv_frozen):
    def __init__(self, means, big_sigma):
        #super(MultivariateNormal, self).__init__([False] * len(means), True)
        self.means = means
        # Variance-covariance matrix
        self.big_sigma = big_sigma

    def rvs(self, size=1):
        return multivariate_normal.rvs(self.means, self.big_sigma, size=size)

    def pdf(self, xxs):
        return multivariate_normal.pdf(xxs, self.means, self.big_sigma)

    def logpdf(self, xxs):
        return multivariate_normal.logpdf(xxs, self.means, self.big_sigma)
