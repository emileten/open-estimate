from models.multivariate_model import MultivariateModel

class MultivariateNormal(MultivariateModel):
    def __init__(self, means, sigma):
        super(MultivariateNormal, self).__init__([False] * len(means), True)
        self.means = means
        self.sigma = sigma
        
    def means(self):
        return self.means

    def sigma(self):
        return self.sigma
    

