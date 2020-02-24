from .model import Model
import numpy as np

# Supports both multiple independent variables and models not specified only at points
class MultivariateModel(Model):
    # xx_is_categoricals is a boolean list with an element for each independent variables
    def __init__(self, xx_is_categoricals, scaled=True):
        super(MultivariateModel, self).__init__(scaled)
        
        self.xx_is_categoricals = xx_is_categoricals

    def numvars(self):
        return len(self.xx_is_categoricals)

    def condition(self, conditions):
        pass # override this!

    def default_condition(self):
        pass # override this!
